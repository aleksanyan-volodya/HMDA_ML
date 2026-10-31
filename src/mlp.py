"""MLP with categorical embeddings (TODO 7.24), a candidate for the blend.

Self-contained on purpose: numpy, pandas, scikit-learn and torch only, no project import, so this
file can be copied as it is into a notebook. It uses the GPU when there is one.

Same interface as the models of src/cv.py:
    model = MLPModel(cfg).fit(X, y, fold)     cfg: {"mlp": {...}, "model_seeds": [seed, ...]}
    model.predict(X)                          X: DataFrame; category dtype = embedded, the rest = numeric
fold holds the inner fold number of each row of X: the scored outer fold must not be in X.

Same nested scheme as the tree count of the boosted models: the number of epochs is chosen on an
inner fold, never on the scored fold. One inner fold only (not all of them): a net costs much more
than a booster. The net is then refitted on all of X for that number of epochs.
"""
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import QuantileTransformer
from torch import nn


class Encoder:
    """DataFrame -> integer codes + floats. Fitted on the rows the net trains on, nothing else.
      categorical -> code of its level; missing = a level of its own; levels with fewer than
                     min_level_rows rows share code 0 with unseen levels, so that code gets trained
                     (84 counties of test are absent from train) and a rare county cannot be memorised
      numeric     -> quantile-normal (neural nets train badly on skewed columns: loan amount goes up
                     to 375M for a median of 285k) + NaN filled with 0, the middle of that scale,
                     + a missing flag for each column that has NaN (missingness is predictive here)"""

    def __init__(self, min_level_rows: int):
        self.min_level_rows = min_level_rows

    @staticmethod
    def _text(s: pd.Series) -> pd.Series:
        return s.astype("string").fillna("missing")

    def _numbers(self, X: pd.DataFrame) -> pd.DataFrame:
        # ratios divide by income, which can be 0: infinities are treated as missing
        return X[self.num_].astype("float64").replace([np.inf, -np.inf], np.nan)

    def fit(self, X: pd.DataFrame) -> "Encoder":
        self.cat_ = [c for c in X.columns if isinstance(X[c].dtype, pd.CategoricalDtype)]
        self.num_ = [c for c in X.columns if c not in self.cat_]
        self.levels_ = {}
        for c in self.cat_:
            counts = self._text(X[c]).value_counts()
            self.levels_[c] = {v: i + 1 for i, v in enumerate(counts.index[counts >= self.min_level_rows])}
        num = self._numbers(X)
        self.flags_ = [c for c in self.num_ if num[c].isna().any()]
        self.quantiles_ = QuantileTransformer(n_quantiles=1000, output_distribution="normal", random_state=0).fit(num)
        return self

    def transform(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        codes = np.column_stack([self._text(X[c]).map(self.levels_[c]).fillna(0).to_numpy(dtype="int64")
                                 for c in self.cat_])
        num = self._numbers(X)
        floats = np.hstack([np.nan_to_num(self.quantiles_.transform(num)), num[self.flags_].isna().to_numpy()])
        return codes, floats.astype("float32")

    @property
    def n_levels(self) -> list[int]:
        return [len(self.levels_[c]) + 1 for c in self.cat_]      # + 1 = code 0, rare or unseen

    @property
    def n_floats(self) -> int:
        return len(self.num_) + len(self.flags_)


class Net(nn.Module):
    def __init__(self, n_levels: list[int], n_floats: int, hidden: list[int], dropout: float, emb_max_dim: int):
        super().__init__()
        # 1.6 * n^0.56: the usual rule of thumb (fastai), capped so county cannot take over the input
        sizes = [min(emb_max_dim, round(1.6 * n ** 0.56)) for n in n_levels]
        self.embeddings = nn.ModuleList(nn.Embedding(n, d) for n, d in zip(n_levels, sizes))
        self.input_dropout = nn.Dropout(dropout)
        layers, width = [], sum(sizes) + n_floats
        for h in hidden:
            layers += [nn.Linear(width, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            width = h
        self.body = nn.Sequential(*layers, nn.Linear(width, 1))

    def forward(self, codes: torch.Tensor, floats: torch.Tensor) -> torch.Tensor:
        embedded = [e(codes[:, i]) for i, e in enumerate(self.embeddings)]
        return self.body(torch.cat([self.input_dropout(torch.cat(embedded, dim=1)), floats], dim=1)).squeeze(1)


class MLPModel:
    def __init__(self, cfg: dict):
        self.p, self.seed = cfg["mlp"], cfg["model_seeds"][0]
        wanted = self.p["device"]
        self.device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if wanted == "auto" else wanted)

    def _tensors(self, encoder: Encoder, X: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
        codes, floats = encoder.transform(X)
        return torch.from_numpy(codes).to(self.device), torch.from_numpy(floats).to(self.device)

    def _logits(self, net: Net, codes: torch.Tensor, floats: torch.Tensor) -> torch.Tensor:
        net.eval()
        with torch.no_grad():
            parts = [net(c, f) for c, f in zip(codes.split(8192), floats.split(8192))]
        return torch.cat(parts)

    def _train(self, encoder: Encoder, X: pd.DataFrame, y: np.ndarray, epochs: int,
               X_val: pd.DataFrame | None = None, y_val: np.ndarray | None = None) -> tuple[Net, int]:
        """Trains for at most `epochs`, returns (net, best epoch). With a validation part, stops after
        `patience` epochs without a better AUC. The learning rate follows the same curve in both uses
        (a cosine over max_epochs, whatever `epochs` is), so the refit is the same training, stopped
        at the chosen epoch."""
        torch.manual_seed(self.seed)
        order = torch.Generator().manual_seed(self.seed)     # on the CPU: gives the same order on any device
        p = self.p
        net = Net(encoder.n_levels, encoder.n_floats, p["hidden"], p["dropout"], p["emb_max_dim"]).to(self.device)
        codes, floats = self._tensors(encoder, X)
        target = torch.as_tensor(y, dtype=torch.float32, device=self.device)
        valid = self._tensors(encoder, X_val) if X_val is not None else None
        steps = len(y) // p["batch_size"]                    # the last partial batch is dropped: batch norm
        optimizer = torch.optim.AdamW(net.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=p["max_epochs"] * steps)
        loss_fn = nn.BCEWithLogitsLoss()
        best_auc, best_epoch = -1.0, epochs
        for epoch in range(1, epochs + 1):
            t0 = time.perf_counter()
            net.train()
            total = 0.0
            for batch in torch.randperm(len(y), generator=order).to(self.device).split(p["batch_size"])[:steps]:
                optimizer.zero_grad()
                loss = loss_fn(net(codes[batch], floats[batch]), target[batch])
                loss.backward()
                optimizer.step()
                schedule.step()
                total += loss.item()
            line = f"    epoch {epoch:3d}/{epochs}  loss {total / steps:.4f}  lr {schedule.get_last_lr()[0]:.5f}"
            if valid is not None:
                auc = roc_auc_score(y_val, self._logits(net, *valid).cpu().numpy())
                if auc > best_auc:
                    best_auc, best_epoch = auc, epoch
                line += f"  inner AUC {auc:.5f}  best {best_auc:.5f} (epoch {best_epoch})"
            print(f"{line}  ({time.perf_counter() - t0:.1f}s)", flush=True)
            if valid is not None and epoch - best_epoch >= p["patience"]:
                print(f"    stopped: no better AUC for {p['patience']} epochs", flush=True)
                break
        return net, best_epoch

    def fit(self, X: pd.DataFrame, y: pd.Series, fold: np.ndarray) -> "MLPModel":
        y = np.asarray(y, dtype="float32")
        inner = fold == np.unique(fold)[0]
        print(f"  choosing the number of epochs on inner fold {np.unique(fold)[0]} ({self.device})", flush=True)
        encoder = Encoder(self.p["min_level_rows"]).fit(X[~inner])
        _, self.n_epochs_ = self._train(encoder, X[~inner], y[~inner], self.p["max_epochs"], X[inner], y[inner])
        print(f"  refit on {len(X)} rows for {self.n_epochs_} epochs", flush=True)
        self.encoder_ = Encoder(self.p["min_level_rows"]).fit(X)
        self.net_, _ = self._train(self.encoder_, X, y, self.n_epochs_)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        logits = self._logits(self.net_, *self._tensors(self.encoder_, X)).double()   # double: no exact 0 or 1
        return torch.sigmoid(logits).cpu().numpy()
