"""Generated tables and figures of reports/report.md, from logged runs only.

    uv run --locked python reports/figs/make_figures.py               # every step
    uv run --locked python reports/figs/make_figures.py slices shap   # some steps

Reads data/raw, oof/<run_id>.parquet, results/runs.csv and reports/submissions.md, writes
reports/figs/, and pastes each generated table into reports/report.md between its
"<!-- begin name -->" and "<!-- end name -->" markers. Trains nothing, except the "shap" step:
one LightGBM fit with the tuned settings on folds 1-4 (about a minute), because importance
needs a model and only predictions are logged.
"""
import json
import re
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from compare import _run, fold_diff, weighted_auc  # noqa: E402
from config import load_config  # noqa: E402
from models import LGBMModel  # noqa: E402
from data import CODED, FEATURES, TARGET, load  # noqa: E402
from features import BLOCKS, LTV_EDGES, LTV_LEVELS, build  # noqa: E402
from folds import load_folds  # noqa: E402

FIGS = ROOT / "reports" / "figs"
REPORT = ROOT / "reports" / "report.md"

RUNS = {  # display name -> run_id (reports/models.md)
    "final blend": "20261031T220901Z_1ff742f2",
    "LightGBM": "20261003T084548Z_406e4210",
    "CatBoost": "20261014T085450Z_d96c8be4",
    "XGBoost": "20261031T215118Z_05b22e57",
    "ExtraTrees": "20261019T060754Z_7bc88a0b",
    "MLP": "20261031T130048Z_752fd022",
    "logistic L1": "20261018T172507Z_6c8c7142",
}
N_BOOT = 1000       # resamples for every bootstrap interval, as in src/compare.py
MIN_SLICE = 1000    # a slice smaller than this is listed but not read (fixed before looking)
SEED = 0

# Colors: reference categorical palette, slots 1-4, validated with the dataviz validator
# (light mode); grey = "not read" / reference marks; text in ink colors, never series colors.
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GREY, INK, INK2, GRID = "#a3a29c", "#0b0b0b", "#52514e", "#e6e5e0"
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight", "font.size": 8.5,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False,
})


# ---------- helpers ----------

def read_oof(run_id: str) -> pd.DataFrame:
    return pd.read_parquet(ROOT / "oof" / f"{run_id}.parquet")


def boot_auc_ci(y: np.ndarray, p: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED) -> tuple[float, float]:
    """95% stratified bootstrap interval of the AUC: positives and negatives resampled apart,
    so every resample keeps the slice's class counts (as src/bootstrap_ci.py)."""
    auc = weighted_auc(y, p)
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    stats = [auc(np.bincount(np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))],
                             minlength=len(y)).astype(float)) for _ in range(n_boot)]
    return tuple(np.percentile(stats, [2.5, 97.5]))


def md_table(df: pd.DataFrame) -> str:
    cells = [[str(c) for c in df.columns]] + [["" if pd.isna(v) else str(v) for v in row]
                                              for row in df.itertuples(index=False)]
    lines = ["| " + " | ".join(r) + " |" for r in cells]
    lines.insert(1, "|" + "---|" * len(df.columns))
    return "\n".join(lines)


def publish(name: str, md: str) -> None:
    """Write reports/figs/<name>.md and paste it into report.md between its markers."""
    (FIGS / f"{name}.md").write_text(md + "\n")
    text = REPORT.read_text()
    pattern = re.compile(rf"(<!-- begin {name} -->\n).*?(<!-- end {name} -->)", re.S)
    if not pattern.search(text):
        print(f"  {name}: no markers in report.md, table left in reports/figs/{name}.md")
        return
    REPORT.write_text(pattern.sub(lambda m: m.group(1) + md + "\n" + m.group(2), text))


def save(fig, name: str) -> None:
    fig.savefig(FIGS / f"{name}.png")
    plt.close(fig)


def fmt(x: float, d: int = 3) -> str:
    return f"{x:.{d}f}"


# ---------- section 2-3: data ----------

def data_facts() -> None:
    """T2: the data facts the report relies on, recomputed from the raw files."""
    train, test = load()
    y = train[TARGET]
    is_1111 = train[CODED].astype("string") == "1111"
    money = pd.concat([train[["loan_amount", "property_value"]], test[["loan_amount", "property_value"]]])
    on_grid = (money.stack().dropna() % 10000 == 5000).mean()  # stack keeps NaN in recent pandas
    unknown_tr, unknown_te = train.tract_population == 0, test.tract_population == 0
    test_only = set(test.county_code.dropna()) - set(train.county_code.dropna())
    rows = [
        ("train rows, test rows, input columns", f"{len(train):,}, {len(test):,}, {len(FEATURES)}"),
        ("accepted share in train", f"{y.mean():.3f} ({y.sum():,} accepted, {(1 - y).sum():,} denied)"),
        ("train ids / test ids", f"{train.id.min():,} to {train.id.max():,} / {test.id.min():,} to "
                                 f"{test.id.max():,}, shared ids {len(set(train.id) & set(test.id))}"),
        ("AUC of the row id alone (no order leak if ~0.5)", fmt(roc_auc_score(y, train.id), 4)),
        ("train rows with code 1111 (Exempt): in any coded column / in all affected columns",
         f"{is_1111.any(axis=1).sum():,} / {(is_1111.sum(axis=1) == is_1111.sum(axis=1).max()).sum():,} "
         f"({is_1111.sum(axis=1).max()} columns), accepted {y[is_1111.any(axis=1)].mean():.3f}"),
        ("loan amount and property value on the $10k midpoint grid (v mod 10,000 = 5,000), both files",
         f"{on_grid:.4f} of non-blank values"),
        ("location unknown (census block all 0): train / test rows, train accept rate",
         f"{unknown_tr.sum():,} / {unknown_te.sum():,}, {y[unknown_tr].mean():.3f}"),
        ("derived_msa-md = 99999 (outside any MSA): train rows, accept rate",
         f"{(train['derived_msa-md'] == '99999').sum():,}, {y[train['derived_msa-md'] == '99999'].mean():.3f}"),
        ("county levels in test but not train, test rows affected",
         f"{len(test_only)}, {test.county_code.isin(test_only).sum()} ({test.county_code.isin(test_only).mean():.2%})"),
        ("exact duplicate rows in train over the 56 inputs", f"{train.duplicated(FEATURES).sum()}"),
    ]
    publish("t2_data_facts", md_table(pd.DataFrame(rows, columns=["fact", "value"])))


def missingness() -> None:
    """T3 + F1: blanks per column, train vs test, and the accept rate with and without a value."""
    train, test = load()
    y = train[TARGET]
    rows = []
    for c in FEATURES:
        miss = train[c].isna()
        rows.append({"column": c, "train blank %": 100 * miss.mean(), "test blank %": 100 * test[c].isna().mean(),
                     "blank rows (train)": int(miss.sum()),
                     "accept if blank": y[miss].mean() if miss.any() else np.nan, "accept if present": y[~miss].mean()})
    t = pd.DataFrame(rows).sort_values("train blank %", ascending=False)
    t["gap, points"] = t["test blank %"] - t["train blank %"]
    t.to_csv(FIGS / "t3_missingness.csv", index=False)
    shown = t[t["blank rows (train)"] > 0].copy()
    none = len(t) - len(shown)
    for c in ["train blank %", "test blank %", "gap, points"]:
        shown[c] = shown[c].map(lambda v: f"{v:.2f}")
    for c in ["accept if blank", "accept if present"]:
        shown[c] = shown[c].map(fmt)
    shown["blank rows (train)"] = shown["blank rows (train)"].map(lambda v: f"{v:,}")
    cols = ["column", "train blank %", "test blank %", "gap, points", "blank rows (train)",
            "accept if blank", "accept if present"]
    md = md_table(shown[cols])
    md += (f"\n\n{none} of the {len(t)} columns have no blank in train. Largest train-test gap: "
           f"{t['gap, points'].abs().max():.2f} points. Blank = empty in the file; codes such as 1111 "
           f"and census zeros are values here (T2).")
    publish("t3_missingness", md)

    # F1: dumbbell, one row per column with > 0.5% blank: present (blue) vs blank (orange).
    f = t[t["train blank %"] > 0.5].sort_values("accept if blank")
    fig, ax = plt.subplots(figsize=(6.4, 0.32 * len(f) + 0.9))
    yy = np.arange(len(f))
    ax.hlines(yy, f["accept if blank"], f["accept if present"], color=GRID, lw=2, zorder=1)
    ax.scatter(f["accept if present"], yy, s=36, color=C1, label="value present", zorder=2,
               edgecolor="white", linewidth=1)
    ax.scatter(f["accept if blank"], yy, s=36, color=C2, label="blank", zorder=2, edgecolor="white", linewidth=1)
    ax.set_yticks(yy, [f"{c}  (n blank {n:,})" for c, n in zip(f.column, f["blank rows (train)"])])
    ax.axvline(y.mean(), color=GREY, lw=1, ls="--")
    ax.text(y.mean(), len(f) - 0.4, f" all rows {y.mean():.3f}", color=INK2, fontsize=7.5, va="bottom")
    ax.set_xlabel("share accepted (train)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower left", ncols=2, bbox_to_anchor=(0, 1.0))
    ax.set_title("Accept rate when a column is blank vs present", loc="left", fontsize=9.5, pad=18)
    save(fig, "f1_missing_accept")


def states() -> None:
    """F2: accept rate by state with a 95% Wilson interval; states under MIN_SLICE rows in grey."""
    train, _ = load()
    # Blank state (824 rows, accepted 0.10) is left out: it is F1's row and would squeeze the axis.
    g = train.groupby(train.state_code.astype("string"))[TARGET].agg(["mean", "size"])
    g = g.sort_values("mean")
    z = 1.96
    n, p = g["size"], g["mean"]
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    read = n >= MIN_SLICE
    fig, ax = plt.subplots(figsize=(6.4, 0.16 * len(g) + 1.0))
    yy = np.arange(len(g))
    for mask, color, label in [(read, C1, f"n >= {MIN_SLICE:,}"), (~read, GREY, f"n < {MIN_SLICE:,}, not read")]:
        ax.errorbar(p[mask], yy[mask], xerr=[p[mask] - (centre[mask] - half[mask]), centre[mask] + half[mask] - p[mask]],
                    fmt="o", ms=3.5, color=color, ecolor=color, elinewidth=1, label=label)
    ax.set_yticks(yy, [f"{s} ({k:,})" for s, k in zip(g.index, n)], fontsize=6.5)
    ax.axvline(train[TARGET].mean(), color=GREY, lw=1, ls="--")
    ax.set_xlim(0.6, 1.0)
    ax.set_xlabel("share accepted (train), 95% Wilson interval")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper left")
    ax.set_title("Accept rate by state (rows in brackets; blank state in F1)", loc="left", fontsize=9.5)
    save(fig, "f2_state_accept")


# ---------- section 5: metric and validation ----------

def transforms() -> None:
    """T5: strictly increasing transforms keep the AUC, while log loss and Brier move."""
    o = read_oof(RUNS["final blend"])
    y, p = o[TARGET].to_numpy(), o.pred.to_numpy()
    logit = np.log(p / (1 - p))
    variants = {
        "as predicted, p": p, "p squared": p ** 2, "square root of p": np.sqrt(p),
        "rank / (n + 1)": rankdata(p) / (len(p) + 1), "sharpened, sigmoid(2 logit p)": 1 / (1 + np.exp(-2 * logit)),
    }
    rows = [{"scores": k, "AUC": f"{roc_auc_score(y, v):.6f}", "log loss": fmt(log_loss(y, v), 4),
             "Brier": fmt(brier_score_loss(y, v), 4)} for k, v in variants.items()]
    md = md_table(pd.DataFrame(rows))
    md += ("\n\nOut-of-fold predictions of the final blend (run 1ff742f2). Every transform is strictly "
           "increasing and keeps scores in (0, 1), so log loss and Brier are defined for all of them.")
    publish("t5_transforms", md)


def lb_noise() -> None:
    """T6: SD of a leaderboard score when it is computed on a random share of 45,000 test rows,
    simulated by drawing that many out-of-fold rows; and SD of the paired gap between two models."""
    both = read_oof(RUNS["final blend"]).merge(read_oof(RUNS["LightGBM"])[["id", "pred"]], on="id",
                                               suffixes=("", "_lgbm"), validate="1:1")
    y = both[TARGET].to_numpy()
    auc_f, auc_l = weighted_auc(y, both.pred.to_numpy()), weighted_auc(y, both.pred_lgbm.to_numpy())
    rng = np.random.default_rng(SEED)
    rows = []
    for share in [0.2, 0.3, 0.5, 1.0]:
        k = int(45000 * share)
        a, d = [], []
        for _ in range(N_BOOT):
            w = np.zeros(len(y))
            w[rng.choice(len(y), k, replace=False)] = 1.0
            fa = auc_f(w)
            a.append(fa)
            d.append(fa - auc_l(w))
        rows.append({"share of test": f"{share:.0%}", "rows": f"{k:,}", "SD of one score": np.std(a, ddof=1),
                     "SD of a paired gap (final blend - LightGBM)": np.std(d, ddof=1)})
    t = pd.DataFrame(rows)
    t.to_csv(FIGS / "t6_lb_noise.csv", index=False)
    for c in t.columns[2:]:
        t[c] = t[c].map(lambda v: f"{v:.4f}")
    md = md_table(t) + (f"\n\n{N_BOOT:,} draws without replacement from the 105,000 out-of-fold rows per "
                        "share. The public leaderboard is the 30% row; the private ranking uses 70%.")
    publish("t6_lb_noise", md)


def cv_lb() -> None:
    """F3: CV against public leaderboard for every model submission, with +-2 SD of a public score."""
    rows = [l.split("|") for l in (ROOT / "reports" / "submissions.md").read_text().splitlines()
            if re.match(r"\| \d+ \|", l)]
    s = pd.DataFrame({"n": [int(r[1]) for r in rows], "cv": [float(r[6]) for r in rows],
                      "lb": [float(r[8]) for r in rows], "config": [r[5].strip().strip("`") for r in rows]})
    s = s[s.cv > 0.5]  # submission 1 is the constant check (0.5 / 0.5)
    sd = pd.read_csv(FIGS / "t6_lb_noise.csv").set_index("share of test").loc["30%", "SD of one score"]
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    lo, hi = s[["cv", "lb"]].min().min() - 2 * sd - 0.001, s[["cv", "lb"]].max().max() + 2 * sd + 0.001
    ax.plot([lo, hi], [lo, hi], color=GREY, lw=1, ls="--")
    ax.text(hi - 0.0005, hi - 0.0012, "LB = CV", color=INK2, ha="right", fontsize=7.5)
    # Fair models (race-blind, reweighed) are told apart by their config name.
    fair = s.config.str.contains("fair_|no_demographics")
    for mask, col, lab in [(~fair, C1, "competition models"), (fair, C2, "fair models")]:
        ax.errorbar(s.cv[mask], s.lb[mask], yerr=2 * sd, fmt="o", ms=5, color=col, ecolor=col,
                    elinewidth=1, capsize=0, label=f"{lab} ({', '.join(map(str, s.n[mask]))})")
    ax.legend(loc="lower right", fontsize=7.5, frameon=False)
    # Clusters (3, 7, 8), (4, 5, 6) and (9, 10) sit within 0.0015 of each other: labels are spread by hand.
    offsets = {4: (-12, 2), 5: (5, -11), 6: (6, 2), 3: (6, -8), 7: (-3, 7), 8: (-12, -3), 9: (6, -1), 10: (-15, -4)}
    for r in s.itertuples():
        ax.annotate(str(r.n), (r.cv, r.lb), color=INK, fontsize=8, xytext=offsets.get(r.n, (6, -3)),
                    textcoords="offset points")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("CV AUC, mean of 5 folds")
    ax.set_ylabel("public leaderboard AUC (30% of test)")
    ax.set_title(f"Submissions 2 to {s.n.max()}: bars = +-2 SD of a public score ({sd:.4f}, T6)",
                 loc="left", fontsize=8.5)
    save(fig, "f3_cv_lb")


# ---------- section 6: models ----------

def roc() -> None:
    """F4: ROC curves, full range and a zoom on the part where the models differ."""
    shown = ["final blend", "LightGBM", "CatBoost", "logistic L1"]
    colors = [C1, C2, C3, C4]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.8))
    for name, color in zip(shown, colors):
        o = read_oof(RUNS[name])
        fpr, tpr, _ = roc_curve(o[TARGET], o.pred)
        auc = roc_auc_score(o[TARGET], o.pred)
        for ax in axes:
            ax.plot(fpr, tpr, color=color, lw=1.6, label=f"{name}  {auc:.4f}")
    axes[0].plot([0, 1], [0, 1], color=GREY, lw=1, ls="--")
    axes[0].set_title("Full range", loc="left", fontsize=9)
    axes[1].set_xlim(0, 0.4)
    axes[1].set_ylim(0.6, 1.0)
    axes[1].set_title("Zoom: false positive rate <= 0.4", loc="left", fontsize=9)
    for ax in axes:
        ax.set_xlabel("false positive rate (denied rows scored above the cut)")
    axes[0].set_ylabel("true positive rate")
    axes[0].legend(loc="lower right", title="pooled out-of-fold AUC", title_fontsize=7.5, fontsize=7.5)
    save(fig, "f4_roc")


def rank_corr() -> None:
    """F5: Spearman correlation between the models' out-of-fold predictions."""
    preds = pd.DataFrame({k: read_oof(v).set_index("id").pred for k, v in RUNS.items()})
    c = preds.corr(method="spearman")
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    im = ax.imshow(c, cmap="Blues", vmin=0.8, vmax=1.0)
    ax.set_xticks(range(len(c)), c.columns, rotation=40, ha="right")
    ax.set_yticks(range(len(c)), c.index)
    for i in range(len(c)):
        for j in range(len(c)):
            ax.text(j, i, f"{c.iloc[i, j]:.3f}", ha="center", va="center", fontsize=7.5,
                    color="white" if c.iloc[i, j] > 0.93 else INK)
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Spearman rank correlation (scale from 0.8)")
    ax.set_title("How alike the models rank the 105,000 training rows", loc="left", fontsize=9)
    save(fig, "f5_rank_corr")


# ---------- section 7: ablations ----------

FULL = "20261003T084548Z_406e4210"
ABLATIONS = [  # (row of T9 / T12, what changes, base run, candidate run)
    ("A1", "drop the 17 demographic fields", FULL, "20261110T192232Z_20a010b4"),
    ("A2", "drop the 4 process fields", FULL, "20261110T192704Z_bc281e8f"),
    ("A3", "drop the 7 census columns", FULL, "20261110T192832Z_29f2ea4d"),
    ("A4c", "drop county rate, L1 logistic", "20261018T172507Z_6c8c7142", "20261110T212542Z_1dc8e77b"),
    ("A5", "raw typed columns only", FULL, "20261110T193003Z_fcc2e23f"),
    ("H1-H3", "add the 3 worst-error flags", FULL, "20261110T200644Z_ebb1bd30"),
    ("A4a", "add county rate, LightGBM", "20261002T141818Z_f7f2dc68", "20261002T143948Z_1114e40c"),
    ("A4b", "drop raw county / MSA ids, CatBoost", "20261014T085450Z_d96c8be4", "20261014T001640Z_6deb776a"),
    ("A6a", "LightGBM -> 50/50 blend", "20261003T084548Z_406e4210", "20261014T115111Z_918eb62b"),
    ("A6a", "LightGBM -> final blend", "20261003T084548Z_406e4210", "20261031T220901Z_1ff742f2"),
    ("A7", "starting -> tuned settings", "20261002T141818Z_f7f2dc68", "20261003T084548Z_406e4210"),
    ("A9", "drop the 2 leaks", "20261003T084548Z_406e4210", "20261109T170831Z_116f6267"),
    ("A10", "strict list (vs clean)", "20261109T170831Z_116f6267", "20261109T170956Z_cd2c2a46"),
]


def ablations() -> None:
    """F6: paired gain +- 2 corrected SE for every measured row of T9; two panels because the
    large losses are ~50x larger than the rest and would flatten them on one scale."""
    rows = []
    for tag, label, base, cand in ABLATIONS:
        d = fold_diff(_run(base), _run(cand))
        rows.append({"tag": tag, "label": f"{tag}  {label}", "gain": d["mean_diff"], "se": d["corrected_se"],
                     "up": int((d["fold_diffs"] > 0).sum())})
    t = pd.DataFrame(rows)
    t.to_csv(FIGS / "f6_ablations.csv", index=False)
    big = t.gain.abs() > 0.01
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 6.2), gridspec_kw={"height_ratios": [(~big).sum(), big.sum()],
                                                                   "hspace": 0.45})
    for ax, part, title in [(axes[0], t[~big], "Changes under 0.01 AUC"), (axes[1], t[big], "Losses over 0.01 AUC")]:
        yy = np.arange(len(part))[::-1]
        colors = [C1 if g > 0 else C2 for g in part.gain]
        ax.barh(yy, part.gain, height=0.55, color=colors)
        ax.errorbar(part.gain, yy, xerr=2 * part.se, fmt="none", ecolor=INK, elinewidth=1, capsize=2.5)
        ax.axvline(0, color=INK2, lw=0.8)
        ax.set_yticks(yy, part.label)
        ax.grid(axis="y", visible=False)
        ax.set_title(title, loc="left", fontsize=9)
    axes[1].set_xlabel("paired CV gain in AUC, error bar = 2 corrected SE (the adoption threshold)")
    save(fig, "f6_ablations")


# ---------- section 8: error analysis ----------

def slice_labels(train: pd.DataFrame) -> dict[str, pd.Series]:
    """The slices fixed for the error analysis; each maps a row to a level (NaN = not in the slice)."""
    s = train.state_code.astype("string")
    top = s.value_counts().index[:15]
    aus_names = {"1": "1 DU", "2": "2 LP / LPA", "3": "3 TOTAL", "4": "4 GUS", "5": "5 other",
                 "6": "6 no automated system", "7": "7 internal system", "1111": "1111 exempt"}
    inc = train.income
    q = pd.qcut(inc, 5)
    inc_label = pd.Series(
        [f"Q{i + 1} ({iv.left:.0f}k to {iv.right:.0f}k)" if isinstance(iv, pd.Interval) else "blank"
         for i, iv in zip(q.cat.codes, q)], index=train.index)
    ltv = pd.Categorical(  # ordered, so the table and F7 list the bands low to high
        pd.cut(train.loan_to_value_ratio, LTV_EDGES, labels=LTV_LEVELS[:-1]).astype("string").fillna("blank"),
        categories=[*LTV_LEVELS[:-1], "blank"])
    ltv = pd.Series(ltv, index=train.index)
    exempt = (train[CODED].astype("string") == "1111").any(axis=1).map({True: "exempt (1111)", False: "not exempt"})
    blank = {"LTV": train.loan_to_value_ratio.isna(), "property value": train.property_value.isna(),
             "DTI": train.debt_to_income_ratio.isna(), "income": train.income.isna(),
             "loan term": train.loan_term.isna(), "location": train.tract_population == 0}
    pattern = pd.DataFrame(blank).apply(lambda r: " + ".join(k for k, v in r.items() if v) or "nothing blank",
                                        axis=1)
    return {
        "state (15 largest)": s.where(s.isin(top)),
        "loan product": train.derived_loan_product_type.astype("string"),
        "aus-1": train["aus-1"].astype("string").map(aus_names),
        "income quintile": inc_label,
        "LTV band (reported)": ltv,
        "exempt record": exempt,
        "blank pattern (LTV, property value, DTI, income, term, location)": pattern,
    }


def slices() -> None:
    """T10 + F7: within-slice AUC and calibration of the final blend."""
    train, _ = load()
    o = read_oof(RUNS["final blend"])
    df = train.merge(o[["id", "pred"]], on="id", validate="1:1")
    y_all, p_all = df[TARGET].to_numpy(), df.pred.to_numpy()
    pooled = roc_auc_score(y_all, p_all)
    rows = []
    for sname, lab in slice_labels(df).items():
        for level, idx in lab.dropna().groupby(lab.dropna(), observed=True, sort=True).groups.items():
            y, p = y_all[df.index.get_indexer(idx)], p_all[df.index.get_indexer(idx)]
            n, acc = len(y), y.mean()
            r = {"slice": sname, "level": level, "n": n, "denied": int((y == 0).sum()), "accept rate": acc,
                 "mean prediction": p.mean(), "residual": acc - p.mean(),
                 "residual SE": np.sqrt(acc * (1 - acc) / n), "read": n >= MIN_SLICE,
                 "AUC": np.nan, "AUC low": np.nan, "AUC high": np.nan}
            if r["read"] and 0 < acc < 1:
                r["AUC"] = roc_auc_score(y, p)
                r["AUC low"], r["AUC high"] = boot_auc_ci(y, p)
            rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(FIGS / "t10_slices.csv", index=False)

    shown = t[t.read].copy()
    shown["AUC [95% CI]"] = [f"{a:.3f} [{lo:.3f}, {hi:.3f}]" if pd.notna(a) else "undefined (one class)"
                             for a, lo, hi in zip(shown.AUC, shown["AUC low"], shown["AUC high"])]
    shown["residual"] = [f"{r:+.3f} ({se:.3f})" for r, se in zip(shown.residual, shown["residual SE"])]
    for c in ["accept rate", "mean prediction"]:
        shown[c] = shown[c].map(fmt)
    for c in ["n", "denied"]:
        shown[c] = shown[c].map(lambda v: f"{v:,}")
    unread = t[~t.read].groupby("slice").agg(levels=("n", "size"), rows=("n", "sum"))
    md = md_table(shown[["slice", "level", "n", "denied", "accept rate", "mean prediction",
                         "residual", "AUC [95% CI]"]].rename(columns={"residual": "residual (SE)"}))
    md += (f"\n\nFinal blend, run 1ff742f2, out-of-fold; pooled AUC {pooled:.4f}. Residual = accept rate - mean "
           f"prediction. AUC = ranking inside the slice only, with a {N_BOOT:,}-resample stratified bootstrap "
           f"interval. Levels under {MIN_SLICE:,} rows are not read: "
           + "; ".join(f"{s}: {r.levels} levels, {r.rows:,} rows" for s, r in unread.iterrows()) + ".")
    publish("t10_slices", md)

    # F7: forest plot of the read levels, grouped by slice.
    f = t[t.read & t.AUC.notna()]
    labels, ys, y0 = [], [], 0
    fig_h = 0.17 * (len(f) + 2 * f.slice.nunique()) + 1
    fig, ax = plt.subplots(figsize=(6.6, fig_h))
    for sname, part in f.groupby("slice", sort=False):
        ax.text(0.5, y0, sname, transform=ax.get_yaxis_transform(), ha="center", fontsize=7.5,
                color=INK, weight="bold")
        y0 += 1
        for level, n, auc, lo, hi in zip(part.level, part.n, part.AUC, part["AUC low"], part["AUC high"]):
            ax.errorbar(auc, y0, xerr=[[auc - lo], [hi - auc]], fmt="o", ms=3.5, color=C1, ecolor=C1, elinewidth=1)
            labels.append((y0, f"{level}  (n {n:,})"))
            y0 += 1
        y0 += 0.6
    ax.set_yticks([a for a, _ in labels], [b for _, b in labels], fontsize=6.5)
    ax.invert_yaxis()
    ax.axvline(pooled, color=GREY, lw=1, ls="--")
    ax.text(pooled, -1.2, f"pooled {pooled:.4f}", color=INK2, fontsize=7, ha="center")
    ax.set_xlabel("AUC inside the slice, 95% bootstrap interval")
    ax.grid(axis="y", visible=False)
    save(fig, "f7_slices")


def worst_errors() -> None:
    """T11 (evidence for T12). (a) The 200 worst false positives (denied, highest scores) and false negatives
    (accepted, lowest scores) against the rows of the other class in the same score range, so a
    difference is something the model saw and still could not use. (b) For each candidate feature
    the reading suggested, the residual with it minus the residual without it, inside the region where
    the errors sit: a feature can only add ranking where the model leaves such a difference."""
    train, _ = load()
    df = train.merge(read_oof(RUNS["final blend"])[["id", "pred"]], on="id", validate="1:1")
    s = lambda c: df[c].astype("string")  # noqa: E731
    term_off_grid = (df.loan_term % 12 != 0) & df.loan_term.notna()
    ltv_blank_value_present = df.loan_to_value_ratio.isna() & df.property_value.notna()
    computed_ltv_above_100 = df.loan_amount > df.property_value
    flags = {
        "manufactured home": s("construction_method") == "2",
        "location unknown": df.tract_population == 0,
        "LTV blank": df.loan_to_value_ratio.isna(),
        "no automated underwriting (aus-1 = 6)": s("aus-1") == "6",
        "DTI > 60%": s("debt_to_income_ratio") == ">60%",
        "conventional loan": s("loan_type") == "1",
        "exempt record (1111)": (df[CODED].astype("string") == "1111").any(axis=1),
        "submitted directly to the lender": s("submission_of_application") == "1",
        "initially payable to the lender": s("initially_payable_to_institution") == "1",
        "single male applicant (derived_sex Male)": s("derived_sex") == "Male",
        "applicant age > 74": s("applicant_age") == ">74",
        "H1 term off the 12-month grid": term_off_grid,
        "H2 LTV blank, property value present": ltv_blank_value_present,
        "H3 computed LTV > 100": computed_ltv_above_100,
    }
    flags = {k: v.fillna(False).astype(bool) for k, v in flags.items()}
    neg, pos = df[TARGET] == 0, df[TARGET] == 1
    fp, fn = df[neg].nlargest(200, "pred").index, df[pos].nsmallest(200, "pred").index
    fp_ref = df.index[pos & df.pred.between(df.pred[fp].min(), df.pred[fp].max())]
    fn_ref = df.index[neg & df.pred.between(df.pred[fn].min(), df.pred[fn].max())]
    pd.concat([df.loc[fp].assign(error="false positive"), df.loc[fn].assign(error="false negative")]).to_csv(
        FIGS / "t11_worst_rows.csv", index=False)

    def z(err, ref, f):  # share in the 200 errors against the binomial SE of the reference share
        p = f[ref].mean()
        return f[err].mean(), p, (f[err].mean() - p) / np.sqrt(max(p * (1 - p), 1e-6) / len(err))

    rows = []
    for name, f in flags.items():
        a, b, za = z(fp, fp_ref, f)
        c, d, zc = z(fn, fn_ref, f)
        rows.append({"rows with": name, "worst FP": fmt(a), "accepted, same scores": fmt(b), "z FP": f"{za:+.1f}",
                     "worst FN": fmt(c), "denied, same scores": fmt(d), "z FN": f"{zc:+.1f}"})
    md = md_table(pd.DataFrame(rows))
    md += (f"\n\nShares of rows. Worst FP: the 200 denied rows with the highest scores ({df.pred[fp].min():.3f} to "
           f"{df.pred[fp].max():.3f}), against the {len(fp_ref):,} accepted rows in that range; worst FN: the 200 "
           f"accepted rows with the lowest scores ({df.pred[fn].min():.3f} to {df.pred[fn].max():.3f}), against the "
           f"{len(fn_ref):,} denied rows in that range. Final blend, run 1ff742f2. The rows: "
           "reports/figs/t11_worst_rows.csv.")

    r = df[TARGET] - df.pred
    regions = {"low scores: manufactured or no AUS": flags["manufactured home"] | flags["no automated underwriting (aus-1 = 6)"],
               "manufactured homes only": flags["manufactured home"], "high scores: prediction > 0.95": df.pred > 0.95}
    screen = []
    for cname in [k for k in flags if k.startswith("H")]:
        for rname, reg in regions.items():
            a, b = r[reg & flags[cname]], r[reg & ~flags[cname]]
            d = a.mean() - b.mean()
            screen.append({"candidate": cname, "region": rname, "rows with": f"{len(a):,}",
                           "residual with": f"{a.mean():+.4f}", "residual without": f"{b.mean():+.4f}",
                           "difference": f"{d:+.4f}", "z": f"{d / np.sqrt(a.var() / len(a) + b.var() / len(b)):+.1f}"})
    md += ("\n\nResidual (accepted - prediction) with and without each candidate, inside the regions where the "
           "worst errors sit:\n\n" + md_table(pd.DataFrame(screen)))

    # Candidates dropped before the screen. Channel: residual of every submission x payable combination
    # (n >= 1,000). Exempt rows: a county accept rate fitted on the exempt rows of the other folds
    # (shrunk to their mean, m = 20, as in src/target_encoding.py) and stacked on the blend's log-odds.
    ch = pd.DataFrame({"r": r, "c": s("submission_of_application") + " x " + s("initially_payable_to_institution"),
                       "y": df[TARGET]}).groupby("c").agg(n=("r", "size"), r=("r", "mean"), a=("y", "mean"))
    ch = ch[(ch.n >= MIN_SLICE) & (ch.a > 0) & (ch.a < 1)]
    ch_z = (ch.r / np.sqrt(ch.a * (1 - ch.a) / ch.n)).abs().max()
    ex = df[flags["exempt record (1111)"]].merge(read_oof(RUNS["final blend"])[["id", "fold"]], on="id")
    rate = np.empty(len(ex))
    for k in range(5):
        fit, val = ex.fold != k, (ex.fold == k).to_numpy()
        g = ex[fit].groupby(ex.county_code.astype("string")[fit])[TARGET].agg(["sum", "size"])
        prior = ex[TARGET][fit].mean()
        rate[val] = ex.county_code.astype("string")[val].map((g["sum"] + 20 * prior) / (g["size"] + 20)) \
            .astype(float).fillna(prior).to_numpy()
    X = np.c_[np.log(ex.pred / (1 - ex.pred)), rate]
    stacked = np.empty(len(ex))
    for k in range(5):
        fit, val = (ex.fold != k).to_numpy(), (ex.fold == k).to_numpy()
        stacked[val] = LogisticRegression().fit(X[fit], ex[TARGET][fit]).predict_proba(X[val])[:, 1]
    dropped = [
        ("application channel (submission x payable)", f"largest absolute z of the residual over {len(ch)} combinations "
         f"with n >= {MIN_SLICE:,}: {ch_z:.1f}"),
        ("county accept rate on exempt rows", f"AUC on the {len(ex):,} exempt rows: blend {roc_auc_score(ex[TARGET], ex.pred):.4f}, "
         f"blend + rate {roc_auc_score(ex[TARGET], stacked):.4f}, rate alone {roc_auc_score(ex[TARGET], rate):.4f}"),
    ]
    md += "\n\nDropped before the screen:\n\n" + md_table(pd.DataFrame(dropped, columns=["candidate", "check"]))
    publish("t11_worst_errors", md)


def checks() -> None:
    """Tests of explanations written in the report (rule: a hypothesis is stated, tested, then
    reported as held or not). Each check reads the data only."""
    train, test = load()
    train, _, _ = build(train, test, ["dti_order"])
    s = lambda c: train[c].astype("string")  # noqa: E731
    out = ["Hypothesis: the DTI rank separates accepted from denied less well inside FHA and VA than inside "
           "conventional loans. AUC of the DTI rank alone (higher DTI = lower score), rows with a DTI, "
           f"products with at least {MIN_SLICE:,} rows:", ""]
    rows = []
    for prod, g in train[train.dti_order.notna()].groupby(s("derived_loan_product_type")[train.dti_order.notna()]):
        if len(g) >= MIN_SLICE:
            rows.append({"loan product": prod, "rows": f"{len(g):,}", "denied": f"{(g[TARGET] == 0).sum():,}",
                         "AUC of the DTI rank": fmt(roc_auc_score(g[TARGET], -g.dti_order))})
    out += [md_table(pd.DataFrame(rows)), ""]
    site = s("construction_method") == "1"
    out += ["Hypothesis: construction_method is repeated by the manufactured-home columns. Site-built rows "
            "(construction_method = 1) and the value 'not applicable' in each manufactured-home column:", ""]
    rows = []
    for col, na in [("manufactured_home_secured_property_type", "3"), ("manufactured_home_land_property_interest", "5")]:
        rows.append({"column": col, "site-built rows": f"{site.sum():,}",
                     "of them not applicable": f"{(site & (s(col) == na)).sum():,}",
                     "of them exempt (1111)": f"{(site & (s(col) == '1111')).sum():,}",
                     "not applicable but not site-built": f"{(~site & (s(col) == na)).sum():,}"})
    out.append(md_table(pd.DataFrame(rows)))
    (FIGS / "checks.md").write_text("\n".join(out) + "\n")


def shap() -> None:
    """F8 + F9: permutation importance and SHAP values of the tuned LightGBM, the largest member
    of the final blend (weight 1/3). One fit on folds 1-4, explained on 10,000 rows of fold 0;
    SHAP values are LightGBM's exact TreeSHAP (pred_contrib), in log-odds."""
    cfg = load_config(ROOT / "configs" / "experiments" / "lgbm_optuna_best.yaml")
    train, test = load()
    train, _, feats = build(train, test, cfg["features"])
    train = train.merge(load_folds(cfg), on="id", how="left", validate="1:1")
    fit, valid = train[train.fold != 0], train[train.fold == 0]
    t0 = time.perf_counter()
    print(f"  fitting the tuned LightGBM on {len(fit):,} rows (folds 1-4)", flush=True)
    model = LGBMModel(cfg).fit(fit[feats], fit[TARGET], fit.fold.to_numpy())
    auc_fold = roc_auc_score(valid[TARGET], model.predict(valid[feats]))
    logged = json.loads(_run(RUNS["LightGBM"]).fold_auc)[0]
    print(f"  fold 0 AUC {auc_fold:.6f} (logged {logged:.6f}), {model.n_trees_} trees, "
          f"{time.perf_counter() - t0:.0f}s", flush=True)
    assert abs(auc_fold - logged) < 1e-6, "refit does not reproduce the logged fold model"

    sample, _ = train_test_split(valid, train_size=10000, stratify=valid[TARGET], random_state=SEED)
    X, y = sample[feats], sample[TARGET].to_numpy()
    contrib = model.booster_.predict(X, pred_contrib=True)
    raw = model.booster_.predict(X, raw_score=True)
    assert np.allclose(contrib.sum(axis=1), raw, atol=1e-6), "SHAP values do not add up to the log-odds"
    phi = pd.DataFrame(contrib[:, :-1], columns=feats, index=X.index)

    base = roc_auc_score(y, model.predict(X))
    rng = np.random.default_rng(SEED)

    def drop(cols: list[str], repeats: int = 5) -> tuple[float, float]:
        out = []
        for _ in range(repeats):
            perm = rng.permutation(len(X))
            Xp = X.copy()
            for c in cols:  # the same permutation for every column of a group keeps the group intact
                s = X[c].iloc[perm]
                s.index = X.index
                Xp[c] = s
            out.append(base - roc_auc_score(y, model.predict(Xp)))
        return float(np.mean(out)), float(np.std(out, ddof=1))

    imp = pd.DataFrame([(f, *drop([f]), phi[f].abs().mean()) for f in feats],
                       columns=["feature", "permutation drop", "drop SD", "mean |SHAP|"])
    imp.sort_values("permutation drop", ascending=False).to_csv(FIGS / "f8_importance.csv", index=False)
    in_blocks = {c for cols in BLOCKS.values() for c in cols}
    groups = {k: [c for c in v if c in feats] for k, v in BLOCKS.items()}
    groups["other"] = [c for c in feats if c not in in_blocks]
    blocks = pd.DataFrame([(k, len(v), *drop(v)) for k, v in groups.items() if v],
                          columns=["block", "columns", "permutation drop", "drop SD"])
    blocks.to_csv(FIGS / "f8_blocks.csv", index=False)
    print(f"  base AUC on the 10,000 rows {base:.4f}; blocks:\n{blocks.to_string(index=False)}", flush=True)

    # F8: two panels, single features (top 20) and blocks permuted together.
    top = imp.sort_values("permutation drop", ascending=False).head(20)[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(9, 5.2), gridspec_kw={"width_ratios": [1.5, 1]})
    axes[0].barh(top.feature, top["permutation drop"], xerr=top["drop SD"], color=C1, height=0.6,
                 error_kw={"ecolor": INK2, "elinewidth": 0.8})
    axes[0].set_title("One column at a time (20 largest)", loc="left", fontsize=9)
    b = blocks.sort_values("permutation drop")
    axes[1].barh([f"{k} ({n})" for k, n in zip(b.block, b["columns"])], b["permutation drop"], xerr=b["drop SD"],
                 color=C1, height=0.6, error_kw={"ecolor": INK2, "elinewidth": 0.8})
    axes[1].set_title("Whole block at once (columns)", loc="left", fontsize=9)
    axes[1].xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
    for ax in axes:
        ax.set_xlabel("AUC drop when permuted (mean of 5, bar = SD)")
        ax.grid(axis="y", visible=False)
    fig.suptitle(f"Permutation importance, tuned LightGBM, 10,000 rows of fold 0 (AUC {base:.4f})",
                 x=0.01, ha="left", fontsize=9.5)
    save(fig, "f8_permutation")

    # F9: SHAP beeswarm of the 15 largest mean |SHAP|; numeric values coloured by their rank.
    order = imp.sort_values("mean |SHAP|", ascending=False).feature.head(15).tolist()[::-1]
    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    cmap = plt.get_cmap("Blues")
    for i, f in enumerate(order):
        v = phi[f].to_numpy()
        jitter = rng.uniform(-0.33, 0.33, len(v))
        if pd.api.types.is_numeric_dtype(X[f]):
            r = X[f].rank(pct=True).to_numpy()
            colors = np.where(np.isnan(r)[:, None], matplotlib.colors.to_rgba(C2), cmap(0.25 + 0.75 * np.nan_to_num(r)))
        else:
            colors = np.tile(matplotlib.colors.to_rgba(GREY), (len(v), 1))
        ax.scatter(v, i + jitter, s=1.5, c=colors, alpha=0.5, linewidths=0, rasterized=True)
    ax.set_yticks(range(len(order)), order)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_xlabel("SHAP value (log-odds contribution to the prediction)")
    ax.grid(axis="y", visible=False)
    ax.set_title("SHAP values, 15 largest mean |SHAP|. Blue: low to high value (rank); "
                 "orange: blank; grey: category", loc="left", fontsize=8)
    save(fig, "f9_shap")


STEPS = {"data_facts": data_facts, "missingness": missingness, "states": states, "transforms": transforms,
         "lb_noise": lb_noise, "cv_lb": cv_lb, "roc": roc, "rank_corr": rank_corr, "ablations": ablations,
         "slices": slices, "worst_errors": worst_errors, "checks": checks, "shap": shap}

if __name__ == "__main__":
    names = sys.argv[1:] or list(STEPS)
    unknown = [n for n in names if n not in STEPS]
    if unknown:
        raise SystemExit(f"unknown steps {unknown}; available: {list(STEPS)}")
    FIGS.mkdir(exist_ok=True)
    for name in names:
        t0 = time.perf_counter()
        print(f"{name} ...", flush=True)
        STEPS[name]()
        print(f"{name} done ({time.perf_counter() - t0:.0f}s)", flush=True)
