"""
Task 6 -- M1, the experimental / FRET knowledge condition.

Question under test: does explicitly encoding FRET measurement uncertainty add
value beyond the existing data-only models?

Four conditions, on the same fixed 5-fold protocol as M0, Task 4 and Task 5:

    P0    plain prototype                        (Task 4, reused unchanged)
    P1a   prototype + engineered FRET feature    FEATURE ENGINEERING
    P1b   uncertainty-aware prototype distance   KNOWLEDGE-ENCODED DISTANCE
    LR0   logistic regression                    (Task 3, reused unchanged)
    LR1   logistic regression + engineered FRET  FEATURE ENGINEERING

P1a/LR1 and P1b are deliberately kept apart. The pair (P1a, LR1) answers
"does a better feature representation help?", while P1b answers "does encoding
the uncertainty in the DISTANCE help?". Without that separation, a gain from
P1b could not be attributed to the knowledge encoding rather than to the extra
column. This separation was required by the brief and is the main reason this
task has four models rather than one.

THE ENGINEERED FRET FEATURE (P1a, LR1)
---------------------------------------
    fret_normalized = fret_error / fret_uncertainty

A single extra column. The two original columns are RETAINED, so nothing is
removed; the model is free to use the raw disagreement, the uncertainty, or
the ratio. This is the brief's requested "uncertainty-normalised FRET feature"
and it is a plain feature-engineering control: a new COLUMN, not a change to
how distance is computed.

Computed per row from observable columns only. No labels, no statistics
estimated from the data, so it cannot leak.

THE UNCERTAINTY-AWARE DISTANCE (P1b)
-------------------------------------
Only the FRET term of the squared distance is modified. All eleven other
features keep the plain squared difference. In standardized feature space, with
`fret_error` the first column and `fret_uncertainty` the second:

    d_1(x) = sum_{j>1} (x_j - p_1j)^2  +  ( (x_0 - p_10) / (sigma_x / sigma_ref) )^2
    d_0(x) = sum_{j>1} (x_j - p_0j)^2  +  ( (x_0 - p_00) / (sigma_x / sigma_ref) )^2

where `sigma_x` is that sample's own `fret_uncertainty` and `sigma_ref` is the
MEAN `fret_uncertainty` of the TRAINING FOLD.

So the same observed disagreement is divided by the measurement's relative
precision. A disagreement measured imprecisely (sigma_x large) is discounted;
one measured precisely is counted fully. That is the intended semantics, and
it is a soft, transparent rule with no fitted parameters beyond sigma_ref.

`sigma_ref` is a training-fold mean, so it is fit on training rows only and
never uses held-out labels. The scaling also keeps the FRET term on a
comparable numerical footing to the other terms, which a bare division by
`sigma_x` does not -- see VARIANT_REJECTED below.

WHY P1b FAILS, STATED PRECISELY
-------------------------------
It is worth separating two things that are easy to conflate.

The DIRECTION of the weighting is conceptually sensible. Evidence measured with
low uncertainty deserves more weight than evidence measured with high
uncertainty, and this rule does exactly that. Nothing about the principle is
wrong.

The failure is one of MAGNITUDE in this feature representation. The FRET term
is multiplied by

    1 / (sigma_x / sigma_ref)^2

and in the low-uncertainty tier that multiplier is large -- median 9.96, up to
704 (see the notebook, Section 4). At that scale the single FRET dimension
comes to dominate the twelve-dimensional prototype distance, so P1b in that
tier is no longer meaningfully a prototype model; it is close to a
FRET-only classifier with a different failure mode.

So the diagnosis is: **a reasonable scientific principle encoded too
aggressively for this representation degrades classification.** It is not that
the model was told to trust unreliable evidence, and it should not be
described that way. The result is a calibration-of-strength problem, not a
reversal of direction.

NOTHING ELSE IS ADDED. No module information, no structural prior, no
hand-written interaction term, no relevance weights, no rejection.

VARIANT REJECTED, AND WHY IT IS RECORDED
-----------------------------------------
A more obvious-looking rule divides the FRET term by the standardized
`fret_uncertainty` directly, with no reference scale. It is numerically
catastrophic here (mean AUROC 0.611 against P0's 0.828) because the
standardized uncertainty is near zero for many rows and can be negative, so the
FRET term explodes and dominates the entire distance. The reference-scale
form above is the defensible reading of the same idea. This is recorded because
a reader may reasonably try the naive form first and deserves to know why it
fails.

SCIENTIFIC STATUS AND SCOPE
---------------------------
The dataset is synthetic and FRET measurements are simulated. AUR differences
here measure recovery of a known generative process, nothing biological. No
biological or clinical validity is claimed, and no prior in this module is
evidence about real RNA structure.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline  # noqa: E402
import cv_protocol  # noqa: E402
import prototype_model  # noqa: E402
from validation_checks import FEATURE_COLUMNS, load_candidates  # noqa: E402

# Position of the two FRET columns in FEATURE_COLUMNS.
FRET_ERROR_COL = 0
FRET_UNCERTAINTY_COL = 1

TIER_LABELS = ["low", "medium", "high"]

VARIANT_REJECTED = {
    "rule": "divide the standardized FRET term by the standardized fret_uncertainty directly",
    "mean_cv_auroc": 0.6111,
    "reference_mean_cv_auroc": 0.8276,
    "reason": (
        "Numerically catastrophic. The standardized fret_uncertainty is near "
        "zero for many rows and can be negative, so the FRET term explodes and "
        "dominates the whole distance. The implemented rule divides by the "
        "RELATIVE precision sigma_x / sigma_ref instead, which expresses the "
        "same idea on a comparable numerical footing."
    ),
}


# --------------------------------------------------------------------------
# Feature engineering (label-free, per row)
# --------------------------------------------------------------------------
def fret_normalized(x: np.ndarray) -> np.ndarray:
    """fret_error / fret_uncertainty, computed row by row.

    Uses no labels and no fitted statistic, so it is computed identically for
    training and held-out rows and cannot leak. The denominator is floored to
    avoid division by an arbitrarily small positive uncertainty.
    """
    return x[:, FRET_ERROR_COL] / np.maximum(x[:, FRET_UNCERTAINTY_COL], 1e-6)


def with_engineered_feature(x: np.ndarray) -> np.ndarray:
    """Append the uncertainty-normalised FRET column, keeping all originals."""
    return np.column_stack([x, fret_normalized(x)])


# --------------------------------------------------------------------------
# Uncertainty-aware prototype distance
# --------------------------------------------------------------------------
def uncertainty_aware_distances(
    x_std: np.ndarray,
    p_plausible: np.ndarray,
    p_implausible: np.ndarray,
    relative_precision: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Squared distances where ONLY the FRET term is precision-discounted.

    All columns j > 1 use the plain squared difference. Column 0, the FRET
    error, is divided by that sample's relative precision
    `sigma_x / sigma_ref` before squaring.
    """
    rel = np.maximum(relative_precision, 1e-6)
    rest_p = ((x_std[:, 1:] - p_plausible[1:]) ** 2).sum(axis=1)
    rest_i = ((x_std[:, 1:] - p_implausible[1:]) ** 2).sum(axis=1)
    fret_p = ((x_std[:, 0] - p_plausible[0]) / rel) ** 2
    fret_i = ((x_std[:, 0] - p_implausible[0]) / rel) ** 2
    return rest_p + fret_p, rest_i + fret_i


# --------------------------------------------------------------------------
# Per-condition evaluation
# --------------------------------------------------------------------------
def _prototype_condition(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    uncertainty_aware: bool,
) -> dict:
    """One prototype model: P0 (plain) or P1b (uncertainty-aware)."""
    scaler = StandardScaler().fit(x_train)  # training rows only
    a = scaler.transform(x_train)
    b = scaler.transform(x_test)
    prototypes = prototype_model.fit_prototypes(a, y_train)

    if uncertainty_aware:
        sigma_ref = float(x_train[:, FRET_UNCERTAINTY_COL].mean())  # training rows only
        relative = x_test[:, FRET_UNCERTAINTY_COL] / sigma_ref
        d_plausible, d_implausible = uncertainty_aware_distances(
            b, prototypes["plausible"], prototypes["implausible"], relative
        )
    else:
        d_plausible = prototype_model.squared_distance(b, prototypes["plausible"])
        d_implausible = prototype_model.squared_distance(b, prototypes["implausible"])

    margin = prototype_model.margin(d_plausible, d_implausible)
    return {
        "margin": margin,
        "probability": prototype_model.probability_from_margin(margin),
        "prediction": (d_plausible < d_implausible).astype(int),
    }


def _logistic_condition(
    x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray
) -> dict:
    """LR0 or LR1. Scaling inside the pipeline, so training rows only."""
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("logisticregression", LogisticRegression()),
    ])
    model.fit(x_train, y_train)
    probability = model.predict_proba(x_test)[:, 1]
    return {
        "margin": probability - 0.5,
        "probability": probability,
        "prediction": (probability >= 0.5).astype(int),
    }


def _stratify_by_uncertainty(
    y_true: np.ndarray, out: dict, uncertainty: np.ndarray
) -> list[dict]:
    """Tier held-out predictions by fret_uncertainty and report each tier.

    The question this answers is not only "did AUROC improve?" but also "did
    the model become less confident where the FRET evidence is less reliable?"
    """
    tiers = pd.qcut(pd.Series(uncertainty), 3, labels=TIER_LABELS)
    rows = []
    for label in TIER_LABELS:
        mask = (tiers == label).to_numpy()
        y_t, m_t, p_t = y_true[mask], out["margin"][mask], out["probability"][mask]
        auroc = (
            float(roc_auc_score(y_t, m_t)) if len(np.unique(y_t)) > 1 else None
        )
        rows.append({
            "tier": label,
            "n": int(mask.sum()),
            "median_uncertainty": float(np.median(uncertainty[mask])),
            "accuracy": float((out["prediction"][mask] == y_t).mean()),
            "auroc": auroc,
            "mean_abs_margin": float(np.abs(m_t).mean()),
            "mean_probability": float(p_t.mean()),
            "brier": float(np.mean((p_t - y_t) ** 2)),
        })
    return rows


def run_cross_validation(candidates: pd.DataFrame) -> dict:
    """PRIMARY protocol: the shared 5 folds, verified explicitly."""
    X = candidates[FEATURE_COLUMNS].copy()
    baseline.assert_model_facing_only(X)
    y = candidates["label"].to_numpy()
    assert_model_features_unchanged = list(FEATURE_COLUMNS) == list(V_FEATURES_SNAPSHOT)
    assert assert_model_features_unchanged, "FEATURE_COLUMNS changed"

    folds = list(cv_protocol.make_cv().split(X, y))
    expected = cv_protocol.fold_assignment(y)
    for k, (_, test_idx) in enumerate(folds):
        assert list(test_idx) == list(expected[k]), f"fold {k} differs from the shared protocol"

    conditions = {
        "P0_plain_prototype": "plain_prototype",
        "P1a_engineered_fret_feature": "engineered_prototype",
        "P1b_uncertainty_aware_distance": "aware_prototype",
        "LR0_logistic_regression": "logistic",
        "LR1_logistic_engineered_fret": "logistic_engineered",
    }

    per_fold: dict = {name: [] for name in conditions}
    tier_rows: dict = {name: [] for name in conditions}

    for k, (train_idx, test_idx) in enumerate(folds):
        x_train, x_test = X.iloc[train_idx].to_numpy(), X.iloc[test_idx].to_numpy()
        y_train, y_test = y[train_idx], y[test_idx]
        uncertainty = x_test[:, FRET_UNCERTAINTY_COL]

        built = {
            "plain_prototype": lambda: _prototype_condition(x_train, y_train, x_test, False),
            "engineered_prototype": lambda: _prototype_condition(
                with_engineered_feature(x_train), y_train,
                with_engineered_feature(x_test), False,
            ),
            "aware_prototype": lambda: _prototype_condition(
                x_train, y_train, x_test, True
            ),
            "logistic": lambda: _logistic_condition(x_train, y_train, x_test),
            "logistic_engineered": lambda: _logistic_condition(
                with_engineered_feature(x_train), y_train,
                with_engineered_feature(x_test),
            ),
        }

        for name, kind in conditions.items():
            out = built[kind]()
            per_fold[name].append({"fold": k, **baseline.evaluate(y_test, out["probability"])})
            for row in _stratify_by_uncertainty(y_test, out, uncertainty):
                tier_rows[name].append({"fold": k, **row})

    summary = {}
    for name, rows in per_fold.items():
        summary[name] = {
            metric: {
                "mean": float(np.mean([r[metric] for r in rows])),
                "std": float(np.std([r[metric] for r in rows], ddof=1)),
                "min": float(np.min([r[metric] for r in rows])),
                "max": float(np.max([r[metric] for r in rows])),
            }
            for metric in cv_protocol.PRIMARY_METRICS
        }

    # Tier summaries pooled across folds, for readability.
    tier_summary = {}
    for name, rows in tier_rows.items():
        tier_summary[name] = {}
        for label in TIER_LABELS:
            sel = [r for r in rows if r["tier"] == label]
            aurocs = [r["auroc"] for r in sel if r["auroc"] is not None]
            tier_summary[name][label] = {
                "n_total": int(sum(r["n"] for r in sel)),
                "median_uncertainty": float(np.mean([r["median_uncertainty"] for r in sel])),
                "accuracy": float(np.mean([r["accuracy"] for r in sel])),
                "auroc": float(np.mean(aurocs)) if aurocs else None,
                "mean_abs_margin": float(np.mean([r["mean_abs_margin"] for r in sel])),
                "mean_probability": float(np.mean([r["mean_probability"] for r in sel])),
                "brier": float(np.mean([r["brier"] for r in sel])),
            }

    return {
        "protocol": {
            "name": "stratified 5-fold cross-validation (shared with M0, Task 4, Task 5)",
            "role": "PRIMARY",
            "n_splits": cv_protocol.CV_N_SPLITS,
            "random_state": cv_protocol.CV_RANDOM_STATE,
            "fold_fingerprint": cv_protocol.fold_fingerprint(y),
            "folds_verified_identical": True,
            "preprocessing": (
                "StandardScaler refit within each training fold for every "
                "condition. The engineered feature is computed per row from "
                "observable columns only, so it uses no labels and no fitted "
                "statistic. sigma_ref for P1b is the training-fold mean of "
                "fret_uncertainty."
            ),
            "base_features": list(FEATURE_COLUMNS),
            "engineered_feature": "fret_error / fret_uncertainty (originals retained)",
            "base_features_retained": True,
            "structural_prior_used": False,
            "network_prior_used": False,
            "relevance_weights_used": False,
            "rejection_used": False,
            "latent_truth_used": False,
            "hyperparameter_tuning": "none; sigma_ref is a training-fold mean, not a tuned value",
        },
        "conditions": {
            "P0_plain_prototype": "Task 4 model, recomputed here unchanged as the reference",
            "P1a_engineered_fret_feature": "prototype + fret_error/fret_uncertainty column (feature engineering)",
            "P1b_uncertainty_aware_distance": "prototype with only the FRET distance term precision-discounted (knowledge encoding)",
            "LR0_logistic_regression": "Task 3 model, recomputed here unchanged as the reference",
            "LR1_logistic_engineered_fret": "logistic regression + the same engineered column (feature engineering)",
        },
        "summary_mean_std": summary,
        "per_fold": per_fold,
        "tier_summary": tier_summary,
        "tier_rows": tier_rows,
        "variant_rejected": VARIANT_REJECTED,
    }


# Snapshot of the feature list, used only to assert it has not drifted.
V_FEATURES_SNAPSHOT = list(FEATURE_COLUMNS)


def compare(cv_result: dict) -> dict:
    """Fold-wise differences against the two no-knowledge references."""
    comparisons = {}
    pairs = {
        "P1a_vs_P0": ("P1a_engineered_fret_feature", "P0_plain_prototype"),
        "P1b_vs_P0": ("P1b_uncertainty_aware_distance", "P0_plain_prototype"),
        "LR1_vs_LR0": ("LR1_logistic_engineered_fret", "LR0_logistic_regression"),
        "P1a_vs_P1b": ("P1a_engineered_fret_feature", "P1b_uncertainty_aware_distance"),
    }
    for label, (new, ref) in pairs.items():
        rows = []
        for metric in cv_protocol.PRIMARY_METRICS:
            a = [f[metric] for f in cv_result["per_fold"][new]]
            b = [f[metric] for f in cv_result["per_fold"][ref]]
            diffs = [x - y for x, y in zip(a, b)]
            rows.append({
                "metric": metric,
                "new_mean": float(np.mean(a)),
                "reference_mean": float(np.mean(b)),
                "mean_difference": float(np.mean(diffs)),
                "difference_std": float(np.std(diffs, ddof=1)),
                "n_folds_new_better": int(sum(d > 0 for d in diffs)),
                "n_folds_reference_better": int(sum(d < 0 for d in diffs)),
                "per_fold_difference": diffs,
            })
        comparisons[label] = rows
    return comparisons


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 6 -- M1 experimental/FRET knowledge")
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()

    candidates = load_candidates()
    result = run_cross_validation(candidates)
    result["comparison"] = compare(result)
    result["versions"] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }

    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / "fret_knowledge_cv.json"
    with open(out_path, "w") as handle:
        json.dump(result, handle, indent=2)
    print(f"wrote {out_path}\n")
    print(f"fold fingerprint: {result['protocol']['fold_fingerprint']} (verified shared)")

    for name in result["summary_mean_std"]:
        print(f"\n  {name}")
        for metric in cv_protocol.PRIMARY_METRICS:
            s = result["summary_mean_std"][name][metric]
            print(f"    {metric:19s} {s['mean']:.4f} +/- {s['std']:.4f}")

    for label, rows in result["comparison"].items():
        print(f"\n{label} (positive = first named model better; brier lower is better)")
        for row in rows:
            if row["metric"] not in ("auroc", "accuracy", "brier"):
                continue
            print(f"  {row['metric']:19s} {row['new_mean']:.4f} vs "
                  f"{row['reference_mean']:.4f} diff={row['mean_difference']:+.4f} "
                  f"({row['n_folds_new_better']}/5 folds better)")

    print("\nconfidence and accuracy by FRET-uncertainty tier:")
    for name in result["tier_summary"]:
        print(f"  {name}")
        for label in TIER_LABELS:
            t = result["tier_summary"][name][label]
            au = "n/a" if t["auroc"] is None else f"{t['auroc']:.4f}"
            print(f"    {label:7s} n={t['n_total']:3d} acc={t['accuracy']:.4f} "
                  f"auroc={au} mean|margin|={t['mean_abs_margin']:.4f} brier={t['brier']:.4f}")


if __name__ == "__main__":
    main()
