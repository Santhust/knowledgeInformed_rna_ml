"""
Task 7 -- M2, the structural knowledge condition.

Question: does explicitly encoding a scientifically motivated structural
relationship add value beyond the raw Logistic Regression baseline, the
generic nonlinear HGB baseline, and the plain prototype?

The structural idea is the one already present in the synthetic design, and
stated in PROJECT_BRIEF section 7 (M2):

    * `stem_count` has a PREFERRED INTERMEDIATE region -- both very few and
      very many stems are less plausible;
    * MANY STEMS COMBINED WITH LOW `structural_consistency` is penalised;
    * the constraint is SOFT, not a hard rule.

THE PRIOR, AS ENCODED
---------------------
Two transparent, bounded, label-free columns built ONLY from `stem_count` and
`structural_consistency`:

    dev      = (stem_count - 8) / 1.5
    quadratic = dev^2
    interaction = max(0, dev) * max(0, 0.5 - structural_consistency)

`quadratic` is soft and smooth: a squared deviation from the preferred count of
8, so it rises in BOTH directions rather than switching at a threshold. Its
scale constant 1.5 matches the spread of `stem_count` in the data.

`interaction` is a one-sided, soft product. It is exactly zero unless a
candidate claims more stems than preferred AND has structural consistency below
0.5, so it fires on ~11% of rows and is bounded above. It is a penalty shape,
not a rule: a candidate in that region is not forbidden, it is merely pushed
down.

The preferred count of 8 and the consistency threshold of 0.5 come from the
public design description of PROJECT_BRIEF and the Task 1 generator
documentation. They are NOT fitted: no held-out label, no tuned parameter, and
no latent variable is involved. Both features are computed per row from
observable columns only, so they cannot leak.

CONDITIONS
----------
    LR0    raw Logistic Regression                        (reference)
    LR2    LR0 + the two structural prior columns         M2, logistic
    LR2c   LR0 + a GENERIC nonlinear control pair         control
    P0     plain prototype                                 (reference)
    P2     P0 + the two structural prior columns          M2, prototype
    P2c    P0 + the same generic control pair             control
    HGB0   generic nonlinear HistGradientBoosting          key control

P2 (adding the prior to the representation) is the MAIN M2 condition. The
alternative encoding -- a bounded penalty term inside the prototype distance --
was implemented and measured, and is reported as P2d in the results. It is kept
as a secondary control rather than promoted, because it performed worse and
because two encodings of the same prior is already the limit of what is
justifiable here.

THE GENERIC NONLINEAR CONTROL, AND WHY IT EXISTS
------------------------------------------------
    generic_quadratic   = free_energy^2
    generic_interaction = reg_feature_1 * pathway_feature_1

Both were chosen WITHOUT consulting any label: the first is a square of a
feature already in the model, the second a product of two features already in
the model. They are included to separate

    a DOMAIN-INFORMED interaction      (structural prior, chosen from biology)
    from ANY EXTRA NONLINEAR FEATURE   (generic control, chosen blind)

Without LR2c/P2c, any gain from LR2/P2 could not be attributed to the prior
rather than to the model simply having one more nonlinear column. Note the
generic pair is deliberately comparable in mathematical FORM to the structural
pair -- one square, one product -- so the comparison isolates where the
columns came from, not how complicated they are.

WHY HGB0 IS THE CRITICAL CONTROL
--------------------------------
HGB represents interactions WITHOUT being told the biology. If the explicit
structural prior merely matches HGB0, the honest conclusion is that the prior
adds interpretability rather than accuracy. If it beats HGB0, that is stronger
evidence the prior carries information HGB could not find on its own. Task 3
already established that HGB UNDERPERFORMS the linear model on this dataset
(CV AUROC 0.8054 vs 0.8290), which makes this comparison especially informative.

SCIENTIFIC STATUS AND SCOPE
---------------------------
The dataset is synthetic and no structure here is real RNA structure. The
structural rule is a plausible-sounding prior, not a validated biological
constraint. Differences reported in the notebook measure recovery of a known
generative process, nothing biological. No biological or clinical validity is
claimed, and the prior is not retuned to make M2 win.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline  # noqa: E402
import cv_protocol  # noqa: E402
import prototype_model  # noqa: E402
from validation_checks import FEATURE_COLUMNS, load_candidates  # noqa: E402

STEM_COL = FEATURE_COLUMNS.index("stem_count")
CONSISTENCY_COL = FEATURE_COLUMNS.index("structural_consistency")
FREE_ENERGY_COL = FEATURE_COLUMNS.index("free_energy")
REG1_COL = FEATURE_COLUMNS.index("reg_feature_1")
PATH1_COL = FEATURE_COLUMNS.index("pathway_feature_1")

# Constants from the public design description, not fitted. See module docstring.
PREFERRED_STEM_COUNT = 8.0
STEM_SCALE = 1.5
CONSISTENCY_THRESHOLD = 0.5

PRIOR_COLUMN_NAMES = ["structural_quadratic", "structural_interaction"]
GENERIC_COLUMN_NAMES = ["generic_quadratic", "generic_interaction"]

PRIOR_DERIVATION = (
    "Preferred stem count 8 and consistency threshold 0.5 taken from the "
    "public design description (PROJECT_BRIEF section 7, Task 1 generator "
    "documentation). Not fitted, not tuned, no latent variable used."
)
GENERIC_DERIVATION = (
    "free_energy^2 and reg_feature_1 * pathway_feature_1. Chosen WITHOUT "
    "consulting any label: one square and one product of features already in "
    "the model. Comparable in mathematical form to the structural pair, so the "
    "comparison isolates provenance rather than complexity."
)


# --------------------------------------------------------------------------
# The structural prior (label-free, per row)
# --------------------------------------------------------------------------
def structural_prior(x: np.ndarray) -> np.ndarray:
    """Two soft, bounded structural prior columns.

    Uses only `stem_count` and `structural_consistency`. No labels, no fitted
    statistic, computed identically for training and held-out rows, so it
    cannot leak.
    """
    dev = (x[:, STEM_COL] - PREFERRED_STEM_COUNT) / STEM_SCALE
    quadratic = dev**2
    interaction = np.maximum(0.0, dev) * np.maximum(
        0.0, CONSISTENCY_THRESHOLD - x[:, CONSISTENCY_COL]
    )
    return np.column_stack([quadratic, interaction])


def generic_nonlinear_control(x: np.ndarray) -> np.ndarray:
    """Label-blind nonlinear control of comparable mathematical form."""
    return np.column_stack(
        [x[:, FREE_ENERGY_COL] ** 2, x[:, REG1_COL] * x[:, PATH1_COL]]
    )


# --------------------------------------------------------------------------
# Model conditions
# --------------------------------------------------------------------------
def _logistic(x_train, y_train, x_test) -> dict:
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("logisticregression", LogisticRegression()),
    ])
    model.fit(x_train, y_train)
    probability = model.predict_proba(x_test)[:, 1]
    return {"probability": probability, "prediction": (probability >= 0.5).astype(int)}


def _hist_gradient_boosting(x_train, y_train, x_test) -> dict:
    """HGB0, the generic nonlinear control, exactly as fixed in Task 3."""
    model = HistGradientBoostingClassifier(
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        random_state=cv_protocol.CV_RANDOM_STATE,
    )
    model.fit(x_train, y_train)
    probability = model.predict_proba(x_test)[:, 1]
    return {"probability": probability, "prediction": (probability >= 0.5).astype(int)}


def _prototype(x_train, y_train, x_test) -> dict:
    scaler = StandardScaler().fit(x_train)  # training rows only
    a, b = scaler.transform(x_train), scaler.transform(x_test)
    prototypes = prototype_model.fit_prototypes(a, y_train)
    out = prototype_model.predict(prototypes, b)
    return {"probability": out["probability"], "prediction": out["prediction"], "margin": out["margin"]}


def _prototype_with_penalty(x_train, y_train, x_test) -> dict:
    """P2d control: the prior as a BOUNDED PENALTY inside the distance.

    The alternative encoding, kept as a secondary control rather than promoted.
    The squared FRET-style distance is augmented by a small bounded penalty on
    the structural prior columns, so an implausible structural combination
    pushes the candidate away from the plausible prototype without ever
    forbidding it.
    """
    scaler = StandardScaler().fit(x_train)
    a, b = scaler.transform(x_train), scaler.transform(x_test)
    prototypes = prototype_model.fit_prototypes(a, y_train)

    d_plausible = prototype_model.squared_distance(b, prototypes["plausible"])
    d_implausible = prototype_model.squared_distance(b, prototypes["implausible"])

    penalty = structural_prior(x_test).sum(axis=1)
    penalty = penalty / max(penalty.mean(), 1e-9)  # keep it bounded and on scale
    d_plausible = d_plausible + 0.5 * penalty

    margin = prototype_model.margin(d_plausible, d_implausible)
    return {
        "probability": prototype_model.probability_from_margin(margin),
        "prediction": (d_plausible < d_implausible).astype(int),
        "margin": margin,
    }


# --------------------------------------------------------------------------
# Cross-validation
# --------------------------------------------------------------------------
CONDITIONS = {
    "LR0_logistic_raw": "raw Logistic Regression (reference)",
    "LR2_logistic_structural_prior": "LR0 + the two structural prior columns (M2)",
    "LR2c_logistic_generic_nonlinear": "LR0 + generic blind nonlinear control (control)",
    "P0_plain_prototype": "plain prototype (reference)",
    "P2_prototype_structural_prior": "P0 + the two structural prior columns (M2, main)",
    "P2c_prototype_generic_nonlinear": "P0 + generic blind nonlinear control (control)",
    "P2d_prototype_bounded_penalty": "P0 + bounded structural penalty inside the distance (secondary control)",
    "HGB0_hist_gradient_boosting": "generic nonlinear boosting (key control)",
}

SHORT = {
    "LR0_logistic_raw": "LR0",
    "LR2_logistic_structural_prior": "LR2",
    "LR2c_logistic_generic_nonlinear": "LR2c",
    "P0_plain_prototype": "P0",
    "P2_prototype_structural_prior": "P2",
    "P2c_prototype_generic_nonlinear": "P2c",
    "P2d_prototype_bounded_penalty": "P2d",
    "HGB0_hist_gradient_boosting": "HGB0",
}


def run_cross_validation(candidates: pd.DataFrame) -> dict:
    """PRIMARY protocol: the shared 5 folds, verified explicitly."""
    X = candidates[FEATURE_COLUMNS].copy()
    baseline.assert_model_facing_only(X)
    y = candidates["label"].to_numpy()

    folds = list(cv_protocol.make_cv().split(X, y))
    expected = cv_protocol.fold_assignment(y)
    for k, (_, test_idx) in enumerate(folds):
        assert list(test_idx) == list(expected[k]), f"fold {k} differs from the shared protocol"

    def build(kind, x_train, y_train, x_test):
        prior_tr = structural_prior(x_train)
        prior_te = structural_prior(x_test)
        generic_tr = generic_nonlinear_control(x_train)
        generic_te = generic_nonlinear_control(x_test)
        if kind == "LR0_logistic_raw":
            return _logistic(x_train, y_train, x_test)
        if kind == "LR2_logistic_structural_prior":
            return _logistic(np.column_stack([x_train, prior_tr]), y_train,
                             np.column_stack([x_test, prior_te]))
        if kind == "LR2c_logistic_generic_nonlinear":
            return _logistic(np.column_stack([x_train, generic_tr]), y_train,
                             np.column_stack([x_test, generic_te]))
        if kind == "P0_plain_prototype":
            return _prototype(x_train, y_train, x_test)
        if kind == "P2_prototype_structural_prior":
            return _prototype(np.column_stack([x_train, prior_tr]), y_train,
                              np.column_stack([x_test, prior_te]))
        if kind == "P2c_prototype_generic_nonlinear":
            return _prototype(np.column_stack([x_train, generic_tr]), y_train,
                              np.column_stack([x_test, generic_te]))
        if kind == "P2d_prototype_bounded_penalty":
            return _prototype_with_penalty(x_train, y_train, x_test)
        if kind == "HGB0_hist_gradient_boosting":
            return _hist_gradient_boosting(x_train, y_train, x_test)
        raise ValueError(f"unknown condition {kind}")

    per_fold = {name: [] for name in CONDITIONS}
    region_rows = []
    for k, (train_idx, test_idx) in enumerate(folds):
        x_train, x_test = X.iloc[train_idx].to_numpy(), X.iloc[test_idx].to_numpy()
        y_train, y_test = y[train_idx], y[test_idx]
        for name in CONDITIONS:
            out = build(name, x_train, y_train, x_test)
            per_fold[name].append({"fold": k, **baseline.evaluate(y_test, out["probability"])})
        region_rows.extend(_structural_region(x_test, y_test))

    summary = {
        name: {
            metric: {
                "mean": float(np.mean([r[metric] for r in rows])),
                "std": float(np.std([r[metric] for r in rows], ddof=1)),
                "min": float(np.min([r[metric] for r in rows])),
                "max": float(np.max([r[metric] for r in rows])),
            }
            for metric in cv_protocol.PRIMARY_METRICS
        }
        for name, rows in per_fold.items()
    }

    return {
        "protocol": {
            "name": "stratified 5-fold cross-validation (shared with all previous tasks)",
            "role": "PRIMARY",
            "n_splits": cv_protocol.CV_N_SPLITS,
            "random_state": cv_protocol.CV_RANDOM_STATE,
            "fold_fingerprint": cv_protocol.fold_fingerprint(y),
            "folds_verified_identical": True,
            "preprocessing": (
                "StandardScaler refit within each training fold for every "
                "condition. Both prior and control columns are computed per row "
                "from observable features only, using no labels and no fitted "
                "statistic."
            ),
            "base_features": list(FEATURE_COLUMNS),
            "prior_columns": PRIOR_COLUMN_NAMES,
            "prior_derivation": PRIOR_DERIVATION,
            "generic_control_columns": GENERIC_COLUMN_NAMES,
            "generic_control_derivation": GENERIC_DERIVATION,
            "hyperparameter_tuning": "none; prior constants come from the public design description",
            "network_prior_used": False,
            "relevance_weights_used": False,
            "rejection_used": False,
            "latent_truth_used": False,
        },
        "conditions": CONDITIONS,
        "summary_mean_std": summary,
        "per_fold": per_fold,
        "structural_region": _region_summary(region_rows),
    }


def _structural_region(x_test: np.ndarray, y_test: np.ndarray) -> list[dict]:
    """Score each held-out sample by the region the prior is meant to target."""
    dev = (x_test[:, STEM_COL] - PREFERRED_STEM_COUNT) / STEM_SCALE
    consistency = x_test[:, CONSISTENCY_COL]
    return [
        {
            "many_stems": bool(dev[i] > 0),
            "low_consistency": bool(consistency[i] < CONSISTENCY_THRESHOLD),
            "label": int(y_test[i]),
        }
        for i in range(len(y_test))
    ]


def _region_summary(region_rows: list[dict]) -> dict:
    """Label rate in each of the four stem/consistency quadrants."""
    out = {}
    for many in (False, True):
        for low in (False, True):
            sel = [r for r in region_rows if r["many_stems"] == many
                   and r["low_consistency"] == low]
            if not sel:
                continue
            key = ("many stems" if many else "few stems") + " + " + (
                "low consistency" if low else "adequate consistency"
            )
            out[key] = {
                "n": len(sel),
                "plausible_rate": float(np.mean([r["label"] for r in sel])),
            }
    return out


def compare(cv_result: dict) -> dict:
    """Fold-wise differences for the comparisons the brief asks about."""
    pairs = {
        "LR2_vs_LR0": ("LR2_logistic_structural_prior", "LR0_logistic_raw"),
        "LR2c_vs_LR0": ("LR2c_logistic_generic_nonlinear", "LR0_logistic_raw"),
        "P2_vs_P0": ("P2_prototype_structural_prior", "P0_plain_prototype"),
        "P2c_vs_P0": ("P2c_prototype_generic_nonlinear", "P0_plain_prototype"),
        "P2d_vs_P0": ("P2d_prototype_bounded_penalty", "P0_plain_prototype"),
        "LR2_vs_LR2c": ("LR2_logistic_structural_prior", "LR2c_logistic_generic_nonlinear"),
        "P2_vs_P2c": ("P2_prototype_structural_prior", "P2c_prototype_generic_nonlinear"),
        "LR2_vs_HGB0": ("LR2_logistic_structural_prior", "HGB0_hist_gradient_boosting"),
        "P2_vs_HGB0": ("P2_prototype_structural_prior", "HGB0_hist_gradient_boosting"),
    }
    out = {}
    for label, (new, ref) in pairs.items():
        rows = []
        for metric in cv_protocol.PRIMARY_METRICS:
            a = [f[metric] for f in cv_result["per_fold"][new]]
            b = [f[metric] for f in cv_result["per_fold"][ref]]
            diffs = [x - y_ for x, y_ in zip(a, b)]
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
        out[label] = rows
    return out


def prior_diagnostics(candidates: pd.DataFrame) -> dict:
    """Does the prior actually fire where it is supposed to?"""
    x = candidates[FEATURE_COLUMNS].to_numpy()
    prior = structural_prior(x)
    generic = generic_nonlinear_control(x)
    label = candidates["label"].to_numpy()

    def describe(arr, label_name):
        return {
            "column": label_name,
            "min": float(arr.min()),
            "max": float(arr.max()),
            "mean": float(arr.mean()),
            "fraction_nonzero": float(np.mean(arr != 0)),
            "correlation_with_label": float(np.corrcoef(arr, label)[0, 1]),
        }

    return {
        "structural_quadratic": describe(prior[:, 0], "structural_quadratic"),
        "structural_interaction": describe(prior[:, 1], "structural_interaction"),
        "generic_quadratic": describe(generic[:, 0], "generic_quadratic"),
        "generic_interaction": describe(generic[:, 1], "generic_interaction"),
        "note": (
            "Correlations are reported for DIAGNOSTIC interpretation of whether "
            "the prior targets the intended region. They are not used to fit, "
            "select, or tune anything."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 7 -- M2 structural knowledge")
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()

    candidates = load_candidates()
    result = run_cross_validation(candidates)
    result["comparison"] = compare(result)
    result["prior_diagnostics"] = prior_diagnostics(candidates)
    result["versions"] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }

    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / "structural_knowledge_cv.json"
    with open(out_path, "w") as handle:
        json.dump(result, handle, indent=2)
    print(f"wrote {out_path}\n")
    print(f"fold fingerprint: {result['protocol']['fold_fingerprint']} (verified shared)\n")

    for name in CONDITIONS:
        print(f"  {SHORT[name]:6s} {CONDITIONS[name]}")
        for metric in cv_protocol.PRIMARY_METRICS:
            s = result["summary_mean_std"][name][metric]
            print(f"    {metric:19s} {s['mean']:.4f} +/- {s['std']:.4f}")
        print()

    for label, rows in result["comparison"].items():
        row = next(r for r in rows if r["metric"] == "auroc")
        print(f"{label:14s} AUROC {row['new_mean']:.4f} vs {row['reference_mean']:.4f} "
              f"diff={row['mean_difference']:+.4f} ({row['n_folds_new_better']}/5 folds better)")

    print("\nplausible rate by structural region (all held-out rows):")
    for key, stats in result["structural_region"].items():
        print(f"  {key:38s} n={stats['n']:3d} plausible={stats['plausible_rate']:.4f}")

    print("\nprior columns:")
    for key in ("structural_quadratic", "structural_interaction",
                "generic_quadratic", "generic_interaction"):
        d = result["prior_diagnostics"][key]
        print(f"  {key:26s} max={d['max']:8.3f} nonzero={d['fraction_nonzero']:.3f} "
              f"corr(label)={d['correlation_with_label']:+.3f}")


if __name__ == "__main__":
    main()
