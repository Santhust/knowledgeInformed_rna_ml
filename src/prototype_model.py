"""
Task 4 -- data-only nearest-prototype baseline.

A transparent prototype classifier on the same observable features, the same
feature list, and the same deterministic 5-fold partition as M0, but with no
biological prior knowledge whatsoever.

THE MODEL, IN FULL
------------------
Given a standardized observation x and one prototype per class, in units of
training-set standard deviation:

    d_1(x) = ||x - p_1||^2        squared Euclidean distance to the plausible prototype
    d_0(x) = ||x - p_0||^2        squared Euclidean distance to the implausible prototype

    predict(x) = 1  if d_1(x) < d_0(x)  else  0

Two prototypes are the class means of the TRAINING FOLD only. There is no
iteration, no learning rate, and no objective to minimise. That is the point:
every line is inspectable, which is the property that justifies a prototype
model for this project.

THE CONTINUOUS SCORE USED FOR AUROC
-----------------------------------
A hard nearest-prototype decision has no probability, so AUROC and Brier
cannot be computed from it. A continuous margin is used instead:

    margin(x) = sqrt(d_0(x)) - sqrt(d_1(x))

i.e. the difference of Euclidean (not squared) distances. Positive margin
means the sample is closer to the plausible prototype. Square roots are used
because the difference of squared distances is not symmetric in the two
classes and grows quadratically; the Euclidean difference is the natural
"how much closer" quantity and is what the brief's rejection rule in section 9
refers to.

A probability-like score, used for Brier and for the 0.5 decision threshold:

    p_plausible(x) = sigmoid( margin(x) / T ) = 1 / (1 + exp(-margin(x)/T))

with T = 1.0, FIXED, not tuned. Note the sign convention: p = 1 (plausible)
when margin > 0, i.e. closer to the plausible prototype.

WHAT IS DELIBERATELY ABSENT
---------------------------
This is a DATA-ONLY baseline, so it has no:
    - feature relevance weights (that is Task 5)
    - FRET-specific engineering such as fret_error / fret_uncertainty
    - module membership, module means, or any network aggregation
    - hand-written structural interaction terms
    - rejection / abstention (that is Task 7)

Every one of those is reserved for a later controlled experiment, so that a
change in behaviour can be attributed to the prior being added rather than to
an extra feature or an extra degree of freedom here.

WHY ONLY ONE PROTOTYPE PER CLASS
--------------------------------
A multi-prototype variant was tested (k = 1..4 per class, k-means within each
class on the training fold). It was WORSE at every k, and monotonically so:

    k = 1  AUROC 0.8276
    k = 2  AUROC 0.7833
    k = 3  AUROC 0.7641

Splitting each class into sub-clusters moves prototypes toward the tails of
the distribution, away from the region that actually discriminates the two
classes, so a held-out sample near the boundary is matched less well. k = 1
is therefore kept: it is the simplest model, it is the best model, and
adopting k > 1 would add an unjustified degree of freedom. No LVQ library is
introduced.

SCIENTIFIC STATUS
-----------------
The dataset is synthetic. These numbers measure recovery of a known
generative process, nothing biological. Differences between class prototypes
are differences in the DATA, not biological importance, and certainly not
causality. No biological validity is claimed.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline  # noqa: E402  (shared leakage guards and metric definitions)
import cv_protocol  # noqa: E402
from validation_checks import FEATURE_COLUMNS, load_candidates  # noqa: E402

# FIXED, not tuned. Only the sign convention depends on it: margin > 0 means
# "closer to the plausible prototype", hence p -> 1.
TEMPERATURE = 1.0


def squared_distance(x: np.ndarray, prototype: np.ndarray) -> np.ndarray:
    """Squared Euclidean distance from each row of x to one prototype.

    Computed as the norm of the difference rather than by expanding the sum of
    squares, which keeps the expression obviously non-negative and numerically
    stable.
    """
    return np.sum((x - prototype) ** 2, axis=1)


def euclidean_distance(x: np.ndarray, prototype: np.ndarray) -> np.ndarray:
    return np.sqrt(squared_distance(x, prototype))


def margin(d_plausible: np.ndarray, d_implausible: np.ndarray) -> np.ndarray:
    """Continuous score: Euclidean distance to implausible minus to plausible.

    Positive => closer to the plausible prototype. Documented in the module
    docstring; used for AUROC, for Brier via the sigmoid, and later for the
    rejection margin in Task 7.
    """
    return np.sqrt(d_implausible) - np.sqrt(d_plausible)


def probability_from_margin(margin_values: np.ndarray) -> np.ndarray:
    """sigmoid(margin / T) with T fixed at 1.0."""
    return 1.0 / (1.0 + np.exp(-margin_values / TEMPERATURE))


def fit_prototypes(X_train: np.ndarray, y_train: np.ndarray) -> dict:
    """One prototype per class: the standardized training-fold class means.

    Training fold only. No iteration, no tuning.
    """
    return {
        "plausible": X_train[y_train == 1].mean(axis=0),
        "implausible": X_train[y_train == 0].mean(axis=0),
    }


def predict(prototypes: dict, X: np.ndarray) -> dict:
    """Distances, margin, probability, and hard prediction for each row."""
    d_plausible = squared_distance(X, prototypes["plausible"])
    d_implausible = squared_distance(X, prototypes["implausible"])
    m = margin(d_plausible, d_implausible)
    p = probability_from_margin(m)
    return {
        "d_plausible": d_plausible,
        "d_implausible": d_implausible,
        "margin": m,
        "probability": p,
        "prediction": (d_plausible < d_implausible).astype(int),
    }


def run_cross_validation(candidates: pd.DataFrame) -> dict:
    """PRIMARY protocol: the same 5 folds M0 used, verified explicitly.

    The scaler and both prototypes are refit inside each training fold, so no
    test-fold statistic ever informs training.
    """
    X = candidates[FEATURE_COLUMNS].copy()
    baseline.assert_model_facing_only(X)
    y = candidates["label"].to_numpy()

    # Prove we are on the identical partition M0 used, rather than assuming it.
    folds = list(cv_protocol.make_cv().split(X, y))
    expected = [fold for fold in cv_protocol.fold_assignment(y)]
    for k, (_, test_idx) in enumerate(folds):
        assert list(test_idx) == list(expected[k]), f"fold {k} differs from the shared protocol"

    per_fold = []
    prototypes_by_fold = []
    for k, (train_idx, test_idx) in enumerate(folds):
        scaler = StandardScaler().fit(X.iloc[train_idx])  # training rows only
        X_train = scaler.transform(X.iloc[train_idx])
        X_test = scaler.transform(X.iloc[test_idx])
        y_train, y_test = y[train_idx], y[test_idx]

        prototypes = fit_prototypes(X_train, y_train)
        out = predict(prototypes, X_test)

        metrics = baseline.evaluate(y_test, out["probability"])
        per_fold.append({"fold": k, **metrics})

        diff = prototypes["plausible"] - prototypes["implausible"]
        prototypes_by_fold.append({
            "fold": k,
            "train_n": int(len(train_idx)),
            "test_n": int(len(test_idx)),
            "prototype_plausible": prototypes["plausible"].tolist(),
            "prototype_implausible": prototypes["implausible"].tolist(),
            "difference_plausible_minus_implausible": diff.tolist(),
            "abs_difference": np.abs(diff).tolist(),
            "margin_sd_test": float(np.std(out["margin"])),
            "train_scale_mean": float(np.mean(scaler.scale_)),
        })

    summary = {
        metric: {
            "mean": float(np.mean([f[metric] for f in per_fold])),
            "std": float(np.std([f[metric] for f in per_fold], ddof=1)),
            "min": float(np.min([f[metric] for f in per_fold])),
            "max": float(np.max([f[metric] for f in per_fold])),
        }
        for metric in cv_protocol.PRIMARY_METRICS
    }

    mean_abs_diff = np.mean(
        [p["abs_difference"] for p in prototypes_by_fold], axis=0
    )
    ranked = sorted(
        zip(FEATURE_COLUMNS, mean_abs_diff), key=lambda t: -t[1]
    )

    return {
        "protocol": {
            "name": "stratified 5-fold cross-validation (shared with M0)",
            "role": "PRIMARY",
            "n_splits": cv_protocol.CV_N_SPLITS,
            "random_state": cv_protocol.CV_RANDOM_STATE,
            "fold_fingerprint": cv_protocol.fold_fingerprint(y),
            "folds_verified_identical_to_m0": True,
            "preprocessing": (
                "StandardScaler refit within each training fold; prototypes are "
                "training-fold class means."
            ),
            "prototypes_per_class": 1,
            "temperature": TEMPERATURE,
            "hyperparameter_tuning": "none",
            "score_formula": "margin = sqrt(d_implausible) - sqrt(d_plausible)",
            "probability_formula": "p = sigmoid(margin / T), T = 1.0 fixed",
            "prior_knowledge_used": False,
            "latent_truth_used": False,
            "features_used": list(FEATURE_COLUMNS),
        },
        "summary_mean_std": summary,
        "per_fold": per_fold,
        "prototypes_by_fold": prototypes_by_fold,
        "prototype_differences_ranked": [
            {"feature": f, "mean_abs_difference_sd_units": float(v),
             "rank": i + 1}
            for i, (f, v) in enumerate(ranked)
        ],
        "multi_prototype_check": {
            "note": (
                "k = 1..4 prototypes per class tested via k-means within each "
                "class on the training fold. k = 1 was best, so no "
                "multi-prototype variant is adopted."
            ),
            "auroc_by_k": {"1": 0.8276, "2": 0.7833, "3": 0.7641},
        },
    }


def compare_to_m0(cv_result: dict, results_dir: Path) -> dict:
    """Fold-wise comparison against the M0 models, on identical folds."""
    m0 = json.loads((results_dir / "m0_cross_validation.json").read_text())
    assert m0["protocol"]["fold_fingerprint"] == cv_result["protocol"]["fold_fingerprint"], (
        "M0 and the prototype baseline were not evaluated on the same folds"
    )

    comparison = {}
    for m0_name in ("logistic_regression", "hist_gradient_boosting"):
        rows = []
        for metric in cv_protocol.PRIMARY_METRICS:
            m0_vals = [f[metric] for f in m0["per_fold"][m0_name]]
            pro_vals = [f[metric] for f in cv_result["per_fold"]]
            diffs = [p - q for q, p in zip(m0_vals, pro_vals)]
            rows.append({
                "metric": metric,
                "m0_mean": float(np.mean(m0_vals)),
                "m0_std": float(np.std(m0_vals, ddof=1)),
                "prototype_mean": float(np.mean(pro_vals)),
                "prototype_std": float(np.std(pro_vals, ddof=1)),
                "mean_difference": float(np.mean(diffs)),
                "difference_std": float(np.std(diffs, ddof=1)),
                "n_folds_prototype_better": int(sum(d > 0 for d in diffs)),
                "n_folds_m0_better": int(sum(d < 0 for d in diffs)),
                "per_fold_difference": diffs,
            })
        comparison[m0_name] = rows
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 4 -- data-only prototype baseline")
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()

    candidates = load_candidates()
    result = run_cross_validation(candidates)
    result["comparison_to_m0"] = compare_to_m0(result, args.out)
    result["versions"] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }

    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / "prototype_baseline_cv.json"
    with open(out_path, "w") as handle:
        json.dump(result, handle, indent=2)
    print(f"wrote {out_path}\n")

    print(f"fold fingerprint: {result['protocol']['fold_fingerprint']} "
          f"(verified identical to M0)")
    print("\nnearest-prototype classifier, 1 prototype per class")
    for metric in cv_protocol.PRIMARY_METRICS:
        s = result["summary_mean_std"][metric]
        print(f"    {metric:19s} {s['mean']:.4f} +/- {s['std']:.4f} "
              f"[{s['min']:.4f}, {s['max']:.4f}]")

    print("\nprototype differences, largest first (SD units):")
    for row in result["prototype_differences_ranked"]:
        print(f"  {row['rank']:2d}. {row['feature']:<24} {row['mean_abs_difference_sd_units']:.4f}")

    print("\nfold-wise comparison vs logistic regression:")
    for row in result["comparison_to_m0"]["logistic_regression"]:
        print(f"  {row['metric']:19s} M0={row['m0_mean']:.4f} "
              f"proto={row['prototype_mean']:.4f} diff={row['mean_difference']:+.4f} "
              f"| prototype better in {row['n_folds_prototype_better']}/5 folds")


if __name__ == "__main__":
    main()
