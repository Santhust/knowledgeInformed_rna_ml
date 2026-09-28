"""
Task 3 -- M0, the data-only baseline.

M0 is the reference point for every later model condition (M1/M2/M3). It sees
ONLY the twelve observable feature columns of `data/generated/candidates.csv`.

WHAT M0 IS ALLOWED TO USE
    - the twelve observable features, standardised
    - a reproducible stratified train/test split
    - fixed random seeds

WHAT M0 IS DELIBERATELY NOT ALLOWED TO USE
    - `latent_truth.csv` in any form (asserted, see the leakage safeguards)
    - module memberships or pathway aggregation
    - any FRET-specific engineered ratio, e.g. fret_error / fret_uncertainty
    - any hand-written structural interaction feature
    - any knowledge-informed relevance weight
    - rejection / abstention (that is Task 7)

The engineered comparisons listed above are intentionally withheld so that
later stages can attribute any change in behaviour to the added prior rather
than to extra features. See RESERVED_COMPARISONS below.

TWO BASELINE MODELS, DELIBERATELY DIFFERENT IN KIND
    1. LogisticRegression, standardised, L2 as shipped. Linear, inspectable.
    2. HistGradientBoostingClassifier, defaults. Captures nonlinearity and
       interactions WITHOUT hand-engineering them, so it is a fair control
       for M2's structural prior: if M2 adds nothing beyond what a generic
       nonlinear learner already finds, that must be visible.

SCIENTIFIC STATUS
    The dataset is synthetic. Accuracy here measures recovery of a known
    generative process, nothing biological. A high M0 score does not mean
    the biology is understood, and no biological validity is claimed.
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
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv_protocol  # noqa: E402
from validation_checks import FEATURE_COLUMNS, load_candidates  # noqa: E402

RANDOM_STATE = cv_protocol.CV_RANDOM_STATE

# Intentionally NOT built in Task 3. Each is reserved for a later controlled
# experiment so that any performance change is attributable to the prior
# being tested, not to an extra feature.
RESERVED_COMPARISONS = {
    "fret_error_over_uncertainty": "fret_error / fret_uncertainty",
    "regulatory_module_mean": "mean(reg_feature_1, reg_feature_2)",
    "pathway_module_mean": "mean(pathway_feature_1, pathway_feature_2)",
    "structural_interaction": "stem_count x structural_consistency term",
    "prototype_distances": "distance to learned prototypes",
    "knowledge_weights": "relevance weights from prior knowledge",
}


# --------------------------------------------------------------------------
# Leakage safeguards
# --------------------------------------------------------------------------
def assert_model_facing_only(
    frame: pd.DataFrame, latent_columns: set[str] | None = None
) -> None:
    """Fail loudly if the feature matrix is contaminated or mis-specified."""
    latent_columns = latent_columns or set(LATENT_ONLY_COLUMNS)

    leaked = sorted(set(frame.columns) & latent_columns)
    assert not leaked, f"latent-truth columns present in feature matrix: {leaked}"

    unexpected = sorted(set(frame.columns) - set(FEATURE_COLUMNS))
    assert not unexpected, f"unexpected columns in feature matrix: {unexpected}"

    missing = sorted(set(FEATURE_COLUMNS) - set(frame.columns))
    assert not missing, f"agreed observable features missing: {missing}"

    assert "label" not in frame.columns, "label must not appear in the features"
    assert "sample_id" not in frame.columns, "sample_id must not appear in the features"


# Names that exist in latent_truth.csv but must never reach a model. Kept as a
# literal list so the check does not require READING the diagnostic file.
LATENT_ONLY_COLUMNS = {
    "q", "a_A", "a_B", "a_C", "a_D", "z_true", "sigma", "m_signed", "m_mag",
    "d_cand", "d_obs", "f_true", "s_true", "c_true", "score", "p_plausible",
}


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
def evaluate(y_true: np.ndarray, proba: np.ndarray) -> dict:
    """All reported metrics for one train or test split.

    Includes a self-consistency assertion: accuracy must equal (TN + TP) / n,
    which catches any confusion-matrix reporting error immediately.
    """
    y_pred = (proba >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in cm.ravel())

    accuracy = float(accuracy_score(y_true, y_pred))
    # Internal consistency: accuracy must reconcile with the confusion matrix.
    assert np.isclose(accuracy, (tn + tp) / len(y_true), atol=1e-12), (
        f"accuracy {accuracy} does not equal (TN+TP)/n = {(tn + tp) / len(y_true)}; "
        "the confusion matrix and accuracy disagree"
    )

    return {
        "accuracy": accuracy,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "auroc": float(roc_auc_score(y_true, proba)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "brier": float(brier_score_loss(y_true, proba)),
        "confusion_matrix": {
            "labels": [0, 1],
            "layout": "[[TN, FP], [FN, TP]]",
            "counts": cm.tolist(),
            "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "n": int(len(y_true)),
            "accuracy_from_counts": float((tn + tp) / len(y_true)),
        },
    }


def performance_gap(train_metrics: dict, test_metrics: dict) -> dict:
    return {
        key: float(train_metrics[key] - test_metrics[key])
        for key in ("accuracy", "balanced_accuracy", "auroc", "f1", "brier")
    }


def logistic_coefficients(model: Pipeline) -> pd.DataFrame:
    """Standardized coefficients.

    These are predictive associations in a fitted linear model. They are NOT
    causal claims, and a non-zero coefficient does not mean the feature drives
    the label.
    """
    coefs = model.named_steps["logisticregression"].coef_.ravel()
    return pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "coefficient": coefs,
            "abs_coefficient": np.abs(coefs),
            "sign": np.sign(coefs).astype(int),
        }
    ).sort_values("abs_coefficient", ascending=False, ignore_index=True)


def build_models() -> dict:
    """The three baseline configurations.

    Exactly two are primary: logistic regression and the early-stopped
    boosting model. The library-default boosting variant is carried as a
    diagnostic so its overfitting stays visible rather than being hidden.

    Specifications are fixed and must not be tuned later, so that M0-M3 remain
    comparable.
    """
    return {
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("logisticregression", LogisticRegression()),
            ]
        ),
        # Primary nonlinear baseline. Early stopping is enabled for one stated
        # technical reason: with library defaults the ensemble reaches
        # train AUROC 1.0000 on 600 rows, i.e. it memorises the training set.
        # A model that fits its training data perfectly cannot serve as a fair
        # control for later knowledge-informed comparisons.
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            random_state=RANDOM_STATE,
        ),
        # Diagnostic only. Not a tuned model; the untouched library default.
        "hist_gradient_boosting_default": HistGradientBoostingClassifier(),
    }


def run_holdout_diagnostic(X: pd.DataFrame, y: np.ndarray) -> dict:
    """Fit on the 75/25 holdout and return every metric.

    DIAGNOSTIC ONLY. See `cv_protocol.make_holdout_split` for why this split
    is not an untouched final test set.
    """
    train_idx, test_idx = cv_protocol.make_holdout_split(y)
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    results: dict = {}
    for name, model in build_models().items():
        model.fit(X_train, y_train)
        train_proba = model.predict_proba(X_train)[:, 1]
        test_proba = model.predict_proba(X_test)[:, 1]

        train_metrics = evaluate(y_train, train_proba)
        test_metrics = evaluate(y_test, test_proba)

        results[name] = {
            "train": train_metrics,
            "test": test_metrics,
            "train_minus_test": performance_gap(train_metrics, test_metrics),
        }

    # Coefficients: the scaler standardises, so these are per-standard-deviation.
    results["logistic_regression"]["standardized_coefficients"] = (
        logistic_coefficients(build_models()["logistic_regression"].fit(X_train, y_train))
        .to_dict("records")
    )

    # Nonlinear model: no coefficient-based importance exists for a tree
    # ensemble, so use permutation importance on the held-out rows.
    # Model-agnostic, uses no latent truth. Predictive relevance only.
    perm = permutation_importance(
        build_models()["hist_gradient_boosting"].fit(X_train, y_train),
        X_test,
        y_test,
        n_repeats=30,
        random_state=RANDOM_STATE,
        scoring="roc_auc",
    )
    results["hist_gradient_boosting"]["permutation_importance"] = {
        "method": "permutation importance, 30 repeats, scored by AUROC",
        "computed_on": "holdout diagnostic split",
        "interpretation": (
            "Model-agnostic predictive relevance. NOT biological causality, and "
            "not evidence that a feature influences the underlying process. "
            "Computed without latent truth."
        ),
        "features": [
            {
                "feature": feature,
                "importance_mean": float(perm.importances_mean[i]),
                "importance_std": float(perm.importances_std[i]),
            }
            for i, feature in enumerate(FEATURE_COLUMNS)
        ],
    }
    return results


def run_cross_validation(X: pd.DataFrame, y: np.ndarray) -> dict:
    """PRIMARY protocol: deterministic stratified 5-fold CV.

    Both primary models are evaluated on exactly the folds defined by
    `cv_protocol`, and each fold is refit from scratch so the scaler is fit
    only on that fold's training rows. No tuning of any kind.
    """
    folds = list(cv_protocol.make_cv().split(X, y))
    fold_ids = [int(test_idx[0]) for _, test_idx in folds]  # bookkeeping only
    primary = ["logistic_regression", "hist_gradient_boosting"]

    per_model: dict = {}
    for name in primary:
        fold_metrics = []
        for k, (train_idx, test_idx) in enumerate(folds):
            # A fresh model per fold: the scaler must never see the test rows.
            model = build_models()[name]
            model.fit(X.iloc[train_idx], y[train_idx])
            proba = model.predict_proba(X.iloc[test_idx])[:, 1]
            fold_metrics.append({"fold": k, **evaluate(y[test_idx], proba)})
        per_model[name] = fold_metrics

    summary = {}
    for name, fold_metrics in per_model.items():
        summary[name] = {
            metric: {
                "mean": float(np.mean([f[metric] for f in fold_metrics])),
                "std": float(np.std([f[metric] for f in fold_metrics], ddof=1)),
                "min": float(np.min([f[metric] for f in fold_metrics])),
                "max": float(np.max([f[metric] for f in fold_metrics])),
            }
            for metric in cv_protocol.PRIMARY_METRICS
        }

    return {
        "protocol": {
            "name": "stratified 5-fold cross-validation",
            "role": "PRIMARY protocol for M0-M3 comparison",
            "n_splits": cv_protocol.CV_N_SPLITS,
            "random_state": cv_protocol.CV_RANDOM_STATE,
            "shuffle": True,
            "fold_fingerprint": cv_protocol.fold_fingerprint(y),
            "n": int(len(y)),
            "preprocessing": (
                "StandardScaler is inside the Pipeline, refit within each "
                "training fold. The test fold is never used for fitting."
            ),
            "hyperparameter_tuning": "none; specifications fixed in build_models()",
            "models": primary,
            "reusable_by": (
                "M1/M2/M3 must call cv_protocol.make_cv() and may assert "
                "cv_protocol.assert_identical_folds(y, folds) to prove they "
                "used the same partition."
            ),
        },
        "fold_test_index_first_row": fold_ids,
        "summary_mean_std": summary,
        "per_fold": per_model,
    }


def run_baseline(candidates: pd.DataFrame, run_cv: bool = True) -> dict:
    """Return the full M0 result: holdout diagnostic plus primary CV."""
    X = candidates[FEATURE_COLUMNS].copy()
    assert_model_facing_only(X)
    y = candidates["label"].to_numpy()

    holdout = run_holdout_diagnostic(X, y)
    train_idx, test_idx = cv_protocol.make_holdout_split(y)
    y_train, y_test = y[train_idx], y[test_idx]

    class_counts = {
        "scope": (
            "counts for the holdout diagnostic split; the primary CV protocol "
            "uses all 800 rows, see cross_validation.protocol.n"
        ),
        "overall": {
            "n": int(len(y)),
            "n_plausible": int(y.sum()),
            "n_implausible": int((1 - y).sum()),
            "plausible_rate": float(y.mean()),
        },
        "holdout_train": {
            "n": int(len(y_train)),
            "n_plausible": int(y_train.sum()),
            "n_implausible": int((1 - y_train).sum()),
            "plausible_rate": float(y_train.mean()),
        },
        "holdout_test": {
            "n": int(len(y_test)),
            "n_plausible": int(y_test.sum()),
            "n_implausible": int((1 - y_test).sum()),
            "plausible_rate": float(y_test.mean()),
        },
    }
    results: dict = {
        "protocol": {
            "primary_protocol": "5-fold stratified cross-validation (see cross_validation)",
            "diagnostic_protocol": "75/25 fixed holdout diagnostic split",
            "diagnostic_split_caveat": (
                "The 75/25 holdout is NOT an untouched final test set. Its "
                "held-out results were inspected while deciding how to handle "
                "the overfitting library-default boosting configuration, so the "
                "split is no longer independent of that modelling decision. It "
                "is retained for transparency and to keep the overfitting "
                "visible. It has NOT been re-run with a different seed, and must "
                "not be: cross-validation is the primary comparison protocol."
            ),
            "holdout_train_fraction": 1.0 - cv_protocol.HOLDOUT_TEST_SIZE,
            "holdout_test_fraction": cv_protocol.HOLDOUT_TEST_SIZE,
            "holdout_random_state": cv_protocol.HOLDOUT_RANDOM_STATE,
            "preprocessing_fit_on": "training data only (StandardScaler inside Pipeline)",
            "hyperparameter_tuning": (
                "none. Logistic regression uses library defaults. The primary "
                "nonlinear baseline uses library defaults plus training-only "
                "early stopping, enabled because the default ensemble reached "
                "train AUROC 1.0000 (memorisation). The default variant is also "
                "reported so the overfitting stays visible."
            ),
            "features_used": list(FEATURE_COLUMNS),
            "reserved_comparisons_withheld": RESERVED_COMPARISONS,
            "latent_truth_used": False,
        },
        "class_counts": class_counts,
    }

    results["holdout_diagnostic"] = holdout

    if run_cv:
        results["cross_validation"] = run_cross_validation(X, y)

    results["versions"] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 3 -- M0 data-only baseline")
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--no-cv", action="store_true",
                        help="Skip cross-validation (diagnostic use only).")
    args = parser.parse_args()

    candidates = load_candidates()
    results = run_baseline(candidates, run_cv=not args.no_cv)

    args.out.mkdir(parents=True, exist_ok=True)
    payloads = [("m0_baseline_metrics.json", results)]
    if "cross_validation" in results:
        payloads.append(("m0_cross_validation.json", results["cross_validation"]))
    for filename, payload in payloads:
        out_path = args.out / filename
        with open(out_path, "w") as handle:
            json.dump(payload, handle, indent=2)
        print(f"wrote {out_path}")

    holdout = results["holdout_diagnostic"]
    print("\nHOLDOUT DIAGNOSTIC (75/25, not an untouched final test set)")
    print("class counts:", json.dumps(results["class_counts"], indent=2))
    for name in (
        "logistic_regression",
        "hist_gradient_boosting",
        "hist_gradient_boosting_default",
    ):
        print(f"\n  {name}")
        for split in ("train", "test"):
            m = holdout[name][split]
            cm = m["confusion_matrix"]
            print(
                f"    {split:5s} acc={m['accuracy']:.4f} bal_acc={m['balanced_accuracy']:.4f} "
                f"auroc={m['auroc']:.4f} prec={m['precision']:.4f} rec={m['recall']:.4f} "
                f"f1={m['f1']:.4f} brier={m['brier']:.4f} "
                f"cm={cm['counts']} (n={cm['n']})"
            )
        gap = holdout[name]["train_minus_test"]
        print(
            f"    gap   acc={gap['accuracy']:+.4f} auroc={gap['auroc']:+.4f} "
            f"f1={gap['f1']:+.4f} brier={gap['brier']:+.4f}"
        )

    if "cross_validation" not in results:
        return
    cv = results["cross_validation"]
    print(f"\nPRIMARY PROTOCOL: {cv['protocol']['n_splits']}-fold stratified CV")
    print(f"  fold fingerprint: {cv['protocol']['fold_fingerprint']}")
    for name, summary in cv["summary_mean_std"].items():
        print(f"\n  {name}")
        for metric in cv_protocol.PRIMARY_METRICS:
            s = summary[metric]
            print(
                f"    {metric:19s} {s['mean']:.4f} +/- {s['std']:.4f} "
                f"[{s['min']:.4f}, {s['max']:.4f}]"
            )


if __name__ == "__main__":
    main()
