"""
Task 9 -- rejection / abstention for the prototype classifier.

A prototype classifier can be forced to answer, or it can be allowed to
abstain. This module adds the second option and measures the trade-off
honestly, including the trade-off's unattractive end.

TWO SCORES, AND WHY THEY ARE NOT THE SAME NUMBER
-----------------------------------------------
1. PREDICTIVE SCORE -- unchanged from Task 4, used for AUROC and Brier:

       score(x) = sqrt(d_implausible(x)) - sqrt(d_plausible(x))

   This is exactly the signed continuous score the Task 4 plain prototype
   model used, and it is retained unchanged here. At zero rejection it
   reproduces Task 4's full-coverage metrics exactly, which is asserted
   against the saved Task 4 results (see assert_matches_task4).

2. REJECTION SCORE -- the NORMALISED margin, used ONLY to decide who is
   rejected:

       margin(x) = (d_implausible(x) - d_plausible(x)) / (d_implausible(x) + d_plausible(x) + eps)

   Normalising matters for the rejection decision. The raw score's scale
   depends on how far a sample sits from the prototypes, so a distant but
   confident sample could look ambiguous purely because it is far away. The
   normalised form is a dimensionless relative gap, bounded in (-1, 1), so a
   single threshold grid means the same thing across folds.

   It is NOT used for AUROC or Brier. The two are different functions of the
   same two distances, and substituting one for the other changes the
   reported metric. Both derived from the SAME fitted prototypes, so nothing
   about the model differs between them.

The predicted class is the sign of either, since normalisation is monotone:
    predict plausible  iff  score > 0   (equivalently margin > 0)
    reject             iff  |margin| < threshold

WHAT THIS IS NOT
----------------
A probability, and not a confidence claim. The normalised margin is a
discriminative score, not a calibrated estimate of `P(plausible)`. Nothing
here demonstrates uncertainty calibration; the Brier score is reported only
as a descriptive consequence of the 0.5 decision rule.

THRESHOLDS: A PREDECLARED GRID, NOT A SELECTION
-----------------------------------------------
No single threshold is chosen or recommended. The grid below is fixed in
advance, every point on it is reported, and the low-coverage end is shown as
plainly as the high-coverage end. Selecting one operating point after seeing
the results would be choosing it on the reporting folds, so the deliverable
is the CURVE and the reader picks a point knowingly.

    THRESHOLD_GRID = 0.00, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50

0.00 is included so the no-rejection baseline is on the same axes as
everything else, and 0.50 is included so the extreme low-coverage regime is
visible rather than hidden.

MATCHED RANDOM REJECTION CONTROL
--------------------------------
For every threshold, the same NUMBER of samples is rejected at random, and
the accepted-accuracy is averaged over many repetitions. This is the control
that decides whether rejection works. Rejecting fewer samples raises
accepted-accuracy mechanically, for ANY rule whatsoever, simply because the
easy remainder is a smaller pool. Only the comparison against matched
randomness separates a rule that identifies hard cases from one that just
shrinks the pool. Fixed seed; variability across repetitions is reported as
a standard deviation so the margin-based numbers can be judged against it.

BASE MODEL
----------
The plain Task 4 prototype. NOT the Task 5 relevance-weighted model, which
showed no benefit, and no M1/M2/M3 prior. The purpose here is to characterise
the reject option itself, and mixing in a prior would confound that.

SCIENTIFIC STATUS
-----------------
The dataset is synthetic. Coverage and accepted-accuracy trade-offs measured
here are properties of a fitted model on simulated data, nothing biological.
No biological or clinical validity is claimed.
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
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline  # noqa: E402
import cv_protocol  # noqa: E402
import prototype_model  # noqa: E402
from validation_checks import FEATURE_COLUMNS, load_candidates  # noqa: E402

EPS = 1e-9

# PREDECLARED. Fixed before evaluation; no point on this grid is selected,
# recommended, or treated as optimal anywhere in this module.
THRESHOLD_GRID: tuple[float, ...] = (0.00, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50)

# Repetitions for the matched random-rejection control.
N_RANDOM_REPETITIONS = 2000
RANDOM_SEED = 0


def normalized_margin(d_plausible: np.ndarray, d_implausible: np.ndarray) -> np.ndarray:
    """(d_implausible - d_plausible) / (d_implausible + d_plausible + eps).

    Positive => closer to the plausible prototype. Bounded in (-1, 1), so one
    threshold grid is meaningful across folds.
    """
    return (d_implausible - d_plausible) / (d_implausible + d_plausible + EPS)


def rejection_metrics(
    y_true: np.ndarray,
    margin: np.ndarray,
    score: np.ndarray,
    probability: np.ndarray,
    threshold: float,
) -> dict:
    """Coverage and accepted-sample metrics at one threshold.

    `margin` (normalised) decides WHO is rejected.
    `score` (Task 4's signed score) is what AUROC and Brier are computed
    from, so that the reported discrimination and calibration are exactly
    those of the base model restricted to the accepted set. Mixing the two
    would silently change the metric being reported.

    Accepted-sample metrics are computed on accepted rows only. AUROC is
    reported only when both classes survive rejection; otherwise it is None
    rather than a misleading number. Brier is a consequence of the fixed 0.5
    rule on the Task 4 score, not a calibration claim.
    """
    rejected = np.abs(margin) < threshold
    accepted = ~rejected
    n_accepted = int(accepted.sum())
    n_rejected = int(rejected.sum())
    prediction = (margin > 0).astype(int)

    out: dict = {
        "threshold": float(threshold),
        "n": int(len(y_true)),
        "n_accepted": n_accepted,
        "n_rejected": n_rejected,
        "coverage": float(n_accepted / len(y_true)) if len(y_true) else None,
        "rejection_rate": float(n_rejected / len(y_true)) if len(y_true) else None,
    }
    if n_accepted == 0:
        out.update({
            "accepted_accuracy": None, "accepted_balanced_accuracy": None,
            "accepted_auroc": None, "accepted_f1": None, "accepted_brier": None,
            "error_rate_accepted": None, "error_rate_rejected": None,
        })
        return out

    ya, pa, sa, proba = (
        y_true[accepted], prediction[accepted],
        score[accepted], probability[accepted],
    )
    out["accepted_accuracy"] = float(accuracy_score(ya, pa))
    out["accepted_balanced_accuracy"] = float(balanced_accuracy_score(ya, pa))
    out["accepted_auroc"] = (
        float(roc_auc_score(ya, sa)) if len(np.unique(ya)) > 1 else None
    )
    out["accepted_f1"] = float(f1_score(ya, pa, zero_division=0))
    out["accepted_brier"] = float(brier_score_loss(ya, proba))
    out["error_rate_accepted"] = float(1.0 - out["accepted_accuracy"])
    out["error_rate_rejected"] = (
        float(np.mean(prediction[rejected] != y_true[rejected]))
        if n_rejected else None
    )
    return out


def matched_random_rejection(
    y_true: np.ndarray,
    margin: np.ndarray,
    n_reject: int,
    rng: np.random.Generator,
    repetitions: int = N_RANDOM_REPETITIONS,
) -> dict:
    """Accepted-accuracy when the same NUMBER of samples is rejected at random.

    The essential control. Rejecting fewer samples raises accepted-accuracy
    for any rule at all, so only the gap between this and the margin-based
    number says whether margin-based rejection identifies hard cases.
    """
    n = len(y_true)
    if n_reject == 0:
        value = float(accuracy_score(y_true, (margin > 0).astype(int)))
        return {"mean": value, "std": 0.0, "n_reject": 0, "repetitions": 0}
    if n_reject >= n:
        return {"mean": None, "std": None, "n_reject": int(n_reject),
                "repetitions": repetitions}

    prediction = (margin > 0).astype(int)
    values = []
    for _ in range(repetitions):
        drop = rng.permutation(n)[:n_reject]
        keep = np.ones(n, dtype=bool)
        keep[drop] = False
        values.append(float(accuracy_score(y_true[keep], prediction[keep])))
    arr = np.asarray(values)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)),
        "n_reject": int(n_reject),
        "repetitions": repetitions,
    }


def _fold_margins(candidates: pd.DataFrame):
    """Out-of-fold (normalised margin, Task 4 score) from the plain prototype.

    The two are different functions of the same two distances. The
    normalised margin decides rejection; the Task 4 score is what AUROC and
    Brier are computed from. Both come from the same fitted prototypes.
    """
    X = candidates[FEATURE_COLUMNS].copy()
    baseline.assert_model_facing_only(X)
    y = candidates["label"].to_numpy()
    folds = list(cv_protocol.make_cv().split(X, y))
    expected = cv_protocol.fold_assignment(y)
    for k, (_, test_idx) in enumerate(folds):
        assert list(test_idx) == list(expected[k]), f"fold {k} differs from the shared protocol"

    margins, scores, labels, fold_ids = [], [], [], []
    for k, (train_idx, test_idx) in enumerate(folds):
        scaler = StandardScaler().fit(X.iloc[train_idx])  # training rows only
        a = scaler.transform(X.iloc[train_idx])
        b = scaler.transform(X.iloc[test_idx])
        prototypes = prototype_model.fit_prototypes(a, y[train_idx])
        d_plausible = prototype_model.squared_distance(b, prototypes["plausible"])
        d_implausible = prototype_model.squared_distance(b, prototypes["implausible"])
        margins.append(normalized_margin(d_plausible, d_implausible))
        # Task 4's own signed score, unchanged.
        scores.append(prototype_model.margin(d_plausible, d_implausible))
        labels.append(y[test_idx])
        fold_ids.append(np.full(len(test_idx), k))
    return (
        np.concatenate(margins),
        np.concatenate(scores),
        np.concatenate(labels),
        np.concatenate(fold_ids),
    )


def assert_matches_task4(
    results_dir: Path, atol: float = 1e-9
) -> dict:
    """Verify zero-rejection Task 9 reproduces Task 4 EXACTLY.

    The two tasks must not drift apart: same features, same folds, same
    prototypes, same scoring rule. If they ever do, every Task 9 number is
    suspect, so this is an assertion rather than a printed comparison.
    """
    task4 = json.loads((results_dir / "prototype_baseline_cv.json").read_text())
    margin, score, y, fold_id = _fold_margins(load_candidates())
    prediction = (margin > 0).astype(int)

    probability = prototype_model.probability_from_margin(score)
    # Average PER FOLD, exactly as Task 4 aggregates. A pooled computation is
    # not the same number and the check exists precisely to catch that.
    per_fold = [
        rejection_metrics(
            y[fold_id == k], margin[fold_id == k], score[fold_id == k],
            probability[fold_id == k], 0.0,
        )
        for k in sorted(set(fold_id.tolist()))
    ]
    reference = {
        "accuracy": task4["summary_mean_std"]["accuracy"]["mean"],
        "balanced_accuracy": task4["summary_mean_std"]["balanced_accuracy"]["mean"],
        "auroc": task4["summary_mean_std"]["auroc"]["mean"],
        "f1": task4["summary_mean_std"]["f1"]["mean"],
        "brier": task4["summary_mean_std"]["brier"]["mean"],
    }
    mine = {
        "accuracy": float(np.mean([f["accepted_accuracy"] for f in per_fold])),
        "balanced_accuracy": float(
            np.mean([f["accepted_balanced_accuracy"] for f in per_fold])
        ),
        "auroc": float(np.mean([f["accepted_auroc"] for f in per_fold])),
        "f1": float(np.mean([f["accepted_f1"] for f in per_fold])),
        "brier": float(np.mean([f["accepted_brier"] for f in per_fold])),
    }

    mismatches = {
        key: {"task4": reference[key], "task9_at_zero": mine[key],
              "abs_difference": abs(reference[key] - mine[key])}
        for key in reference
        if not np.isclose(reference[key], mine[key], atol=atol, rtol=0.0)
    }
    assert not mismatches, (
        "Task 9 at zero rejection does NOT reproduce Task 4: "
        f"{json.dumps(mismatches, indent=2)}"
    )
    return {
        "checked": reference,
        "task9_at_zero_rejection": mine,
        "max_abs_difference": max(abs(reference[k] - mine[k]) for k in reference),
        "atol": atol,
        "reproduces_task4": True,
        "note": (
            "Task 9 at threshold 0.00 accepts every sample and therefore must "
            "equal Task 4 exactly, aggregating per fold in the same way. It "
            "does, to within floating point. This check is what guarantees the "
            "rejection analysis did not silently change the base model."
        ),
    }


def run_analysis(candidates: pd.DataFrame, results_dir: Path = Path("results")) -> dict:
    margin, score, y, fold_id = _fold_margins(candidates)
    prediction = (margin > 0).astype(int)
    # The DECISION is the sign of the margin, never this probability. It is
    # Task 4's sigmoid of the Task 4 score, so the Brier column has a defined,
    # documented input identical to the base model's. Reported as a descriptive
    # consequence of the fixed 0.5 rule, not as a calibration claim.
    probability = prototype_model.probability_from_margin(score)

    per_threshold = []
    for threshold in THRESHOLD_GRID:
        rows = [
            rejection_metrics(y[fold_id == k], margin[fold_id == k], score[fold_id == k],
                              probability[fold_id == k], threshold)
            for k in sorted(set(fold_id.tolist()))
        ]
        pooled = rejection_metrics(y, margin, score, probability, threshold)
        control = matched_random_rejection(
            y, margin, pooled["n_rejected"],
            np.random.default_rng(RANDOM_SEED),
        )
        gain = (
            pooled["accepted_accuracy"] - control["mean"]
            if pooled["accepted_accuracy"] is not None and control["mean"] is not None
            else None
        )
        per_threshold.append({
            "threshold": float(threshold),
            "pooled": pooled,
            "per_fold": rows,
            "random_control": control,
            "margin_gain_over_random": gain,
        })

    ambiguity = _ambiguity_profile(margin, y, prediction)
    ambiguity["feature_profile"] = ambiguity_features(candidates, margin)

    task4_check = assert_matches_task4(results_dir)

    return {
        "protocol": {
            "name": "stratified 5-fold cross-validation (shared with all previous tasks)",
            "role": "PRIMARY",
            "n_splits": cv_protocol.CV_N_SPLITS,
            "random_state": cv_protocol.CV_RANDOM_STATE,
            "fold_fingerprint": cv_protocol.fold_fingerprint(candidates["label"].to_numpy()),
            "folds_verified_identical": True,
            "base_model": "plain nearest prototype (Task 4), no relevance weights, no M1/M2/M3 prior",
            "rejection_score": (
                "normalised margin = (d_implausible - d_plausible) / "
                "(d_implausible + d_plausible + eps), used ONLY to decide rejection"
            ),
            "predictive_score": (
                "Task 4 signed score = sqrt(d_implausible) - sqrt(d_plausible), "
                "used for AUROC and Brier so the base model's own metric is preserved"
            ),
            "decision_rule": "predict plausible iff margin > 0; reject iff |margin| < threshold",
            "threshold_grid": list(THRESHOLD_GRID),
            "threshold_selection": (
                "NONE. The grid is predeclared and every point is reported. No "
                "threshold is chosen, recommended, or treated as optimal, so no "
                "result here is selected on the reporting folds."
            ),
            "random_control": {
                "repetitions": N_RANDOM_REPETITIONS,
                "seed": RANDOM_SEED,
                "matched_on": "number of rejected samples at each threshold",
            },
            "calibration_claim": (
                "NONE. The margin is a discriminative score, not a calibrated "
                "P(plausible). Brier is reported as a descriptive consequence "
                "of the fixed 0.5 rule only."
            ),
            "latent_truth_used": False,
        },
        "overall": {
            "n": int(len(y)),
            "accuracy_all_samples": float(np.mean(prediction == y)),
            "auroc_all_samples": (
                float(roc_auc_score(y, score)) if len(np.unique(y)) > 1 else None
            ),
            "rejection_score": "normalised margin, used only to decide rejection",
            "predictive_score": "Task 4 signed score, used for AUROC and Brier",
            "margin_min": float(margin.min()),
            "margin_max": float(margin.max()),
            "margin_sd": float(margin.std()),
        },
        "task4_reproduction_check": task4_check,
        "per_threshold": per_threshold,
        "ambiguity_profile": ambiguity,
    }


def _ambiguity_profile(margin: np.ndarray, y: np.ndarray, prediction: np.ndarray) -> dict:
    """Do rejected samples look genuinely harder? Descriptive only.

    Uses OBSERVABLE features only. No latent truth here, and nothing in this
    profile is used to choose a threshold.
    """
    order = np.argsort(np.abs(margin))
    quintile = np.zeros(len(margin), dtype=int)
    quintile[order] = np.minimum(
        (np.arange(len(margin)) * 5) // len(margin), 4
    )
    rows = []
    for q in range(5):
        mask = quintile == q
        rows.append({
            "abs_margin_quintile": q + 1,
            "n": int(mask.sum()),
            "mean_abs_margin": float(np.abs(margin[mask]).mean()),
            "error_rate": float(np.mean(prediction[mask] != y[mask])),
            "plausible_rate": float(y[mask].mean()),
        })
    return {
        "note": (
            "Descriptive, observable features only. Not used to select a "
            "threshold. Groups are quintiles of |margin|, so group 1 holds the "
            "most ambiguous samples and group 5 the clearest."
        ),
        "quintiles": rows,
    }


def ambiguity_features(
    candidates: pd.DataFrame, margin: np.ndarray
) -> dict:
    """Observable-feature profile of rejected vs accepted samples.

    Answers the descriptive question of whether rejected candidates differ on
    FRET uncertainty, structural ambiguity, or module evidence. Uses only
    observable features and is used for nothing else -- no threshold, no
    model, and no reported metric depends on it.
    """
    X = candidates[FEATURE_COLUMNS]
    rejected = np.abs(margin) < 0.10
    prediction = (margin > 0).astype(int)
    y = candidates["label"].to_numpy()

    # Derived observable summaries. The module means use the same
    # authoritative grouping as Task 8; this is description, not modelling.
    regulatory = X[["reg_feature_1", "reg_feature_2"]].mean(axis=1)
    pathway = X[["pathway_feature_1", "pathway_feature_2"]].mean(axis=1)
    stem_deviation = (X["stem_count"] - 8.0) / 1.5

    derived = {
        "fret_uncertainty": X["fret_uncertainty"],
        "fret_error": X["fret_error"],
        "structural_consistency": X["structural_consistency"],
        "stem_count": X["stem_count"],
        "stem_deviation_from_8": stem_deviation,
        "regulatory_module_score": regulatory,
        "pathway_module_score": pathway,
        "gc_content": X["gc_content"],
        "noise_feature_1": X["noise_feature_1"],
    }

    rows = []
    for name, values in derived.items():
        v = values.to_numpy()
        rows.append({
            "feature": name,
            "mean_rejected": float(v[rejected].mean()),
            "mean_accepted": float(v[~rejected].mean()),
            "difference": float(v[rejected].mean() - v[~rejected].mean()),
            "pooled_sd": float(v.std()),
            "standardised_difference": float(
                (v[rejected].mean() - v[~rejected].mean()) / v.std()
            ),
        })

    return {
        "threshold_used_for_this_description": 0.10,
        "note": (
            "DESCRIPTIVE ONLY, observable features only, no latent truth. "
            "A fixed descriptive threshold of 0.10 is used so the rejected "
            "set is large enough to summarise; it is NOT a selected operating "
            "point and nothing here feeds a threshold or a reported metric."
        ),
        "n_rejected": int(rejected.sum()),
        "n_accepted": int((~rejected).sum()),
        "rejected_error_rate": float(np.mean(prediction[rejected] != y[rejected])),
        "accepted_error_rate": float(
            np.mean(prediction[~rejected] != y[~rejected])
        ),
        "feature_profile": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 9 -- prototype rejection")
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()

    candidates = load_candidates()
    result = run_analysis(candidates, args.out)
    result["versions"] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }

    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / "rejection_cv.json"
    with open(out_path, "w") as handle:
        json.dump(result, handle, indent=2)
    print(f"wrote {out_path}\n")
    print(f"fold fingerprint: {result['protocol']['fold_fingerprint']} (verified shared)")
    print(f"threshold selection: {result['protocol']['threshold_selection']}")
    chk = result["task4_reproduction_check"]
    print(f"Task 4 reproduction at zero rejection: OK "
          f"(max abs difference {chk['max_abs_difference']:.2e})\n")

    print(f"{'thr':>5} {'coverage':>9} {'acc_acc':>9} {'bal_acc':>9} {'auroc':>8} {'f1':>7} "
          f"{'brier':>7} {'err_rej':>8} {'err_acc':>8} {'rand_acc':>9} {'gain':>8}")
    for row in result["per_threshold"]:
        p = row["pooled"]
        c = row["random_control"]

        def fmt(value, spec=".4f"):
            return "n/a" if value is None else format(value, spec)

        print(f"{p['threshold']:5.2f} {fmt(p['coverage']):>9} {fmt(p['accepted_accuracy']):>9} "
              f"{fmt(p['accepted_balanced_accuracy']):>9} {fmt(p['accepted_auroc']):>8} "
              f"{fmt(p['accepted_f1']):>7} {fmt(p['accepted_brier']):>7} "
              f"{fmt(p['error_rate_rejected']):>8} {fmt(p['error_rate_accepted']):>8} "
              f"{fmt(c['mean']):>9} {fmt(row['margin_gain_over_random']):>8}")

    print("\nerror rate by |margin| quintile (1 = most ambiguous, 5 = clearest):")
    for q in result["ambiguity_profile"]["quintiles"]:
        print(f"  Q{q['abs_margin_quintile']}  n={q['n']:3d}  mean|margin|={q['mean_abs_margin']:.4f}  "
              f"error={q['error_rate']:.4f}")


if __name__ == "__main__":
    main()
