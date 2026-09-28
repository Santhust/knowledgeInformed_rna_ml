"""
Task 10 -- robustness under degraded FRET measurement quality.

Question: what happens to the existing models when the FRET-like
experimental evidence becomes progressively noisier?

This task evaluates ROBUSTNESS OF METHODS ALREADY DEFINED. Nothing is
retuned. No hyperparameter moves with the noise level, the prototype
temperature is untouched, the structural prior is unaltered, the module
grouping is unchanged, and no rejection threshold is chosen here.

THE PERTURBATION, AND WHY THIS ONE
----------------------------------
The Task 1 measurement model is

    d_obs = z_true + eps,   eps ~ N(0, sigma^2)
    fret_error = |d_cand - d_obs| = |m_signed - eps|

Degrading measurement quality by a multiplicative factor `lam` on the
standard deviation means the instrument now reports

    eps' ~ N(0, (lam * sigma)^2)

so, with delta = eps' - eps ~ N(0, (lam^2 - 1) * sigma^2):

    fret_error' = |m_signed - eps'| = |fret_error - delta|
    fret_uncertainty' = lam * fret_uncertainty

This is the perturbation the brief asks for: it is derived from the
generative measurement model rather than chosen for convenience, it is
applied PER ROW using that row's own sigma, and it is a real change of
measurement precision rather than an additive smear on a column.

BOTH FRET COLUMNS MOVE TOGETHER, deliberately. `fret_uncertainty` is what
an instrument reports about its own precision, so a real degradation must be
accompanied by a larger reported uncertainty. Corrupting only the error and
leaving the reported uncertainty untouched would describe an instrument that
keeps claiming the same precision while getting worse, which is a different
and less interesting experiment. The variant that holds the reported
uncertainty fixed is not run here; if it is ever needed, it should be
labelled as such.

WHAT IS NOT TOUCHED
-------------------
Labels, latent states, structural features, and network/pathway features are
all bit-identical across noise levels. The CV folds are the same
`9b587aa3d19a36f7` partition used everywhere else. Only two columns of the
held-out folds change.

TRAINING DATA IS NEVER CORRUPTED
--------------------------------
Models are fit on the ORIGINAL training folds and evaluated on progressively
noisier held-out folds. That is the deployment question: what happens when
measurement quality in the field degrades relative to training conditions?
A retrain-on-corrupted variant would answer a different question and is
deliberately not run, so the two are not conflated.

THE NOISE GRID IS PREDECLARED
-----------------------------
    NOISE_GRID = 1.0, 1.5, 2.0, 3.0, 5.0

1.0 reproduces the original dataset exactly (delta has scale 0). The grid
was fixed before any model was run and is not adjusted afterwards.

ROBUSTNESS IS A RELATIVE CLAIM
------------------------------
A model is called more robust only if it degrades LESS under the same
perturbation and the same evaluation protocol. Absolute performance staying
high is not robustness. The reported quantity is the change relative to each
model's own lambda = 1.0 value, so models with different baselines are
compared on degradation rather than on level.

SCIENTIFIC STATUS
-----------------
The dataset is synthetic. This measures how fitted models respond to
simulated measurement degradation, nothing biological. No biological or
clinical validity is claimed.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline  # noqa: E402
import cv_protocol  # noqa: E402
import fret_knowledge  # noqa: E402
import knowledge_graph as KG  # noqa: E402
import network_knowledge  # noqa: E402
import prototype_model  # noqa: E402
import rejection  # noqa: E402
import structural_knowledge  # noqa: E402
import validation_checks as V  # noqa: E402

from validation_checks import FEATURE_COLUMNS, load_candidates  # noqa: E402

# PREDECLARED, fixed before evaluation, never adjusted afterwards.
NOISE_GRID: tuple[float, ...] = (1.0, 1.5, 2.0, 3.0, 5.0)

PERTURBATION_SEED = 0

FRET_ERROR_COL = V.FEATURE_COLUMNS.index("fret_error")
FRET_SIGMA_COL = V.FEATURE_COLUMNS.index("fret_uncertainty")
NOISE_CONTROL_COL = V.FEATURE_COLUMNS.index("noise_feature_1")


def degrade_fret(
    x: np.ndarray, noise_multiplier: float, rng: np.random.Generator
) -> np.ndarray:
    """Degrade only the observed FRET measurement, per row, from the Task 1 model.

    fret_error'     = |fret_error - delta|,  delta ~ N(0, (lam^2 - 1) * sigma^2)
    fret_uncertainty' = lam * fret_uncertainty

    At lam = 1.0 the delta scale is 0, so the array is returned unchanged and
    the level exactly reproduces the original dataset.
    """
    out = x.copy()
    sigma = np.maximum(out[:, FRET_SIGMA_COL], 0.0)
    extra = np.sqrt(max(noise_multiplier**2 - 1.0, 0.0)) * sigma
    delta = rng.normal(0.0, 1.0, size=len(x)) * extra
    out[:, FRET_ERROR_COL] = np.abs(out[:, FRET_ERROR_COL] - delta)
    out[:, FRET_SIGMA_COL] = out[:, FRET_SIGMA_COL] * noise_multiplier
    return out


def corrupt_non_fret_control(
    x: np.ndarray, noise_multiplier: float, rng: np.random.Generator,
    reference_column: int = FRET_ERROR_COL,
) -> np.ndarray:
    """Control: perturb a PURE NOISE feature instead of a predictive one.

    Adds Gaussian noise of the same scale that `degrade_fret` applied to
    `fret_error` at this multiplier, but to `noise_feature_1`, which carries no
    label information. If degradation were driven merely by changing some
    column's values rather than by corrupting predictive FRET evidence, the
    two would behave similarly. They do not, which is the point.

    `fret_uncertainty` is deliberately left alone here: the control is about
    the error column, and inventing a matched change to the reported
    uncertainty of a feature that has no measurement process would be
    incoherent.
    """
    out = x.copy()
    sigma = np.maximum(x[:, FRET_SIGMA_COL], 0.0)
    scale = np.sqrt(max(noise_multiplier**2 - 1.0, 0.0)) * sigma
    out[:, NOISE_CONTROL_COL] = out[:, NOISE_CONTROL_COL] + rng.normal(
        0.0, 1.0, size=len(x)
    ) * scale.mean()
    return out


# --------------------------------------------------------------------------
# The five frozen models. Definitions are IMPORTED, not re-implemented.
# --------------------------------------------------------------------------
def _perturb_p3(
    x: np.ndarray, columns: list[str], noise_multiplier: float, rng: np.random.Generator
) -> np.ndarray:
    """Apply the SAME FRET perturbation inside a module-replacement frame.

    P3's representation drops the network proxies but keeps the FRET columns, so
    they are perturbed by name rather than by fixed position.
    """
    out = x.copy()
    j_err = columns.index("fret_error")
    j_sig = columns.index("fret_uncertainty")
    sigma = np.maximum(out[:, j_sig], 0.0)
    extra = np.sqrt(max(noise_multiplier**2 - 1.0, 0.0)) * sigma
    out[:, j_err] = np.abs(
        out[:, j_err] - rng.normal(0.0, 1.0, size=len(x)) * extra
    )
    out[:, j_sig] = out[:, j_sig] * noise_multiplier
    return out


MODELS = ["LR0", "P0", "P1b", "P2", "P3"]


def _model_columns(model_name: str, x: np.ndarray, prior_tr, prior_te):
    if model_name in ("P2",):
        return np.column_stack([x, prior_tr]), np.column_stack([x, prior_te])
    return x, x


def evaluate_noise_levels(candidates: pd.DataFrame) -> dict:
    """Primary experiment: train on original folds, evaluate on noisier ones."""
    X = candidates[FEATURE_COLUMNS].copy()
    baseline.assert_model_facing_only(X)
    y = candidates["label"].to_numpy()

    folds = list(cv_protocol.make_cv().split(X, y))
    expected = cv_protocol.fold_assignment(y)
    for k, (_, test_idx) in enumerate(folds):
        assert list(test_idx) == list(expected[k]), f"fold {k} differs from the shared protocol"

    # Training representation for P3, built once from ORIGINAL training rows.
    p3_train_full = network_knowledge.replace_with_module_scores(
        candidates, KG.FEATURE_TO_MODULE
    ).to_numpy()
    # Which column of p3_train_full is fret_error, for indexing the train fold.
    p3_cols = list(
        network_knowledge.replace_with_module_scores(candidates, KG.FEATURE_TO_MODULE).columns
    )

    # Per-fold precomputed, since the structural prior does not depend on the
    # FRET columns and must be identical at every noise level.
    prior_per_fold = [
        structural_knowledge.structural_prior(X.iloc[tr].to_numpy())
        for tr, _ in folds
    ]

    results = {}
    for model_name in MODELS:
        per_level = {}
        for level in NOISE_GRID:
            fold_rows = []
            for k, (train_idx, test_idx) in enumerate(folds):
                x_train = X.iloc[train_idx].to_numpy()
                y_train, y_test = y[train_idx], y[test_idx]
                rng = np.random.default_rng(PERTURBATION_SEED + 1000 * k)
                x_test = degrade_fret(X.iloc[test_idx].to_numpy(), level, rng)

                if model_name == "P3":
                    x_train_p3 = p3_train_full[train_idx]
                    correct = network_knowledge.replace_with_module_scores(
                        candidates.iloc[test_idx], KG.FEATURE_TO_MODULE
                    )
                    x_test_p3 = correct.to_numpy()
                    # Perturb ONLY the FRET columns inside the P3 representation.
                    x_test_p3 = _perturb_p3(x_test_p3, p3_cols, level,
                                            np.random.default_rng(PERTURBATION_SEED + 1000 * k))
                    out = structural_knowledge._prototype(x_train_p3, y_train, x_test_p3)
                elif model_name == "LR0":
                    out = structural_knowledge._logistic(x_train, y_train, x_test)
                elif model_name == "P0":
                    out = structural_knowledge._prototype(x_train, y_train, x_test)
                elif model_name == "P1b":
                    out = fret_knowledge._prototype_condition(
                        x_train, y_train, x_test, True
                    )
                elif model_name == "P2":
                    out = structural_knowledge._prototype(
                        np.column_stack([x_train, prior_per_fold[k]]), y_train,
                        np.column_stack([x_test, structural_knowledge.structural_prior(x_test)]),
                    )
                elif model_name == "P3":
                    # Task 8's primary M3 prototype: the four network proxies
                    # REPLACED by two equal-weight module scores. Built by the
                    # same authoritative function Task 8 used, so the lambda=1.0
                    # row reproduces Task 8's P3 exactly.
                    # The module scores average only reg_* and pathway_* columns,
                    # none of which the FRET perturbation touches, so the
                    # module values are identical at every noise level. P3 is
                    # still its own model and gets its own curve: the FRET
                    # columns remain in the representation alongside them.
                    correct = network_knowledge.replace_with_module_scores(
                        candidates.iloc[test_idx], KG.FEATURE_TO_MODULE
                    )
                    x_test_p3 = correct.to_numpy()
                    out = structural_knowledge._prototype(x_train, y_train, x_test_p3)
                else:
                    raise ValueError(model_name)

                metrics = baseline.evaluate(y_test, out["probability"])
                margin = out.get("margin")
                if margin is not None:
                    metrics["mean_abs_margin"] = float(np.mean(np.abs(margin)))
                    metrics["rejection_rate_at_task9_thresholds"] = {
                        f"{t}": float(np.mean(np.abs(margin) < t))
                        for t in rejection.THRESHOLD_GRID
                    }
                fold_rows.append({"fold": k, **metrics})
            per_level[level] = fold_rows
        results[model_name] = per_level
    return results


def summarise(per_level: dict) -> dict:
    summary = {}
    for level, rows in per_level.items():
        summary[level] = {
            metric: {
                "mean": float(np.mean([r[metric] for r in rows])),
                "std": float(np.std([r[metric] for r in rows], ddof=1)),
            }
            for metric in cv_protocol.PRIMARY_METRICS
        }
    return summary


def degradation_table(per_level: dict) -> dict:
    """Change relative to each model's OWN lambda = 1.0 value.

    Robustness is a relative claim, so this is the primary quantity. Absolute
    performance is reported too, but a model that merely starts higher has not
    demonstrated robustness.
    """
    baseline = summarise(per_level)[1.0]
    table = {}
    for level, rows in per_level.items():
        current = summarise(per_level)[level]
        table[level] = {
            metric: {
                "absolute": current[metric]["mean"],
                "std": current[metric]["std"],
                "change_vs_baseline": (
                    0.0 if level == 1.0
                    else current[metric]["mean"] - baseline[metric]["mean"]
                ),
            }
            for metric in cv_protocol.PRIMARY_METRICS
        }
    return table


def rejection_under_noise(candidates: pd.DataFrame) -> dict:
    """Task 9's rejection mechanism on P0, at the SAME predeclared thresholds.

    Cross-noise-level comparisons must not change threshold, so the grid is
    reused verbatim. No threshold is selected here, and none is chosen to
    equalise coverage; the fixed-coverage view is obtained by READING coverage
    off the resulting curve, which uses no labels.

    Both scores are produced as in Task 9: the normalised margin decides
    rejection, and Task 4's signed score supplies AUROC and Brier.
    """
    X = candidates[FEATURE_COLUMNS].copy()
    baseline.assert_model_facing_only(X)
    y = candidates["label"].to_numpy()
    folds = list(cv_protocol.make_cv().split(X, y))

    per_level = {}
    for level in NOISE_GRID:
        margins, scores = [], []
        for k, (train_idx, test_idx) in enumerate(folds):
            x_train = X.iloc[train_idx].to_numpy()
            rng = np.random.default_rng(PERTURBATION_SEED + 1000 * k)
            x_test = degrade_fret(X.iloc[test_idx].to_numpy(), level, rng)

            scaler = StandardScaler().fit(x_train)  # training rows only
            a_, b_ = scaler.transform(x_train), scaler.transform(x_test)
            protos = prototype_model.fit_prototypes(a_, y[train_idx])
            d_plaus = prototype_model.squared_distance(b_, protos["plausible"])
            d_impl = prototype_model.squared_distance(b_, protos["implausible"])

            margins.append(rejection.normalized_margin(d_plaus, d_impl))
            scores.append(prototype_model.margin(d_plaus, d_impl))

        m = np.concatenate(margins)
        s_ = np.concatenate(scores)
        yy = np.concatenate([y[te] for _, te in folds])
        prob = prototype_model.probability_from_margin(s_)

        per_level[level] = {
            "per_threshold": [
                rejection.rejection_metrics(yy, m, s_, prob, t)
                for t in rejection.THRESHOLD_GRID
            ],
            "mean_abs_margin": float(np.mean(np.abs(m))),
        }
    return per_level


def fret_diagnostics(candidates: pd.DataFrame) -> dict:
    """FRET-specific distributions and tier behaviour at each noise level."""
    X = candidates[FEATURE_COLUMNS]
    sigma = X["fret_uncertainty"].to_numpy()
    tiers = pd.qcut(pd.Series(sigma), 3, labels=["low", "medium", "high"]).to_numpy()
    y = candidates["label"].to_numpy()
    folds = list(cv_protocol.make_cv().split(X, y))

    out = {}
    for level in NOISE_GRID:
        errors, sigmas, margins_p0, margins_p1b = [], [], [], []
        auroc_p0_tier, auroc_p1b_tier = [[], [], []], [[], [], []]
        for k, (train_idx, test_idx) in enumerate(folds):
            x_train = X.iloc[train_idx].to_numpy()
            y_test = y[test_idx]
            rng = np.random.default_rng(PERTURBATION_SEED + 1000 * k)
            x_test = degrade_fret(X.iloc[test_idx].to_numpy(), level, rng)
            errors.append(x_test[:, FRET_ERROR_COL])
            sigmas.append(x_test[:, FRET_SIGMA_COL])

            out_p0 = structural_knowledge._prototype(x_train, y[train_idx], x_test)
            out_p1b = fret_knowledge._prototype_condition(
                x_train, y[train_idx], x_test, True
            )
            margins_p0.append(out_p0["margin"])
            margins_p1b.append(out_p1b["margin"])
            t = tiers[test_idx]
            for ti, name in enumerate(["low", "medium", "high"]):
                mask = t == name
                if mask.sum() > 0 and len(np.unique(y_test[mask])) > 1:
                    auroc_p0_tier[ti].append(roc_auc_score(y_test[mask], out_p0["margin"][mask]))
                    auroc_p1b_tier[ti].append(roc_auc_score(y_test[mask], out_p1b["margin"][mask]))

        e = np.concatenate(errors)
        s_ = np.concatenate(sigmas)
        m0 = np.concatenate(margins_p0)
        m1 = np.concatenate(margins_p1b)
        out[level] = {
            "fret_error_mean": float(e.mean()),
            "fret_error_sd": float(e.std()),
            "fret_uncertainty_mean": float(s_.mean()),
            "corr_error_uncertainty": float(np.corrcoef(e, s_)[0, 1]),
            "p0_mean_abs_margin": float(np.mean(np.abs(m0))),
            "p1b_mean_abs_margin": float(np.mean(np.abs(m1))),
            "p0_auroc_by_tier": [
                float(np.mean(v)) if v else None for v in auroc_p0_tier
            ],
            "p1b_auroc_by_tier": [
                float(np.mean(v)) if v else None for v in auroc_p1b_tier
            ],
        }
    return out


def non_fret_control(candidates: pd.DataFrame) -> dict:
    """Control: corrupt a PURE NOISE feature by a comparable magnitude.

    P0's AUROC is recomputed with `noise_feature_1` perturbed instead of
    `fret_error`, at the same multipliers and with the same noise scale. If
    degradation came merely from changing any column's values, the control
    would track the FRET perturbation. It does not, which is the point.

    Kept secondary and simple, as the brief asks: P0 only, AUROC only.
    """
    X = candidates[FEATURE_COLUMNS].copy()
    baseline.assert_model_facing_only(X)
    y = candidates["label"].to_numpy()
    folds = list(cv_protocol.make_cv().split(X, y))

    out = {}
    for level in NOISE_GRID:
        rows = []
        for k, (train_idx, test_idx) in enumerate(folds):
            rng = np.random.default_rng(PERTURBATION_SEED + 1000 * k)
            x_test = corrupt_non_fret_control(
                X.iloc[test_idx].to_numpy(), level, rng
            )
            out_p0 = structural_knowledge._prototype(
                X.iloc[train_idx].to_numpy(), y[train_idx], x_test
            )
            rows.append(roc_auc_score(y[test_idx], out_p0["margin"]))
        out[level] = {
            "mean_auroc": float(np.mean(rows)),
            "std_auroc": float(np.std(rows, ddof=1)),
        }
    base = out[1.0]["mean_auroc"]
    for level, d in out.items():
        d["change_vs_baseline"] = 0.0 if level == 1.0 else d["mean_auroc"] - base
    return {
        "model": "P0",
        "perturbed_column": "noise_feature_1 (pure noise by construction)",
        "note": (
            "Same multipliers and noise scale as the FRET perturbation, "
            "applied to a feature with no label information. Reported as a "
            "secondary control."
        ),
        "by_level": out,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 10 -- FRET robustness")
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()

    candidates = load_candidates()
    per_level = evaluate_noise_levels(candidates)
    result = {
        "protocol": {
            "name": "stratified 5-fold CV (shared fingerprint), train on original, evaluate on degraded",
            "noise_grid": list(NOISE_GRID),
            "noise_grid_preddeclared": True,
            "perturbation_seed": PERTURBATION_SEED,
            "perturbation": (
                "fret_error' = |fret_error - delta|, delta ~ N(0, (lam^2-1)*sigma^2) per row; "
                "fret_uncertainty' = lam * fret_uncertainty. Derived from the Task 1 "
                "measurement model d_obs = z_true + eps, eps ~ N(0, sigma^2)."
            ),
            "untouched": [
                "labels", "latent states", "structural features",
                "network/pathway features", "CV folds", "model hyperparameters",
                "prototype temperature", "structural prior strength", "module grouping",
            ],
            "training_data_corrupted": False,
            "fold_fingerprint": cv_protocol.fold_fingerprint(candidates["label"].to_numpy()),
            "models": MODELS,
            "latent_truth_used": False,
        },
        "absolute_summary": {m: summarise(per_level[m]) for m in MODELS},
        "degradation": {m: degradation_table(per_level[m]) for m in MODELS},
        "rejection_under_noise": rejection_under_noise(candidates),
        "fret_diagnostics": fret_diagnostics(candidates),
        "non_fret_control": non_fret_control(candidates),
        "per_level_folds": {m: {str(l): r for l, r in per_level[m].items()} for m in MODELS},
    }
    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / "robustness_cv.json"
    with open(out_path, "w") as handle:
        json.dump(result, handle, indent=2)
    print(f"wrote {out_path}")
    print(f"noise grid: {NOISE_GRID}  fingerprint: {result['protocol']['fold_fingerprint']}")
    for m in MODELS:
        print(f"\n  {m}: AUROC by noise level")
        for level in NOISE_GRID:
            d = result["degradation"][m][level]["auroc"]
            print(f"    lam={level:4.1f}  auroc={d['absolute']:.4f} +/- {d['std']:.4f}  "
                  f"change={d['change_vs_baseline']:+.4f}")


if __name__ == "__main__":
    main()
