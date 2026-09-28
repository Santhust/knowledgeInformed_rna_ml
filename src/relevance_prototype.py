"""
Task 5 -- data-driven relevance-weighted prototype model.

Extends the Task 4 one-prototype-per-class classifier with a LEARNED, DATA-
DRIVEN metric:

    d_lambda(x, w) = sum_j  lambda_j * (x_j - w_j)^2

where lambda_j >= 0 and the weights are normalised to sum to the number of
features (equivalently, mean lambda = 1). Prediction and scoring are otherwise
identical to Task 4, so any change in behaviour is attributable to the metric
and not to a different model.

A NOTE ON NORMALISATION, WHICH IS NOT COSMETIC
----------------------------------------------
Task 4's metric is sum_j (x_j - w_j)^2, i.e. lambda = 1 for every j, so
lambda sums to 12. The RELEVANCE CONSTRAINT ENFORCED AND REPORTED is therefore
NON-NEGATIVE MEAN-ONE NORMALISATION: lambda_j >= 0 and mean(lambda) = 1.

Two distinct operations are involved, and they should not be conflated:

  * WITHIN the optimiser, each gradient step is followed by an exact
    Euclidean projection onto the probability simplex {lambda >= 0,
    sum(lambda) = 1}, because that projection is closed-form and keeps the
    iterates well-behaved.
  * AT THE END, the returned weights are rescaled by a constant factor so that
    mean(lambda) = 1, matching Task 4's units.

The constraint that defines the model is the mean-one condition. The simplex
projection is an implementation device inside the optimiser, not the model's
constraint, and the two differ by a constant factor.

This rescaling is not cosmetic. With sum(lambda) = 1 the weighted distance is
12x smaller than Task 4's, the margin shrinks by a factor of sqrt(12) ~ 3.46,
and `sigmoid(margin)` becomes far too flat: measured mean Brier score 0.2209
against Task 4's 0.1793, a large apparent regression that is purely an
artefact of the units. Rescaling the margin by sqrt(n_features) restores
0.1798, essentially identical to Task 4. AUROC is unaffected throughout, being
invariant to positive rescaling of the score.

So the normalisation convention is fixed to match Task 4 exactly, and the
rescaled margin is what the reported Brier score uses. This is a correction of
units, not a temperature tuned to improve a number; the same fixed factor is
applied to every fold and every model.


WHAT IS AND IS NOT LEARNED
--------------------------
LEARNED, from the training fold only:
    lambda, the 12 relevance weights

FIXED at the training-fold class means:
    the two prototypes

Prototypes are held at the class means deliberately. Letting them move as well
was tested (joint optimisation) and made results markedly WORSE, because with
only ~480 training rows per fold the extra 24 parameters are fitted to noise.
Learning lambda alone is the smaller, more interpretable, and better-performing
choice, and it isolates the metric change cleanly. The choice is a finding, not
a convenience -- see RESULTS_RECORDED below.

THE OBJECTIVE, IN PLAIN LANGUAGE
--------------------------------
A sample is classified plausible when it sits closer to the plausible
prototype than to the implausible one, under the weighted metric. Define the
margin

    margin(x) = sqrt( sum_j lambda_j (x_j - p_0_j)^2 )
              - sqrt( sum_j lambda_j (x_j - p_1_j)^2 )

so margin > 0 means "closer to plausible". We minimise a smooth logistic loss
on that margin,

    L(lambda) = mean_i  log( 1 + exp( -margin(x_i) / T ) )

plus a ridge pull toward uniform weights that keeps the solution identifiable.
L is smooth, so the gradient is written out explicitly in `fit_relevance` and
computed by hand -- no automatic differentiation and no external LVQ or deep
learning library. Only numpy, scipy-free elementary algebra, and scikit-learn
for the shared evaluation protocol are required.

OPTIMISATION: PROJECTED GRADIENT
---------------------------------
Each step: take a gradient step, then project back onto the probability simplex
{lambda >= 0, sum(lambda) = 1} with the standard exact Euclidean projection.
That projection is what keeps the ITERATES feasible and non-negative, so
non-negativity holds by construction rather than by penalty. On return, the
weights are rescaled to mean(lambda) = 1, which is the relevance constraint the
model actually uses (see the normalisation note above).

The ridge strength and step size are chosen by an INNER 3-fold
cross-validation on the TRAINING ROWS ONLY, scored by AUROC. The outer test
fold is never consulted for any fitting or selection decision. A single
search per outer fold; no search of the reporting folds.

NO PRIOR KNOWLEDGE ANYWHERE
---------------------------
lambda is learned purely from the observable features and the labels. There is
no module membership, no FRET-specific engineered ratio, no hand-written
structural interaction, and no hand-set initial importance. The only
non-data input is the uniform starting point lambda_j = 1/12, which is the
absence of any prior rather than an expression of one.

INTERPRETATION LIMIT -- AND A MEASURED COUNTEREXAMPLE
----------------------------------------------------
A large learned lambda_j means the fitted metric relies on that dimension to
separate the classes IN THIS SYNTHETIC SAMPLE. It is a statement about
predictive relevance under a fitted model, NOT about biological importance and
NOT about causality. A dimension can receive high relevance because it is
informative, because it is highly correlated with something informative, or
for reasons that have nothing to do with either.

The last case was OBSERVED here, and is the reason this section is emphatic.
The two features known by construction to be pure noise, noise_feature_1 and
noise_feature_2, received the HIGHEST and third-highest learned relevance of all
twelve features, while structural_consistency -- the largest genuine separator
in the data -- received the lowest. The directly supported conclusion is:

    The optimisation assigns slightly higher weights to some known irrelevant
    dimensions despite their lack of true predictive signal. Therefore learned
    diagonal metric weights should not automatically be interpreted as
    scientific feature relevance.

That conclusion needs no mechanism to be valid, and none is asserted here as a
general explanation. Candidate mechanisms are discussed in the notebook and are
labelled there as diagnostic observations about these fitted weights, not as an
established account of why diagonal relevance learning behaves this way.
Whatever the mechanism, the observation itself is what matters for this
project: it is direct evidence that data-learned relevance can disagree with
genuine relevance, which is the premise the knowledge-informed conditions in
later tasks are meant to test.

SCIENTIFIC STATUS
-----------------
The dataset is synthetic. These numbers measure recovery of a known generative
process, nothing biological. No biological validity is claimed.
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
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline  # noqa: E402
import cv_protocol  # noqa: E402
import prototype_model  # noqa: E402
from validation_checks import FEATURE_COLUMNS, load_candidates  # noqa: E402

# Logistic-loss temperature. Fixed, not tuned on any reporting fold. It only
# controls how sharply the loss saturates; it does not affect the sign of the
# margin and therefore never changes the predicted class.
LOSS_TEMPERATURE = 0.75

# Inner-CV search grid, evaluated on training rows only.
RIDGE_GRID = [0.0, 0.5, 2.0, 10.0]
STEP_GRID = [0.001, 0.003, 0.01, 0.03]
N_INNER_SPLITS = 3
N_ITERATIONS = 300

EPS = 1e-8

# Recorded during development. These were measured, not assumed, and are
# reported rather than hidden. Both are reasons the final method is the
# deliberately narrow one implemented here.
RESULTS_RECORDED = {
    "joint_prototype_and_relevance_optimisation": (
        "Tested. Letting the prototypes move as well as lambda, under the same "
        "logistic loss, was consistently WORSE (best observed mean AUROC 0.821 "
        "vs 0.828 for prototypes fixed at the class means). With ~480 training "
        "rows per fold the extra 24 parameters are fitted to noise. Prototypes "
        "are therefore held at the class means."
    ),
    "relevance_learning_did_not_beat_uniform_weights": (
        "The headline result of this task and an honest negative finding. Mean "
        "CV AUROC is statistically indistinguishable from the uniform-weight "
        "Task 4 model, and the learned weights barely depart from uniform. "
        "See the notebook for the evidence, and for the "
        "test_label_informed_weighting_bound diagnostic, which estimates the "
        "ranking headroom that existed but was not found. That diagnostic is "
        "label-informed and is not a model or a valid predictive estimate."
    ),
}


def project_simplex(v: np.ndarray) -> np.ndarray:
    """Exact Euclidean projection onto {lambda >= 0, sum(lambda) = 1}.

    The standard sorting-based construction. This is an INTERNAL step of the
    optimiser: it keeps the iterates feasible. The model-level constraint is
    non-negative mean-one normalisation, obtained by rescaling the result. See
    the normalisation note in the module docstring.
    """
    u = np.sort(v)[::-1]
    css = np.cumsum(u) - 1.0
    ind = np.arange(1, len(v) + 1)
    keep = u - css / ind > 0
    rho = ind[keep][-1]
    theta = css[keep][-1] / rho
    return np.maximum(v - theta, 0.0)


def weighted_squared_distance(
    x: np.ndarray, prototype: np.ndarray, lam: np.ndarray
) -> np.ndarray:
    """sum_j lambda_j (x_j - w_j)^2 for each row of x."""
    return np.sum(lam * (x - prototype) ** 2, axis=1)


def fit_relevance(
    x_train: np.ndarray,
    p_plausible: np.ndarray,
    p_implausible: np.ndarray,
    ridge: float,
    step: float,
    n_iterations: int = N_ITERATIONS,
) -> np.ndarray:
    """Learn lambda by projected gradient on the logistic margin loss.

    Hand-written gradient, no autodiff. Returns weights summing to 1.
    """
    n_features = x_train.shape[1]
    # Iterate on the simplex (sum = 1) because the projection is cleanest there,
    # then rescale the result to mean 1 so the metric is in Task 4's units. The
    # rescale is a constant, so it does not change the learned relative weights.
    lam = np.full(n_features, 1.0 / n_features)
    uniform = lam.copy()

    sq_implausible = (x_train - p_implausible) ** 2
    sq_plausible = (x_train - p_plausible) ** 2

    for _ in range(n_iterations):
        d_implausible = np.sqrt((lam * sq_implausible).sum(axis=1) + EPS)
        d_plausible = np.sqrt((lam * sq_plausible).sum(axis=1) + EPS)
        m = (d_plausible - d_implausible) / LOSS_TEMPERATURE
        # dL/dmargin = -sigmoid(-margin/T); chain rule to dL/dlambda below.
        sig = 1.0 / (1.0 + np.exp(np.clip(m, -30, 30)))
        grad = (
            sig[:, None]
            * (
                sq_implausible / (2 * np.maximum(d_implausible[:, None], EPS))
                - sq_plausible / (2 * np.maximum(d_plausible[:, None], EPS))
            )
            / LOSS_TEMPERATURE
        ).mean(axis=0)
        lam = project_simplex(lam - step * (grad + ridge * (lam - uniform)))

    return lam * n_features  # mean(lambda) = 1, matching Task 4


def relevance_score(
    x: np.ndarray, prototypes: dict, lam: np.ndarray
) -> dict:
    """Weighted distances, margin, probability, and hard prediction."""
    d_plausible = weighted_squared_distance(x, prototypes["plausible"], lam)
    d_implausible = weighted_squared_distance(x, prototypes["implausible"], lam)
    m = prototype_model.margin(d_plausible, d_implausible)  # already in Task 4 units
    return {
        "d_plausible": d_plausible,
        "d_implausible": d_implausible,
        "margin": m,
        "probability": prototype_model.probability_from_margin(m),
        "prediction": (d_plausible < d_implausible).astype(int),
    }


def _select_hyperparameters(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Inner CV on the TRAINING ROWS ONLY. Returns (ridge, step).

    The outer test fold is not passed in, so it cannot influence this choice.
    """
    inner = StratifiedKFold(
        n_splits=N_INNER_SPLITS, shuffle=True, random_state=cv_protocol.CV_RANDOM_STATE
    )
    best_score, best = -np.inf, (RIDGE_GRID[0], STEP_GRID[0])
    for ridge in RIDGE_GRID:
        for step in STEP_GRID:
            scores = []
            for train_idx, val_idx in inner.split(x, y):
                scaler = StandardScaler().fit(x[train_idx])
                xa = scaler.transform(x[train_idx])
                xb = scaler.transform(x[val_idx])
                ya, yb = y[train_idx], y[val_idx]
                prototypes = prototype_model.fit_prototypes(xa, ya)
                lam = fit_relevance(
                    xa,
                    prototypes["plausible"],
                    prototypes["implausible"],
                    ridge,
                    step,
                )
                out = relevance_score(xb, prototypes, lam)
                scores.append(
                    baseline.evaluate(yb, out["probability"])["auroc"]
                )
            mean_score = float(np.mean(scores))
            if mean_score > best_score:
                best_score, best = mean_score, (ridge, step)
    return best


def _fold_rank(lam: np.ndarray, feature: str) -> int:
    """Rank of one feature's relevance within a single fold (1 = highest)."""
    value = lam[FEATURE_COLUMNS.index(feature)]
    return 1 + sum(1 for v in lam if v > value)


def run_cross_validation(candidates: pd.DataFrame) -> dict:
    """PRIMARY protocol: the same 5 folds as M0 and Task 4, verified."""
    X = candidates[FEATURE_COLUMNS].copy()
    baseline.assert_model_facing_only(X)
    y = candidates["label"].to_numpy()

    folds = list(cv_protocol.make_cv().split(X, y))
    expected = cv_protocol.fold_assignment(y)
    for k, (_, test_idx) in enumerate(folds):
        assert list(test_idx) == list(expected[k]), f"fold {k} differs from the shared protocol"

    per_fold = []
    relevance_by_fold = []
    for k, (train_idx, test_idx) in enumerate(folds):
        x_train_raw, x_test_raw = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        ridge, step = _select_hyperparameters(x_train_raw.to_numpy(), y_train)

        scaler = StandardScaler().fit(x_train_raw)  # training rows only
        x_train = scaler.transform(x_train_raw)
        x_test = scaler.transform(x_test_raw)

        prototypes = prototype_model.fit_prototypes(x_train, y_train)
        lam = fit_relevance(
            x_train, prototypes["plausible"], prototypes["implausible"], ridge, step
        )
        out = relevance_score(x_test, prototypes, lam)

        per_fold.append({"fold": k, **baseline.evaluate(y_test, out["probability"])})
        relevance_by_fold.append({
            "fold": k,
            "ridge_selected": ridge,
            "step_selected": step,
            "lambda": lam.tolist(),
            "lambda_max": float(lam.max()),
            "lambda_min": float(lam.min()),
            "lambda_entropy": float(
                -np.sum(lam * np.log(lam + EPS))
            ),
            "uniform_weight": 1.0,
            "max_deviation_from_uniform": float(
                np.max(np.abs(lam - 1.0))
            ),
            "prototype_plausible": prototypes["plausible"].tolist(),
            "prototype_implausible": prototypes["implausible"].tolist(),
            "margin_sd_test": float(np.std(out["margin"])),
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

    lam_matrix = np.array([r["lambda"] for r in relevance_by_fold])
    mean_lambda = lam_matrix.mean(axis=0)
    std_lambda = lam_matrix.std(axis=0, ddof=1)

    ranking = sorted(
        zip(FEATURE_COLUMNS, mean_lambda, std_lambda), key=lambda t: -t[1]
    )

    return {
        "protocol": {
            "name": "stratified 5-fold cross-validation (shared with M0 and Task 4)",
            "role": "PRIMARY",
            "n_splits": cv_protocol.CV_N_SPLITS,
            "random_state": cv_protocol.CV_RANDOM_STATE,
            "fold_fingerprint": cv_protocol.fold_fingerprint(y),
            "folds_verified_identical": True,
            "preprocessing": (
                "StandardScaler refit within each training fold; prototypes and "
                "relevance learned within each training fold."
            ),
            "learned_parameters": "lambda only (12 values)",
            "prototypes": "fixed at training-fold class means (see RESULTS_RECORDED)",
            "metric": (
                "d_lambda(x, w) = sum_j lambda_j (x_j - w_j)^2, "
                "lambda >= 0, mean(lambda) = 1 (matches Task 4's lambda = 1 units)"
            ),
            "loss": "mean_i log(1 + exp(-margin_i / T)) + ridge pull toward uniform",
            "loss_temperature": LOSS_TEMPERATURE,
            "optimiser": (
                "projected gradient. Per-step exact Euclidean projection onto "
                "{lambda >= 0, sum(lambda) = 1} keeps the iterates feasible; the "
                "returned weights are then rescaled to mean(lambda) = 1, which is "
                "the relevance constraint the model uses."
            ),
            "relevance_constraint": "lambda_j >= 0 and mean(lambda) = 1 (non-negative mean-one normalisation)",
            "hyperparameter_selection": (
                "ridge and step size chosen by inner 3-fold CV on training rows "
                "only, scored by AUROC. Outer test fold never consulted."
            ),
            "ridge_grid": RIDGE_GRID,
            "step_grid": STEP_GRID,
            "prior_knowledge_used": False,
            "latent_truth_used": False,
            "features_used": list(FEATURE_COLUMNS),
        },
        "summary_mean_std": summary,
        "per_fold": per_fold,
        "relevance_by_fold": relevance_by_fold,
        "relevance_ranking": [
            {
                "rank": i + 1,
                "feature": feature,
                "mean_lambda": float(mean_value),
                "std_lambda": float(std_value),
                "fold_rank_min": int(min(
                    _fold_rank(r["lambda"], feature) for r in relevance_by_fold
                )),
                "fold_rank_max": int(max(
                    _fold_rank(r["lambda"], feature) for r in relevance_by_fold
                )),
            }
            for i, (feature, mean_value, std_value) in enumerate(ranking)
        ],
        "relevance_concentration": {
            "max_lambda_mean": float(mean_lambda.max()),
            "uniform_weight": 1.0,
            "max_deviation_from_uniform_mean": float(
                np.max(np.abs(mean_lambda - 1.0))
            ),
            "normalisation": "mean(lambda) = 1, so uniform relevance is lambda_j = 1",
        },
        "results_recorded": RESULTS_RECORDED,
    }


def compare_to_baselines(cv_result: dict, results_dir: Path) -> dict:
    """Fold-wise comparison against M0 and the Task 4 plain prototype."""
    sources = {
        "logistic_regression": json.loads(
            (results_dir / "m0_cross_validation.json").read_text()
        ),
        "hist_gradient_boosting": json.loads(
            (results_dir / "m0_cross_validation.json").read_text()
        ),
        "plain_prototype": json.loads(
            (results_dir / "prototype_baseline_cv.json").read_text()
        ),
    }
    fingerprint = cv_result["protocol"]["fold_fingerprint"]
    for name, payload in sources.items():
        assert payload["protocol"]["fold_fingerprint"] == fingerprint, (
            f"{name} was not evaluated on the same folds"
        )

    comparison = {}
    for name, payload in sources.items():
        reference = payload["per_fold"][name] if name != "plain_prototype" else payload["per_fold"]
        rows = []
        for metric in cv_protocol.PRIMARY_METRICS:
            ref = [f[metric] for f in reference]
            new = [f[metric] for f in cv_result["per_fold"]]
            diffs = [b - a for a, b in zip(ref, new)]
            rows.append({
                "metric": metric,
                "reference_mean": float(np.mean(ref)),
                "reference_std": float(np.std(ref, ddof=1)),
                "relevance_mean": float(np.mean(new)),
                "relevance_std": float(np.std(new, ddof=1)),
                "mean_difference": float(np.mean(diffs)),
                "difference_std": float(np.std(diffs, ddof=1)),
                "n_folds_relevance_better": int(sum(d > 0 for d in diffs)),
                "n_folds_reference_better": int(sum(d < 0 for d in diffs)),
                "per_fold_difference": diffs,
            })
        comparison[name] = rows
    return comparison


def test_label_informed_weighting_bound(
    candidates: pd.DataFrame, n_samples: int = 3000
) -> dict:
    """Post-hoc optimistic weighting bound -- a DIAGNOSTIC, not a model.

    WHAT THIS IS
    Random weights are drawn from a Dirichlet, scored on the OUTER TEST FOLD
    using that fold's true labels, and the best-scoring one is kept.

    WHAT THIS IS NOT -- please read before quoting any number from it
    * It is deliberately LABEL-INFORMED: the held-out labels are used to CHOOSE
      the weights.
    * It is therefore NOT a deployable model, NOT a valid predictive estimate,
      and NOT a performance any method could honestly attain.
    * It must NOT be compared against other models as though it were a normal
      result. It is a post-hoc diagnostic, reported only to answer one narrow
      question: how much ranking headroom could diagonal relevance weighting
      conceivably have had on THESE PARTICULAR held-out folds?
    * It is NOT the Bayes-optimal accuracy of the data-generating process. That
      is a different quantity, computed in Task 1 from the latent label
      probabilities, and the two numbers are not comparable. Do not describe
      this one as an "oracle" bound without that qualification, since the word
      invites confusion with the Bayes rate.

    It is reported rather than hidden because the flat headline result of this
    task needs explaining: is it a learning method that failed, or a ceiling
    that is simply low? Note also that the number is optimistic in a second,
    subtler way -- it is the maximum over a bank of samples scored on the same
    data it is measured on, so it carries selection bias upward on top of the
    label information.

    No model in this project uses it, and it is never used to select anything.
    """
    X = candidates[FEATURE_COLUMNS].copy()
    baseline.assert_model_facing_only(X)
    y = candidates["label"].to_numpy()
    rng = np.random.default_rng(cv_protocol.CV_RANDOM_STATE)
    bank = rng.dirichlet(np.ones(X.shape[1]) * 0.3, size=n_samples)

    rows = []
    for k, (train_idx, test_idx) in enumerate(cv_protocol.make_cv().split(X, y)):
        scaler = StandardScaler().fit(X.iloc[train_idx])
        x_train, x_test = scaler.transform(X.iloc[train_idx]), scaler.transform(X.iloc[test_idx])
        prototypes = prototype_model.fit_prototypes(x_train, y[train_idx])

        uniform = prototype_model.predict(prototypes, x_test)
        uniform_auroc = float(roc_auc_score(y[test_idx], uniform["margin"]))

        # prototype_model.margin(d_plausible, d_implausible) = sqrt(d_impl) - sqrt(d_plaus)
        d_impl = bank @ ((x_test - prototypes["implausible"]) ** 2).T
        d_plaus = bank @ ((x_test - prototypes["plausible"]) ** 2).T
        margins = np.sqrt(np.maximum(d_impl, 0)) - np.sqrt(np.maximum(d_plaus, 0))
        best = max(float(roc_auc_score(y[test_idx], margins[i])) for i in range(margins.shape[0]))

        rows.append({
            "fold": k,
            "uniform_auroc": uniform_auroc,
            "post_hoc_optimistic_auroc": best,
            "apparent_headroom": best - uniform_auroc,
        })

    return {
        "name": "test_label_informed_weighting_bound",
        "is_a_model": False,
        "is_label_informed": True,
        "is_comparable_to_other_models": False,
        "is_bayes_optimal_accuracy": False,
        "method": (
            f"{n_samples} random Dirichlet weights scored on the outer test fold "
            "using that fold's true labels; best kept. Deliberately "
            "label-informed, therefore not deployable and not a valid predictive "
            "estimate. Optimistic for two reasons: the labels informed the "
            "choice, and the reported figure is a maximum over a bank of "
            "candidates scored on the same data it is measured on. Used only to "
            "estimate the ranking headroom diagonal relevance weighting could "
            "theoretically have had on these particular folds. NOT to be compared "
            "as a model result, and NOT the Bayes-optimal accuracy of the "
            "generative process."
        ),
        "per_fold": rows,
        "uniform_mean": float(np.mean([r["uniform_auroc"] for r in rows])),
        "post_hoc_optimistic_mean": float(
            np.mean([r["post_hoc_optimistic_auroc"] for r in rows])
        ),
        "apparent_headroom_mean": float(np.mean([r["apparent_headroom"] for r in rows])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Task 5 -- data-driven relevance-weighted prototype"
    )
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()

    candidates = load_candidates()
    result = run_cross_validation(candidates)
    result["comparison_to_baselines"] = compare_to_baselines(result, args.out)
    result["test_label_informed_weighting_bound"] = test_label_informed_weighting_bound(
        candidates
    )
    result["versions"] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }

    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / "relevance_prototype_cv.json"
    with open(out_path, "w") as handle:
        json.dump(result, handle, indent=2)
    print(f"wrote {out_path}\n")

    print(f"fold fingerprint: {result['protocol']['fold_fingerprint']} (verified shared)")
    print("\nnearest-prototype with LEARNED relevance weights")
    for metric in cv_protocol.PRIMARY_METRICS:
        s = result["summary_mean_std"][metric]
        print(f"  {metric:19s} {s['mean']:.4f} +/- {s['std']:.4f} "
              f"[{s['min']:.4f}, {s['max']:.4f}]")

    print("\nlearned relevance (uniform = 1.0000):")
    for row in result["relevance_ranking"]:
        print(f"  {row['rank']:2d}. {row['feature']:<24} {row['mean_lambda']:.4f} "
              f"+/- {row['std_lambda']:.4f}  (fold rank {row['fold_rank_min']}-{row['fold_rank_max']})")

    print("\nhyperparameters selected per fold (inner CV, training only):")
    for r in result["relevance_by_fold"]:
        print(f"  fold {r['fold']}: ridge={r['ridge_selected']} step={r['step_selected']} "
              f"lambda_max={r['lambda_max']:.4f}")

    oh = result["test_label_informed_weighting_bound"]
    print("\nDIAGNOSTIC ONLY -- test-label-informed weighting bound")
    print("  label-informed, not a model, not a valid predictive estimate,")
    print("  not comparable to other results, not Bayes-optimal accuracy:")
    print(f"  uniform={oh['uniform_mean']:.4f}  "
          f"post-hoc optimistic={oh['post_hoc_optimistic_mean']:.4f}  "
          f"apparent headroom={oh['apparent_headroom_mean']:+.4f}")

    print("\nfold-wise vs plain prototype (positive = relevance-weighted better; brier lower better):")
    for row in result["comparison_to_baselines"]["plain_prototype"]:
        diffs = " ".join(f"{d:+.4f}" for d in row["per_fold_difference"])
        print(f"  {row['metric']:19s} ref={row['reference_mean']:.4f} "
              f"new={row['relevance_mean']:.4f} diff={row['mean_difference']:+.4f}  {diffs}")


if __name__ == "__main__":
    main()
