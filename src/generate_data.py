"""
Task 1 -- Synthetic candidate-structure data generation.

This module generates a fully synthetic dataset of candidate biological
structures, each described by experimental (FRET-like), structural, and
biological-network/pathway evidence, and labelled `plausible` or
`implausible`.

SCIENTIFIC STATUS
-----------------
Everything here is synthetic. FRET distances are simulated, not measured.
Nothing in this module performs or approximates real RNA folding or real
RNA 3D-structure prediction, and no biological validity is claimed.

GENERATIVE ARCHITECTURE (three separated stages)
------------------------------------------------
Stage 0  shared latent factors
Stage 1  latent experimental / structural / module states
Stage 2  latent plausibility score -> logistic -> p -> label ~ Bernoulli(p)
Stage 3  observed features as noisy partial read-outs of the latent states

The label is generated from the LATENT STATES ONLY. It is never computed
from the engineered feature columns that later model conditions (M1/M2/M3)
will receive. Observed features are generated afterwards, as lossy
measurements. A model's success is therefore about recovering hidden
structure, never about reproducing a formula that was handed to it.

FRET MATHEMATICS
----------------
    m_signed ~ N(0, 0.7^2)            signed positioning deviation
    m_mag    = |m_signed|             non-negative mismatch magnitude
    d_cand   = z_true + m_signed
    d_obs    = z_true + eps,  eps ~ N(0, sigma^2)

    d_cand - d_obs = m_signed - eps

so the pre-absolute-value quantity has variance 0.49 + sigma^2. Because
`fret_error` is an absolute value, that expression is NOT its variance and
no simple variance formula is claimed for it. Qualitatively: increasing
the measurement uncertainty sigma broadens the observed FRET disagreement
and reduces its evidential value. A large `fret_error` measured with large
`fret_uncertainty` is weak evidence, not strong evidence of implausibility.
The label depends on the magnitude `m_mag` and never on the sign of
`m_signed`; the sign only positions the candidate above or below the true
distance.

MODULE STRUCTURE
----------------
`a_m = rho * q + sqrt(1 - rho^2) * m_m`  with rho = 0.6 means:

  * each module activity is CORRELATED 0.60 with the shared quality factor q;
  * the shared q component accounts for rho^2 = 36% of each module's variance;
  * the module-specific component accounts for (1 - rho^2) = 64%;
  * module activities are pairwise correlated at rho^2 = 0.36.

    Module C (regulatory interaction)   -> reg_feature_1, reg_feature_2
    Module B (FRET-supported geometry)   -> pathway_feature_1, pathway_feature_2
    Module A (structural stability)      -> no observed proxy
    Module D (weak / noisy biology)      -> no observed proxy

Modules C and B are recoverable: each has two noisy proxies of one shared
state, so aggregating within a module reduces proxy noise.

Modules A and D carry no observed proxy, so their MODULE-SPECIFIC variation
is unavailable to any model. Their shared q component may still be inferred
indirectly from the observed structural and experimental features. This is a
source of irreducible uncertainty, not a fully unrecoverable component. No
quantity is claimed here for the size of that contribution.

`gc_content` is a weak indirect proxy for the shared quality factor only. It
belongs to no module and has no module-specific state. The two
`noise_feature_*` columns are the only genuinely label-irrelevant
distractors: they are correlated with each other and carry no information
about q, any module state, or the label.

CHANGES MADE DURING TASK 1
--------------------------
These are recorded by category, because they are not equivalent in kind.

Calibration adjustment
  * Weakened `gc_content` loading: `gc_q` 0.05 -> 0.01. With `gc_sd = 0.10`
    a loading of 0.05 gives correlation 0.45 with q and AUC ~0.64, far too
    strong for a weak indirect proxy. 0.01 yields AUC ~0.54, weak but
    non-zero, as intended.

Design / bug fixes
  * Rescaled the stem-related latent terms (`s_true_q` 0.80 -> 0.35,
    `s_true_sd` 0.7 -> 1.00, deviation scale 6 -> 1.5) because the intended
    structural interaction was effectively inactive. With the original
    values the U-shaped term had sd 0.017 against a total score sd of 1.06,
    and the stem-inconsistency interaction NEVER FIRED (P = 0.000), since
    `s_true` and `c_true` both loaded positively on q, making "many stems
    with low consistency" nearly impossible. Now quadratic sd 0.28,
    interaction sd 0.06, P(fires) = 0.20. The interaction is real but
    modest, the intended level for a soft structural prior.
  * Corrected the correlated-noise construction so the intended correlation
    is achieved. The original shared-plus-independent form mixed unequal
    variances and delivered correlation ~0.52 instead of the designed 0.70.
    The unit-variance shared-factor form gives the target directly.
  * Corrected the metadata interpretation of shared variance. It reported
    1 - rho^2; the shared q component accounts for rho^2.

Mechanism refinement
  * Widened the `fret_uncertainty` distribution (log sigma mean -1.6 -> -1.9,
    sd 0.55 -> 0.85) so the uncertainty mechanism spans a meaningful range.
    Quartiles had been 0.14 / 0.20 / 0.29, too narrow for the uncertainty
    mechanism to carry weight.

KNOWN LIMITATION OF THE FRET DESIGN
-----------------------------------
Uncertainty-normalized FRET disagreement does NOT reliably beat raw
`fret_error` in this dataset, and a Bayes-optimal posterior statistic for
`m_mag` given (`fret_error`, `sigma`) does not beat it either. The reason is
structural: `sigma` is drawn independently of q, so it carries no label
information, and within any single uncertainty tier the raw disagreement is
already close to a sufficient statistic for the mismatch magnitude. A ratio
is a crude approximation to the correct posterior and can even be slightly
worse in the high-uncertainty tier.

This was checked directly and is reported rather than tuned away, per the
project's scientific-integrity rules. Consequence for later stages: any M1
advantage must NOT be assumed to come from the uncertainty ratio, and if
M1 fails to beat M0 that is a legitimate, reportable outcome.

SINGLE-FEATURE AUC INTERPRETATION
---------------------------------
The 0.55-0.78 univariate AUC band applies to features whose contribution to
the label is monotonic: `fret_error`, `free_energy`, `structural_consistency`,
`gc_content`, and the four network features. All sit inside it.

`stem_count` is explicitly EXEMPT from that band. Its intended contribution
is nonlinear (a U-shaped preference around 8 stems) and partly
interaction-dependent (a penalty for many stems combined with low internal
consistency). Consequently BOTH its raw AUC and the AUC of the quadratic
monotonic transform may remain near 0.5, and in the published dataset the
transformed value is ~0.47. This is the expected and acceptable outcome,
not a defect.

`stem_count` must NOT be retuned merely to raise its univariate AUC. Doing
so would distort the latent design to flatter a univariate diagnostic.
Its usefulness is to be evaluated later, by models able to represent
nonlinear structure and interactions (prototype-based models with learned
relevance, M2, and any model evaluated jointly rather than one feature at a
time). Whether it proves useful is an open question to be reported
honestly, including the possibility that it does not.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

DEFAULT_SEED = 20240617
N_SAMPLES = 800

# Seeds used only to check that the frozen parameters behave consistently.
# The published dataset uses DEFAULT_SEED and is never one of these.
CALIBRATION_SEEDS = tuple(range(1001, 1018))

PARAMS = {
    "n_samples": N_SAMPLES,
    "rho": 0.6,
    "z_true_mean": 7.0,
    "z_true_sd": 1.5,
    "sigma_log_mean": -1.9,
    "sigma_log_sd": 0.85,
    "m_signed_sd": 0.7,
    "f_true_q": 0.80,
    "f_true_sd": 0.6,
    "s_true_q": 0.35,
    "s_true_sd": 1.0,
    "stem_deviation_scale": 1.5,
    "c_true_q": 0.85,
    "c_true_sd": 0.5,
    "stem_preferred": 8.0,
    "stem_max": 20.0,
    "w_fret": -0.85,
    "w_module_a": 0.55,
    "w_module_b": 0.45,
    "w_module_c": 0.40,
    "w_module_d": 0.12,
    "w_c_true": 0.35,
    "w_f_true": 0.40,
    "w_stem_preference": -0.35,
    "w_stem_inconsistency": -0.30,
    "logit_scale": 1.1,
    "free_energy_base": 1.60,
    "free_energy_scale": 0.50,
    "free_energy_sd": 0.70,
    "consistency_base": 0.50,
    "consistency_scale": 0.18,
    "consistency_sd": 0.12,
    "gc_q": 0.01,
    "gc_sd": 0.10,
    "reg_proxy_sds": (0.70, 0.85),
    "pathway_proxy_sds": (0.70, 0.85),
    "noise_correlation": 0.70,
}

FEATURES = [
    "fret_error",
    "fret_uncertainty",
    "free_energy",
    "stem_count",
    "structural_consistency",
    "gc_content",
    "reg_feature_1",
    "reg_feature_2",
    "pathway_feature_1",
    "pathway_feature_2",
    "noise_feature_1",
    "noise_feature_2",
]

# Features whose contribution to the label is not a simple increasing or
# decreasing function of the raw column, together with a monotonic transform
# that exposes their signal. `free_energy` is sign-flipped (more negative is
# more stable, hence more plausible), so its raw signed AUC sits below 0.5
# and only the direction-agnostic AUC is meaningful. `stem_count` enters
# through a U-shaped preference, so its monotonic transform can score below
# 0.5 while the quadratic term still contributes to the label.
MONOTONIC_TRANSFORMS = {
    "stem_count": lambda s: (
        (s - PARAMS["stem_preferred"]) / PARAMS["stem_deviation_scale"]
    ) ** 2,
    "free_energy": lambda s: -s,
}


def latent_plausibility_score(latent: dict[str, np.ndarray]) -> np.ndarray:
    """Latent-state-only plausibility score. Uses no observed feature."""
    p = PARAMS
    stem_deviation = (
        latent["s_true"] - p["stem_preferred"]
    ) / p["stem_deviation_scale"]

    return (
        p["w_fret"] * latent["m_mag"]
        + p["w_module_a"] * latent["a_A"]
        + p["w_module_b"] * latent["a_B"]
        + p["w_module_c"] * latent["a_C"]
        + p["w_module_d"] * latent["a_D"]
        + p["w_c_true"] * latent["c_true"]
        + p["w_f_true"] * latent["f_true"]
        + p["w_stem_preference"] * stem_deviation**2
        + p["w_stem_inconsistency"]
        * np.maximum(0.0, stem_deviation)
        * np.maximum(0.0, -latent["c_true"])
    )


def generate(seed: int, n: int = N_SAMPLES) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Generate one synthetic dataset. Identical for a given (seed, n)."""
    p = PARAMS
    rng = np.random.default_rng(seed)
    n = int(n)

    # ---- Stage 0: shared latent factor and module activities -------------
    q = rng.standard_normal(n)
    module_sd = np.sqrt(1.0 - p["rho"] ** 2)
    a = {name: p["rho"] * q + module_sd * rng.standard_normal(n) for name in "ABCD"}

    # ---- Stage 1: latent experimental, structural, module states ---------
    z_true = rng.normal(p["z_true_mean"], p["z_true_sd"], n)
    sigma = np.exp(rng.normal(p["sigma_log_mean"], p["sigma_log_sd"], n))
    m_signed = rng.normal(0.0, p["m_signed_sd"], n)
    m_mag = np.abs(m_signed)
    d_cand = z_true + m_signed
    d_obs = z_true + rng.normal(0.0, 1.0, n) * sigma

    f_true = p["f_true_q"] * q + rng.normal(0.0, p["f_true_sd"], n)
    s_true = p["stem_preferred"] + p["s_true_q"] * q + rng.normal(0.0, p["s_true_sd"], n)
    c_true = p["c_true_q"] * q + rng.normal(0.0, p["c_true_sd"], n)

    latent = {
        "q": q,
        "a_A": a["A"],
        "a_B": a["B"],
        "a_C": a["C"],
        "a_D": a["D"],
        "z_true": z_true,
        "sigma": sigma,
        "m_signed": m_signed,
        "m_mag": m_mag,
        "d_cand": d_cand,
        "d_obs": d_obs,
        "f_true": f_true,
        "s_true": s_true,
        "c_true": c_true,
    }

    # ---- Stage 2: latent score -> probability -> sampled label -----------
    score = latent_plausibility_score(latent)
    p_plausible = 1.0 / (1.0 + np.exp(-p["logit_scale"] * score))
    label = (rng.random(n) < p_plausible).astype(int)

    # ---- Stage 3: observed features as noisy read-outs -------------------
    reg_sds = p["reg_proxy_sds"]
    path_sds = p["pathway_proxy_sds"]
    e_shared = rng.standard_normal(n)
    u1 = rng.standard_normal(n)
    u2 = rng.standard_normal(n)
    noise_corr = p["noise_correlation"]
    # Unit-variance shared-factor construction giving corr = noise_corr exactly:
    # both variables carry a shared component plus their own idiosyncratic part.
    shared = np.sqrt(noise_corr / (1.0 - noise_corr))

    features = pd.DataFrame(
        {
            "fret_error": np.abs(d_cand - d_obs),
            "fret_uncertainty": sigma,
            "free_energy": -(
                p["free_energy_base"] + p["free_energy_scale"] * f_true
            ) + rng.normal(0.0, p["free_energy_sd"], n),
            "stem_count": np.rint(np.clip(s_true, 0.0, p["stem_max"])),
            "structural_consistency": np.clip(
                p["consistency_base"] + p["consistency_scale"] * c_true
                + rng.normal(0.0, p["consistency_sd"], n),
                0.0,
                1.0,
            ),
            "gc_content": np.clip(
                0.50 + p["gc_q"] * q + rng.normal(0.0, p["gc_sd"], n), 0.20, 0.80
            ),
            "reg_feature_1": a["C"] + rng.normal(0.0, reg_sds[0], n),
            "reg_feature_2": a["C"] + rng.normal(0.0, reg_sds[1], n),
            "pathway_feature_1": a["B"] + rng.normal(0.0, path_sds[0], n),
            "pathway_feature_2": a["B"] + rng.normal(0.0, path_sds[1], n),
            "noise_feature_1": shared * e_shared + u1,
            "noise_feature_2": shared * e_shared + u2,
        }
    )[FEATURES]

    candidates = features.copy()
    candidates.insert(0, "sample_id", np.arange(n))
    candidates["label"] = label

    oracle_ceiling = float(np.mean(np.maximum(p_plausible, 1.0 - p_plausible)))

    latent_truth = pd.DataFrame(
        {"sample_id": np.arange(n), **{k: v for k, v in latent.items()},
         "score": score, "p_plausible": p_plausible}
    )

    return candidates, latent_truth, {"oracle_ceiling": oracle_ceiling}


def feature_aucs(candidates: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Signed and direction-agnostic AUC for every feature."""
    y = candidates["label"].to_numpy()
    out: dict[str, dict[str, float]] = {}
    for name in FEATURES:
        signed = float(roc_auc_score(y, candidates[name].to_numpy()))
        out[name] = {
            "auc_signed": signed,
            "auc_strong": max(signed, 1.0 - signed),
        }
    return out


def monotonic_aucs(candidates: pd.DataFrame) -> dict[str, float]:
    """AUC of the two non-monotonic features after their informative transform."""
    y = candidates["label"].to_numpy()
    return {
        name: float(roc_auc_score(y, transform(candidates[name].to_numpy())))
        for name, transform in MONOTONIC_TRANSFORMS.items()
    }


def calibration_logreg_accuracy(candidates: pd.DataFrame) -> float:
    """Generator calibration only.

    A quick linear sanity probe used while freezing the generating
    parameters. This is NOT the M0 experiment: it is not saved, not
    reported as a model result, and carries no model artifact.
    """
    x = candidates[FEATURES].to_numpy()
    y = candidates["label"].to_numpy()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    return float(
        cross_val_score(
            LogisticRegression(max_iter=1000), x, y, cv=cv, scoring="accuracy"
        ).mean()
    )


def run_calibration(n: int = N_SAMPLES) -> dict:
    """Check the frozen parameters across the calibration seeds."""
    records = []
    for seed in CALIBRATION_SEEDS:
        candidates, _, info = generate(seed, n)
        aucs = feature_aucs(candidates)
        records.append(
            {
                "seed": seed,
                "positive_rate": float(candidates["label"].mean()),
                "bayes_optimal_accuracy": info["oracle_ceiling"],
                "calibration_logreg_accuracy": calibration_logreg_accuracy(candidates),
                "auc_strong": {k: v["auc_strong"] for k, v in aucs.items()},
                "auc_strong_monotonic": monotonic_aucs(candidates),
            }
        )

    def stat(values):
        arr = np.asarray(values, dtype=float)
        return {"mean": float(arr.mean()), "min": float(arr.min()), "max": float(arr.max())}

    per_feature = {
        name: stat([r["auc_strong"][name] for r in records]) for name in FEATURES
    }
    per_feature_monotonic = {
        name: stat([r["auc_strong_monotonic"][name] for r in records])
        for name, transform in MONOTONIC_TRANSFORMS.items()
    }
    return {
        "seeds": list(CALIBRATION_SEEDS),
        "n_seeds": len(CALIBRATION_SEEDS),
        "positive_rate": stat([r["positive_rate"] for r in records]),
        "bayes_optimal_accuracy": stat(
            [r["bayes_optimal_accuracy"] for r in records]
        ),
        "calibration_logreg_accuracy": stat(
            [r["calibration_logreg_accuracy"] for r in records]
        ),
        "auc_strong_per_feature": per_feature,
        "auc_strong_monotonic_per_feature": per_feature_monotonic,
        "note": (
            "The logreg accuracy is a generator calibration probe only. It is "
            "not the M0 experiment and no fitted model is saved."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--n", type=int, default=N_SAMPLES)
    parser.add_argument("--out", type=Path, default=Path("data/generated"))
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Also run the multi-seed parameter calibration and print a summary.",
    )
    args = parser.parse_args()

    candidates, latent_truth, info = generate(args.seed, args.n)
    args.out.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(args.out / "candidates.csv", index=False)
    latent_truth.to_csv(args.out / "latent_truth.csv", index=False)

    aucs = feature_aucs(candidates)
    latent_by_id = latent_truth.set_index("sample_id")
    q = latent_by_id.loc[candidates["sample_id"], "q"].to_numpy()
    label = candidates["label"].to_numpy()

    correlations = {
        name: {
            "pearson_with_label": float(np.corrcoef(candidates[name], label)[0, 1]),
            "pearson_with_q": float(np.corrcoef(candidates[name], q)[0, 1]),
        }
        for name in FEATURES
    }
    pair_correlations = {
        "reg_feature_1__reg_feature_2": float(
            candidates["reg_feature_1"].corr(candidates["reg_feature_2"])
        ),
        "pathway_feature_1__pathway_feature_2": float(
            candidates["pathway_feature_1"].corr(candidates["pathway_feature_2"])
        ),
        "noise_feature_1__noise_feature_2": float(
            candidates["noise_feature_1"].corr(candidates["noise_feature_2"])
        ),
        "reg_feature_1__pathway_feature_1": float(
            candidates["reg_feature_1"].corr(candidates["pathway_feature_1"])
        ),
    }

    metadata = {
        "synthetic": True,
        "fret_measurements_are_simulated": True,
        "no_biological_validation_claimed": True,
        "default_seed": DEFAULT_SEED,
        "seed_used": args.seed,
        "n_samples": int(args.n),
        "parameters": {
            k: (list(v) if isinstance(v, tuple) else v) for k, v in PARAMS.items()
        },
        "rho_interpretation": {
            "rho": PARAMS["rho"],
            "module_correlation_with_q": PARAMS["rho"],
            "shared_variance_fraction_from_q": PARAMS["rho"] ** 2,
            "module_specific_variance_fraction": 1.0 - PARAMS["rho"] ** 2,
            "pairwise_module_correlation": PARAMS["rho"] ** 2,
            "note": (
                "rho is the correlation of each module activity with the "
                "shared quality factor q. The shared factor accounts for "
                "rho^2 of each module's variance; the module-specific part "
                "accounts for 1 - rho^2. The two are not interchangeable."
            ),
        },
        "class_balance": {
            "n_positive": int(candidates["label"].sum()),
            "n_negative": int((1 - candidates["label"]).sum()),
            "positive_rate": float(candidates["label"].mean()),
        },
        "bayes_optimal_accuracy": info["oracle_ceiling"],
        "bayes_optimal_accuracy_definition": (
            "E[max(p, 1-p)] over the sampled label probabilities. This is the "
            "expected Bayes-optimal classification accuracy under the "
            "data-generating process, because the label was drawn from "
            "Bernoulli(p). It is NOT a hard finite-sample upper bound: an "
            "empirical model fitted to one finite sampled dataset can exceed "
            "this number through sampling variation, so a model scoring above "
            "it has not necessarily beaten the Bayes rule."
        ),
        "per_feature_auc": aucs,
        "per_feature_auc_monotonic_transform": {
            name: float(roc_auc_score(label, transform(candidates[name].to_numpy())))
            for name, transform in MONOTONIC_TRANSFORMS.items()
        },
        "auc_band_interpretation": {
            "band": [0.55, 0.78],
            "applies_to": [
                "fret_error",
                "free_energy",
                "structural_consistency",
                "gc_content",
                "reg_feature_1",
                "reg_feature_2",
                "pathway_feature_1",
                "pathway_feature_2",
            ],
            "exempt": ["stem_count"],
            "exempt_reason": (
                "stem_count enters the label through a U-shaped preference and "
                "an interaction with internal consistency, so both its raw AUC "
                "and the AUC of its quadratic monotonic transform may remain "
                "near 0.5. That is the expected outcome, not a defect. It must "
                "not be retuned merely to raise its univariate AUC. Its "
                "usefulness is to be evaluated later by models that can "
                "represent nonlinear structure and interactions, and the "
                "honest possibility that it proves unhelpful is left open."
            ),
        },
        "changes_during_task_1": {
            "calibration_adjustment": [
                "Weakened gc_content loading: gc_q 0.05 -> 0.01. At gc_sd 0.10 "
                "a loading of 0.05 gives correlation 0.45 with q and AUC ~0.64, "
                "too strong for a weak indirect proxy; 0.01 gives AUC ~0.54."
            ],
            "design_and_bug_fixes": [
                "Rescaled the stem-related latent terms (s_true_q 0.80 -> 0.35, "
                "s_true_sd 0.7 -> 1.00, deviation scale 6 -> 1.5) because the "
                "intended structural interaction was effectively inactive: the "
                "U-shaped term had sd 0.017 against score sd 1.06 and the "
                "inconsistency interaction never fired (P = 0.000). Now "
                "quadratic sd 0.28, interaction sd 0.06, P(fires) = 0.20.",
                "Corrected the correlated-noise construction so the intended "
                "0.70 correlation is achieved; the previous shared-plus-"
                "independent form mixed unequal variances and delivered ~0.52.",
                "Corrected the metadata interpretation of shared variance from "
                "1 - rho^2 to rho^2, since the shared q component accounts for "
                "rho^2 of each module's variance.",
            ],
            "mechanism_refinement": [
                "Widened the fret_uncertainty distribution (log sigma mean "
                "-1.6 -> -1.9, sd 0.55 -> 0.85) so the uncertainty mechanism "
                "spans a meaningful range; quartiles had been "
                "0.14 / 0.20 / 0.29, too narrow to carry weight."
            ],
            "known_limitation": (
                "Uncertainty-normalized FRET disagreement does not reliably beat "
                "raw fret_error here, and a Bayes-optimal posterior statistic "
                "for m_mag given (fret_error, sigma) does not beat it either, "
                "because sigma is independent of q and carries no label "
                "information. Reported, not tuned away. Any future M1 "
                "advantage must not be assumed to come from the uncertainty "
                "ratio."
            ),
        },
        "correlations": correlations,
        "pair_correlations": pair_correlations,
        "calibration": run_calibration(args.n) if args.calibrate else None,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }

    with open(args.out / "metadata.json", "w") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"wrote {args.out / 'candidates.csv'}")
    print(f"wrote {args.out / 'latent_truth.csv'}")
    print(f"wrote {args.out / 'metadata.json'}")
    print(f"seed={args.seed} n={args.n} positive_rate={candidates['label'].mean():.4f}")
    print(f"bayes_optimal_accuracy={info['oracle_ceiling']:.4f}")
    for name in FEATURES:
        stats = aucs[name]
        print(
            f"  {name:<24} auc_signed={stats['auc_signed']:.4f} "
            f"auc_strong={stats['auc_strong']:.4f}"
        )
    for name, value in metadata["per_feature_auc_monotonic_transform"].items():
        print(f"  {name + ' (monotonic transform)':<24} auc={value:.4f}")

    if args.calibrate:
        print(json.dumps(metadata["calibration"], indent=2))


if __name__ == "__main__":
    main()
