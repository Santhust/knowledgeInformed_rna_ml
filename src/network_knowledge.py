"""
Task 8 -- M3, the biological network / module knowledge condition.

Question: does knowing that certain noisy features belong together in a
biological module improve prediction or interpretability, compared with

  * treating the raw proxies independently, and
  * an equally complex but BIOLOGICALLY INCORRECT grouping.

THE ONE THING THE PRIOR SUPPLIES
---------------------------------
Which observed variables belong together. That is all.

  * No weights. Module scores are unweighted means.
  * No learned parameters.
  * No labels, latent states, or known noise variances.
  * No ordering, no ranking, no importance attached to modules.

Membership is imported from `knowledge_graph`, which is the single
authoritative source. It is NOT restated here, so a "network-informed" model
cannot silently drift onto a different grouping from the documented one.

REPLACEMENT, NOT APPENDING -- THE PRIMARY DESIGN
------------------------------------------------
The four raw proxies are REPLACED by two module scores:

    regulatory_module_score = mean(reg_feature_1, reg_feature_2)
    pathway_module_score    = mean(pathway_feature_1, pathway_feature_2)

so the representation goes from 12 columns to 10. The incorrect control
replaces exactly the same four columns with two incorrect-grouping scores,
giving the same 12 -> 10 reduction. All other observable features are
unchanged, and no FRET prior, structural prior, or relevance weighting is
added: this task isolates the network prior.

WHY REPLACEMENT RATHER THAN APPENDING
-------------------------------------
Appending four raw columns plus two means would let the model keep the raw
proxies and simply ignore the scores, so a null result would be
uninterpretable -- we would not know whether the grouping failed or whether
the model sidestepped it. Replacement forces the grouping to carry the
information, which is what makes the comparison informative either way.

A raw-plus-means condition is included as a clearly SECONDARY diagnostic
(LR3-append) purely to show what was lost, never as the headline M3 result.

THE INCORRECT GROUPING
----------------------
    incorrect_module_1 = mean(reg_feature_1,  pathway_feature_1)
    incorrect_module_2 = mean(reg_feature_2,  pathway_feature_2)

Identical in module count (2), module sizes (2 each), and aggregation form
(equal-weight mean). Fixed in `knowledge_graph` before any evaluation and
chosen without labels or performance, by crossing the correct module
boundaries. It differs from the correct grouping ONLY in membership, which is
exactly what makes it a control: if correct and incorrect aggregation behave
the same, the grouping is not what is driving any difference.

SCIENTIFIC STATUS
-----------------
The knowledge graph is SYNTHETIC. It is not real RNA biology, is not derived
from KEGG or Reactome, and no biological validation is claimed. A module-level
coefficient is a PREDICTIVE association under a fitted model, not a causal
claim and not a statement about real regulatory or pathway biology.
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline  # noqa: E402
import cv_protocol  # noqa: E402
import knowledge_graph as KG  # noqa: E402
import prototype_model  # noqa: E402
from validation_checks import FEATURE_COLUMNS, load_candidates  # noqa: E402

MODULE_SCORE_NAMES = KG.module_score_names(KG.FEATURE_TO_MODULE)
INCORRECT_SCORE_NAMES = KG.module_score_names(KG.INCORRECT_FEATURE_TO_MODULE)


# --------------------------------------------------------------------------
# Module scores and representation (label-free)
# --------------------------------------------------------------------------
def module_scores(
    frame: pd.DataFrame, mapping: dict[str, str] = KG.FEATURE_TO_MODULE
) -> np.ndarray:
    """Equal-weight means of each module's members.

    The prior is membership only. Weights are equal by construction, which is
    a modelling choice (no proxy is privileged) and NOT something fitted from
    labels. No latent state and no known noise variance is used.
    """
    members = KG.module_members(mapping)
    return np.column_stack([
        frame[features].to_numpy().mean(axis=1) for features in members.values()
    ])


def replace_with_module_scores(
    frame: pd.DataFrame, mapping: dict[str, str] = KG.FEATURE_TO_MODULE
) -> pd.DataFrame:
    """REPLACE the four proxies with their module scores. 12 columns -> 10.

    Used for the primary M3 conditions and for the incorrect control, so the
    dimensionality reduction is identical in both.
    """
    members = KG.module_members(mapping)
    names = KG.module_score_names(mapping)
    scores = pd.DataFrame(
        {
            names[module]: frame[features].to_numpy().mean(axis=1)
            for module, features in members.items()
        },
        index=frame.index,
    )
    kept = [c for c in FEATURE_COLUMNS if c not in KG.AGGREGATED_FEATURES]
    return pd.concat([frame[kept], scores], axis=1)


def append_module_scores(
    frame: pd.DataFrame, mapping: dict[str, str] = KG.FEATURE_TO_MODULE
) -> pd.DataFrame:
    """SECONDARY DIAGNOSTIC ONLY: keep the proxies AND add the scores.

    Never the headline M3 result. Included to show what replacement discards.
    """
    scores = pd.DataFrame(
        module_scores(frame, mapping),
        columns=list(MODULE_SCORE_NAMES.values()),
        index=frame.index,
    )
    return pd.concat([frame[list(FEATURE_COLUMNS)], scores], axis=1)


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
def _logistic(x_train, y_train, x_test) -> dict:
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("logisticregression", LogisticRegression()),
    ])
    model.fit(x_train, y_train)
    probability = model.predict_proba(x_test)[:, 1]
    return {"probability": probability, "prediction": (probability >= 0.5).astype(int)}


def _prototype(x_train, y_train, x_test) -> dict:
    scaler = StandardScaler().fit(x_train)  # training rows only
    a, b = scaler.transform(x_train), scaler.transform(x_test)
    prototypes = prototype_model.fit_prototypes(a, y_train)
    out = prototype_model.predict(prototypes, b)
    return {"probability": out["probability"], "prediction": out["prediction"]}


def _hist_gradient_boosting(x_train, y_train, x_test) -> dict:
    model = HistGradientBoostingClassifier(
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        random_state=cv_protocol.CV_RANDOM_STATE,
    )
    model.fit(x_train, y_train)
    probability = model.predict_proba(x_test)[:, 1]
    return {"probability": probability, "prediction": (probability >= 0.5).astype(int)}


CONDITIONS = {
    "LR0_logistic_raw": "raw Logistic Regression (reference)",
    "LR3_logistic_module_replacement": "LR0 with correct module REPLACEMENT (M3, main)",
    "LR3c_logistic_incorrect_grouping": "LR0 with incorrect-grouping replacement (control)",
    "LR3a_logistic_module_appended": "SECONDARY DIAGNOSTIC: raw proxies + module scores",
    "P0_plain_prototype": "plain prototype (reference)",
    "P3_prototype_module_replacement": "P0 with correct module REPLACEMENT (M3, main)",
    "P3c_prototype_incorrect_grouping": "P0 with incorrect-grouping replacement (control)",
    "HGB0_hist_gradient_boosting": "generic nonlinear boosting (context only)",
}

SHORT = {
    "LR0_logistic_raw": "LR0",
    "LR3_logistic_module_replacement": "LR3",
    "LR3c_logistic_incorrect_grouping": "LR3c",
    "LR3a_logistic_module_appended": "LR3a",
    "P0_plain_prototype": "P0",
    "P3_prototype_module_replacement": "P3",
    "P3c_prototype_incorrect_grouping": "P3c",
    "HGB0_hist_gradient_boosting": "HGB0",
}


def run_cross_validation(candidates: pd.DataFrame) -> dict:
    """PRIMARY protocol: the shared 5 folds, verified explicitly."""
    KG.validate_groupings()

    X_raw = candidates[FEATURE_COLUMNS].copy()
    baseline.assert_model_facing_only(X_raw)
    y = candidates["label"].to_numpy()

    # Representations are built from OBSERVABLE features only, per row, with
    # no labels and no fitted statistic, so they are computed identically for
    # training and held-out rows and cannot leak.
    X_correct = replace_with_module_scores(candidates, KG.FEATURE_TO_MODULE)
    X_incorrect = replace_with_module_scores(candidates, KG.INCORRECT_FEATURE_TO_MODULE)
    X_appended = append_module_scores(candidates, KG.FEATURE_TO_MODULE)

    folds = list(cv_protocol.make_cv().split(X_raw, y))
    expected = cv_protocol.fold_assignment(y)
    for k, (_, test_idx) in enumerate(folds):
        assert list(test_idx) == list(expected[k]), f"fold {k} differs from the shared protocol"

    def model_for(name, train_idx, test_idx):
        raw_tr = X_raw.iloc[train_idx]
        raw_te = X_raw.iloc[test_idx]
        y_train = y[train_idx]
        if name == "LR0_logistic_raw":
            return _logistic(raw_tr.to_numpy(), y_train, raw_te.to_numpy())
        if name == "LR3_logistic_module_replacement":
            return _logistic(X_correct.iloc[train_idx].to_numpy(), y_train,
                             X_correct.iloc[test_idx].to_numpy())
        if name == "LR3c_logistic_incorrect_grouping":
            return _logistic(X_incorrect.iloc[train_idx].to_numpy(), y_train,
                             X_incorrect.iloc[test_idx].to_numpy())
        if name == "LR3a_logistic_module_appended":
            return _logistic(X_appended.iloc[train_idx].to_numpy(), y_train,
                             X_appended.iloc[test_idx].to_numpy())
        if name == "P0_plain_prototype":
            return _prototype(raw_tr.to_numpy(), y_train, raw_te.to_numpy())
        if name == "P3_prototype_module_replacement":
            return _prototype(X_correct.iloc[train_idx].to_numpy(), y_train,
                              X_correct.iloc[test_idx].to_numpy())
        if name == "P3c_prototype_incorrect_grouping":
            return _prototype(X_incorrect.iloc[train_idx].to_numpy(), y_train,
                              X_incorrect.iloc[test_idx].to_numpy())
        if name == "HGB0_hist_gradient_boosting":
            return _hist_gradient_boosting(raw_tr.to_numpy(), y_train, raw_te.to_numpy())
        raise ValueError(f"unknown condition {name}")

    per_fold = {name: [] for name in CONDITIONS}
    for k, (train_idx, test_idx) in enumerate(folds):
        y_test = y[test_idx]
        for name in CONDITIONS:
            out = model_for(name, train_idx, test_idx)
            per_fold[name].append({"fold": k, **baseline.evaluate(y_test, out["probability"])})

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
                "condition. Module scores are equal-weight means computed per "
                "row from observable features only: no labels, no latent state, "
                "no known noise variance, no fitted statistic."
            ),
            "primary_design": "replacement (12 -> 10 columns), identical for correct and incorrect",
            "base_features": list(FEATURE_COLUMNS),
            "aggregated_features": list(KG.AGGREGATED_FEATURES),
            "module_scores": MODULE_SCORE_NAMES,
            "incorrect_grouping_scores": INCORRECT_SCORE_NAMES,
            "grouping_source": "src/knowledge_graph.py, single authoritative source",
            "grouping_fixed_before_evaluation": True,
            "aggregation_weights": "equal weight by construction, not fitted",
            "no_fret_prior_used": True,
            "no_structural_prior_used": True,
            "no_relevance_weighting_used": True,
            "rejection_used": False,
            "latent_truth_used": False,
            "hyperparameter_tuning": "none",
        },
        "conditions": CONDITIONS,
        "summary_mean_std": summary,
        "per_fold": per_fold,
    }


def compare(cv_result: dict) -> dict:
    """Fold-wise differences. Primary comparisons first, then context."""
    primary = {
        "LR3_vs_LR0": ("LR3_logistic_module_replacement", "LR0_logistic_raw"),
        "LR3_vs_LR3c": ("LR3_logistic_module_replacement", "LR3c_logistic_incorrect_grouping"),
        "P3_vs_P0": ("P3_prototype_module_replacement", "P0_plain_prototype"),
        "P3_vs_P3c": ("P3_prototype_module_replacement", "P3c_prototype_incorrect_grouping"),
    }
    context = {
        "LR3_vs_HGB0": ("LR3_logistic_module_replacement", "HGB0_hist_gradient_boosting"),
        "P3_vs_HGB0": ("P3_prototype_module_replacement", "HGB0_hist_gradient_boosting"),
        "LR3a_vs_LR0": ("LR3a_logistic_module_appended", "LR0_logistic_raw"),
    }
    out = {}
    for group, pairs in (("primary", primary), ("context", context)):
        out[group] = {}
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
            out[group][label] = rows
    return out


def interpretability(candidates: pd.DataFrame) -> dict:
    """Module-level coefficients (LR3) and prototype separations (P3).

    Predictive only. The grouping was SUPPLIED by prior knowledge; the model
    did not discover module membership.
    """
    X_correct = replace_with_module_scores(candidates, KG.FEATURE_TO_MODULE)
    X_incorrect = replace_with_module_scores(candidates, KG.INCORRECT_FEATURE_TO_MODULE)
    y = candidates["label"].to_numpy()

    # LR3, fitted on ALL data purely to obtain interpretable coefficients. This
    # is a descriptive fit for inspection, NOT a performance estimate; all
    # reported metrics come from the cross-validated runs above.
    model = Pipeline([("scaler", StandardScaler()), ("logisticregression", LogisticRegression())])
    model.fit(X_correct.to_numpy(), y)
    coefs = model.named_steps["logisticregression"].coef_.ravel()
    names = list(X_correct.columns)
    coefficient_rows = sorted(
        ({"feature": n, "coefficient": float(c), "abs_coefficient": float(abs(c))}
         for n, c in zip(names, coefs)),
        key=lambda r: -r["abs_coefficient"],
    )

    # P3 module-level prototype separation, in training-SD units.
    scaler = StandardScaler().fit(X_correct.to_numpy())
    a = scaler.transform(X_correct.to_numpy())
    protos = prototype_model.fit_prototypes(a, y)
    separation = {
        name: float(abs(protos["plausible"][i] - protos["implausible"][i]))
        for i, name in enumerate(names)
    }

    # The same, for the incorrect grouping, for comparison.
    scaler_i = StandardScaler().fit(X_incorrect.to_numpy())
    ai = scaler_i.transform(X_incorrect.to_numpy())
    protos_i = prototype_model.fit_prototypes(ai, y)
    separation_incorrect = {
        name: float(abs(protos_i["plausible"][i] - protos_i["implausible"][i]))
        for i, name in enumerate(X_incorrect.columns)
    }

    # Raw-proxy separations from the Task 4 plain prototype, for context.
    raw = StandardScaler().fit(candidates[FEATURE_COLUMNS].to_numpy())
    raw_protos = prototype_model.fit_prototypes(
        raw.transform(candidates[FEATURE_COLUMNS].to_numpy()), y
    )
    raw_separation = {
        name: float(abs(raw_protos["plausible"][i] - raw_protos["implausible"][i]))
        for i, name in enumerate(FEATURE_COLUMNS)
    }

    return {
        "note": (
            "Coefficients and separations are PREDICTIVE associations from "
            "descriptive fits, not causal claims. The module grouping was "
            "SUPPLIED by prior knowledge from the knowledge graph; the model "
            "did not discover membership."
        ),
        "lr3_standardized_coefficients": coefficient_rows,
        "p3_prototype_separation_correct": separation,
        "p3_prototype_separation_incorrect": separation_incorrect,
        "p0_prototype_separation_raw_proxies": {
            f: raw_separation[f] for f in KG.AGGREGATED_FEATURES
        },
    }


def information_loss(candidates: pd.DataFrame) -> dict:
    """What is gained and lost by replacing two proxies with their mean.

    Descriptive statistics computed on observable data only. No claim that
    denoising must help.

    A caveat on interpreting the pairwise proxy correlation as a noise
    estimate. The proxy pairs already share substantial variation
    (correlation around 0.64), so equal-weight averaging provides only
    limited additional denoising. Whether that shared variation is
    measurement noise or genuine shared signal CANNOT be determined from
    these data: there is no separately measured noise estimate for any
    proxy, and latent states are not consulted here. Any noise-based
    reading is therefore tentative.
    """
    label = candidates["label"].to_numpy()
    rows = {}
    for module, members in KG.module_members(KG.FEATURE_TO_MODULE).items():
        values = candidates[members].to_numpy()
        mean_score = values.mean(axis=1)
        rows[module] = {
            "members": members,
            "mean_pairwise_correlation": float(np.corrcoef(values[:, 0], values[:, 1])[0, 1]),
            "sd_of_each_proxy": [float(values[:, i].std()) for i in range(values.shape[1])],
            "sd_of_mean_score": float(mean_score.std()),
            "mean_score_correlation_with_label": float(np.corrcoef(mean_score, label)[0, 1]),
            "best_single_proxy_correlation_with_label": float(max(
                np.corrcoef(values[:, i], label)[0, 1] for i in range(values.shape[1])
            )),
        }
    return {
        "note": (
            "Descriptive only. Averaging two proxies of one module yields a "
            "score that tracks the label slightly more closely than either "
            "member alone, but it also DELETES proxy-specific variation. The "
            "proxy pairs already share substantial variation (correlation "
            "around 0.64), so equal-weight averaging provides only limited "
            "additional denoising. Whether the shared variation is measurement "
            "noise or shared signal cannot be determined here, so that reading "
            "stays tentative. Whether aggregation helps, hurts, or changes "
            "nothing is an empirical question answered by the CV results, not "
            "an assumption."
        ),
        "per_module": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 8 -- M3 network knowledge")
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()

    KG.validate_groupings()
    candidates = load_candidates()
    result = run_cross_validation(candidates)
    result["comparison"] = compare(result)
    result["interpretability"] = interpretability(candidates)
    result["information_loss"] = information_loss(candidates)
    result["knowledge_graph"] = KG.describe()
    result["versions"] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }

    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / "network_knowledge_cv.json"
    with open(out_path, "w") as handle:
        json.dump(result, handle, indent=2)
    print(f"wrote {out_path}\n")
    print(f"fold fingerprint: {result['protocol']['fold_fingerprint']} (verified shared)")
    print(f"primary design: {result['protocol']['primary_design']}\n")

    for name in CONDITIONS:
        print(f"  {SHORT[name]:6s} {CONDITIONS[name]}")
        for metric in cv_protocol.PRIMARY_METRICS:
            s = result["summary_mean_std"][name][metric]
            print(f"    {metric:19s} {s['mean']:.4f} +/- {s['std']:.4f}")
        print()

    for group, pairs in result["comparison"].items():
        print(f"{group.upper()} comparisons (positive = first-named better; brier lower better)")
        for label, rows in pairs.items():
            row = next(r for r in rows if r["metric"] == "auroc")
            print(f"  {label:14s} AUROC {row['new_mean']:.4f} vs {row['reference_mean']:.4f} "
                  f"diff={row['mean_difference']:+.4f} ({row['n_folds_new_better']}/5 folds better)")
        print()

    print("module-level standardized coefficients (LR3):")
    for row in result["interpretability"]["lr3_standardized_coefficients"][:6]:
        print(f"  {row['feature']:28s} {row['coefficient']:+.4f}")


if __name__ == "__main__":
    main()
