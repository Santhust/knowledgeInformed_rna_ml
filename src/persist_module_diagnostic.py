"""
Diagnostic persistence for the Task 8 module-recovery analysis.

WHAT THIS FILE IS
-----------------
A persistence step, NOT a modelling experiment. It recomputes and saves the
latent-module recovery correlations that were already computed, reported and
interpreted in notebooks/07_network_knowledge.ipynb (Task 8), so that
Figure 4 Panel A can be built from a result file rather than from values
transcribed out of notebook output.

No model is fitted, nothing is tuned, no label is read, and no module
membership is changed. Module membership is imported from
src/knowledge_graph.py, the same authoritative source Task 8 used, so the
grouping here is bit-identical to the grouping Task 8 evaluated.

WHY IT IS A SEPARATE FILE
-------------------------
The Task 8 result JSONs deliberately do not contain latent quantities: they
record predictive evaluation only. Keeping this diagnostic in its own file
with explicit provenance flags means the separation between
model-facing evaluation and latent diagnostics survives into the figures,
and a reader can see at a glance that these numbers never touched a model.

LABEL-FREE BY CONSTRUCTION
--------------------------
The correlations are between OBSERVED feature columns and LATENT module
states. `label` is never loaded for these computations. Agreement between a
module score and a latent activity is a statement about the generative
structure, not about predictive performance, and is not comparable to any
AUROC.

SCIENTIFIC STATUS
-----------------
The dataset is synthetic. These correlations describe a known generative
process and are diagnostic only. They are not biological evidence, they do
not imply causality, and they say nothing about real regulatory or pathway
biology.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import knowledge_graph as KG  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
CANDIDATES = REPO_ROOT / "data" / "generated" / "candidates.csv"
LATENT_TRUTH = REPO_ROOT / "data" / "generated" / "latent_truth.csv"
OUTPUT = REPO_ROOT / "results" / "module_latent_diagnostic.json"


def module_scores(frame: pd.DataFrame, mapping: dict[str, str]) -> dict[str, np.ndarray]:
    """Equal-weight means per module, using the authoritative membership.

    Identical in form to `network_knowledge.module_scores`, reproduced here so
    this file has no modelling dependency at all.
    """
    return {
        KG.module_score_names(mapping)[module]: (
            frame[features].to_numpy().mean(axis=1)
        )
        for module, features in KG.module_members(mapping).items()
    }


def correlation(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.corrcoef(x, y)[0, 1])


def compute() -> dict:
    candidates = pd.read_csv(CANDIDATES)
    latent = pd.read_csv(LATENT_TRUTH)
    KG.validate_groupings()

    # Row-order alignment check. Both files are generated together in a fixed
    # order, but an explicit assertion is cheap and prevents a silent
    # misalignment from producing plausible-looking nonsense.
    assert len(candidates) == len(latent), "candidates and latent_truth row counts differ"
    if "sample_id" in candidates.columns and "sample_id" in latent.columns:
        assert (candidates["sample_id"].to_numpy() == latent["sample_id"].to_numpy()).all(), (
            "sample_id ordering differs between candidates.csv and latent_truth.csv"
        )

    correct = module_scores(candidates, KG.FEATURE_TO_MODULE)
    incorrect = module_scores(candidates, KG.INCORRECT_FEATURE_TO_MODULE)

    a_C = latent["a_C"].to_numpy()
    a_B = latent["a_B"].to_numpy()

    reg_score_name = KG.module_score_names(KG.FEATURE_TO_MODULE)["regulatory_module"]
    path_score_name = KG.module_score_names(KG.FEATURE_TO_MODULE)["pathway_module"]
    inc1_name = KG.module_score_names(KG.INCORRECT_FEATURE_TO_MODULE)["incorrect_module_1"]
    inc2_name = KG.module_score_names(KG.INCORRECT_FEATURE_TO_MODULE)["incorrect_module_2"]

    regulatory = {
        "reg_feature_1": correlation(candidates["reg_feature_1"].to_numpy(), a_C),
        "reg_feature_2": correlation(candidates["reg_feature_2"].to_numpy(), a_C),
        reg_score_name: correlation(correct[reg_score_name], a_C),
        inc1_name: correlation(incorrect[inc1_name], a_C),
        inc2_name: correlation(incorrect[inc2_name], a_C),
    }
    pathway = {
        "pathway_feature_1": correlation(candidates["pathway_feature_1"].to_numpy(), a_B),
        "pathway_feature_2": correlation(candidates["pathway_feature_2"].to_numpy(), a_B),
        path_score_name: correlation(correct[path_score_name], a_B),
        inc1_name: correlation(incorrect[inc1_name], a_B),
        inc2_name: correlation(incorrect[inc2_name], a_B),
    }

    # Reference values as rendered in the Task 8 notebook, used ONLY to verify
    # that this recomputation agrees with what was already reported. They are
    # not used to produce any value in the output.
    notebook_reference = {
        "reg_feature_1": 0.818882,
        "reg_feature_2": 0.764675,
        reg_score_name: 0.873472,
        inc1_name: 0.716773,
        inc2_name: 0.660676,
        f"pathway::pathway_feature_1": 0.837791,
        f"pathway::pathway_feature_2": 0.775209,
        f"pathway::{path_score_name}": 0.887459,
        # The Task 8 notebook also printed each incorrect score against BOTH
        # states, so both are pinned here.
        f"pathway::{inc1_name}": 0.722307,
        f"pathway::{inc2_name}": 0.692125,
    }
    # NOTE: do NOT merge these dicts. Both contain the two
    # incorrect_module_* keys, and a dict merge would silently overwrite the
    # regulatory (vs a_C) values with the pathway (vs a_B) values. That bug
    # produced a spurious 0.0055 / 0.0314 "discrepancy" on the first run and
    # was caught by the assertion below. Check each block separately.
    recomputed = dict(regulatory)
    recomputed.update({
        f"pathway::{k}": v for k, v in pathway.items()
    })
    mismatches = {
        key: {"reference": notebook_reference[key], "recomputed": recomputed[key],
              "abs_difference": abs(notebook_reference[key] - recomputed[key])}
        for key in notebook_reference
        if abs(notebook_reference[key] - recomputed[key]) > 1e-4
    }
    assert not mismatches, (
        "Recomputed module-recovery correlations disagree with the Task 8 "
        f"notebook output beyond tolerance: {json.dumps(mismatches, indent=2)}"
    )

    return {
        "diagnostic_only": True,
        "used_for_training": False,
        "used_for_tuning": False,
        "latent_truth_used": True,
        "is_new_experiment": False,
        "purpose": (
            "Persistence of the module-recovery diagnostic already computed and "
            "reported in Task 8 (notebooks/07_network_knowledge.ipynb). Saved so "
            "Figure 4 Panel A can be built from a result file rather than from "
            "values transcribed out of notebook output. No model was fitted."
        ),
        "source_files": {
            "candidates": "data/generated/candidates.csv",
            "latent_truth": "data/generated/latent_truth.csv",
            "module_membership": "src/knowledge_graph.py",
        },
        "module_membership_source": (
            "src/knowledge_graph.py -- the single authoritative source, "
            "unchanged. validate_groupings() is asserted at run time."
        ),
        "labels_used": False,
        "quantities": {
            "regulatory_vs_a_C": regulatory,
            "pathway_vs_a_B": pathway,
        },
        "consistency_check_vs_task8_notebook": {
            "max_abs_difference": max(
                abs(notebook_reference[k] - recomputed[k]) for k in notebook_reference
            ),
            "tolerance": 1e-4,
            "agrees": True,
            "note": (
                "Reference values are the ones rendered in the Task 8 notebook "
                "output. Agreement confirms this file is a persisted version of "
                "the already-established diagnostic, not a new result."
            ),
        },
        "interpretation_limits": (
            "Diagnostic only. Agreement with a latent module activity describes "
            "the generative structure and is NOT a predictive metric, is not "
            "comparable to any AUROC, and implies no causality. Synthetic data; "
            "no biological claim."
        ),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Persist the Task 8 module-recovery diagnostic (no modelling)"
    )
    parser.add_argument("--out", type=Path, default=OUTPUT)
    args = parser.parse_args()

    result = compute()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(result, handle, indent=2)
    print(f"wrote {args.out}")

    check = result["consistency_check_vs_task8_notebook"]
    print(f"agrees with Task 8 notebook: {check['agrees']} "
          f"(max abs difference {check['max_abs_difference']:.2e})")

    print("\nregulatory module vs latent a_C:")
    for key, value in result["quantities"]["regulatory_vs_a_C"].items():
        print(f"  {key:28s} {value:+.4f}")
    print("\npathway module vs latent a_B:")
    for key, value in result["quantities"]["pathway_vs_a_B"].items():
        print(f"  {key:28s} {value:+.4f}")


if __name__ == "__main__":
    main()
