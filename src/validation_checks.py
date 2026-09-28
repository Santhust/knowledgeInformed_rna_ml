"""
Task 2 -- small reusable helpers for data validation.

Deliberately minimal. These functions only READ the generated dataset; they
contain no model, no fitting, and no knowledge weighting.

DATA ACCESS RULE
----------------
`candidates.csv` is model-facing. `latent_truth.csv` is DIAGNOSTIC ONLY and
must never be merged into a model-facing feature matrix, used for fitting, or
imported by any later model module. The only function here that touches it
is `load_latent_truth`, which is diagnostic by name and used exclusively in
the notebook's labelled Diagnostic Ground Truth section.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

FEATURE_COLUMNS = [
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

# Ranges the generator guarantees by construction, used for sanity checks.
# A violation means the dataset is not what the generator claims, not that
# the values are biologically out of range.
EXPECTED_RANGES = {
    "fret_error": (0.0, np.inf),
    "fret_uncertainty": (0.0, np.inf),
    "free_energy": (-np.inf, np.inf),
    "stem_count": (0.0, 20.0),
    "structural_consistency": (0.0, 1.0),
    "gc_content": (0.20, 0.80),
    "reg_feature_1": (-np.inf, np.inf),
    "reg_feature_2": (-np.inf, np.inf),
    "pathway_feature_1": (-np.inf, np.inf),
    "pathway_feature_2": (-np.inf, np.inf),
    "noise_feature_1": (-np.inf, np.inf),
    "noise_feature_2": (-np.inf, np.inf),
}

# Resolved relative to the repository root so the module works regardless of
# the caller's working directory. Override by assigning `DATA_DIR`.
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "generated"


def load_candidates(data_dir: Path | None = None) -> pd.DataFrame:
    """Model-facing data. Safe to use in any later modelling stage."""
    return pd.read_csv((data_dir or DATA_DIR) / "candidates.csv")


def load_latent_truth(data_dir: Path | None = None) -> pd.DataFrame:
    """DIAGNOSTIC ONLY. Never merge into a model-facing feature matrix."""
    return pd.read_csv((data_dir or DATA_DIR) / "latent_truth.csv")


def latent_only_columns(candidates: pd.DataFrame, latent: pd.DataFrame) -> list[str]:
    """Columns that exist in latent truth but are absent from the model data.

    `sample_id` is excluded: it is the legitimate join key present in both
    files, not hidden state.
    """
    return sorted((set(latent.columns) - set(candidates.columns)) - {"sample_id"})


def per_feature_auc(candidates: pd.DataFrame) -> pd.DataFrame:
    """Signed and direction-agnostic univariate AUC for every feature.

    NOTE: univariate AUC cannot capture nonlinear (U-shaped) or
    interaction-dependent contributions. A value near 0.5 is therefore NOT
    evidence that a feature is irrelevant -- see `stem_count`, which enters
    the label through a quadratic and an interaction term.
    """
    y = candidates["label"].to_numpy()
    rows = []
    for name in FEATURE_COLUMNS:
        signed = float(roc_auc_score(y, candidates[name].to_numpy()))
        rows.append(
            {
                "feature": name,
                "auc_signed": signed,
                "auc_strong": max(signed, 1.0 - signed),
            }
        )
    return pd.DataFrame(rows)


def structural_summary(candidates: pd.DataFrame) -> pd.DataFrame:
    """Label rate by stem_count, to inspect structure jointly, not univariate."""
    grouped = candidates.groupby("stem_count")["label"]
    return pd.DataFrame(
        {
            "n": grouped.size(),
            "label_rate": grouped.mean(),
            "mean_structural_consistency": candidates.groupby("stem_count")[
                "structural_consistency"
            ].mean(),
        }
    ).reset_index()


def fret_uncertainty_tiers(
    candidates: pd.DataFrame, tiers: int = 3
) -> pd.DataFrame:
    """Split samples into `tiers` equal-count groups by fret_uncertainty."""
    labelled = candidates.copy()
    labelled["tier"] = pd.qcut(
        labelled["fret_uncertainty"], tiers, labels=False, duplicates="drop"
    )
    return labelled


def scan_for_latent_reads(src_dir: Path | None = None) -> pd.DataFrame:
    """Leakage check: which files under src/ reference latent_truth.csv?

    The generator itself is expected to appear, since it writes that file.
    Any OTHER module reading it is a leakage violation to report.
    """
    pattern = re.compile(r"latent_truth\.csv")
    src_dir = src_dir or Path(__file__).resolve().parent
    rows = []
    for path in sorted(src_dir.rglob("*.py")):
        try:
            text = path.read_text()
        except (UnicodeDecodeError, OSError):
            continue
        hits = len(pattern.findall(text))
        rows.append({"file": path.name, "latent_truth_references": hits})
    return pd.DataFrame(rows)
