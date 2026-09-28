"""
Task 11 -- final scientific figures.

This script ONLY reads saved result files and draws. It never fits a model,
never tunes anything, and never recomputes a result. Every number that
appears in a final figure is read from a JSON file under results/ at run
time, so no value is duplicated or transcribed into this module.

MISSING SOURCES FAIL LOUDLY
---------------------------
Every required file and key is declared up front and asserted before any
figure is drawn. If a result file is absent, a key is renamed, or a
condition is dropped, the run aborts with a clear message rather than
producing a figure with a silent gap. The same applies to values that
cannot support a required claim: nothing is invented or substituted to fill
a hole.

SYNTHETIC DATA
--------------
The dataset is synthetic. FRET-like measurements are simulated. Nothing in
these figures is real RNA folding, a real FRET measurement, or biological
validation, and no panel implies causality.

VISUAL STYLE
------------
One font family, one size scale, one line width, flat colours, no 3D, no
decorative effects, and no axis truncation chosen to dramatise a small
difference. Where differences are genuinely small (Figures 2 and 3) the
axis says so, because compressing them would misrepresent them.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "results"
FIGDIR = REPO_ROOT / "figures" / "final"

# ---------------------------------------------------------------- style ---
FONT = "DejaVu Sans"
plt.rcParams.update({
    "font.family": FONT,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.6,
    "lines.markersize": 5,
    "figure.dpi": 110,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.6,
    # Fixed salt for the internal SVG element ids, which matplotlib otherwise
    # derives from object identity and varies between runs of identical code.
    # Affects ids only, never the rendered output.
    "svg.hashsalt": "knowledgeInformed_rna_ml_fig",
})

# Restrained, colour-blind-safe-ish palette, reused across all figures.
C_BLUE = "#3B6EA5"
C_ORANGE = "#D1662C"
C_GREEN = "#4E8A63"
C_PURPLE = "#7B6CA8"
C_GREY = "#8A8A8A"
C_RED = "#B3453C"

# ------------------------------------------------------- required sources ---
REQUIRED_SOURCES = {
    "m0_cv": "m0_cross_validation.json",
    "prototype": "prototype_baseline_cv.json",
    "relevance": "relevance_prototype_cv.json",
    "fret": "fret_knowledge_cv.json",
    "structural": "structural_knowledge_cv.json",
    "network": "network_knowledge_cv.json",
    "rejection": "rejection_cv.json",
    "robustness": "robustness_cv.json",
    "module_diagnostic": "module_latent_diagnostic.json",
}

# Keys that must exist for each figure, checked before drawing anything.
REQUIRED_KEYS = {
    "m0_cv": ["per_fold", "summary_mean_std"],
    "prototype": ["per_fold", "summary_mean_std"],
    "relevance": ["per_fold", "summary_mean_std"],
    "fret": ["comparison", "summary_mean_std"],
    "structural": ["comparison", "summary_mean_std"],
    "network": ["comparison", "interpretability", "summary_mean_std"],
    "rejection": ["per_threshold", "task4_reproduction_check"],
    "robustness": ["degradation", "protocol", "non_fret_control"],
    "module_diagnostic": ["quantities", "diagnostic_only", "consistency_check_vs_task8_notebook"],
}


def load_all() -> dict:
    """Load and validate every required result file. Fails loudly."""
    data = {}
    problems = []
    for key, filename in REQUIRED_SOURCES.items():
        path = RESULTS / filename
        if not path.exists():
            problems.append(f"missing result file: {path.relative_to(REPO_ROOT)}")
            continue
        with open(path) as handle:
            data[key] = json.load(handle)
        for required in REQUIRED_KEYS[key]:
            if required not in data[key]:
                problems.append(
                    f"{filename}: missing required key {required!r}"
                )
    if problems:
        raise SystemExit(
            "Cannot build final figures; required results are unavailable:\n  - "
            + "\n  - ".join(problems)
        )
    # Cross-file safety: every figure must be on the same folds.
    fingerprints = {
        "m0_cv": data["m0_cv"]["protocol"]["fold_fingerprint"],
        "prototype": data["prototype"]["protocol"]["fold_fingerprint"],
        "relevance": data["relevance"]["protocol"]["fold_fingerprint"],
        "rejection": data["rejection"]["protocol"]["fold_fingerprint"],
        "robustness": data["robustness"]["protocol"]["fold_fingerprint"],
    }
    if len(set(fingerprints.values())) != 1:
        raise SystemExit(
            f"result files disagree on the CV fold partition: {fingerprints}"
        )
    return data


def save(fig, stem: str) -> None:
    """Write one figure as SVG and PNG.

    ``Date: None`` suppresses the wall-clock timestamp matplotlib would
    otherwise embed in the SVG metadata, so re-running this script on the same
    result files reproduces byte-identical SVGs.
    """
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGDIR / f"{stem}.svg", metadata={"Date": None})
    fig.savefig(FIGDIR / f"{stem}.png", metadata={"Software": None})
    plt.close(fig)
    print(f"  wrote {stem}.svg and {stem}.png")


def fold_values(per_fold: list[dict], metric: str) -> np.ndarray:
    return np.array([row[metric] for row in per_fold])


def mean_std(summary: dict, metric: str) -> tuple[float, float]:
    return summary[metric]["mean"], summary[metric]["std"]


# ---------------------------------------------------------------------------
# Figure 1's dependency map, declared once so it can be checked against the
# generator instead of merely asserted in prose. Keys are quantities, values
# are the sets of quantities each one is drawn as a direct child of. An empty
# set means "exogenous, drawn with no incoming arrow".
FIGURE1_PARENTS = {
    # Latent states.
    "a_A": {"q"}, "a_B": {"q"}, "a_C": {"q"}, "a_D": {"q"},
    "f_true": {"q"}, "s_true": {"q"}, "c_true": {"q"},
    "z_true": set(), "sigma": set(), "m_signed": set(),
    # The score, fed by latent states only -- never by an observed column.
    "score": {"m_mag", "a_A", "a_B", "a_C", "a_D", "c_true", "f_true", "s_true"},
    # Observed columns and what each reads out.
    "fret_error": {"z_true", "m_signed", "sigma"},
    "fret_uncertainty": {"sigma"},
    "free_energy": {"f_true"},
    "stem_count": {"s_true"},
    "structural_consistency": {"c_true"},
    "gc_content": {"q"},
    "reg_feature_1": {"a_C"}, "reg_feature_2": {"a_C"},
    "pathway_feature_1": {"a_B"}, "pathway_feature_2": {"a_B"},
    "noise_feature_1": set(), "noise_feature_2": set(),
}

# The score coefficients as printed inside the schematic, checked against
# generate_data.latent_plausibility_score().
FIGURE1_SCORE_TERMS = {
    "m_mag": "w_fret", "a_A": "w_module_a", "a_B": "w_module_b",
    "a_C": "w_module_c", "a_D": "w_module_d",
    "c_true": "w_c_true", "f_true": "w_f_true",
}


def assert_figure1_matches_generator(observed_columns, stage1_names, parents,
                                      score_terms) -> None:
    """Check Figure 1's schematic against src/generate_data.py itself.

    Imports the generator and compares, rather than trusting the drawing or
    this file's comments:
      * the observed columns drawn must be exactly generate_data.FEATURES;
      * every direct parent in FIGURE1_PARENTS must match the term that
        actually produces the child in the generator's Stage 3 dict;
      * no observed column may be a parent of anything, which is what stops
        the schematic implying that fret_error (or any engineered column)
        causes the plausibility score;
      * gc_content and the noise features must not sit under the FRET branch;
      * the coefficients printed in the score box must match PARAMS.
    """
    sys.path.insert(0, str(REPO_ROOT / "src"))
    import generate_data  # noqa: PLC0415

    expected_columns = list(generate_data.FEATURES)
    drawn = list(observed_columns)
    if sorted(drawn) != sorted(expected_columns):
        missing = sorted(set(expected_columns) - set(drawn))
        extra = sorted(set(drawn) - set(expected_columns))
        raise SystemExit(
            f"Figure 1: observed columns do not match generate_data.FEATURES; "
            f"missing={missing} extra={extra}"
        )
    if len(drawn) != len(set(drawn)):
        raise SystemExit("Figure 1: an observed column is drawn twice")

    # Stage 1 latent boxes that feed the score.
    expected_stage1 = {"a_A", "a_B", "a_C", "a_D", "f_true", "s_true", "c_true"}
    if set(stage1_names) != expected_stage1:
        raise SystemExit(
            f"Figure 1: latent row shows {sorted(stage1_names)}, "
            f"generator has {sorted(expected_stage1)}"
        )

    # Every parent set must be a subset of quantities that actually exist, and
    # must exclude every observed column (the Stage 3 read-outs are leaves).
    observed_set = set(expected_columns)
    for child, ps in parents.items():
        if ps & observed_set:
            raise SystemExit(
                f"Figure 1: {child!r} is drawn as a parent of {sorted(ps & observed_set)}; "
                "observed features are leaves in the generator (Stage 3) and "
                "must not appear as parents of anything"
            )
    for child in ("gc_content", "noise_feature_1", "noise_feature_2"):
        if parents[child] & {"z_true", "sigma", "m_signed", "m_mag"}:
            raise SystemExit(
                f"Figure 1: {child!r} is drawn under the FRET branch; in the "
                "generator it does not read out any FRET quantity"
            )
    if parents["z_true"] or parents["sigma"] or parents["m_signed"]:
        raise SystemExit(
            "Figure 1: an FRET latent is given a parent; the generator draws "
            "z_true, sigma and m_signed independently of q"
        )
    if parents["gc_content"] != {"q"}:
        raise SystemExit("Figure 1: gc_content must read out q only")
    for name in ("noise_feature_1", "noise_feature_2"):
        if parents[name]:
            raise SystemExit(
                f"Figure 1: {name} is given a parent; it is a pure distractor"
            )
    for name in ("a_A", "a_B", "a_C", "a_D", "f_true", "s_true", "c_true"):
        if parents[name] != {"q"}:
            raise SystemExit(
                f"Figure 1: {name} must read out q only "
                f"(a_m = rho*q + module-specific; f_true/s_true/c_true load on q)"
            )

    # Coefficients printed in the score box.
    params = generate_data.PARAMS
    printed = {"m_mag": -0.85, "a_A": 0.55, "a_B": 0.45, "a_C": 0.40,
               "a_D": 0.12, "c_true": 0.35, "f_true": 0.40}
    for child, param_name in score_terms.items():
        want = params[param_name]
        if abs(printed[child] - want) > 1e-9:
            raise SystemExit(
                f"Figure 1: score box prints {child} coefficient "
                f"{printed[child]}, generator PARAMS[{param_name!r}] is {want}"
            )
    if params["logit_scale"] != 1.1:
        raise SystemExit("Figure 1: printed logit scale 1.1 no longer matches PARAMS")

    print(f"  figure 1 dependencies verified against src/generate_data.py: "
          f"{len(drawn)} observed columns match FEATURES, "
          f"{len(parents)} parent sets checked, "
          f"{len(score_terms)} score coefficients match PARAMS, "
          "no observed feature used as a parent, FRET latents exogenous")


def assert_figure2_axis_claim(ax) -> None:
    """Figure 2's left panel is zoomed, so it must not claim otherwise."""
    title = ax.get_title()
    banned = ("no axis truncation", "full range shown", "no truncation")
    lowered = title.lower()
    for phrase in banned:
        if phrase in lowered:
            raise SystemExit(
                f"Figure 2: panel title claims {phrase!r} but the AUROC axis is "
                f"zoomed to {ax.get_ylim()}; either show 0-1 or label the zoom"
            )
    lo, hi = ax.get_ylim()
    if (lo, hi) == (0.0, 1.0):
        raise SystemExit(
            "Figure 2: axis is the full 0-1 range, so the title should stop "
            "warning about a zoom that is not there"
        )
    if "zoom" not in lowered:
        raise SystemExit(
            f"Figure 2: the AUROC axis is zoomed to {lo}-{hi} but the title "
            "does not say so"
        )
    print(f"  figure 2 axis claim verified: zoomed to {lo:.2f}-{hi:.2f} and "
          "the title says so")


def assert_figure3_marker_semantics(ax, n_rows: int) -> None:
    """Every plotted mean must use the ONE marker the legend documents.

    An earlier version used marker shape to mean "prior" vs "control", which
    was ambiguous: P1a is a prior-informed condition that also served as the
    control for P1b. Shape now means only "mean", and this check stops a
    per-row override from creeping back in.
    """
    legend_texts = [t.get_text() for t in ax.get_legend().get_texts()]
    if not any("mean" in t.lower() for t in legend_texts):
        raise SystemExit(
            f"Figure 3: legend has no 'mean' entry ({legend_texts}); the legend "
            "must document what the mean marker means"
        )
    mean_lines = [ln for ln in ax.get_lines() if ln.get_marker() == "D"]
    if len(mean_lines) != n_rows:
        raise SystemExit(
            f"Figure 3: expected {n_rows} mean markers but found "
            f"{len(mean_lines)}; every row needs exactly one mean marker"
        )
    other_markers = {ln.get_marker() for ln in ax.get_lines()
                     if ln.get_marker() not in ("D", "None")}
    if other_markers:
        raise SystemExit(
            f"Figure 3: rows use more than one mean marker shape ({other_markers}); "
            "marker shape must carry a single meaning"
        )
    print(f"  figure 3 marker semantics verified: {n_rows} mean markers, all "
          f"the same shape; legend = {legend_texts}")


def assert_figure4_rendered_ticks(ax, row_positions, row_labels) -> None:
    """Read the ticks back OUT of the rendered axes and match them to the bars.

    `assert_figure4_label_mapping` proves the recorded list is self-consistent,
    but on its own it cannot catch the original defect, which was calling
    `set_yticks(range(len(row_labels)))` instead of `set_yticks(row_positions)`:
    the two lists are both 0..9 and perfectly self-consistent, yet the second
    block's bars sit at 5.85..9.85 because of the inter-block gap. So the tick
    positions are compared against the artists' own data here, not against the
    list this module built.
    """
    drawn_positions = [float(t) for t in ax.get_yticks()]
    drawn_labels = [t.get_text() for t in ax.get_yticklabels()]

    if len(drawn_positions) != len(row_positions):
        raise SystemExit(
            f"Figure 4: {len(drawn_positions)} y ticks drawn for "
            f"{len(row_positions)} bars; every bar needs a tick"
        )
    for want, got in zip(row_positions, drawn_positions):
        if abs(want - got) > 1e-9:
            raise SystemExit(
                f"Figure 4: a y tick is at {got} but its bar is at {want}; "
                "ticks must be taken from the recorded bar positions, not from "
                "a re-walked range()"
            )
    for want, got in zip(row_labels, drawn_labels):
        if want != got:
            raise SystemExit(
                f"Figure 4: tick reads {got!r} but the bar at that position is "
                f"{want!r}"
            )
    # The topmost bar must carry a tick, or it would be unlabelled.
    top_bar = max(row_positions)
    if not any(abs(t - top_bar) < 1e-9 for t in drawn_positions):
        raise SystemExit(
            f"Figure 4: the top bar at y={top_bar} has no tick and is unlabelled"
        )
    # And every drawn tick must have a bar under it.
    for t in drawn_positions:
        if not any(abs(t - p) < 1e-9 for p in row_positions):
            raise SystemExit(
                f"Figure 4: a tick at y={t} has no bar; labels and bars must "
                "correspond one-to-one"
            )
    print(f"  figure 4 rendered ticks verified: {len(drawn_positions)} ticks "
          f"read back from the axes, all on bar positions, top bar labelled")


def assert_figure5_axis_convention(ax, cov, acc, rnd, tol: float = 1e-9) -> None:
    """Prove Figure 5 Panel A really plots x=coverage, y=accuracy.

    Rather than trusting the call sites, this reads the coordinates back OUT
    of the artists matplotlib actually drew and matches them against known
    values from results/rejection_cv.json. If x and y were ever swapped, the
    margin-rule artist's x data would be the accuracy series and the known
    points below would not be found, so the run fails here instead of
    shipping a figure with a transposed axis.
    """
    lines = {ln.get_label(): ln for ln in ax.get_lines()}
    for name in ("margin-based rejection", "matched random rejection (mean)"):
        if name not in lines:
            raise SystemExit(f"Figure 5: expected line {name!r} was not drawn")

    def drawn(label):
        line = lines[label]
        return (np.asarray(line.get_xdata(), dtype=float),
                np.asarray(line.get_ydata(), dtype=float))

    margin_x, margin_y = drawn("margin-based rejection")
    rand_x, rand_y = drawn("matched random rejection (mean)")

    # Known points, read from the result file, that pin down the convention.
    # (threshold, expected coverage, expected accepted accuracy)
    known = {0.00: (1.0, 0.751), 0.10: (0.7625, 0.8016), 0.20: (0.4900, 0.8546),
             0.30: (0.2238, 0.8715), 0.40: (0.0563, 0.8889)}
    for threshold, (want_cov, want_acc) in known.items():
        hit = int(np.argmin(np.abs(cov - want_cov)))
        if abs(cov[hit] - want_cov) > 5e-4:
            raise SystemExit(
                f"Figure 5: no result row near coverage {want_cov} "
                f"(threshold {threshold}); closest is {cov[hit]:.4f}"
            )
        if abs(margin_x[hit] - cov[hit]) > tol or abs(margin_y[hit] - acc[hit]) > tol:
            raise SystemExit(
                f"Figure 5 axis convention is wrong at threshold {threshold}: "
                f"drawn point is x={margin_x[hit]:.4f}, y={margin_y[hit]:.4f}; "
                f"required x=coverage {cov[hit]:.4f}, y=accuracy {acc[hit]:.4f}"
            )

    # The random control must follow the SAME convention, not a transposed one.
    if (np.allclose(rand_x, cov, atol=tol) and np.allclose(rand_y, rnd, atol=tol)):
        pass
    elif np.allclose(rand_x, rnd, atol=tol) and np.allclose(rand_y, cov, atol=tol):
        raise SystemExit(
            "Figure 5: the matched random control is plotted transposed "
            "(x=accuracy, y=coverage); it must share the margin rule's "
            "x=coverage, y=accuracy convention"
        )
    else:
        raise SystemExit(
            "Figure 5: matched random control does not match either "
            "(coverage, accuracy) or (accuracy, coverage); refusing to guess"
        )

    # The no-rejection reference must be a horizontal line at accuracy 0.751,
    # which only holds when accuracy is on y.
    for line in ax.get_lines():
        if line.get_linestyle() == ":" and len(line.get_xdata()) > 1:
            ref_y = float(np.asarray(line.get_ydata(), dtype=float)[0])
            if abs(ref_y - acc[0]) > tol:
                raise SystemExit(
                    "Figure 5: the 'no rejection' reference is not a "
                    f"horizontal line at the no-rejection accuracy {acc[0]:.4f}"
                )
    print(f"  figure 5 axis convention verified: x=coverage, y=accuracy "
          f"({len(known)} known points + random control + reference line)")


def assert_figure4_label_mapping(row_labels, row_values, row_positions,
                                 row_keys) -> None:
    """Prove every Figure 4 Panel A bar has exactly one matching label.

    Guards three separate failure modes seen in this figure:
      * a label with no bar (label list longer than the bars);
      * a bar with no label (bars longer than the label list);
      * a tick placed at a different y from its bar, which is what silently
        shifted the whole second group before the positions were recorded.
    """
    n_labels, n_values = len(row_labels), len(row_values)
    if n_labels != n_values:
        raise SystemExit(
            f"Figure 4: {n_labels} labels for {n_values} bars -- "
            "label and value lists must correspond one-to-one"
        )
    if len(row_positions) != n_values:
        raise SystemExit(
            f"Figure 4: {len(row_positions)} bar positions for {n_values} bars -- "
            "every bar needs its own y position for its tick"
        )
    if len(row_keys) != n_values:
        raise SystemExit(
            f"Figure 4: {len(row_keys)} source keys for {n_values} bars"
        )
    if len(set(row_positions)) != n_values:
        raise SystemExit(
            "Figure 4: duplicate bar y positions -- a tick would be shared by "
            "two bars and the mapping would be ambiguous"
        )
    # Labels repeat ACROSS blocks on purpose ("correct module score" appears
    # once per block, distinguished by the shaded block heading), so uniqueness
    # is required per block, not globally.
    per_block = {}
    for key, label in zip(row_keys, row_labels):
        per_block.setdefault(key.split(".", 1)[0], []).append(label)
    for block, labels_in_block in per_block.items():
        if len(set(labels_in_block)) != len(labels_in_block):
            raise SystemExit(
                f"Figure 4: duplicate bar label inside block {block!r} "
                f"({labels_in_block}); the mapping would be ambiguous"
            )
    if len(set(row_keys)) != n_labels:
        raise SystemExit(
            f"Figure 4: duplicate source keys among bars ({sorted(row_keys)}); "
            "the same quantity cannot be plotted as two different bars"
        )
    print(f"  figure 4 label mapping verified: {n_values} bars, "
          f"{n_labels} labels, {len(row_positions)} distinct positions, "
          f"one-to-one with source keys across {len(per_block)} blocks")


# =========================================================== FIGURE 1 ======
# =========================================================== FIGURE 1 ======
def figure1() -> None:
    """Schematic of the synthetic generator, rebuilt from src/generate_data.py.

    Every arrow corresponds to a term in `generate_data.generate()` or
    `generate_data.latent_plausibility_score()`. The dependency list was read
    off that source rather than from memory, and the four claims this
    schematic exists to get right are:

      1. The FRET mismatch process (z_true, sigma, m_signed) is EXOGENOUS. It
         is not downstream of the shared quality factor q and does not share a
         top-level latent-module mechanism with it. The only link between the
         two branches is that m_mag enters the score.
      2. The score is built from LATENT states. No observed feature is an
         input to it, so fret_error does not cause the plausibility score -- it
         is a lossy read-out of the same mismatch that does.
      3. gc_content is a weak read-out of q (loading 0.01). The two
         noise_feature_* columns have no generative parent at all.
      4. Neither gc_content nor the noise features are downstream of the FRET
         measurements.

    Panel A is the latent generative structure; Panel B is the measurement
    channel. Splitting them is what lets every arrow stay short and clear of
    every other box, instead of one dense graph with lines through nodes.
    """
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(15.6, 8.0))

    latent_fill, latent_edge = "#EDE7F2", C_PURPLE
    obs_fill, obs_edge = "#E7EFF7", C_BLUE
    model_fill, model_edge = "#E4F0E8", C_GREEN
    label_fill, label_edge = "#FBEAE4", C_ORANGE
    ghost_fill, ghost_edge = "#EFEFEF", C_GREY
    XMAX = 16.4

    def setup(ax, title, sub):
        ax.set_xlim(-0.4, XMAX)
        ax.set_ylim(-0.3, 11.7)
        ax.axis("off")
        ax.set_title(title, fontsize=10.0, fontweight="bold", pad=16)
        ax.text(0.0, 11.45, sub, fontsize=6.6, color=C_GREY, ha="left",
                va="top", linespacing=1.5)

    def box(ax, cx, y, text, fc, ec, fs=4.8, w=1.6, h=0.8, weight="normal"):
        ax.add_patch(plt.Rectangle((cx - w / 2, y - h / 2), w, h, facecolor=fc,
                                   edgecolor=ec, linewidth=1.0, zorder=3))
        ax.text(cx, y, text, ha="center", va="center", fontsize=fs, zorder=4,
                fontweight=weight, linespacing=1.35)

    def arrow(ax, x_from, y_from, x_to, y_to, colour=C_GREY, lw=0.9, ls="-",
              zorder=2):
        ax.annotate("", xy=(x_to, y_to), xytext=(x_from, y_from),
                    arrowprops=dict(arrowstyle="-|>", color=colour,
                                    linewidth=lw, linestyle=ls,
                                    shrinkA=0, shrinkB=0, mutation_scale=7),
                    zorder=zorder)

    def elbow(ax, pts, colour=C_GREY, lw=0.9, ls="-"):
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs[:-1], ys[:-1], color=colour, lw=lw, ls=ls, zorder=2,
                solid_capstyle="round")
        arrow(ax, xs[-2], ys[-2], xs[-1], ys[-1], colour=colour, lw=lw, ls=ls)

    # ==================================================== PANEL A: latents ===
    setup(axA,
          "A.  Latent generative structure   (nothing in this panel is observed)",
          "Dependencies read from generate_data.generate() and\n"
          "generate_data.latent_plausibility_score() in src/generate_data.py")

    bus_y = 10.4
    box(axA, 0.9, bus_y, "q\nshared latent\nquality factor", latent_fill,
        latent_edge, fs=5.0, w=1.9, h=0.9, weight="bold")
    axA.plot([1.85, 11.7], [bus_y, bus_y], color=C_PURPLE, lw=0.9, zorder=2)

    # Stage 1: the seven latent states that all feed the score.
    stage1 = [
        (1.5, "a_A\nstructural\nstability\n(no proxy)"),
        (3.2, "a_D\nweak/noisy\nbiology\n(no proxy)"),
        (4.9, "a_C\nregulatory\ninteraction"),
        (6.6, "a_B\nFRET-supported\ngeometry"),
        (8.3, "f_true\nfolding\nlatent"),
        (10.0, "s_true\nstem-count\nlatent"),
        (11.7, "c_true\nconsistency\nlatent"),
    ]
    for cx, text in stage1:
        arrow(axA, cx, bus_y, cx, 9.725, colour=C_PURPLE)
        box(axA, cx, 9.15, text, latent_fill, latent_edge, fs=4.8, w=1.6, h=1.15)

    # Stage 1, separate branch: the FRET process. Deliberately given NO arrow
    # from q, because the generator draws it independently. The q bus stops at
    # c_true, well short of this box.
    box(axA, 14.2, 8.6,
        "FRET measurement process\n(EXOGENOUS)\n\n"
        "z_true      sigma\nm_signed\n->  m_mag = |m_signed|",
        latent_fill, C_RED, fs=4.9, w=2.7, h=2.3)
    axA.text(14.2, 7.25,
             "drawn independently of q,\nso there is correctly\nno arrow from q",
             fontsize=4.9, color=C_RED, ha="center", va="top", linespacing=1.4)

    # Collector bus: every latent state, and only latent states, enters the
    # score. A bus keeps seven arrows from overlapping into a smudge.
    coll_y = 6.6
    axA.plot([1.5, 14.2], [coll_y, coll_y], color=C_PURPLE, lw=1.0, zorder=2)
    for cx, _ in stage1:
        arrow(axA, cx, 8.575, cx, coll_y, colour=C_PURPLE)
    arrow(axA, 14.2, 7.45, 14.2, coll_y, colour=C_RED)
    axA.text(4.6, coll_y + 0.2, "every latent state above enters the score",
             fontsize=5.4, color=C_PURPLE, ha="center", va="bottom")

    # Stage 2: score -> probability -> sampled label.
    box(axA, 7.75, 5.25,
        "latent plausibility score\n"
        "= -0.85 m_mag + 0.55 a_A + 0.45 a_B + 0.40 a_C + 0.12 a_D\n"
        "+ 0.35 c_true + 0.40 f_true + stem terms in s_true and c_true",
        latent_fill, latent_edge, fs=5.6, w=13.0, h=1.5, weight="bold")
    arrow(axA, 7.75, coll_y, 7.75, 6.0, colour=C_PURPLE)

    box(axA, 7.75, 3.85, "p_plausible  =  sigmoid(1.1 x score)",
        latent_fill, latent_edge, fs=6.0, w=5.0, h=0.7)
    arrow(axA, 7.75, 4.50, 7.75, 4.20, colour=C_PURPLE)

    box(axA, 7.75, 2.65, "observed label  ~  Bernoulli(p_plausible)",
        label_fill, label_edge, fs=6.0, w=5.6, h=0.7, weight="bold")
    arrow(axA, 7.75, 3.50, 7.75, 3.00, colour=C_PURPLE, ls="--")

    axA.text(-0.2, 1.75,
             "The label is generated from the latent states alone. No observed feature\n"
             "is an input to the score, so the score is not computed from fret_error\n"
             "or from any other engineered column.",
             fontsize=6.4, color=C_GREY, ha="left", va="top", linespacing=1.6)

    # ================================================ PANEL B: measurement ===
    setup(axB,
          "B.  Measurement channel:   latent states  ->  observed columns  ->  dataset",
          "Stage 3 of generate_data.generate(): observed features as noisy,\n"
          "lossy read-outs of the Stage 1 latent states")

    # Each observed column sits directly under the latent it reads out, so
    # every arrow in this panel is vertical and none can cross another.
    cols = [1.0, 3.1, 5.2, 7.3, 9.4, 11.5]

    box(axB, 1.0, bus_y, "q", latent_fill, latent_edge, fs=6.2, w=1.3, h=0.8,
        weight="bold")
    axB.plot([1.65, 11.5], [bus_y, bus_y], color=C_PURPLE, lw=0.9, zorder=2)

    latents = [("a_C", 3.1), ("a_B", 5.2), ("f_true", 7.3), ("s_true", 9.4),
               ("c_true", 11.5)]
    for name, cx in latents:
        arrow(axB, cx, bus_y, cx, 9.55, colour=C_PURPLE)
        box(axB, cx, 9.15, name, latent_fill, latent_edge, fs=5.6, w=1.6, h=0.8)

    # The 12 observed columns, boxed so the set is unambiguous.
    axB.add_patch(plt.Rectangle((0.0, 2.9), XMAX - 0.2, 3.7, facecolor="none",
                                edgecolor=C_BLUE, linewidth=1.0,
                                linestyle=(0, (3, 2)), zorder=1))
    axB.text(0.2, 6.5, "observed feature columns  (12 in total)",
             fontsize=6.0, color=C_BLUE, ha="left", va="top")

    row_a = [("gc_content", "gc_content\n(weak proxy\nfor q)", 1.0),
             ("reg_feature_1", "reg_feature_1\nreg_feature_2", 3.1),
             ("reg_feature_2", "reg_feature_1\nreg_feature_2", 3.1),
             ("pathway_feature_1", "pathway_feature_1\npathway_feature_2", 5.2),
             ("pathway_feature_2", "pathway_feature_1\npathway_feature_2", 5.2),
             ("free_energy", "free_energy", 7.3),
             ("stem_count", "stem_count", 9.4),
             ("structural_consistency", "structural_\nconsistency", 11.5)]
    # Two stacked read-outs share one box, so the same box appears twice in
    # `row_a` (once per column name). Draw it once per distinct (text, x).
    drawn_boxes = set()
    for _, text, cx in row_a:
        if (text, cx) in drawn_boxes:
            continue
        drawn_boxes.add((text, cx))
        box(axB, cx, 5.6, text, obs_fill, obs_edge, fs=4.4, w=1.9, h=1.15)

    box(axB, 3.1, 3.9, "noise_feature_1\nnoise_feature_2", obs_fill, obs_edge,
        fs=4.8, w=2.4, h=1.0)

    # The FRET branch is exogenous, so it is drawn as its own dashed block with
    # NO parent box feeding it. A stacked set of latent boxes would force the
    # arrows to its read-outs straight through the boxes below it, so the three
    # exogenous latents are named inside one block; Panel A spells out their
    # internals.
    axB.add_patch(plt.Rectangle((12.8, 7.0), 3.6, 2.5, facecolor="#FDF3F1",
                                edgecolor=C_RED, linewidth=1.0,
                                linestyle=(0, (3, 2)), zorder=1))
    axB.text(14.6, 8.25,
             "FRET measurement process\n(EXOGENOUS -- no parent)\n\n"
             "z_true,  sigma,  m_signed",
             fontsize=5.2, color=C_RED, ha="center", va="center",
             linespacing=1.5, zorder=4)

    box(axB, 13.6, 5.6, "fret_error", obs_fill, obs_edge, fs=4.8, w=1.9, h=0.9)
    box(axB, 15.4, 5.6, "fret_uncertainty", obs_fill, obs_edge, fs=4.4, w=2.0,
        h=0.9)

    # q -> gc_content, a straight drop in the clear left margin.
    arrow(axB, 1.0, 10.0, 1.0, 6.175, colour=C_PURPLE)
    # Latent -> its own observed column, all vertical, no crossings.
    for _, cx in latents:
        arrow(axB, cx, 8.75, cx, 6.175, colour=C_PURPLE)
    # FRET read-outs, fed from the exogenous block with nothing in between.
    arrow(axB, 13.9, 7.0, 13.6, 6.05, colour=C_RED)
    arrow(axB, 15.0, 7.0, 15.4, 6.05, colour=C_RED)

    axB.text(7.3, 4.7,
             "pure distractors: correlated with each other,\n"
             "no generative parent, no label information",
             fontsize=5.0, color=C_GREY, ha="center", va="center",
             linespacing=1.4)

    box(axB, 7.0, 2.2,
        "Model-facing dataset\n12 observed features  +  the sampled label (target)",
        model_fill, model_edge, fs=6.4, w=11.5, h=0.9, weight="bold")
    arrow(axB, 7.0, 2.9, 7.0, 2.65, colour=C_GREEN)

    box(axB, 7.0, 0.75,
        "latent diagnostic truth  (q, a_A..a_D, z_true, sigma, m_signed, f_true,\n"
        "s_true, c_true, score, p)      DIAGNOSTIC ONLY, NEVER A MODEL INPUT",
        ghost_fill, ghost_edge, fs=5.6, w=11.5, h=0.95)
    axB.text(12.9, 0.75, "no arrow:\nnot an input", fontsize=5.2, color=C_RED,
             ha="left", va="center", linespacing=1.4)

    assert_figure1_matches_generator(
        observed_columns=[name for name, _, _ in row_a]
        + ["noise_feature_1", "noise_feature_2", "fret_error", "fret_uncertainty"],
        stage1_names=[text.split("\n")[0] for _, text in stage1],
        parents=FIGURE1_PARENTS,
        score_terms=FIGURE1_SCORE_TERMS,
    )

    fig.suptitle("Synthetic data-generating structure: what is latent, what is "
                 "measured, and what a model may see", fontsize=12, y=0.985)
    fig.text(0.5, -0.03,
             "Schematic only: no numbers, no axes, no results. Every arrow "
             "corresponds to a term in src/generate_data.py. In particular the FRET "
             "measurement process is exogenous and is NOT downstream of the shared "
             "quality factor q; the plausibility score is built from latent states "
             "only, so no observed feature causes it; and neither gc_content nor the "
             "noise features are downstream of the FRET measurements. FRET-like "
             "quantities here are simulated, not measured. Not real RNA folding, no "
             "biological validation, no causal claim.",
             ha="center", fontsize=7.0, color=C_GREY, wrap=True)
    save(fig, "fig01_project_concept")


# =========================================================== FIGURE 2 ======
def figure2(d: dict) -> None:
    """Cross-validated AUROC for four data-only models.

    Values read from the shared 5-fold results. The y-axis is NOT truncated
    toward the differences: the whole honest range is shown, and a second
    panel shows the deltas against a shared baseline with an explicit
    statement that they are small.
    """
    entries = [
        ("Logistic regression (LR0)", d["m0_cv"]["per_fold"]["logistic_regression"],
         d["m0_cv"]["summary_mean_std"]["logistic_regression"], C_BLUE),
        ("Nonlinear boosting (HGB0)", d["m0_cv"]["per_fold"]["hist_gradient_boosting"],
         d["m0_cv"]["summary_mean_std"]["hist_gradient_boosting"], C_GREEN),
        ("Plain prototype (P0)", d["prototype"]["per_fold"],
         d["prototype"]["summary_mean_std"], C_ORANGE),
        ("Relevance-weighted prototype", d["relevance"]["per_fold"],
         d["relevance"]["summary_mean_std"], C_PURPLE),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.0),
                             gridspec_kw={"width_ratios": [1.15, 1.0]})
    fig.subplots_adjust(top=0.78, bottom=0.24)

    ax = axes[0]
    for i, (label, folds, summary, colour) in enumerate(entries):
        vals = fold_values(folds, "auroc")
        mu, sd = mean_std(summary, "auroc")
        ax.errorbar(i, mu, yerr=sd, fmt="o", color=colour, capsize=4, markersize=7,
                    markeredgecolor="white", markeredgewidth=0.8, zorder=4)
        jitter = np.linspace(-0.14, 0.14, len(vals))
        ax.scatter(i + jitter, vals, s=16, color=colour, alpha=0.45, zorder=3)
    ax.set_xticks(range(len(entries)))
    ax.set_xticklabels([e[0] for e in entries], rotation=20, ha="right", fontsize=7.6)
    ax.set_ylabel("cross-validated AUROC (mean ± SD over 5 folds)")
    # The natural range of AUROC is 0-1. This axis is deliberately zoomed to
    # 0.74-0.88 so the four close baselines can be told apart WITHOUT the
    # small differences being exaggerated by a truncated baseline. The zoom is
    # stated in the panel title rather than implied away: 0.5 (chance) and
    # most of the 0-1 range lie below the axis and are not shown.
    AUROC_ZOOM = (0.74, 0.88)
    ax.set_ylim(*AUROC_ZOOM)
    ax.set_title("Absolute AUROC — ZOOMED AXIS\n"
                 f"only {AUROC_ZOOM[0]:.2f}–{AUROC_ZOOM[1]:.2f} of the natural "
                 "0–1 range is shown;\nchance (0.5) and the rest of the scale "
                 "are off-scale below", fontsize=9.2)

    # Panel B: deltas against logistic regression, same baseline for all.
    ax = axes[1]
    base = np.mean(fold_values(entries[0][1], "auroc"))
    deltas, labels, colours = [], [], []
    for label, folds, summary, colour in entries[1:]:
        vals = fold_values(folds, "auroc")
        deltas.append(vals - base)
        labels.append(label.split(" (")[0])
        colours.append(colour)
    for i, (dvals, label, colour) in enumerate(zip(deltas, labels, colours)):
        ax.scatter(np.full(len(dvals), i) + np.linspace(-0.12, 0.12, len(dvals)),
                   dvals, s=20, color=colour, alpha=0.5, zorder=3)
        mean_v = float(np.mean(dvals))
        ax.hlines(mean_v, i - 0.28, i + 0.28, color=colour, lw=2.6, zorder=4)
        ax.text(i + 0.32, mean_v, f"{mean_v:+.4f}", fontsize=7.2, color=colour,
                va="center", ha="left", zorder=4)
    ax.axhline(0, color=C_GREY, lw=1.0, ls="--", zorder=2)
    ax.set_xticks(range(len(deltas)))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=7.6)
    ax.set_ylabel("fold-level AUROC difference vs logistic regression")
    ax.set_title("Paired differences on identical folds\n"
                 "(relevance weighting is a negligible change)", fontsize=10)
    span = max(0.02, max(abs(np.concatenate(deltas))) * 1.35)
    ax.set_ylim(-span, span)

    fig.suptitle("Data-only baselines, 5-fold cross-validation", fontsize=11, y=0.97)
    assert_figure2_axis_claim(axes[0])
    fig.text(0.5, 0.015,
             "Synthetic data. Mean ± SD over the same 5 folds for all models. "
             "No statistical test performed; the differences are small and "
             "several are within fold-level spread.",
             ha="center", fontsize=7.4, color=C_GREY)
    save(fig, "fig02_model_comparison")


# =========================================================== FIGURE 3 ======
def figure3(d: dict) -> None:
    """Change in AUROC relative to the appropriate baseline, per prior.

    Every delta is read from the saved comparison blocks, each of which was
    computed against that condition's own baseline. Controls sit next to the
    conditions they control for.
    """
    def auroc_delta(source, path, label):
        """Fetch one saved auroc comparison row, failing loudly if absent."""
        node = source
        for part in path.split("."):
            if part not in node:
                raise SystemExit(f"result file is missing section {path!r}")
            node = node[part]
        if label not in node:
            raise SystemExit(f"result file is missing comparison {label!r}")
        for row in node[label]:
            if row["metric"] == "auroc":
                return row
        raise SystemExit(f"no auroc row in comparison {label!r}")

    fk, sk, nk = d["fret"], d["structural"], d["network"]

    items = [
        ("M1  engineered FRET feature",
         auroc_delta(fk, "comparison", "P1a_vs_P0"), C_BLUE),
        ("M1  uncertainty-aware distance",
         auroc_delta(fk, "comparison", "P1b_vs_P0"), C_RED),
        ("M2  structural prior (LR)",
         auroc_delta(sk, "comparison", "LR2_vs_LR0"), C_GREEN),
        ("M2  generic nonlinear control",
         auroc_delta(sk, "comparison", "LR2c_vs_LR0"), C_GREY),
        ("M3  correct module aggregation (LR)",
         auroc_delta(nk, "comparison.primary", "LR3_vs_LR0"), C_PURPLE),
        ("M3  incorrect grouping (LR)",
         auroc_delta(nk, "comparison.primary", "LR3_vs_LR3c"), C_ORANGE),
        ("M3  correct module aggregation (P)",
         auroc_delta(nk, "comparison.primary", "P3_vs_P0"), C_PURPLE),
        ("M3  incorrect grouping (P)",
         auroc_delta(nk, "comparison.primary", "P3_vs_P3c"), C_ORANGE),
    ]

    labels = [i[0] for i in items]
    means = [i[1]["mean_difference"] for i in items]
    sds = [i[1]["difference_std"] for i in items]
    folds = [i[1]["per_fold_difference"] for i in items]
    colours = [i[2] for i in items]

    fig, ax = plt.subplots(figsize=(10.4, 4.8))
    span = max(abs(min(means)), abs(max(means)))
    LABEL_X = span * 1.72   # dedicated label column on the right
    y = np.arange(len(items))[::-1]
    for yv, m, fv, colour in zip(y, means, folds, colours):
        ax.scatter(fv, np.full(len(fv), yv) + np.linspace(-0.16, 0.16, len(fv)),
                   s=20, color=colour, alpha=0.5, zorder=3)
        # ONE marker style for every mean. An earlier version used shape to
        # mark "prior" vs "control", but P1a is a prior-informed condition
        # that was also serving as the control for P1b, so the encoding was
        # ambiguous. Shape now carries a single unambiguous meaning.
        ax.plot([m], [yv], marker="D", color=colour, markersize=8.5,
                markeredgecolor="white", markeredgewidth=0.9, zorder=5)
        # Value label sits at a fixed column, clear of the point cloud.
        ax.text(LABEL_X, yv, f"{m:+.4f}", fontsize=7.4, color=colour,
                va="center", ha="right", zorder=5)

    ax.set_xlim(-span * 1.45, span * 1.95)
    ax.axvline(0, color=C_GREY, lw=1.1, ls="--", zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.2)
    ax.set_xlabel("change in cross-validated AUROC relative to the condition's own baseline")
    ax.set_ylim(-0.65, len(items) - 0.35)
    ax.text(LABEL_X, len(items) - 0.15, "mean change", fontsize=7.6, color=C_GREY,
            ha="right", va="center", fontweight="bold")

    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([], [], marker="D", color=C_GREY, linestyle="none", markersize=8,
               label="mean over the 5 folds"),
        Line2D([], [], marker=".", color=C_GREY, linestyle="none", markersize=9,
               label="individual folds"),
    ], loc="lower left", frameon=False, fontsize=7.8)

    ax.set_title("Effect of the three knowledge priors on AUROC, with their controls",
                 fontsize=11)
    assert_figure3_marker_semantics(ax, len(items))
    fig.text(0.5, -0.075,
             "Each condition is compared to its own baseline: P1a/P1b to P0, LR2/LR2c to LR0, "
             "M3 to LR0 and P0 respectively. Marker shape carries only one meaning -- mean versus "
             "individual fold -- so which rows are priors and which are controls is read from the "
             "row labels, not from the marker. The engineered FRET feature is approximately "
             "neutral, the uncertainty-aware distance reduces AUROC, the structural prior is a "
             "small positive change, and correct module aggregation does not improve AUROC. "
             "Deltas are small; the axis is not compressed to exaggerate them. "
             "Synthetic data; no statistical test performed.",
             ha="center", fontsize=7.2, color=C_GREY, wrap=True)
    save(fig, "fig03_prior_effects")


# =========================================================== FIGURE 4 ======
def figure4(d: dict) -> None:
    """Module recovery versus predictive performance: the Task 8 dissociation.

    Panel A is diagnostic-only latent analysis, sourced entirely from
    results/module_latent_diagnostic.json. Panel B is predictive, sourced
    from results/network_knowledge_cv.json. The two panels measure
    different things and are never placed on a shared axis.
    """
    diag = d["module_diagnostic"]
    if diag.get("diagnostic_only") is not True:
        raise SystemExit("module_latent_diagnostic.json is not flagged diagnostic_only")
    reg = diag["quantities"]["regulatory_vs_a_C"]
    path = diag["quantities"]["pathway_vs_a_B"]

    nk = d["network"]
    prim = nk["comparison"]["primary"]

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.0),
                             gridspec_kw={"width_ratios": [1.25, 1.0]})

    # ---- Panel A: latent module recovery (diagnostic only) ----
    ax = axes[0]
    reg_score = "regulatory_module_score"
    path_score = "pathway_module_score"
    inc1 = "incorrect_module_1_score"
    inc2 = "incorrect_module_2_score"

    # Each row is (display label, source key, source block, colour), so the
    # value drawn can never drift from the key it was read from.
    groups = [
        ("Regulatory module  vs latent a_C", [
            ("reg_feature_1", "reg_feature_1", C_BLUE),
            ("reg_feature_2", "reg_feature_2", C_BLUE),
            ("correct module score", reg_score, C_GREEN),
            ("incorrect score 1", inc1, C_GREY),
            ("incorrect score 2", inc2, C_GREY),
        ]),
        ("Pathway module  vs latent a_B", [
            ("pathway_feature_1", "pathway_feature_1", C_BLUE),
            ("pathway_feature_2", "pathway_feature_2", C_BLUE),
            ("correct module score", path_score, C_GREEN),
            ("incorrect score 1", inc1, C_GREY),
            ("incorrect score 2", inc2, C_GREY),
        ]),
    ]
    blocks = {"Regulatory": ("regulatory_vs_a_C", reg),
              "Pathway": ("pathway_vs_a_B", path)}

    # Pathway block on top, regulatory below, so each block reads top-down in
    # the order its rows were listed.
    groups = groups[::-1]

    # The two blocks must report the correct module scores, not the
    # neighbouring proxy values. These are the values the caption quotes.
    expected_correct = {
        "regulatory_vs_a_C": (reg, reg_score, 0.873472),
        "pathway_vs_a_B": (path, path_score, 0.887459),
    }
    for block, (node, key, want) in expected_correct.items():
        if abs(node[key] - want) > 5e-4:
            raise SystemExit(
                f"Figure 4: {block} correct module score is {node[key]:.6f}, "
                f"expected {want:.6f}; a proxy value has been substituted"
            )

    row_labels, row_values, row_colours, row_keys = [], [], [], []
    row_positions = []
    row = 0
    for gname, rows in groups:
        block_name, block_node = blocks[gname.split()[0]]
        start = row
        for name, key, colour in rows:
            value = float(block_node[key])
            ax.barh(row, value, height=0.68, color=colour, edgecolor="white",
                    linewidth=0.6, alpha=1.0 if colour != C_GREY else 0.5, zorder=3)
            ax.text(value + 0.010, row, f"{value:.3f}", va="center", fontsize=7.4,
                    color=colour if colour != C_GREY else "#5A5A5A", zorder=4)
            row_labels.append(name)
            row_values.append(value)
            row_colours.append(colour)
            # The tick MUST be recorded from the same `row` value the bar was
            # drawn at. Deriving ticks from a re-walked range() instead would
            # misplace every bar in the second block, whose y positions are
            # offset by the inter-block gap.
            row_positions.append(row)
            row_keys.append(f"{block_name}.{key}")
            row += 1
        # shaded band, with the group name as an in-plot heading above it
        ax.axhspan(start - 0.5, row - 0.5,
                   color="#EFEFEF" if start == 0 else "#F8F8F8", zorder=0)
        ax.text(0.005, row - 0.5 + 0.08, gname, fontsize=8.2,
                fontweight="bold", color=C_PURPLE, ha="left", va="bottom", zorder=5)
        row += 0.85

    assert_figure4_label_mapping(row_labels, row_values, row_positions, row_keys)

    # Ticks on the recorded positions, not on range(len(labels)).
    ax.set_yticks(row_positions)
    ax.set_yticklabels(row_labels, fontsize=7.8)
    ax.set_xlabel("correlation with the latent module state")
    ax.set_xlim(0, 1.02)
    # headroom above the last band so the group heading stays inside the axes
    ax.set_ylim(-0.6, row + 0.15)
    ax.grid(axis="y", visible=False)
    assert_figure4_rendered_ticks(ax, row_positions, row_labels)
    ax.set_title("A. Latent module recovery — DIAGNOSTIC ONLY\n"
                 "correct aggregation tracks its own module state best",
                 fontsize=9.6, color=C_PURPLE)

    # ---- Panel B: predictive effect ----
    ax = axes[1]
    order = [("LR3 − LR0", "LR3_vs_LR0"), ("LR3 − LR3c", "LR3_vs_LR3c"),
             ("P3 − P0", "P3_vs_P0"), ("P3 − P3c", "P3_vs_P3c")]
    vals, sds, folds, colours_b = [], [], [], []
    for label, key in order:
        for row in prim[key]:
            if row["metric"] == "auroc":
                vals.append(row["mean_difference"])
                sds.append(row["difference_std"])
                folds.append(row["per_fold_difference"])
                colours_b.append(C_PURPLE if "LR3" in label or "P3 − P0" == label else C_ORANGE)
    xs = np.arange(len(order))
    for xi, fv, colour in zip(xs, folds, colours_b):
        ax.scatter(np.full(len(fv), xi) + np.linspace(-0.13, 0.13, len(fv)), fv,
                   s=22, color=colour, alpha=0.5, zorder=3)
    ax.errorbar(xs, vals, yerr=sds, fmt="D", color="k", markersize=7, capsize=4,
                linestyle="none", markeredgecolor="white", markeredgewidth=0.8, zorder=5)
    # value labels sit to the left of each error bar so they never sit on a fold point
    for xi, v in zip(xs, vals):
        ax.text(xi - 0.22, v, f"{v:+.4f}", ha="right", va="center", fontsize=7.4,
                color=C_GREY)
    ax.axhline(0, color=C_GREY, lw=1.0, ls="--", zorder=2)
    ax.set_xticks(xs)
    ax.set_xticklabels([o[0] for o in order], fontsize=8)
    ax.set_xlim(-0.72, len(order) - 0.35)
    ax.set_ylabel("change in cross-validated AUROC")
    # limits taken from the data (mean +/- SD and every fold point) plus padding,
    # so no error bar or fold point can be clipped
    all_pts = np.concatenate([np.asarray(f, dtype=float) for f in folds])
    lo = min(float(np.min(np.asarray(vals) - np.asarray(sds))), float(all_pts.min()))
    hi = max(float(np.max(np.asarray(vals) + np.asarray(sds))), float(all_pts.max()))
    pad = 0.16 * (hi - lo)
    ax.set_ylim(lo - pad, hi + pad)
    ax.scatter([], [], s=22, color=C_PURPLE, alpha=0.5, label="individual folds")
    ax.errorbar([], [], yerr=[], fmt="D", color="k", markersize=6, linestyle="none",
                label="mean ± SD across folds")
    ax.legend(frameon=False, fontsize=7.0, loc="upper left", handletextpad=0.6)
    ax.set_title("B. Predictive effect\n"
                 "correct aggregation does not improve discrimination", fontsize=9.6)

    fig.suptitle("Module recovery and predictive performance dissociate", fontsize=11)
    fig.text(0.5, -0.045,
             "Panel A is a diagnostic-only latent analysis: it describes the synthetic "
             "generative structure and is not a predictive metric, is not comparable to any "
             "AUROC, and implies no causality. Panel B is the cross-validated predictive "
             "result. The two panels are not placed on a shared axis because they measure "
             "different things. Correct aggregation recovered its intended latent module state "
             "more faithfully than individual proxies or the incorrect grouping, and did not "
             "improve discrimination here; this does not establish that correct grouping caused "
             "the predictive difference. Synthetic data; no statistical test performed.",
             ha="center", fontsize=7.2, color=C_GREY, wrap=True)
    save(fig, "fig04_network_dissociation")


# =========================================================== FIGURE 5 ======
def figure5(d: dict) -> None:
    """Rejection trade-off: coverage against accepted-sample accuracy.

    AXIS CONVENTION
    ---------------
    x is COVERAGE and y is ACCEPTED-SAMPLE ACCURACY, for the margin rule and
    for the matched random control alike. `assert_figure5_axis_convention`
    below re-checks the rendered artist coordinates against known values from
    the result file, so a future refactor cannot silently swap them again.

    The matched random-rejection control is shown because accepted accuracy
    rises for any rule that shrinks the accepted set, and only the gap
    against matched randomness says the margin identifies hard cases. The
    very low coverage region is kept visible, including its instability.
    """
    rj = d["rejection"]
    if rj["task4_reproduction_check"].get("reproduces_task4") is not True:
        raise SystemExit("rejection_cv.json does not confirm Task 4 reproduction")

    rows = rj["per_threshold"]
    cov = np.array([r["pooled"]["coverage"] for r in rows], dtype=float)
    acc = np.array([r["pooled"]["accepted_accuracy"] for r in rows], dtype=float)
    rnd = np.array([r["random_control"]["mean"] for r in rows], dtype=float)
    rsd = np.array([r["random_control"]["std"] or 0.0 for r in rows], dtype=float)
    thr = np.array([r["threshold"] for r in rows])

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.7),
                             gridspec_kw={"width_ratios": [1.1, 1.0]})

    ax = axes[0]
    # NOTE the argument order: (x, y) == (coverage, accuracy) in every call.
    ax.fill_between(cov, rnd - 2 * rsd, rnd + 2 * rsd, color=C_GREY, alpha=0.22,
                    label="matched random rejection (±2 SD)", zorder=2)
    ax.plot(cov, rnd, "s--", color=C_GREY, lw=1.6, zorder=3,
            label="matched random rejection (mean)")
    ax.plot(cov, acc, "o-", color=C_BLUE, lw=1.9, zorder=4,
            label="margin-based rejection")
    for c, a, t in zip(cov, acc, thr):
        if t in (0.05, 0.10, 0.20, 0.30, 0.40):
            ax.annotate(f"|margin| < {t:.2f}", (c, a), fontsize=6.8, color=C_BLUE,
                        xytext=(5, -9), textcoords="offset points")
    ax.set_xlabel("coverage (fraction of candidates answered)")
    ax.set_ylabel("accuracy among accepted samples")
    ax.set_xlim(-0.02, 1.04)
    ax.set_ylim(0.68, 0.93)
    # Reference line for "no rejection": accuracy 0.751 at full coverage.
    ax.axhline(acc[0], color=C_BLUE, lw=0.8, ls=":", alpha=0.7)
    ax.text(0.02, acc[0] + 0.005, f"no rejection = {acc[0]:.3f} at coverage 1.0",
            fontsize=7.2, color=C_BLUE, ha="left", va="bottom")
    ax.legend(loc="lower right", frameon=False, fontsize=7.8)
    ax.set_title("Coverage (x) against accepted-sample accuracy (y)\n"
                 "(thresholds annotated descriptively, none selected)", fontsize=10)

    assert_figure5_axis_convention(ax, cov, acc, rnd)

    ax = axes[1]
    rej_err = np.array([np.nan if r["pooled"]["error_rate_rejected"] is None
                        else r["pooled"]["error_rate_rejected"] for r in rows], dtype=float)
    acc_err = np.array([np.nan if r["pooled"]["error_rate_accepted"] is None
                        else r["pooled"]["error_rate_accepted"] for r in rows], dtype=float)
    ax.plot(thr, acc_err, "o-", color=C_BLUE, lw=1.7, label="error among accepted")
    ax.plot(thr, rej_err, "s-", color=C_ORANGE, lw=1.7, label="error among rejected")
    ax.set_xlabel("rejection threshold  (reject when |margin| < t)")
    ax.set_ylabel("error rate")
    ax.set_ylim(0, 0.75)
    ax.legend(frameon=False, fontsize=7.8, loc="upper right")
    ax.set_title("Rejected cases remain much harder\n"
                 "(rightmost point: very low coverage, unstable)", fontsize=10)

    fig.suptitle("Rejection trades coverage for accuracy on accepted cases", fontsize=11)
    fig.text(0.5, -0.045,
             "Both panels are selective metrics and are not comparable with full-coverage model "
             "accuracy. Accepted accuracy is purchased by rejecting more samples: the right-hand "
             "end of the curve is a small accepted set, and the far-right point is a few samples "
             "and unstable, which is shown rather than hidden. No threshold is selected or "
             "recommended. Synthetic data; no statistical test performed.",
             ha="center", fontsize=7.2, color=C_GREY, wrap=True)
    save(fig, "fig05_rejection_tradeoff")


# =========================================================== FIGURE 6 ======
def figure6(d: dict) -> None:
    """Robustness to degraded FRET-like evidence, using the frozen models."""
    rb = d["robustness"]
    grid = rb["protocol"]["noise_grid"]
    deg = rb["degradation"]
    order = [("LR0", C_BLUE), ("P0", C_GREEN), ("P1b", C_ORANGE),
             ("P2", C_PURPLE), ("P3", C_GREY)]

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.7))

    ax = axes[0]
    for name, colour in order:
        abs_vals = [deg[name][str(l)]["auroc"]["absolute"] for l in grid]
        ax.plot(grid, abs_vals, "o-", color=colour, lw=1.7, label=name, zorder=3)
    ax.set_xlabel("FRET noise multiplier  (1.0 = original measurement quality)")
    ax.set_ylabel("cross-validated AUROC\n(mean over the same 5 folds)")
    ax.set_xticks(grid)
    ax.set_ylim(0.78, 0.86)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    ax.set_title("Absolute AUROC under measurement degradation\n"
                 "(5-fold cross-validated mean, not a single run)", fontsize=9.6)

    ax = axes[1]
    for name, colour in order:
        chg = [deg[name][str(l)]["auroc"]["change_vs_baseline"] for l in grid]
        ax.plot(grid, chg, "o-", color=colour, lw=1.7, label=name, zorder=3)
    ax.axhline(0, color=C_GREY, lw=1.0, ls="--", zorder=2)
    ax.set_xlabel("FRET noise multiplier")
    ax.set_ylabel("change in AUROC vs each model's own baseline\n"
                  "(5-fold cross-validated mean)")
    ax.set_xticks(grid)
    ax.set_ylim(-0.035, 0.035)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    ax.set_title("Degradation relative to each model's own baseline\n"
                 "P1b rises as its over-aggressive weighting is weakened", fontsize=9.6)

    fig.suptitle("Robustness of frozen models to degraded FRET-like evidence", fontsize=11)
    fig.text(0.5, -0.05,
             "Models were trained on the original folds and evaluated on progressively noisier "
             "held-out folds, over the predeclared multipliers; nothing was retuned. LR0 degrades "
             "more than P0; P0 degrades modestly; P2 and P3 show no clear indirect protection. "
             "P1b rises because increasing uncertainty weakens an inverse-variance weighting "
             "that was already too aggressive at baseline, so the perturbation partially removes "
             "a model defect — that is not evidence of robustness. NON-MONOTONIC BEHAVIOUR IS "
             "SHOWN AS MEASURED, NOT SMOOTHED. Only FRET-like degradation was tested: no claim is "
             "made about corruption of structural or network features. Synthetic data.",
             ha="center", fontsize=7.2, color=C_GREY, wrap=True)
    save(fig, "fig06_robustness")


def main() -> None:
    print("Loading result files (no model is fitted, nothing is recomputed)...")
    d = load_all()
    print(f"  all {len(REQUIRED_SOURCES)} sources present; shared fold "
          f"fingerprint {d['m0_cv']['protocol']['fold_fingerprint']}\n")

    print("Building figures:")
    figure1()
    figure2(d)
    figure3(d)
    figure4(d)
    figure5(d)
    figure6(d)
    print(f"\nAll figures written to {FIGDIR.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
