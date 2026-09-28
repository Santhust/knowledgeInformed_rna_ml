"""
The synthetic biological knowledge graph for M3.

This module is the SINGLE AUTHORITATIVE SOURCE of feature-to-module
membership. Every other module that needs to know which observed features
belong together must import it from here rather than restating the
membership. That is a deliberate design constraint: duplicated membership
lists drift apart silently, and then a "network-informed" model is quietly
using a different grouping from the one the documentation describes.

WHAT THE GRAPH ENCODES
----------------------
A small three-level hierarchy over OBSERVED VARIABLES ONLY:

    feature nodes  ->  module nodes

    reg_feature_1      ->  regulatory_module  -> pathway_level
    reg_feature_2      ->  regulatory_module
    pathway_feature_1  ->  pathway_module     -> pathway_level
    pathway_feature_2  ->  pathway_module

Deliberately absent:

  * LATENT STATES. `q`, `a_A`..`a_D`, `m_mag` and every other hidden variable
    are NOT nodes here. The graph describes structure a scientist would know
    from outside the dataset, not the generative process that produced it.
  * MODULES WITHOUT PROXIES. Module A (structural stability) and module D
    (weak/noisy biology) have no observed proxy in this dataset, so they are
    NOT nodes. A module with no measured member could not contribute a score.
  * ANY EDGE AMONG PROXES. The two regulatory proxies are members of one
    module, not neighbours of each other. Module membership is the only
    relation asserted, and that is the entire prior being tested.

WHY THE GRAPH IS SO SMALL
-------------------------
PROJECT_BRIEF section 6 asks for `features -> modules -> pathways` and
suggests four modules (A structural stability, B FRET-supported geometry,
C regulatory interaction, D weak/noisy biology). Only B-equivalent
(the pathway module) and C-equivalent (the regulatory module) have observed
proxies, so only those two are real nodes here. A, B/C's siblings are left out
rather than represented by empty nodes, because an empty module node would
imply knowledge that cannot be acted on.

HOW THE MEMBERSHIP IS USED
---------------------------
The graph supplies exactly one thing downstream: which observed variables
belong together. `network_knowledge.py` converts that into equal-weight
module scores. It supplies no weights, no learned parameters, and no ordering.

SCIENTIFIC STATUS
-----------------
The graph is SYNTHETIC. It is a small, interpretable structure used to
demonstrate how domain grouping can enter a model. It is not real RNA
biology, it is not derived from KEGG or Reactome, and no biological
validation is claimed.
"""

from __future__ import annotations

import networkx as nx

# --------------------------------------------------------------------------
# Feature -> module membership. Defined ONCE, here, and nowhere else.
# --------------------------------------------------------------------------
FEATURE_TO_MODULE: dict[str, str] = {
    "reg_feature_1": "regulatory_module",
    "reg_feature_2": "regulatory_module",
    "pathway_feature_1": "pathway_module",
    "pathway_feature_2": "pathway_module",
}

# The fixed, equally complex, BIOLOGICALLY INCORRECT grouping used as the
# control. It is declared here so it is versioned next to the correct
# grouping, and it was fixed BEFORE any evaluation: it crosses the correct
# module boundaries (regulatory with pathway) while keeping exactly the same
# number of modules, the same module sizes, and the same equal-weight means.
# Nothing about it was chosen using labels or performance.
INCORRECT_FEATURE_TO_MODULE: dict[str, str] = {
    "reg_feature_1": "incorrect_module_1",
    "pathway_feature_1": "incorrect_module_1",
    "reg_feature_2": "incorrect_module_2",
    "pathway_feature_2": "incorrect_module_2",
}

CORRECT_MODULES: tuple[str, ...] = ("regulatory_module", "pathway_module")
INCORRECT_MODULES: tuple[str, ...] = ("incorrect_module_1", "incorrect_module_2")

MODULE_TO_LEVEL: dict[str, str] = {
    "regulatory_module": "pathway_level",
    "pathway_module": "pathway_level",
    "incorrect_module_1": "pathway_level",
    "incorrect_module_2": "pathway_level",
}

# The four observed features that module scores replace. Every condition
# swaps exactly these, so dimensionality drops identically in the correct and
# incorrect cases.
AGGREGATED_FEATURES: tuple[str, ...] = (
    "reg_feature_1",
    "reg_feature_2",
    "pathway_feature_1",
    "pathway_feature_2",
)


def _build(mapping: dict[str, str], with_levels: bool) -> nx.DiGraph:
    graph = nx.DiGraph()
    for feature, module in mapping.items():
        graph.add_node(feature, kind="feature")
        graph.add_node(module, kind="module")
        graph.add_edge(feature, module, relation="member_of")
    if with_levels:
        for module, level in MODULE_TO_LEVEL.items():
            if module in graph:
                graph.add_node(level, kind="pathway_level")
                graph.add_edge(module, level, relation="part_of")
    return graph


def build_knowledge_graph() -> nx.DiGraph:
    """The authoritative knowledge graph: features -> modules -> pathway level.

    Only observed feature nodes, module nodes for the two modules that have
    observed members, and a single pathway-level node. Contains no latent
    states and no edges among proxies.
    """
    return _build(FEATURE_TO_MODULE, with_levels=True)


def build_incorrect_grouping_graph() -> nx.DiGraph:
    """The fixed control grouping, identical in size and shape to the real one."""
    return _build(INCORRECT_FEATURE_TO_MODULE, with_levels=True)


def module_members(mapping: dict[str, str] = FEATURE_TO_MODULE) -> dict[str, list[str]]:
    """Group feature names by module, preserving declaration order."""
    members: dict[str, list[str]] = {module: [] for module in mapping.values()}
    for feature, module in mapping.items():
        members[module].append(feature)
    return members


def module_score_names(mapping: dict[str, str] = FEATURE_TO_MODULE) -> dict[str, str]:
    """Column name to use for each module's score.

    'regulatory_module' -> 'regulatory_module_score', and so on.
    """
    return {module: f"{module}_score" for module in dict.fromkeys(mapping.values())}


def validate_groupings() -> None:
    """Fail loudly if the control stops being complexity-matched.

    The whole value of the incorrect grouping is that it differs from the
    correct one ONLY in membership. If the two ever differ in module count,
    module size, or which columns get replaced, the comparison stops testing
    what it claims to test.
    """
    correct = module_members(FEATURE_TO_MODULE)
    incorrect = module_members(INCORRECT_FEATURE_TO_MODULE)
    correct_features = {f for group in correct.values() for f in group}
    incorrect_features = {f for group in incorrect.values() for f in group}

    assert len(correct) == len(incorrect), (
        f"groupings differ in module count: {len(correct)} vs {len(incorrect)}"
    )
    assert correct_features == set(AGGREGATED_FEATURES), (
        f"correct grouping does not cover exactly the four aggregated features: "
        f"{sorted(correct_features)}"
    )
    assert incorrect_features == set(AGGREGATED_FEATURES), (
        f"incorrect grouping does not cover exactly the four aggregated features: "
        f"{sorted(incorrect_features)}"
    )
    correct_sizes = sorted(len(v) for v in correct.values())
    incorrect_sizes = sorted(len(v) for v in incorrect.values())
    assert correct_sizes == incorrect_sizes, (
        f"groupings differ in module sizes: {correct_sizes} vs {incorrect_sizes}"
    )
    # The control must actually differ from the correct grouping, otherwise it
    # is not a control at all.
    assert correct != incorrect, "incorrect grouping is identical to the correct one"


def describe() -> dict:
    """A small JSON-friendly summary, useful for logging and the notebook."""
    graph = build_knowledge_graph()
    validate_groupings()
    return {
        "n_nodes": graph.number_of_nodes(),
        "n_edges": graph.number_of_edges(),
        "nodes": sorted(graph.nodes),
        "feature_to_module": FEATURE_TO_MODULE,
        "module_members": module_members(FEATURE_TO_MODULE),
        "module_score_names": module_score_names(FEATURE_TO_MODULE),
        "incorrect_feature_to_module": INCORRECT_FEATURE_TO_MODULE,
        "incorrect_module_members": module_members(INCORRECT_FEATURE_TO_MODULE),
        "aggregated_features": list(AGGREGATED_FEATURES),
        "contains_latent_states": False,
        "synthetic": True,
    }
