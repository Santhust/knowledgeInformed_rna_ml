# AGENTS.md

## Purpose

This file defines how coding agents should work inside the
`knowledgeInformed_rna_ml` repository.

The scientific design is defined in `PROJECT_BRIEF.md`.
This file defines the implementation workflow, coding preferences, review rules,
and repository discipline.

---

## Current Repository Status

The core exploratory study is **complete** through robustness analysis,
selective prediction, and final figure generation. Existing generated
datasets (`data/generated/`), persisted result JSON files (`results/`), and
final figures (`figures/final/`) should be treated as **frozen** unless a
task explicitly requests a new experiment.

- `docs/` is the **public scientific narrative**. It is the authoritative
  public account of the project's findings; it is not a scratch file.
- Documentation or presentation work **must not** silently regenerate or
  alter established results.
- New scientific investigations should be implemented as **clearly separated
  extensions** — new result files, new figures, new site sections — not as
  retroactive modifications of completed experiments.
- When editing public-facing text, preserve the distinction between
  **measured results**, **diagnostic-only latent analyses** (see
  `results/module_latent_diagnostic.json`), **limitations**, and **future
  work**. Do not let one blur into another.
- Do not invent numbers. Every numeric claim in public-facing text must be
  traceable to a saved result file.

---

## 1. Read First

Before making any change:

1. Read `PROJECT_BRIEF.md` completely.
2. Read this `AGENTS.md` completely.
3. Inspect the current repository state before editing.
4. Preserve the scientific design unless the user explicitly asks to change it.

If an implementation choice would materially alter the scientific question,
data-generating process, model comparison, biological interpretation, or
evaluation logic, stop and ask before proceeding.

---

## 2. Core Working Principle

The project is a learning and methodological exploration.

Therefore:

- prefer transparent code over clever abstractions;
- prefer small, reviewable changes over large rewrites;
- optimize for scientific clarity and explainability;
- every important method should be understandable by the project owner;
- avoid unnecessary framework complexity.

The coding agent is an implementation assistant, not the scientific decision-maker.

---

## 3. Repository Environment

Use the repository-local Python virtual environment:

```text
.venv/
```

For all Python execution and dependency installation:

- activate `.venv`;
- do not install project packages globally;
- do not replace the environment unless explicitly instructed;
- if a new dependency is needed, explain why before adding it;
- update `requirements.txt` when dependencies change.

---

## 4. Version-Control Rules

This repository uses Git.

Before editing:

- inspect `git status`;
- do not discard existing user changes;
- do not rewrite unrelated files;
- do not reset, rebase, force-push, or amend history unless explicitly asked.

After each task:

- summarize which files changed;
- summarize what behavior changed;
- mention any assumptions or limitations;
- suggest an appropriate Git commit message.

Do **not** create a Git commit unless explicitly instructed.

Keep changes small enough to review with `git diff`.

---

## 5. Scientific Integrity

Do not:

- invent biological claims;
- invent experimental results;
- claim synthetic data are real;
- imply biological or clinical validation;
- interpret predictive relevance as biological causality;
- hard-code expected results to make the experiment succeed;
- hide negative or unexpected results.

Always preserve the distinction between:

- synthetic vs real data;
- model relevance vs biological interpretation;
- prediction confidence vs correctness;
- association vs causality.

If a result contradicts the original expectation, report it honestly.

---

## 6. Scope Discipline

Do not introduce these unless explicitly requested:

- deep learning;
- large neural-network frameworks;
- external LVQ packages before the simple prototype model is understood;
- React, backend frameworks, or complex frontend tooling;
- Docker;
- CI/CD;
- database infrastructure;
- MLOps tooling;
- unnecessary package architecture.

The first goal is a compact, interpretable scientific workflow.

---

## 7. Implementation Sequence

Follow this order unless explicitly changed:

1. synthetic data generation;
2. data sanity checks / concise EDA;
3. logistic-regression baseline;
4. simple prototype classifier;
5. relevance-weighted prototype classifier;
6. M1/M2/M3 knowledge-informed variants;
7. reject / abstention mechanism;
8. robustness experiments;
9. figures and result tables;
10. GitHub Pages site.

Do not jump ahead merely because later stages are easy to generate.

---

## 8. Coding-Agent Task Style

For each substantial task:

1. explain the proposed approach briefly;
2. identify important assumptions;
3. implement only the requested stage;
4. run the relevant code/tests;
5. inspect outputs for obvious inconsistencies;
6. summarize results and changed files;
7. suggest the next logical step.

If asked to implement only one file or stage, do not silently implement later stages.

---

## 9. Synthetic Data Rules

The data-generating process must remain:

- transparent;
- reproducible;
- scientifically interpretable;
- non-trivial;
- documented.

Use fixed random seeds.

The hidden label-generating rule should be easy to explain and should create
reasonable overlap between plausible and implausible classes.

Avoid creating a single trivial feature that perfectly predicts the label.

---

## 10. Model Rules

Start with simple models.

For prototype-based classification:

- implement a transparent version first;
- expose prototypes explicitly;
- expose distances explicitly;
- expose feature relevance / weights explicitly;
- avoid opaque abstractions that hide model behavior.

For knowledge-informed variants:

- keep M0, M1, M2, and M3 separable;
- make it possible to attribute changes in behavior to the newly added prior;
- keep comparisons fair.

---

## 11. Biological Network / Pathway Component

Use `networkx` for the initial graph representation.

The first version should use a small, interpretable synthetic hierarchy such as:

```text
features / genes
      ↓
modules
      ↓
pathways
```

If curated KEGG or Reactome information is added later:

- state exactly what is real;
- state what remains synthetic;
- document the source;
- do not imply that the toy classifier has biological validation.

The network component should support understanding of knowledge-informed ML,
not become a separate network-biology project.

---

## 12. Robustness and Rejection

The classifier must be allowed to abstain.

Do not force every sample into a class.

For robustness experiments:

- vary simulated FRET noise systematically;
- evaluate the same model conditions consistently;
- report degradation honestly;
- do not tune thresholds solely to make the preferred model look better.

---

## 13. Results and Figures

Prefer a small number of high-value outputs.

Required concepts include:

- prototype/class visualization;
- feature relevance;
- module/pathway relevance;
- robustness vs FRET noise;
- coverage vs accepted-sample accuracy.

Figures should be:

- readable;
- scientifically labeled;
- reproducible from code;
- generated from actual outputs.

Do not manually edit figures in ways that change the underlying result.

---

## 14. GitHub Pages

The GitHub Pages site lives in `docs/`.

Use static HTML/CSS unless explicitly asked otherwise.

The site should be a concise scientific summary, not a frontend-development project.

Every claim on the site must be traceable to:

- project code;
- actual generated results;
- documented assumptions.

Do not invent numbers, conclusions, or biological implications.

---

## 15. Documentation Style

Write documentation for a scientifically literate reader.

Prefer:

- short explanations;
- explicit assumptions;
- diagrams where helpful;
- plain language before equations;
- equations only where they clarify the method.

Avoid:

- marketing language;
- exaggerated claims;
- unexplained jargon;
- unnecessary verbosity.

---

## 16. Dependency Policy

Initial intended dependencies are:

- numpy
- pandas
- scikit-learn
- matplotlib
- networkx
- jupyter

Do not add dependencies casually.

Before adding a package:

1. explain what problem it solves;
2. check whether an existing dependency is sufficient;
3. prefer well-maintained, lightweight packages;
4. update `requirements.txt`.

---

## 17. Testing and Reproducibility

At minimum:

- use deterministic seeds where possible;
- keep train/test splitting reproducible;
- verify scripts run from the repository root;
- avoid hidden notebook-only state;
- save important results to `results/`;
- save generated figures to `figures/`.

As the codebase grows, add focused tests for:

- data generation;
- distance calculations;
- rejection logic;
- knowledge mapping.

---

## 18. When to Stop and Ask

Stop and ask before:

- changing the scientific question;
- changing the meaning of M0/M1/M2/M3;
- replacing the data-generating logic;
- adding real biological datasets;
- introducing a new modeling framework;
- adding deep learning;
- changing the Git history;
- deleting user files;
- making a scientifically consequential assumption not covered in
  `PROJECT_BRIEF.md`.

---

## 19. Definition of a Good Change

A good change is:

- scientifically justified;
- small enough to review;
- reproducible;
- easy to explain;
- consistent with `PROJECT_BRIEF.md`;
- covered by an appropriate sanity check or test;
- documented clearly;
- honest about limitations.

The project should remain something the owner can confidently explain from first principles.
