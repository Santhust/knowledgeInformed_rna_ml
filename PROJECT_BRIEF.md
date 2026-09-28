# PROJECT_BRIEF.md

> Sections 1–19 are the **original brief**, written before the experiments
> were run. They are preserved as written and have not been rewritten to
> match the final results.
> Sections 20–22 record how the project actually evolved, the research
> questions that emerged, and the current status.

## Project Title

**Interpretable Knowledge-Informed Biological Classification**  
*A toy study combining experimental, structural, and biological-network priors for an RNA/FRET-inspired classification problem*

## 1. Purpose

This project is a small, controlled, reproducible learning study designed to explore how different forms of biological prior knowledge can influence an interpretable prototype-based classifier.

The project is inspired by methodological themes relevant to:

- interpretable machine learning,
- prototype-based classification,
- relevance learning,
- knowledge-informed AI,
- FRET-constrained biological structure validation,
- uncertainty and rejection,
- robustness under noisy measurements,
- integration of pathway / gene-association knowledge.

The project is **not** intended to perform real RNA folding, real FRET inference, or validated RNA 3D-structure prediction.

Its purpose is to provide a transparent vertical slice of the scientific reasoning behind knowledge-informed biological ML.

---

## 2. Main Scientific Question

> **How does progressively adding experimental, structural, and biological-network prior knowledge affect the interpretability, robustness, and uncertainty handling of a prototype-based classifier in an RNA/FRET-inspired toy problem?**

The project should compare four model conditions:

- **M0 — Data only**
- **M1 — Data + experimental/FRET prior**
- **M2 — Data + experimental/FRET prior + structural prior**
- **M3 — Data + experimental/FRET prior + structural prior + biological network/pathway prior**

The design should make it possible to compare these models fairly and transparently.

---

## 3. Conceptual Background

The toy problem assumes that multiple candidate biological structures are available and must be classified as:

- **plausible**
- **implausible**

The classifier receives heterogeneous descriptors that represent several kinds of evidence.

### Experimental evidence
Examples:
- FRET distance disagreement
- FRET uncertainty

### Structural evidence
Examples:
- free-energy-like score
- stem count
- structural consistency score

### Biological-network / pathway evidence
Examples:
- regulatory features
- pathway-level features
- feature membership in predefined biological modules
- gene/pathway association structure

### Weak or noisy evidence
Examples:
- GC content
- unrelated synthetic noise features

The project should demonstrate that scientific prior knowledge can enter the model in more than one way.

---

## 4. Key Principle

The central concept is:

> **Do not ask the model to rediscover known science unnecessarily.**

Knowledge-informed AI should combine:

**data + prior biological knowledge**

while preserving the distinction between:

- predictive relevance,
- model confidence,
- biological interpretation,
- causality.

A feature that is important for classification should **not** automatically be described as biologically causal.

---

## 5. Synthetic Dataset Design

Create a synthetic dataset of approximately **500–1000 candidate structures**.

Suggested features:

| Feature | Category | Meaning |
|---|---|---|
| `fret_error` | Experimental | Difference between candidate-predicted and observed FRET-like distance |
| `fret_uncertainty` | Experimental | Simulated uncertainty of the FRET-like measurement |
| `free_energy` | Structural | Synthetic free-energy-like score |
| `stem_count` | Structural | Simplified structural descriptor |
| `structural_consistency` | Structural | Score for internal structural plausibility |
| `gc_content` | Sequence / weak | Sequence-level descriptor |
| `reg_feature_1` | Network | Regulatory or association-related feature |
| `reg_feature_2` | Network | Regulatory or association-related feature |
| `pathway_feature_1` | Network | Pathway-level feature |
| `pathway_feature_2` | Network | Pathway-level feature |
| `noise_feature_1` | Noise | Weak / irrelevant feature |
| `noise_feature_2` | Noise | Weak / irrelevant feature |

The hidden label-generating rule must be transparent and documented.

The rule should create overlap between the classes so that the problem is not trivial.

The synthetic ground truth should depend most strongly on:

- FRET agreement,
- structural consistency,
- selected biological-network/pathway features.

Some noise should be added so that:

- not every plausible sample looks perfect,
- not every implausible sample looks obviously wrong,
- rejection / uncertainty becomes meaningful.

Use a fixed random seed for reproducibility.

---

## 6. Biological Knowledge Graph

Use **NetworkX** to represent a small synthetic biological knowledge graph.

The first version should remain simple and interpretable.

Example hierarchy:

```text
gene/feature nodes
      ↓
biological modules
      ↓
pathways
```

Possible synthetic organization:

```text
Module A — structural stability
Module B — FRET-supported geometry
Module C — regulatory interaction
Module D — weak/noisy biology
```

The graph is not intended to claim real RNA biology.

Its purpose is to demonstrate how structured domain knowledge can influence analysis.

### Optional extension

If time permits, add one **small real curated pathway topology** from KEGG or Reactome while keeping the predictive dataset synthetic.

If this is done:

- clearly label which component is real,
- clearly label which data remain synthetic,
- do not imply biological validation.

---

## 7. Model Conditions

### M0 — Data-only baseline

Use a simple baseline such as:

- Logistic Regression

Potential secondary baseline:

- Random Forest

Purpose:

- establish how well the task can be solved without explicit prior knowledge.

Metrics:

- Accuracy
- AUROC
- confusion matrix

Optional:
- calibration / Brier score if easy to implement

---

### M1 — Experimental/FRET prior

Introduce experimental knowledge.

At minimum:

- give FRET-related evidence more importance than weak features,
- incorporate FRET uncertainty when deciding how strongly disagreement should matter.

A simple uncertainty-aware penalty may conceptually follow:

```text
penalty ∝ (predicted_distance - observed_distance)^2 / uncertainty^2
```

This does not need to be implemented as a complex optimization objective if a simpler transparent weighting scheme demonstrates the concept.

---

### M2 — Structural prior

Add prior structural knowledge.

Examples:

- structural consistency and free-energy-like descriptors receive coordinated treatment,
- biologically implausible combinations may receive a soft penalty,
- structural features may be grouped or jointly weighted.

Prefer **soft constraints** over brittle hard rules where uncertainty exists.

---

### M3 — Biological network/pathway prior

Add structured biological knowledge.

Use the predefined feature → module → pathway relationships.

Possible simple implementations:

1. aggregate selected gene-level features into module-level scores;
2. apply module-aware relevance weights;
3. compare feature-level vs module-level importance;
4. use the graph to regularize or structure the feature representation.

The implementation should remain transparent.

Do **not** attempt to reproduce the full mathematical framework of GMLVQ knowledge decomposition unless time and understanding clearly justify it.

The goal is to demonstrate the principle of network-informed learning.

---

## 8. Prototype-Based Classifier

Implement a simple prototype classifier before using any external LVQ library.

At minimum:

- learn or calculate one or more prototypes for the plausible class,
- learn or calculate one or more prototypes for the implausible class,
- classify a sample according to distance to the nearest relevant prototype.

Then introduce a weighted distance:

```text
d(x, w) = Σ λ_j (x_j - w_j)^2
```

where `λ_j` represents feature relevance.

The project should make it possible to inspect:

- learned or assigned prototypes,
- feature relevance weights,
- module/pathway relevance.

The implementation should be simple enough that the project owner can explain every important line.

---

## 9. Rejection / Uncertainty

The classifier should be allowed to abstain.

For each sample, compare:

- distance to the nearest plausible prototype,
- distance to the nearest implausible prototype.

If the margin between them is too small, classify as:

**uncertain / reject**

Report:

- coverage,
- accuracy among accepted samples,
- rejection rate.

The exact rejection threshold should be documented and, ideally, explored over a small range.

---

## 10. Robustness Experiment

Create a controlled robustness experiment by progressively adding noise to FRET-related measurements.

Example noise levels:

```text
0.0
0.1
0.2
0.5
1.0
```

For each model M0–M3, evaluate:

- Accuracy
- AUROC
- rejection rate
- coverage
- confidence / margin behavior

Main question:

> **Does prior knowledge make the classifier degrade more gracefully under noisy experimental evidence?**

The expected result must **not** be hard-coded.

If prior knowledge does not help, that result should be reported honestly.

---

## 11. Interpretability Outputs

Generate a small number of high-value figures.

Required:

1. **Prototype / class visualization**
   - show candidate samples
   - show plausible / implausible prototypes
   - use a 2D projection only for visualization

2. **Feature relevance plot**
   - show which features influence classification most

3. **Module / pathway relevance plot**
   - show biological-group-level contributions

4. **Robustness curve**
   - x-axis: FRET noise
   - y-axis: performance metric(s)

5. **Coverage vs accepted-sample accuracy**
   - demonstrate the effect of the reject option

Avoid unnecessary decorative plots.

---

## 12. Repository Structure

```text
knowledgeInformed_rna_ml/
├── PROJECT_BRIEF.md
├── AGENTS.md
├── .gitignore
├── requirements.txt
├── .venv/                  # local only; never commit
├── README.md
├── src/
│   ├── generate_data.py
│   ├── knowledge_graph.py
│   ├── baseline.py
│   ├── prototype_model.py
│   ├── knowledge_models.py
│   ├── rejection.py
│   ├── robustness.py
│   └── make_figures.py
├── notebooks/
│   └── walkthrough.ipynb
├── data/
├── results/
├── figures/
└── docs/
    ├── index.html
    ├── style.css
    └── assets/
```

Do not create unnecessary framework complexity.

---

## 13. Development Environment and Version Control

This project uses Git from the beginning.

Repository conventions:

- main branch: `main`;
- use a repository-local Python virtual environment at `.venv/`;
- `.venv/` must never be committed;
- install project dependencies inside `.venv`;
- initial dependencies are defined in `requirements.txt`;
- keep changes small and reviewable with `git diff`;
- coding agents should summarize changed files after each task;
- coding agents should suggest an appropriate commit message;
- coding agents must not create commits unless explicitly instructed;
- do not rewrite unrelated files or discard existing user changes.

The initial intended dependencies are:

- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `networkx`
- `jupyter`

If a new dependency is proposed, explain why it is needed before adding it and update `requirements.txt`.

Coding agents must read both:

1. `PROJECT_BRIEF.md` — scientific purpose, scope, experimental design, model conditions;
2. `AGENTS.md` — workflow, coding style, Git discipline, reproducibility, and agent behavior.

The scientific brief takes precedence for project goals. `AGENTS.md` governs implementation behavior.

---

## 14. GitHub Pages Site

The `docs/` site should function as a concise scientific summary.

Suggested sections:

1. **Scientific Question**
2. **Why Knowledge-Informed AI?**
3. **Toy Experimental Design**
4. **Three Forms of Prior Knowledge**
   - experimental
   - structural
   - biological-network/pathway
5. **Methods**
6. **Results**
7. **Biological-Network View**
8. **Robustness & Rejection**
9. **Limitations**
10. **Broader Relevance and Possible Extensions**

Use static HTML/CSS and generated plots.

Do not build React, a backend, or a complex web application.

The site must only state results that were actually produced by the code.

---

## 15. Scope and Limitations

The project must clearly state:

- the dataset is synthetic,
- FRET measurements are simulated,
- the project does not predict real RNA 3D structures,
- the project does not model real RNA folding,
- the pathway/network component is synthetic unless explicitly replaced by a curated network,
- the work is a methodological learning study,
- model relevance does not imply biological causality,
- no biological or clinical validation is claimed.

---

## 16. Coding-Agent Instructions

Before making changes:

1. Read this `PROJECT_BRIEF.md` completely.
2. Read `AGENTS.md` completely.
3. Inspect the current Git state before editing.
4. Do not change the scientific question or model conditions without explicit instruction.
3. Prefer transparent code over clever abstractions.
4. Explain important assumptions before implementing them.
5. Implement one stage at a time.
6. Do not invent biological claims.
7. Do not invent results.
8. Do not introduce deep learning unless explicitly requested.
9. Do not add external LVQ packages until the simple prototype implementation is complete and understood.
10. Keep all experiments reproducible with fixed seeds.

---

## 17. Recommended Coding-Agent Task Sequence

### Task 1 — Synthetic data only
Implement `src/generate_data.py`.

Before coding:
- explain the hidden label-generating rule,
- explain why each feature is included,
- explain how class overlap and noise are introduced.

Do not implement models yet.

### Task 2 — Inspect and validate data
Create concise EDA.

Verify:
- feature distributions,
- class balance,
- signal/noise behavior,
- no accidental trivial predictor.

### Task 3 — Baseline
Implement logistic regression baseline.

Save metrics to `results/`.

### Task 4 — Simple prototype model
Implement prototype classification using transparent Euclidean distance.

Do not add knowledge weighting yet.

### Task 5 — Relevance-weighted prototype model
Add feature weighting / relevance.

Expose weights clearly.

### Task 6 — Knowledge models
Implement M1, M2, M3 progressively.

Keep each modification explicit and independently testable.

### Task 7 — Rejection
Add uncertainty / abstention based on prototype-distance margin.

### Task 8 — Robustness
Add controlled FRET noise and evaluate M0–M3.

### Task 9 — Figures
Generate only the figures specified in this brief.

### Task 10 — GitHub Pages
Create the static `docs/` site using only actual project outputs.

---

## 18. Acceptance Criteria

The project is successful if:

- every model condition M0–M3 can be explained in plain language,
- the effect of each type of prior knowledge is measurable,
- the prototype classifier is inspectable,
- relevance weights can be visualized,
- pathway/module-level relevance is visible,
- rejection behavior can be demonstrated,
- robustness under FRET noise is quantified,
- results are reproducible,
- limitations are explicit,
- the GitHub Pages site tells the same scientific story as the code,
- the project owner can explain every important methodological choice from first principles.

---

## 19. Scientific Narrative

The intended high-level story is:

> This project explores how different forms of scientific prior knowledge can be integrated into an interpretable biological classifier. Starting from a data-only baseline, it progressively adds simulated experimental FRET constraints, structural knowledge, and biological network/pathway information. Prototype-based decisions are used because they remain inspectable, and uncertain cases can be rejected rather than forced into a class. The project then tests whether prior knowledge improves robustness when experimental evidence becomes noisy.

This project is a **learning and methodological exploration**, not a solved RNA-structure model.

---

## 20. Original Objective

> *Recorded after the fact, to distinguish the starting intent from what the
> experiments actually established. Sections 1–19 above remain the original
> brief.*

The project began with the objective:

> **Explore interpretable, knowledge-informed machine learning in a controlled
> RNA/FRET-inspired synthetic classification problem.**

The starting expectation, stated in section 10, was that prior knowledge
*should* make the classifier degrade more gracefully under noisy experimental
evidence, and the expectation was explicitly not to be hard-coded. It was also
anticipated that domain knowledge would generally be useful. Both assumptions
turned out to be more optimistic than the data supported, which is itself part
of what the study now reports.

---

## 21. Current Project Orientation

As the experiments developed, the study narrowed and broadened at the same
time. The specific question of section 2 — whether adding prior knowledge
improves a prototype classifier in this toy problem — turned out to be
answerable but not very informative on its own, because the data-only
baselines were all close together and the prior effects were small.

The study therefore became focused on a broader **methodological** question:

> **How do different forms of prior knowledge affect predictive performance,
> biological representation, selective prediction, and robustness?**

The emphasis moved from *does this specific prior help this specific model* to
*what does it actually take for injected knowledge to become predictive
utility* — and, as a result, several negative results became the scientifically
interesting outputs rather than disappointments.

### Research questions

1. Can a simple prototype classifier remain competitive with conventional
   baselines?
2. Do experimental, structural, and module-level priors improve prediction?
3. Can biological structure be represented more faithfully without improving
   discrimination?
4. Can prototype geometry identify difficult predictions and support
   abstention?
5. How do frozen models respond to degraded experimental evidence?
6. What important questions remain open — especially sample size and sample
   efficiency, nonlinear problem complexity, and imperfect priors?

### Where the answers live

The full answers, with figures and exact values, are in `docs/index.html`.
The repository intentionally does **not** carry a second, parallel results
document; the hierarchy is:

```text
README.md           concise repository entry point
PROJECT_BRIEF.md    rationale and evolved research questions (this file)
docs/               full public scientific narrative
notebooks/, results/ evidence and reproducibility
AGENTS.md           repository-development rules
```

---

## 22. Current Status

- The **core synthetic study is complete** through robustness analysis,
  selective prediction, and final figure generation.
- **Final figures** are in `figures/final/`, with authoritative descriptions in
  `figures/final/CAPTIONS.md`. They are generated deterministically from the
  persisted result JSON files by `src/make_final_figures.py`, which fits no
  model and recomputes no metric.
- The **public scientific narrative** is in `docs/`, built as a static
  GitHub Pages site.
- **Future experiments proposed in the website and in section 21 above are not
  implemented.** In particular, the learning-curve / sample-efficiency study
  identified as the priority next experiment is a proposal only.
- Persisted results, generated data, and final figures should be treated as
  **frozen** unless a task explicitly requests a new experiment.

The project remains a **synthetic methodological study**. Nothing in it
constitutes real RNA-folding analysis, real FRET inference, or biological
validation, and predictive relevance is never causal evidence.
