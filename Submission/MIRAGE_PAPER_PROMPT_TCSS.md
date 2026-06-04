# MIRAGE Paper Generation Prompt — Target Venue: IEEE Transactions on Computational Social Systems

This file is a complete instruction set for GitHub Copilot (running in VS Code with access to project codes, results, internet, and research files) to generate a submission-ready paper for **IEEE Transactions on Computational Social Systems (TCSS)**. Read this entire file before producing any output. Follow every instruction. The acceptance probability of the resulting paper depends on how closely the output adheres to these instructions.

---

## 0. Project Context

You are writing a single-author paper for Koushik Debnath, a final-year PhD candidate at IIIT Kalyani working under Dr. Imon Mukherjee (IIIT Kalyani) and Dr. Debarshi Kumar Sanyal (IACS). The paper presents **MIRAGE — Mechanism-Indexed Reliability Audit for Group-bias Evaluation**, a discriminative-validity audit framework for LLM bias benchmarks that operationalizes the Epistematics methodology from Kalaitzidis (2026).

The full project specification is in `MIRAGE_MASTER_PROMPT.md` at the repository root. The codebase, experimental results, parquet files, leaderboard data, and figures are in their respective folders. You have access to all of them. Use them as the source of every empirical claim — do not fabricate numbers.

**Target venue: IEEE TCSS.** Every choice in this paper — title framing, abstract emphasis, related-work coverage, results presentation, discussion focus — must be made to maximize acceptance probability at this specific venue.

---

## 1. Understanding IEEE TCSS — Why Framing Matters

IEEE TCSS is not a generic AI/ML journal. Its identity is **computational systems that address questions about social systems and behavior**. Papers that read as pure NLP methodology get desk-rejected as out of scope. Papers that show how a computational method illuminates, audits, or improves a sociotechnical system land well.

TCSS reviewers reward:

- **Sociotechnical framing** — the bias problem is positioned as a harm to specific social groups in specific deployment contexts (hiring, healthcare, content moderation, education), not as a model-internal property.
- **Cross-disciplinary engagement** — the related work must cite both ML/NLP bias literature *and* social-science measurement theory (Cronbach 1955, Messick 1995, Campbell & Fiske 1959).
- **Practical artifacts for practitioners** — the validity leaderboard and audit toolkit are deliverables for benchmark designers and AI auditors, not just researchers.
- **Statistical rigor with explicit uncertainty** — bootstrap CIs, McNemar's test, power analysis, effect sizes are expected, not optional.
- **Ethical and policy implications** — connect findings to NIST AI RMF, the EU AI Act, or sectoral regulation. TCSS published several such papers in 2024–2026.
- **Reproducibility statements** — full artifact release is the norm here, not the exception.

TCSS reviewers penalize:

- Pure methodology with no social-systems framing.
- Tiny incremental contributions over a single prior NLP paper.
- Engineering work without theoretical grounding.
- Cherry-picked results without statistical testing.
- Vague claims of "ethical concern" without specific harm pathways.

**Frame the paper as: a measurement-validity instrument for auditing the sociotechnical reliability of bias benchmarks used to certify LLMs before deployment in social systems.** That single sentence is the spine of the paper. Every paragraph either supports it or is cut.

---

## 2. Acceptance Maximization Strategy

The paper must execute these four moves to hit the highest probability bucket:

### Move 1 — Positioning above "yet another bias paper"

TCSS sees dozens of bias-in-LLM submissions per year. Most are rejected as incremental. To rise above the pile, the opening 500 words must do all of these:

1. Name a concrete social harm caused by *biased benchmarks* (not biased models): regulatory false-confidence, deployment in social services, audit failure cascades.
2. State that the problem is not measuring bias more — it is measuring it *validly*. Cite Bean et al. (NeurIPS 2025) as evidence that the field already noticed the problem at scale.
3. Identify the specific gap Bean et al. left open: their work is a checklist; ours is an instrument that operationalizes Kalaitzidis's discriminative validity with mechanism-level causal intervention.
4. Preview the validity leaderboard as a concrete deliverable.

### Move 2 — Theoretical scaffolding from measurement science

Section 2 (Related Work) must have a subsection explicitly titled **"Measurement Validity and Sociotechnical Auditing"** that cites:

- Cronbach & Meehl (1955), "Construct validity in psychological tests" — the foundational definition of discriminative and predictive validity.
- Messick (1995), "Validity of psychological assessment" — extension to consequential validity, which connects directly to deployment harms.
- Campbell & Fiske (1959), "Convergent and discriminant validation by the multitrait-multimethod matrix" — the original framework MIRAGE operationalizes.
- Jacobs & Wallach (2021), "Measurement and fairness" — the FAccT paper that brought this framework into ML fairness.
- Bean et al. (2025), Kalaitzidis (2026), Kearns (2026) — the recent LLM-specific construct-validity work.

This subsection signals to the reviewer immediately that the author understands measurement theory, not just ML benchmarks. Reviewers from social-science backgrounds (which TCSS recruits) will accept the paper into review at this signal alone.

### Move 3 — Causal mechanism as the methodological wedge

The CDVA section must lean hard into causal language. Cite Pearl (2009) for do-calculus, Meng et al. (2022, ROME) for activation patching, and explicitly write: *"We operationalize Kalaitzidis's discriminative validity criterion as an interventional do-test: a benchmark item exhibits valid construct measurement only when output probability is invariant to interventional substitution of the protected-attribute representation."*

This sentence is the methodological core. It elevates MIRAGE from "another behavioral audit" to "a causal instrument." TCSS reviewers from the computational social systems wing respond to causal framing because their field's central question is causal (does intervention X cause social outcome Y).

### Move 4 — Practical leaderboard + predictive validity

Lead the results section with the **validity leaderboard** (the 4×5 matrix of benchmarks × failure modes) before any per-model results. Practitioners can cite the leaderboard immediately; per-model results are research artifacts.

Then present **predictive validity** (held-out WinoBias generalization) as the second result. This demonstrates that MIRAGE produces transferable claims, not just per-benchmark observations.

Save the per-model behavioral and CDVA breakdowns for after these two headline results.

---

## 3. Paper Structure and Length Budget

Target: 12 pages, two-column IEEEtran format, including references. This is the TCSS regular paper budget. Do not exceed.

| Section | Length (single-column words) | Two-col pages |
|---|---|---|
| Title + Abstract + Keywords | 250 | 0.3 |
| 1. Introduction | 1500 | 1.5 |
| 2. Related Work | 1700 | 1.7 |
| 3. The MIRAGE Framework | 2000 | 2.0 |
| 4. Causal Discriminative Validity Audit (CDVA) | 1500 | 1.5 |
| 5. Experimental Setup | 1000 | 1.0 |
| 6. Results | 2200 | 2.2 |
| 7. Discussion: Sociotechnical Implications | 800 | 0.8 |
| 8. Limitations and Ethical Considerations | 400 | 0.4 |
| 9. Conclusion | 300 | 0.3 |
| References (~70 entries) | — | 1.3 |
| **Total** | **~11700** | **~13 pages including refs** |

If the draft exceeds 12 pages, cut from Section 3 first (tighten the formal definitions), then from Section 6 (collapse per-model breakdowns into appendix), then from Section 2 (consolidate citation paragraphs).

### 3.1 Section-by-section content brief

**Title.** Single line, must contain "Causal," "Validity," and "Bias" or "Audit." Avoid acronym-only titles. Two acceptable patterns:
- "MIRAGE: Causal Discriminative Validity Audit for Bias Benchmarks in Large Language Models"
- "Causal Mechanism-Coupled Validity Audit of LLM Bias Benchmarks for Sociotechnical Deployment"

The first is shorter and acronym-led; the second is longer but emphasizes the TCSS angle. Test both in the abstract preview and pick the one that flows better with the abstract's first sentence.

**Abstract.** 200 words. Mandatory structure:
- 2 sentences: the social problem (bias benchmarks used to certify LLMs for deployment in social systems may not measure what they claim).
- 1 sentence: the gap (existing audits are behavioral and correlational).
- 2 sentences: the contribution (MIRAGE pentad probe design + CDVA causal patching + validity leaderboard + predictive validity).
- 3 sentences: the empirical findings, with one specific quantitative result (e.g., "models passing BBQ at X% pass MIRAGE-Full at Y%").
- 1 sentence: the artifact release and implications for auditors/regulators.

**Keywords (Index Terms).** Exactly 6, all lowercase, comma-separated. Required: "large language models," "bias evaluation," "construct validity," "causal intervention," "sociotechnical auditing," "computational social systems." The last term is non-negotiable — it signals venue fit in the metadata index.

**Section 1 — Introduction.** Four paragraphs:
- Para 1: The deployment context. LLMs in hiring, healthcare, content moderation, education. Cite 3–4 recent deployment papers (Goh et al. 2024 medical, Bigman et al. 2023 hiring, Solaiman et al. 2024 governance).
- Para 2: The certification gap. Regulators and auditors rely on bias benchmarks (BBQ, CrowS-Pairs, StereoSet, WinoBias) to certify pre-deployment fairness. Cite the EU AI Act high-risk system requirements and NIST AI RMF measurement guidance.
- Para 3: The validity question. Recent work (Bean et al. 2025, Kalaitzidis 2026, Kearns 2026) shows benchmarks may not measure what they claim. None operationalize a causal, mechanism-coupled test.
- Para 4: Contributions. State exactly four: (a) MIRAGE pentad design with deterministic and adversarial probes, (b) CDVA causal patching as a discriminative validity test, (c) validity leaderboard for 4 widely-used bias benchmarks, (d) predictive validity demonstrated on held-out WinoBias. End with a one-sentence pointer to the artifact.

**Section 2 — Related Work.** Three subsections in this order:
- 2.1 Measurement Validity and Sociotechnical Auditing (the move-2 subsection).
- 2.2 Bias Benchmarks for LLMs (covers BBQ, CrowS, StereoSet, WinoBias, BharatBBQ, HolisticBias).
- 2.3 Mechanistic Interpretability and Causal Intervention (ROME, attention patching, DIFFHEADS, IPE, MedEqualQA).

End the section with a paragraph titled "Position of this Work" that has a table or paragraph comparing MIRAGE against Bean et al., Wang et al., Kalaitzidis, Kearns, LLMCert-B, and DIFFHEADS on four axes: mechanism-coupled, causal, instrument-vs-checklist, predictive-validity-demonstrated.

**Section 3 — The MIRAGE Framework.** This is where the pentad design lives. Subsections:
- 3.1 Theoretical Foundation (link to Cronbach-Meehl discriminative validity; introduce the four operational criteria: convergent, discriminant, internal coherence, contextual stability)
- 3.2 The Pentad Probe Design (formal definitions of slots a, b, c, d, e; one worked example threaded through using the BBQ Religion category seed about the Sikh/Christian/bombing example)
- 3.3 Pentad Generation Pipeline (template-deterministic for (a)–(c), DeepSeek-assisted with author verification for (d)–(e); cite Shaikh et al. 2023 and DIFFHEADS for CoT-attack provenance)
- 3.4 Behavioral Scoring (MIRAGE-B definition with formal AND criterion)

**Section 4 — Causal Discriminative Validity Audit (CDVA).** The methodological wedge. Subsections:
- 4.1 Interventional Discriminative Validity (the Pearl/Meng framing; formal definition: a benchmark item exhibits valid measurement iff output is invariant under do(demographic_token := alternative))
- 4.2 Operationalization via Activation Patching (the patching procedure, with TransformerLens and nnsight implementations). **Implementation note for the writer:** `HookedTransformer.from_pretrained` is called with `fold_ln=False`, `center_writing_weights=False`, `center_unembed=False`. These flags are not defaults — they must be stated and justified. The justification is: folding LayerNorm into preceding weight matrices or centering the unembedding projection changes the absolute scale of logits, making `delta_logit` values numerically incomparable across patched and unpatched runs. The paper must include one sentence in §4.2 to pre-empt reviewer 2 asking "why not use default TL settings?"
- 4.3 Frequency-Normalized CDVA Score (the per-token unigram correction)
- 4.4 Threshold Calibration (τ on a held-out dev split of 50 seeds, with documented selection procedure)
- 4.5 MIRAGE-Full Composite (CDVA + behavioral)

**Section 5 — Experimental Setup.** Subsections:
- 5.1 Source Benchmarks and Seed Selection (stratified sampling, 270+200+200+200=870 seeds)
- 5.2 Models Evaluated (4 OSM + 4 API; one table with HF IDs, parameter counts, instruction-tuning status, mechanism-library used). **CDVA library column is mandatory in Table:** Llama-3.1-8B and Gemma-2-2B use TransformerLens; Qwen-2.5-7B and Phi-4-mini use nnsight. This split is because TransformerLens does not yet support Qwen2/Phi3 architectures cleanly. State this in a table footnote or a one-sentence aside.
- 5.3 Generation Settings (constrained JSON decoding, temperature, sampling, retry/fallback policy). **Note for the writer:** observed JSON parse failure rates from the production run are: Qwen-2.5-7B-Instruct 1.7% (174/10132 behavioral rows), Gemma-2-2B-IT 0.01% (1/10132). These rows carry `success_flag=False` and are excluded from MIRAGE-B scoring automatically. Report these rates in §5.3 as a transparency note — do not hide them. They reflect natural variation in instruct-model adherence to constrained formats, not a pipeline error.
- 5.4 Statistical Methodology (bootstrap CIs at 5000 resamples, McNemar's test for paired binary, Holm-Bonferroni for 32 confirmatory tests, BH-FDR for exploratory)

**Section 6 — Results.** This section is dense. Lead with the headline result, then drill down. Subsections:
- 6.1 Validity Leaderboard (the 4×5 matrix; benchmark composite scores; first figure of the section is the leaderboard heatmap)
- 6.2 Predictive Validity on Held-Out WinoBias (the killer #3 result; ROC curves figure; comparison table with known WinoBias weaknesses from Zhao et al. 2018, Stanovsky et al. 2019, Webster et al. 2020)
- 6.3 Native-vs-MIRAGE Pass-Rate Gap per Model (bar chart figure showing the headline gap)
- 6.4 CDVA Score Distribution per OSM Model (violin plot or CDF figure)
- 6.5 Failure-Mode Empirical Distribution (which failure modes dominate per benchmark)
- 6.6 Robustness Checks (frequency-normalized CDVA, sample-size sensitivity, τ-threshold sensitivity)

**Section 7 — Discussion: Sociotechnical Implications.** Five paragraphs:
- Para 1: What the leaderboard means for AI auditors and compliance officers. Specifically address the EU AI Act Article 10 requirements on data governance and bias measurement.
- Para 2: What the predictive-validity result means for benchmark designers. The MIRAGE classifier can flag construct-validity weaknesses in new benchmarks before they propagate.
- Para 3: What the CoT-attack robustness gap means for deployment. Adversarial users can defeat surface-suppressed models.
- Para 4: The behavioral-vs-causal gap as a regulatory blind spot. Models can pass behavioral audits while remaining causally biased; current regulatory frameworks do not require causal verification.
- Para 5: A specific policy recommendation. One concrete sentence: "We recommend that AI auditing standards (e.g., NIST AI RMF, ISO 42001) explicitly require mechanism-coupled validity tests for high-stakes deployments."

**Section 8 — Limitations and Ethical Considerations.** Four paragraphs:
- Limitations 1: English-only evaluation. Multilingual extension is published follow-on work.
- Limitations 2: Mid-tier OSM (3.8B–9B); frontier models accessed only via API for behavioral audit.
- Limitations 3: Pentad slot (d) and (e) generated with author-only verification; no inter-annotator agreement.
- Limitations 4 (CDVA position detection): CDVA delta_logit is valid only when the demographic swap token is found in the tokenised prompt (`position_fallback_used=False`). The primary analysis excludes fallback-position pairs (where `pos_a = pos_b = 1`, a non-demographic token) because patching the wrong position produces a trivially-zero delta. After the June 4 normalisation fix (underscore→space + char-level search), the fallback rate dropped from ~53% to < 10%. All reported CDVA statistics are computed on `position_fallback_used=False` rows only; the fallback rate per model is reported in a table footnote in §6.4. State the exact counts (total pairs and position-detected pairs) per model.
- Ethical considerations: Dual-use risk (audit techniques can be inverted to construct adversarial CoT attacks), data sensitivity in source benchmarks, accessibility of the audit toolkit for under-resourced researchers and institutions.

**Section 9 — Conclusion.** Three paragraphs, no new content. Restate contributions; preview multilingual v2; final sentence on the social-systems implication.

### 3.2 What goes in supplementary, not main paper

Move to `supplementary.tex`:
- Full equivalence-set lists for all bias categories.
- Per-category breakdowns within each benchmark.
- Hyperparameter ablations.
- Pilot study confirming CoT-attack bites a weak baseline.
- Frequency-normalized vs. raw CDVA score comparison plots.
- Full retry/fallback statistics per API.
- All prompts used for DeepSeek generation.
- Manifest of source-benchmark seed IDs.

The supplementary should be a separate PDF; TCSS allows supplementary material at submission.

---

## 4. Writing Style — Calibrated to Supervisors

This paper must satisfy two reviewer populations: TCSS external reviewers and the user's supervisors. Both reject AI-flavored prose. The user has an `agentrev` skill calibrated specifically to Dr. Sanyal (weight 0.7) and Dr. Mukherjee (weight 0.3). Follow these style rules verbatim.

### 4.1 Hard style rules

- **No em-dashes ("—") anywhere in the body text.** Use commas, colons, or parentheses. Em-dashes are an AI-tell that Dr. Sanyal flags immediately.
- **No three-item lists that read synthetically.** Either use a real numbered list with full justification per item, or reduce to two items.
- **Active voice as the default.** "We define X" not "X is defined." "The model fails" not "the model is found to fail." Passive only for truly impersonal observations.
- **Verbs to avoid entirely:** delve, leverage, harness, foster, unlock, navigate (when not literal), unleash, embark, illuminate. These are AI-tells.
- **Phrases to avoid entirely:** "It is worth noting," "Furthermore," at paragraph start, "In recent years," "With the advent of," "In this paper, we propose."
- **Hedging:** specific, calibrated, and minimal. "May" is acceptable when uncertainty is real; "potentially possibly" is unacceptable. "Suggests" is acceptable; "tends to suggest" is unacceptable.
- **Adjectives:** earn every "novel," "comprehensive," "robust." If you cannot back the adjective in the same paragraph with a specific contrast or evidence, drop it.
- **No emoji or special characters in body text.** ASCII only.

### 4.2 Paragraph structure

Each paragraph follows: **claim → evidence → implication.** No paragraph longer than 6 sentences. No paragraph shorter than 3 sentences except deliberate transitional ones. Each paragraph's first sentence is the strongest standalone claim.

### 4.3 Equation and notation rules

- Define every symbol at first use.
- Use \mathbb, \mathcal, \mathbf appropriately and consistently.
- Equations numbered only if referenced later.
- Avoid notation collisions across sections.
- Pearl-style do-calculus notation should appear at least once: $P(Y \mid do(X = x))$.

### 4.4 Citations

- Use \cite{} consistently; never inline a citation as "Bean et al. (2025) showed" without also having the \cite key resolved.
- All ~70 references in the bib file must be cited in the body.
- Every empirical claim about a prior work has a specific citation, not a vague "prior work."
- Prefer venue names in the bib file (NeurIPS, ACL, TACL, IEEE TCSS) over publisher names.

### 4.5 The "Sanyal test"

Before finalizing any paragraph, ask: would a senior systems professor read this sentence and think "this was written by an LLM"? If yes, rewrite until no. The user's supervisor will reject the paper at first sight if it reads as AI-flavored. The agentrev skill in the user's environment encodes the full test if you need to invoke it.

---

## 5. Output Structure

Produce these files in the repository under `paper/`:

```
paper/
├── main.tex                 # The paper, IEEEtran two-column
├── references.bib           # All ~70 references
├── IEEEtran.cls             # IEEE template (download if not present)
├── IEEEtran.bst             # IEEE bib style
├── supplementary.tex        # Separate supplementary doc
├── cover_letter.txt         # For the editor submission portal
├── abstract.txt             # Plain-text abstract for the portal
├── README.md                # What's where, how to compile
└── images/
    ├── fig1_pentad_design.png
    ├── fig2_cdva_procedure.png
    ├── fig3_leaderboard_heatmap.png
    ├── fig4_predictive_validity_roc.png
    ├── fig5_native_vs_mirage_bars.png
    ├── fig6_cdva_distribution.png
    └── fig7_failure_mode_distribution.png
```

### 5.1 LaTeX setup specifics

- `\documentclass[10pt,journal,compsoc]{IEEEtran}` — `compsoc` mode is required for TCSS.
- `\usepackage{cite}` for IEEE-style numeric citations.
- `\usepackage{graphicx}` for figures; figure paths `{images/figX_name.png}`.
- `\usepackage{algorithm,algorithmicx,algpseudocode}` for the CDVA algorithm box.
- `\usepackage{booktabs}` for clean tables (no vertical rules).
- `\usepackage{amsmath,amssymb}` for math.
- `\usepackage{hyperref}` last (must be last), with options `[hidelinks]` to avoid colored boxes in TCSS print version.
- Two-column figures use `figure*` environment; single-column use `figure`.
- All tables in two-column wrap with `table*` only if necessary; default to single-column.

### 5.2 Figure requirements

Every figure must:
- Be 300 DPI minimum.
- Use only colorblind-safe palettes (viridis, cividis, or ColorBrewer's "Set2"). No red-green contrasts.
- Have a self-contained caption that explains the figure without requiring body text.
- Use vector-equivalent rendering (export from matplotlib with `bbox_inches='tight'`, `dpi=300`).
- Be saved as PNG (not JPEG; PNG preserves text rendering for IEEE print).
- Font in figure must match LaTeX body font size at final placement (use matplotlib `rcParams.update({'font.size': 9})`).

Figures to generate (with source from existing results parquet):

1. **fig1_pentad_design.png** — Schematic diagram with one BBQ Religion seed expanded into all 5 slots (a, b, c, d, e). Use boxes-and-arrows style, not screenshots.
2. **fig2_cdva_procedure.png** — CDVA computation flowchart: forward pass A (cache), forward pass B (patched), delta computation, score.
3. **fig3_leaderboard_heatmap.png** — 4 benchmarks × 5 failure modes, color-coded composite scores. Annotated cells with values.
4. **fig4_predictive_validity_roc.png** — 5 ROC curves (one per failure mode) for the held-out WinoBias predictions, AUCs in legend.
5. **fig5_native_vs_mirage_bars.png** — Grouped bar chart, 8 models × 2 (native pass rate, MIRAGE-Full pass rate). Error bars from bootstrap CIs.
6. **fig6_cdva_distribution.png** — Violin plot of CDVA scores per OSM model. Horizontal line at τ.
7. **fig7_failure_mode_distribution.png** — Stacked bars showing which of the 5 failure modes dominate per source benchmark.

### 5.3 Cover letter content

`cover_letter.txt` should be ~250 words addressed to the Editor-in-Chief of IEEE TCSS. Structure:
- Opening: brief author affiliation and the paper title.
- One paragraph on the problem and contribution, in the language of computational social systems (deployment certification, sociotechnical auditing, regulatory implications).
- One paragraph stating: no part of this work is under review or published elsewhere; the paper is single-authored; ethical considerations are addressed in Section 8.
- Closing: list 3–5 suggested reviewers with affiliations (do not invent — leave a TODO placeholder for the author to fill in).

---

## 6. Source Material for Empirical Claims

Every quantitative claim in the paper must be sourced from a file in the project. Do not invent numbers. The following files contain the data:

- `results/leaderboard.parquet` — for Section 6.1 leaderboard values.
- `results/predictive_validity_results.parquet` — for Section 6.2 ROC AUCs and prediction accuracies.
- `results/scored_results.parquet` — for Section 6.3 per-model native vs MIRAGE pass rates with bootstrap CIs.
- `results/cdva_results.parquet` — for Section 6.4 CDVA distributions.
- `results/failure_mode_distribution.parquet` — for Section 6.5.
- `results/robustness_checks.parquet` — for Section 6.6.
- `Dataset/seeds/pentad_dataset.parquet` — for example items in Section 3.

If any of these files are absent at write time, mark the corresponding paragraph with `\todo{populate with values from <file>}` and continue. Do not block the entire paper for one missing file.

Statistics text must always carry: point estimate, 95% bootstrap CI in square brackets, n, and (when relevant) effect size. Format example: "Llama-3.1-8B passed BBQ at 78.4% [76.1, 80.6] (n = 270), Cohen's h vs MIRAGE-Full = 0.84."

### 6.1 Known data quality facts from the production run (June 4, 2026)

These facts are established from the production GPU run. **Populate every number from the actual parquet files before writing** — do not use stale figures from earlier runs.

#### 6.1.1 Behavioral evaluation

| Stat | Value | Where to use |
|---|---|---|
| Behavioral rows (OSM, total) | 40,528 (4 × 10,132) | §5.2 table footnote |
| Llama-3.1-8B behavioral parse failures | 0 / 10,132 (0.00%) | §5.3 |
| Qwen-2.5-7B behavioral parse failures | 122 / 10,132 (1.20%) | §5.3 |
| Gemma-2-2B behavioral parse failures | 1 / 10,132 (0.01%) | §5.3 |
| Phi-4-mini behavioral parse failures | populate from `behavioral_results.parquet` | §5.3 |
| A100 GPU | 40 GB VRAM, sequential model loading | §5.2 |
| TransformerLens models | Llama-3.1-8B-Instruct, Gemma-2-2B-IT | §4.2, Table 5.2 |
| nnsight models | Qwen-2.5-7B-Instruct, Phi-4-mini-instruct | §4.2, Table 5.2 |

Note: earlier internal figures showed 174 Qwen failures; the current production run shows 122. Use the value from the final `behavioral_results.parquet` file.

#### 6.1.2 CDVA results (post position-detection fix — June 4 rerun)

| Stat | Value | Where to use |
|---|---|---|
| CDVA pairs total (4 OSM × 596 seeds × 10 pairs) | 23,840 | §4.2 |
| CDVA rows with `position_fallback_used=False` | populate from `cdva_results.parquet` | §6.4 primary analysis |
| Llama CDVA: position-detected pairs (fallback=False) | populate from `cdva_results.parquet` | §6.4 |
| Qwen CDVA: position-detected pairs (fallback=False) | populate from `cdva_results.parquet` | §6.4 |
| Gemma CDVA: position-detected pairs (fallback=False) | populate from `cdva_results.parquet` | §6.4 |
| Phi CDVA: position-detected pairs (fallback=False) | populate from `cdva_results.parquet` | §6.4 |
| Expected fallback rate after fix | < 10% (down from 53% before Jun 4 fix) | §4.2 footnote |

**Critical: use only `position_fallback_used=False` rows for all CDVA analysis.** When the fallback fires, `pos_a = pos_b = 1` (second token, i.e., the BOS prefix token), which is wrong; patching a non-demographic position produces delta_logit = 0 for 91.5% of those pairs, inflating the zero mass and diluting the bias signal. The primary analysis filter must be `success_flag=True AND position_fallback_used=False`. Report both counts — total successful pairs and position-detected pairs — in §6.4.

#### 6.1.3 CDVA position detection — technical note for §4.2

The `_get_token_position` function locates the demographic swap token in each tokenised prompt. Swap tokens are stored with underscores in the pentad dataset (e.g., `a_girl`, `middle_aged`, `a_trailer_park`) because underscores are used as multi-word delimiters in the seed vocabulary. Before the June 4 fix, the function searched for the literal string including underscores; since tokenizers split words on spaces, not underscores, the token was never found for ~53% of pairs — all multi-word swap tokens. The fix applies a three-pass strategy:

1. Normalise underscores to spaces: `a_girl` → `a girl`.
2. Search for the full phrase in the concatenated decoded token string (char-level, mapping back to token index).
3. If not found, try each word of the phrase in reverse order (skipping words of length ≤ 2) to handle cases where the full phrase is partially tokenised.

This reduced the fallback rate from ~53% to < 10%. For the paper: add one sentence to §4.2 stating the swap-token normalisation step and the measured fallback rate after the fix.

---

## 7. Bibliography Requirements

`references.bib` must contain entries for at least these works (use BibTeX format with full venue names, complete author lists, and DOIs/URLs where available):

**Measurement theory:**
- Cronbach, L. J., & Meehl, P. E. (1955). Construct validity in psychological tests. *Psychological Bulletin*.
- Campbell, D. T., & Fiske, D. W. (1959). Convergent and discriminant validation by the multitrait-multimethod matrix. *Psychological Bulletin*.
- Messick, S. (1995). Validity of psychological assessment. *American Psychologist*.
- Jacobs, A. Z., & Wallach, H. (2021). Measurement and fairness. *FAccT 2021*.

**LLM bias and construct validity (2024–2026):**
- Kalaitzidis, M. (2026). The Evaluation Trap: Benchmark Design as Theoretical Commitment. *arXiv:2605.14167*.
- Bean, A., et al. (2025). Measuring what Matters: Construct Validity in Large Language Model Benchmarks. *NeurIPS 2025 Datasets and Benchmarks*.
- Kearns, R. O. (2026). Quantifying construct validity in large language model evaluations. *arXiv*.
- Wang, A., et al. (2025). Fairness through Difference Awareness: Measuring Desired Group Discrimination in LLMs. *ACL 2025*.

**Source benchmarks:**
- Parrish, A., et al. (2022). BBQ: A Hand-Built Bias Benchmark for Question Answering. *Findings of ACL 2022*.
- Nangia, N., et al. (2020). CrowS-Pairs: A Challenge Dataset for Measuring Social Biases in Masked Language Models. *EMNLP 2020*.
- Nadeem, M., et al. (2021). StereoSet. *ACL-IJCNLP 2021*.
- Zhao, J., et al. (2018). Gender Bias in Coreference Resolution. *NAACL 2018* (WinoBias).
- Smith, E. M., et al. (2022). "I'm Sorry to Hear That": Finding New Biases in Language Models with a Holistic Descriptor Dataset. *EMNLP 2022* (HolisticBias).
- Sahoo, N., et al. (2024). IndiBias. *NAACL 2024*.
- BharatBBQ — TACL 2025 entry.

**Mechanism / causal:**
- Pearl, J. (2009). *Causality: Models, Reasoning, and Inference*. Cambridge University Press.
- Meng, K., et al. (2022). Locating and Editing Factual Associations in GPT. *NeurIPS 2022* (ROME).
- Liu, Y., et al. (2026). DIFFHEADS: Differential Head Analysis for Bias. *AAAI 2026*.
- Path Effect / IPE paper from BlackboxNLP 2025.
- LLMCert-B paper (counterfactual bias certification).
- MedEqualQA (arXiv 2510.12818).
- Shaikh, O., et al. (2023). On Second Thought, Let's Not Think Step by Step! *ACL 2023*.

**Statistics:**
- Efron, B., & Tibshirani, R. (1993). *An Introduction to the Bootstrap*. Chapman & Hall.
- McNemar, Q. (1947). Note on the sampling error of the difference between correlated proportions or percentages. *Psychometrika*.
- Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences*.

**Deployment / policy:**
- NIST AI RMF 1.0 (2023). NIST AI 100-1.
- European Commission (2024). Regulation (EU) 2024/1689 (EU AI Act).
- Solaiman, I., et al. (2024). Evaluating the Social Impact of Generative AI Systems. *FAccT 2024*.
- Goh, E., et al. (2024). Large Language Model Influence on Diagnostic Reasoning. *JAMA Network Open*.

**Models:**
- Llama-3.1 technical report (Meta 2024).
- Qwen-2.5 technical report (Alibaba 2024).
- Gemma-2 technical report (Google 2024).
- Phi-4 technical report (Microsoft 2024).

**Recent TCSS bias papers to cite for venue-fit signaling (find 3 from the 2024–2026 issues):**
- Search TCSS issues for "bias," "fairness," "LLM," "social impact." Cite at least three to demonstrate venue familiarity. Use the journal's website to verify exact citations.

Citations must be numerically ordered in IEEE style (`[1], [2], ...`), and all entries must have complete bibliographic information. Do not produce "et al." in the bib entries themselves — list full author names.

---

## 8. Quality Bar Before You Stop Generating

Before declaring the paper complete, verify:

1. The paper compiles cleanly with `pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex` producing a 12-page PDF without overfull boxes.
2. Every figure renders correctly with the specified caption.
3. Every numerical claim traces to a results file or is marked `\todo{}`.
4. Every reference in the bib file is cited; every citation in the body has a bib entry.
5. The paragraph "Sanyal test" passes — no em-dashes, no AI-tell verbs, no synthetic three-item lists.
6. The cover letter exists and mentions IEEE TCSS by name.
7. The abstract has the mandatory structure from §3.1 with one specific quantitative result.
8. The Introduction's first 500 words execute Move 1 from §2.
9. Section 2 has the "Measurement Validity and Sociotechnical Auditing" subsection executing Move 2.
10. Section 4 contains the Pearl/Meng causal-framing sentence verbatim or near-verbatim, executing Move 3.
11. Section 6 leads with the leaderboard before per-model results, executing Move 4.
12. The README.md inside `paper/` explains the compile commands, the figure regeneration script, and the location of every supplementary file.

If any check fails, fix before finalizing.

---

## 9. Pre-Compile Self-Review

After producing all files, perform this self-review and report findings to the user before they upload to Overleaf:

1. Count em-dashes in `main.tex`. Report the count. If non-zero, list each line and replace.
2. Count AI-tell verbs (delve, leverage, harness, foster, unlock, navigate, illuminate). Report.
3. Verify all seven figures are present in `images/`.
4. Verify the bibliography compiles standalone (`bibtex main` with no warnings).
5. Verify the paper title contains "Causal" or "Validity" and "Bias" or "Audit."
6. Verify abstract word count is 195–215.
7. Verify keywords/index terms list contains "computational social systems."
8. Verify Section 8 (Ethical Considerations) explicitly mentions dual-use and accessibility.
9. Verify the cover letter mentions at least three suggested reviewers (even as TODO placeholders).
10. Print a final summary table: file name, line/word count, status.

---

## 10. If You Are Missing Information

If specific empirical values are not yet available because experiments are still running:

- Insert `\todo[fancyline]{Populate from results/<filename>.parquet column <X>}` instead of fabricating numbers.
- Continue writing all sections that do not depend on those values.
- At the end, report which `\todo{}` markers remain so the author can complete them after experiments finish.

Do not skip the bibliography building because results are missing. The bib file is independent of empirical results and must be complete.

---

## 11. Anti-Patterns to Reject

If you find yourself doing any of these, stop and rewrite:

- Padding the related work section with citations that are not engaged in the body. Every citation should serve a sentence.
- Writing "extensive experiments" or "comprehensive evaluation" without specifying scale.
- Using "Our framework" possessively across sections — alternate with "MIRAGE" and "the framework."
- Hiding negative results. If a model performs well on MIRAGE-Full unexpectedly, that's part of the story; do not omit it.
- Concluding sentences that promise future work without specifying what that work is.
- Acknowledgments section larger than 3 sentences. (Single author, no funding to disclose unless the user supplies one.)

---

## 12. Final Note

This paper has been designed for IEEE TCSS over five conversational iterations and is positioned to maximize acceptance probability. The single most important sentence in the paper is the causal-framing sentence in Section 4.1. The single most important figure is the validity leaderboard (fig3). The single most important contribution to foreground in the abstract is the predictive-validity claim, because it is what most clearly differentiates MIRAGE from Bean et al. (NeurIPS 2025).

Write the paper. Then run the self-review. Then return the file tree and a one-paragraph summary of what is complete and what remains as `\todo{}`. Do not produce more output than required by these instructions. Do not editorialize about the project. Do not add a personal note or sign-off in the paper or in the response.

End of instructions.
