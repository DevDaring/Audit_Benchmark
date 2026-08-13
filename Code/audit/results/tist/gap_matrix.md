# Gap matrix — COMJNL rejection to TIST resubmission

Every reviewer objection maps to an experiment, a result artifact, and a section of
`Submission/TIST_Audit_Benchmark.tex`. Status is updated at every checkpoint.

Source manuscript: `Submission/COMJNL_Audit.tex` (rejected, The Computer Journal).
Target manuscript: `Submission/TIST_Audit_Benchmark.tex` (ACM TIST).
Results root: `Code/audit/results/tist/` (the mission file says `Code/mirage/results/tist/`;
that folder was renamed to `Code/audit/` before this work began).

---

## Reviewer 1 — foundational

> "The causal interpretation of CDVA is not sufficiently established. Because MIRAGE-Full,
> the validity leaderboard and several of the main conclusions depend on CDVA, this issue
> is foundational."

The rejected manuscript defended CDVA with one analysis, the patch-site recovery table.
Six independent lines of evidence replace it.

| ID | Experiment | Question it answers | Artifact | Paper section | Status |
|----|-----------|---------------------|----------|---------------|--------|
| E1.1 | Placebo patch controls | Is the commutator specific to the protected-attribute position, or does any patch move the logit? | `e1/battery_<model>.jsonl` -> `e1/stats_e1_1_placebo.csv` | Sec. "Establishing CDVA as a Causal Probe", Fig. F-A | code ready, awaiting GPU |
| E1.2 | Layer-wise localisation sweep | Does the effect concentrate where attribute information is encoded, or is it diffuse numerical perturbation? | `e1/layersweep_<model>.jsonl` -> `e1/stats_e1_2_layers.csv` | same section, Fig. F-B | code ready, awaiting GPU |
| E1.3 | Noising and denoising directions | Is the pass/fail label robust to patch direction? | same battery file -> `e1/stats_e1_3_direction.csv` | same section | code ready, awaiting GPU |
| E1.4 | Ground-truth synthetic controls | Criterion validity: does CDVA fail items that must depend on the attribute and pass items where it is provably irrelevant? | `Dataset/seeds/synthetic_controls.parquet` (**built, 200 seeds**), `e1/controls_<model>.jsonl` -> `e1/stats_e1_4_controls.csv` | same section, Fig. F-C | seeds built, awaiting GPU |
| E1.5 | Convergent validity vs causal mediation | Does an independent published causal method agree with the commutator? | `e1/mediation_<model>.jsonl` -> `e1/stats_e1_5_mediation.csv` | same section | code ready, awaiting GPU |
| E1.6 | Metric robustness | Do conclusions survive logit-difference and KL metrics? | same battery file -> `e1/stats_e1_6_metrics.csv` | same section | code ready, awaiting GPU |

**Additional evidence for Reviewer 1, not requested but directly on point.** E0 found that
19.75% of the CDVA pairs behind the published numbers came from items offering the same
answer option twice, and that those pairs carry systematically larger |C| on all four
models. Part of the commutator demonstrably was an item-construction artefact. It is
quantified, excluded, and the audit rerun. See `e0/FINDINGS.md`. Status: **complete**.

## Reviewer 2 — four points

| # | Objection | Experiment | Artifact | Paper section | Status |
|---|-----------|-----------|----------|---------------|--------|
| 1 | No annotation guidelines or inter-annotator agreement for pentad slots (d)/(e) | E3 | `e3/annotation_guidelines.md`, `e3/iaa_*.csv`, `e3/FINDINGS.md` | Methods (pentad construction) + appendix | pending |
| 2 | tau as the 75th percentile is heuristic; absolute pass rates are sensitive to it | E2 | `e2/tau_sweep.parquet`, `e2/stats_tau_sensitivity.csv` | Methods (threshold) + Results, Fig. F-D | pending |
| 3a | Study is English-only | E4 | `e4/*`, `Dataset/seeds/pentad_hi.parquet`, `pentad_bn.parquet` | New section "Multilingual Auditing", Fig. F-E | pending |
| 3b | CDVA cannot reach closed-API models | E5 | `e5/surrogate_*.csv` | Discussion (policy argument for white-box audit access) | pending |
| 4 | Dual-use risk of the released activation-patching toolkit | W4 (writing) | none | Limitations and Ethical Considerations | pending |

---

## Repositioning for TIST

| Change | Rationale | Status |
|--------|-----------|--------|
| Measurement-theory paper to deployable auditing system | TIST scope: intelligent systems and applicable technology | pending |
| New System Architecture section with pipeline diagram and runtime/cost table | TIST readers value deployability numbers | pending |
| Multilingual results promoted to a first-class contribution | Converts the reviewer's limitation into a contribution | pending |
| Exactly 3 contributions | Hard style rule | pending |
| ACM `acmart` acmsmall, <= 25 pages including references, ~10k words | TIST auto-rejects overlength | pending |

## Verification gates

- [ ] Every reviewer point maps to a completed experiment and a paper section
- [ ] `number_provenance.csv` traces every numeral in the .tex to a file under `results/tist/`
- [ ] E1.1 placebo separation significant per model, or the paper says where it is not
- [ ] Page count <= 25, word count ~10k, overflow in appendix
- [ ] Clean LaTeX build, vector figures, script-generated tables in adjustbox
- [ ] Style lint pass (no em-dashes, no banned verbs or connectives, we/our budget, sentence-length discipline)
- [ ] Repo committed and pushed
- [ ] Premortem: top 3 reasons TIST would reject, each fixed
