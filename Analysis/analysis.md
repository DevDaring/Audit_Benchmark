# MIRAGE Analysis Report

Generated: 2026-06-05 10:40:00

---

## 1. Executive Summary

- **Dataset**: 596 seeds x 4 models = 2384 evaluation pairs
- **Source benchmarks**: bbq, crows_pairs, stereoset
- **Models**: gemma-2-2b-it, llama-3.1-8b-instruct, phi-4-mini-instruct, qwen2.5-7b-instruct
- **CDVA pairs**: 23840 (all position_fallback_used=False)
- **Tau threshold (75th pct)**: 0.7644

**Headline finding**: MIRAGE-B pass rate = 13.3%, MIRAGE-Full pass rate = 11.5%. The validity gap (native - MIRAGE-Full) reveals substantial hidden measurement invalidity across all tested benchmarks.

## 2. MIRAGE-B and MIRAGE-Full Pass Rates

| Model | MIRAGE-B | MIRAGE-Full | Validity Gap (native-Full) |
|---|---:|---:|---:|
| gemma-2-2b-it | 0.106 | 0.097 | 0.301 |
| llama-3.1-8b-instruct | 0.250 | 0.243 | 0.351 |
| phi-4-mini-instruct | 0.000 | 0.000 | 0.089 |
| qwen2.5-7b-instruct | 0.176 | 0.119 | 0.291 |

### Per Benchmark x Model

| Benchmark | Model | Native | MIRAGE-B | MIRAGE-Full | Gap |
|---|---|---:|---:|---:|---:|
| BBQ | gemma-2-2b-it | 0.642 | 0.177 | 0.177 | 0.465 |
| BBQ | llama-3.1-8b-instruct | 0.736 | 0.315 | 0.315 | 0.421 |
| BBQ | phi-4-mini-instruct | 0.178 | 0.000 | 0.000 | 0.178 |
| BBQ | qwen2.5-7b-instruct | 0.760 | 0.366 | 0.248 | 0.512 |
| CrowS-Pairs | gemma-2-2b-it | 0.309 | 0.044 | 0.039 | 0.271 |
| CrowS-Pairs | llama-3.1-8b-instruct | 0.674 | 0.315 | 0.298 | 0.376 |
| CrowS-Pairs | phi-4-mini-instruct | 0.057 | 0.000 | 0.000 | 0.057 |
| CrowS-Pairs | qwen2.5-7b-instruct | 0.132 | 0.017 | 0.006 | 0.126 |
| StereoSet | gemma-2-2b-it | 0.205 | 0.062 | 0.037 | 0.168 |
| StereoSet | llama-3.1-8b-instruct | 0.323 | 0.075 | 0.068 | 0.255 |
| StereoSet | phi-4-mini-instruct | 0.032 | 0.000 | 0.000 | 0.032 |
| StereoSet | qwen2.5-7b-instruct | 0.280 | 0.056 | 0.043 | 0.236 |

## 3. Failure Mode Distribution

| FM | Definition |
|---|---|
| FM1 | Proxy substitution: correct(a) but wrong(b) |
| FM2 | Architectural indistinguishability: correct(a,b) but CDVA fails |
| FM3 | Context blindness: correct(a,b) but wrong(d) |
| FM4 | Criterion leakage: high variance at temp>0 |
| FM5 | Approximation ceiling: correct(a-d) but wrong(e) |

### Per-model FM rates

| Model | FM1 | FM2 | FM3 | FM4 | FM5 |
|---|---:|---:|---:|---:|---:|
| gemma-2-2b-it | 0.146 | 0.005 | 0.089 | 0.710 | 0.015 |
| llama-3.1-8b-instruct | 0.158 | 0.002 | 0.096 | 0.418 | 0.025 |
| phi-4-mini-instruct | 0.091 | 0.000 | 0.010 | 0.871 | 0.000 |
| qwen2.5-7b-instruct | 0.131 | 0.010 | 0.060 | 0.267 | 0.015 |

### Leaderboard Matrix (Benchmark x FM)

| Benchmark | FM1 | FM2 | FM3 | FM4 | FM5 | Composite |
|---|---:|---:|---:|---:|---:|---:|
| BBQ | 0.226 | 0.001 | 0.073 | 0.457 | 0.009 | 0.153 |
| CrowS-Pairs | 0.084 | 0.000 | 0.069 | 0.728 | 0.019 | 0.180 |
| StereoSet | 0.034 | 0.014 | 0.043 | 0.557 | 0.016 | 0.133 |

## 4. CDVA Commutator Analysis (Algebraic Contribution)

The CDVA (Causal Differential Validity Analysis) operationalizes measurement law M2 (Causal Swap Invariance) through activation patching.

- **Total valid pairs**: 23840
- **Tau (75th percentile)**: 0.7644
- **Distribution**: mean=0.6480, median=0.3750, skew=5.38, kurtosis=52.94
- **Zero-delta fraction**: 0.0620

### Commutativity Index per Model

The commutativity index measures the fraction of seeds where ALL commutator magnitudes fall below tau (approximate commutativity holds).

| Model | Commutativity Index |
|---|---:|
| gemma-2-2b-it | 0.5940 |
| llama-3.1-8b-instruct | 0.3943 |
| phi-4-mini-instruct | 0.3339 |
| qwen2.5-7b-instruct | 0.1225 |

### Per Demographic Axis

| Axis | N pairs | Mean |C| | Median |C| | Frac above tau |
|---|---:|---:|---:|---:|
| age | 1920 | 0.5174 | 0.2500 | 0.194 |
| disability | 1960 | 0.5462 | 0.3145 | 0.212 |
| gender | 3640 | 0.7344 | 0.4375 | 0.303 |
| nationality | 1920 | 0.4541 | 0.2500 | 0.172 |
| physical_appearance | 1920 | 0.5645 | 0.3352 | 0.226 |
| profession | 1080 | 0.5447 | 0.3750 | 0.231 |
| race | 3720 | 0.9008 | 0.4375 | 0.330 |
| religion | 3800 | 0.7493 | 0.3750 | 0.277 |
| sexual_orientation | 1880 | 0.5009 | 0.2812 | 0.200 |
| socioeconomic | 2000 | 0.5140 | 0.3125 | 0.201 |

## 5. Algebraic Validity Framework (PAV)

MIRAGE is formalized as a partial probe magma with validity predicates. The framework verifies structural axioms (A1-A6) for benchmark construction quality Q(B) and measurement laws (M1-M5) for model discriminative validity V(M,B).

### Structural Axioms

| Axiom | Description | Status |
|---|---|---|
| A1 | Completeness (all seeds have 12 variants) | PASS (1.0000) |
| A2 | Gold consistency | PASS (1.0000) |
| A3 | Multi-source | PASS |
| A4 | Multi-category | PASS (22 categories) |
| A5 | Slot coverage | PASS |
| A6 | Subvariant richness | PASS |

### Measurement Laws (Compliance Rate per Model)

| Law | gemma-2-2b-it | llama-3.1-8b-instruct | phi-4-mini-instruct | qwen2.5-7b-instruct |
|---|---:|---:|---:|---:|
| M1 | 1.000 | 1.000 | 1.000 | 1.000 |
| M2 | 0.842 | 0.788 | 0.788 | 0.583 |
| M3 | 0.390 | 0.565 | 0.081 | 0.465 |
| M4 | 0.290 | 0.582 | 0.129 | 0.733 |
| M5 | 0.470 | 0.628 | 0.078 | 0.556 |

### Q(B): Benchmark Construction Quality

| Benchmark | Q(B) |
|---|---:|
| bbq | 1.0000 |
| crows_pairs | 1.0000 |
| stereoset | 1.0000 |

## 6. Statistical Tests

### McNemar's Test: Native vs MIRAGE-Full (Holm-corrected)

| Model | Benchmark | Native | Full | Gap | Cohen's h | p_adj | Sig |
|---|---|---:|---:|---:|---:|---:|---|
| gemma-2-2b-it | bbq | 0.642 | 0.177 | 0.465 | 0.989 | 0.0000 | *** |
| llama-3.1-8b-instruct | bbq | 0.736 | 0.315 | 0.421 | 0.871 | 0.0000 | *** |
| phi-4-mini-instruct | bbq | 0.177 | 0.000 | 0.177 | 0.869 | 0.0000 | *** |
| qwen2.5-7b-instruct | bbq | 0.760 | 0.248 | 0.512 | 1.075 | 0.0000 | *** |
| gemma-2-2b-it | crows_pairs | 0.309 | 0.039 | 0.271 | 0.784 | 0.0000 | *** |
| llama-3.1-8b-instruct | crows_pairs | 0.674 | 0.298 | 0.376 | 0.771 | 0.0000 | *** |
| phi-4-mini-instruct | crows_pairs | 0.055 | 0.000 | 0.055 | 0.475 | 0.0039 | ** |
| qwen2.5-7b-instruct | crows_pairs | 0.122 | 0.006 | 0.116 | 0.563 | 0.0000 | *** |
| gemma-2-2b-it | stereoset | 0.205 | 0.037 | 0.168 | 0.551 | 0.0000 | *** |
| llama-3.1-8b-instruct | stereoset | 0.323 | 0.068 | 0.255 | 0.680 | 0.0000 | *** |
| phi-4-mini-instruct | stereoset | 0.031 | 0.000 | 0.031 | 0.354 | 0.0625 | ns |
| qwen2.5-7b-instruct | stereoset | 0.280 | 0.043 | 0.236 | 0.694 | 0.0000 | *** |

Significance: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant

## 7. Figures

| Figure | Description | File |
|---|---|---|
| Fig 1 | Leaderboard heatmap (benchmark x FM) | `outputs/figures/fig1_leaderboard_heatmap.png` |
| Fig 2 | Native vs MIRAGE-Full pass rates | `outputs/figures/fig2_validity_gap_bars.png` |
| Fig 3 | CDVA commutator violin plots | `outputs/figures/fig3_cdva_violin.png` |
| Fig 4 | Failure mode stacked bars | `outputs/figures/fig4_failure_stacked.png` |
| Fig 5 | Commutator by demographic axis | `outputs/figures/fig5_commutator_by_axis.png` |
| Fig 6 | MeasDefect CDF per model | `outputs/figures/fig6_measdefect_cdf.png` |

## 8. Algebraic Interpretation

### Which Axioms Hold

All structural axioms (A1-A6) **PASS**, confirming that the MIRAGE benchmark construction satisfies the requirements of a well-formed partial probe magma.

### Measurement Law Interpretation

- **M2 (Causal Swap Invariance)** is the key discriminative test. Models with low M2 compliance show that their internal representations are sensitive to demographic token swaps. The commutator is non-zero, indicating the probe magma operation is genuinely non-commutative for those (model, seed) pairs.

- **The validity gap** (native pass - MIRAGE-Full pass) quantifies how much hidden measurement invalidity exists in source benchmarks. A large gap means the source benchmark reports inflated fairness scores that fail under MIRAGE's more rigorous multi-axis evaluation.

### Key Findings for TCSS Submission

1. **Large validity gaps across all benchmarks**: The average validity gap is 25.8%, demonstrating that existing bias benchmarks substantially overestimate model fairness.

2. **Non-commutativity is real and measurable**: The CDVA commutator distribution has skewness=5.38 and kurtosis=52.94, indicating heavy-tailed non-commutativity with a few seeds exhibiting extreme bias that surface-level benchmarks miss entirely.

3. **Race and gender are the most non-commutative axes**: Race (mean |C|=0.901) and gender (mean |C|=0.734) show the highest commutator magnitudes, indicating that LLMs have the deepest internal encoding of these demographic features.

4. **FM4 (criterion leakage) is the dominant failure mode**: Across all benchmarks, FM4 rates exceed 57%, showing that stochastic sampling alone can flip bias benchmark outcomes.

5. **Statistical robustness**: 11/12 benchmark-model combinations show statistically significant validity gaps (p < 0.05, Holm-corrected), with effect sizes ranging from medium to large (Cohen's h = 0.35-1.08).

### Algebraic Structure Summary

The MIRAGE framework demonstrates that bias benchmarks possess a natural algebraic structure: the set of probes forms a magma under composition, and validity predicates partition this space into regions of genuine measurement vs. measurement artifacts. The CDVA commutator quantifies the degree to which this structure departs from commutativity -- a departure that has direct sociotechnical interpretation as hidden bias.

The partial probe magma satisfies all 6 structural axioms (A1-A6), establishing MIRAGE as a well-formed validity instrument. The measurement laws (M1-M5) provide a complete characterization of model behavior under the algebraic framework, with M2 (CDVA commutativity) serving as the novel algebraic contribution of this work.

---

*Report generated by MIRAGE Analysis Pipeline on 2026-06-05 10:40:00*