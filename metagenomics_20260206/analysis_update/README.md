## Contents (with hyperlinks)

- [Methods](#methods)
- [Modeling of host genomic DNA fraction](#modeling-of-host-genomic-dna-fraction)
- [Similarity analysis of shotgun metagenomic sequencing](#similarity-analysis-of-shotgun-metagenomic-sequencing)
- [Comparison between shotgun metagenomic sequencing and culture results](#comparison-between-shotgun-metagenomic-sequencing-and-culture-results)
- [Results](#results)
- [Host genomic DNA content in wound skin swabs is associated with structured batch effects and acute-like wound status](#host-genomic-dna-content-in-wound-skin-swabs-is-associated-with-structured-batch-effects-and-acute-like-wound-status)
- [Patient identity and wound chronicity remain associated with metagenomic similarity after technical adjustment](#patient-identity-and-wound-chronicity-remain-associated-with-metagenomic-similarity-after-technical-adjustment)
- [Shotgun metagenomic detection agrees with culture in a nonparametric descriptive analysis and remains significant after rank-based technical adjustment](#shotgun-metagenomic-detection-agrees-with-culture-in-a-nonparametric-descriptive-analysis-and-remains-significant-after-rank-based-technical-adjustment)
- [Third party packages used (including both python and R packages)](#third-party-packaged-used-including-both-python-and-r-packages)
- [Reproducibility](#reproducibility)

## Methods

### Modeling of host genomic DNA fraction

#### Host fraction definition

Host genomic DNA content was quantified for each sequenced sample from pre-host-filter Bracken/Kraken output as the fraction of total Bracken reads assigned to Homo sapiens, where total Bracken reads were defined as classified species reads plus Kraken unclassified reads:
$$
h_i \;=\; \frac{\text{human\_species\_reads}_i}{\text{total\_bracken\_reads}_i},
\qquad
\text{total\_bracken\_reads}_i
=
\text{classified\_species\_reads}_i + \text{kraken\_unclassified\_reads}_i.
$$
In this cohort, the non-bacterial residual was usually small: the unclassified fraction was median 0.082% (95th percentile 1.04%), and the additional classified non-bacterial fraction (including archaeal, fungal, and viral assignments) was median <0.001% (95th percentile 1.60%) ([table_02_01_qc_metrics.tsv](tables/table_02_01_qc_metrics.tsv)); therefore, this quantity was approximately equivalent to human/(human+bacterial) for most samples.
To obtain an approximately unbounded response for Gaussian modeling, host fraction was logit-transformed after truncation away from 0 and 1:
$$
y_i \;=\; \log\!\left(\frac{\tilde h_i}{1-\tilde h_i}\right),
\qquad
\tilde h_i = \min\{\max(h_i,10^{-4}),1-10^{-4}\}.
$$

#### Metadata encoding

All 74 sequenced samples were retained for host-fraction modeling after metadata cleaning and harmonization.
Culture date was treated as a technical batch variable (batch identifier).
Patient-relative biological time was defined as years since the first sampled date for that patient, $t_i = \frac{\text{culture\_date}_i - \min(\text{culture\_date for patient }p_i)}{365.25}$.
Categorical variables with incomplete clinical annotation were not numerically imputed; instead, uncertainty was retained as explicit levels such as unknown.

#### Primary Gaussian mixed model

The main host-fraction model was a Gaussian linear mixed model:
$$
y_i
=
\beta_0
+
\sum_{r} \beta_r\,I(\text{body\_region}_i=r)
+
\sum_{c} \gamma_c\,I(\text{chronicity}_i=c)
+
\delta\,I(\text{culture\_positive}_i=\text{positive})
+
\eta\, t_i
+
u_{p(i)}
+
b_{k(i)}
+
\varepsilon_i.
$$
In this formulation, $r$ indexes the non-reference body-region levels (head/neck, upper extremity, and trunk/perineum), and $c$ indexes the non-reference chronicity levels (acute-like, chronic-like, and mixed).
The random components were $u_{p(i)} \sim N(0,\sigma^2_{\text{patient}})$ for patient-specific intercepts, $b_{k(i)} \sim N(0,\sigma^2_{\text{batch}})$ for culture-date batch intercepts, and $\varepsilon_i \sim N(0,\sigma^2)$ for residual error.
Patient and batch were modeled as random effects because they define correlation structure among observations rather than scientific contrasts of interest.
Body region, chronicity, broad culture positivity, and patient-relative elapsed time were modeled as fixed effects because the goal was to estimate interpretable cohort-level associations for those variables.

#### Model fitting and inference

Models were fit by maximum likelihood (reml = FALSE) so that likelihood-ratio tests between nested models were valid.
To reduce dependence on a single optimizer, each model was attempted with lbfgs, powell, and bfgs, and the lowest-AIC converged fit was retained.
All final reported Gaussian host models converged.

Coefficient-level fixed-effect inference was based on Wald tests from the fitted Gaussian mixed model, with $z_j = \hat\beta_j / \operatorname{SE}(\hat\beta_j)$.
Two-sided p values were taken from the asymptotic normal approximation implemented in statsmodels.
Multiple-testing adjustment used the Benjamini-Hochberg false discovery rate (BH-FDR) procedure.
Two adjustment schemes were considered: global BH, where $(q_1,\dots,q_m)=\mathrm{BH}(p_1,\dots,p_m)$ across all fixed-effect coefficients in the host model, and factor-specific post hoc BH, where $(q^{\text{chronicity}}_1,\dots,q^{\text{chronicity}}_k)=\mathrm{BH}(p^{\text{chronicity}}_1,\dots,p^{\text{chronicity}}_k)$ within a single multi-level factor.

Significance thresholds were defined as $p < 0.05$ for prespecified single tests and $q < 0.10$ for FDR-adjusted results.

Random-effect terms were assessed by nested likelihood-ratio tests comparing models with and without the relevant variance component.
For a nested comparison, $\Lambda = 2\{\ell(\text{full}) - \ell(\text{reduced})\}$.
Because the null hypothesis for a variance component is on the boundary of the parameter space ($H_0:\sigma^2 = 0$), p values were computed using the standard one-parameter boundary correction $p_{\text{boundary}} = \tfrac{1}{2}P(\chi^2_1 \ge \Lambda)$.

To test whether an entire categorical factor contributed to host fraction, we used likelihood-ratio omnibus tests comparing the full model to reduced models with the factor removed.
For chronicity group, the null hypothesis was $H_0:\gamma_{\text{acute}}=\gamma_{\text{chronic}}=\gamma_{\text{mixed}}=0$.
For body region, the null hypothesis was $H_0:\beta_{\text{head/neck}}=\beta_{\text{upper extremity}}=\beta_{\text{trunk/perineum}}=0$.
P values were obtained from a $\chi^2_d$ distribution with $d$ equal to the number of removed coefficients.

To increase power for biologically targeted hypotheses, we also fit prespecified 1-degree-of-freedom contrast models.
These replaced the original multi-level factor with a binary indicator corresponding to the contrast of interest.

For acute-like chronicity:
$$
y_i \sim \text{body region} + I(\text{acute-like}) + \text{culture positivity} + t_i + (1|\text{patient}) + (1|\text{batch}).
$$

For upper-extremity location:
$$
y_i \sim I(\text{upper extremity}) + \text{chronicity} + \text{culture positivity} + t_i + (1|\text{patient}) + (1|\text{batch}).
$$

These planned-contrast p values were BH-adjusted only within the planned-contrast family.

### Similarity analysis of shotgun metagenomic sequencing

#### Outcome definition and distance metric

To assess whether samples sharing biological covariates were more similar in shotgun metagenomic composition, we analyzed Bracken-derived bacterial community profiles from QC-passing samples and quantified between-sample compositional dissimilarity using Bray-Curtis distance.
If $p_{is}$ denotes the relative abundance of species $s$ in sample $i$, Bray-Curtis distance between samples $i$ and $j$ was defined as
$$
d_{ij}=\frac{\sum_s |p_{is}-p_{js}|}{\sum_s (p_{is}+p_{js})}.
$$
Distances range from 0 for identical communities to 1 for maximally dissimilar communities.
All community-level analyses in this section were restricted to the QC-passing sample set used for composition-sensitive inference.

#### Descriptive pairwise comparisons

As an initial descriptive analysis, all sample pairs were partitioned into three groups: same patient, same batch date; same patient, different batch date; and different patient.
Median Bray-Curtis distances for the two within-patient groups were compared against the different-patient reference group using one-sided Mann-Whitney tests with the alternative hypothesis that within-group distances were smaller.
Resulting p values were adjusted with the Benjamini-Hochberg false discovery rate procedure.

#### Pairwise mixed-effects similarity models

Because the biological question is inherently pairwise, we also modeled Bray-Curtis distance directly at the sample-pair level.
The pairwise dataset included all unordered pairs of QC-passing samples.
For each pair $(i,j)$, we defined binary indicators for whether the two samples shared the same patient, batch date, broad body region, exact location, chronicity group, or culture-positivity status.
We also defined continuous pairwise nuisance covariates as the mean host fraction, $\overline{h}_{ij}=(h_i+h_j)/2$, the mean bacterial read depth, $\overline{z}_{ij}=\{\log_{10}(\text{bacterial reads}_i)+\log_{10}(\text{bacterial reads}_j)\}/2$, and the absolute elapsed-time difference, $\Delta t_{ij}=|t_i-t_j|$,
where $t_i$ is years since the first sample from the same patient.

Two linear mixed-effects models were fit.
In the first, site similarity was represented at the level of broad anatomical region:
$$
d_{ij}
=
\beta_0
+
\beta_1 I(\text{same patient}_{ij})
+
\beta_2 I(\text{same batch}_{ij})
+
\beta_3 I(\text{same body region}_{ij})
+
\beta_4 I(\text{same chronicity}_{ij})
+
\beta_5 I(\text{same culture positivity}_{ij})
+
\beta_6 \Delta t_{ij}
+
\beta_7 \overline{h}_{ij}
+
\beta_8 \overline{z}_{ij}
+
a_i + a_j + \varepsilon_{ij}.
$$
In the second, same body region was replaced by same exact location:
$$
d_{ij}
=
\beta_0
+
\beta_1 I(\text{same patient}_{ij})
+
\beta_2 I(\text{same batch}_{ij})
+
\beta_3 I(\text{same exact location}_{ij})
+
\beta_4 I(\text{same chronicity}_{ij})
+
\beta_5 I(\text{same culture positivity}_{ij})
+
\beta_6 \Delta t_{ij}
+
\beta_7 \overline{h}_{ij}
+
\beta_8 \overline{z}_{ij}
+
a_i + a_j + \varepsilon_{ij}.
$$
Here, $a_i$ and $a_j$ are crossed random intercepts for the two samples contributing to the pair, and $\varepsilon_{ij}$ is residual error.
This crossed random-effects structure accounts for non-independence arising because each sample contributes to many pairwise distances.

#### Fixed-effect inference and interpretation

For the pairwise mixed models, coefficient-level inference was based on Wald tests.
For each coefficient $\beta_k$, the test statistic was $t_k = \hat\beta_k / \operatorname{SE}(\hat\beta_k)$, with two-sided p values obtained from the fitted mixed model.
P values across coefficients within each pairwise model family were adjusted using the Benjamini-Hochberg procedure to generate q values.
Significance for FDR-controlled inference was interpreted at $q < 0.10$, while nominal p values were also reported.

Broad body region and exact location were evaluated separately and were not treated as interchangeable site definitions.

### Comparison between shotgun metagenomic sequencing and culture results

#### Organism-group abundance definition

We compared culture results against Bracken-derived bacterial relative abundances aggregated at clinically relevant organism-group level.
For each sample $i$, species-level Bracken bacterial counts $x_{is}$ were converted to relative abundances and organism-group abundance was defined as the sum across the species assigned to that group: $A_i^{(g)} = \sum_{s \in g} p_{is}.$
For example, S. aureus was represented by Staphylococcus aureus, whereas Klebsiella spp. was represented by the sum of Klebsiella pneumoniae and Klebsiella oxytoca.

#### Descriptive nonparametric concordance analysis

The primary descriptive concordance analysis compared organism-group abundance between culture-positive and culture-negative samples using a one-sided Mann-Whitney U test with alternative hypothesis culture positive > culture negative.
For each organism group, if $A_i$ denotes the relative abundance in sample $i$, the null hypothesis was

$$
H_0: A_i^{(+)} \text{ and } A_i^{(-)} \text{ come from the same distribution},
$$

against the one-sided alternative that abundances were stochastically larger in culture-positive samples.
As a complementary discrimination summary, AUROC was computed from empirical ROC curves using organism-group abundance as the continuous score and culture positivity as the binary label.
Benjamini-Hochberg correction was applied across organism groups to obtain q values.
This descriptive analysis used all 74 sequenced samples.

#### Threshold sweep and overlap summaries

To characterize practical detection agreement across abundance cutoffs, we evaluated thresholds from 0 to 0.10 relative abundance in increments of 0.001.
At each threshold, sequencing calls were defined as positive if $A_i^{(g)}\ge \tau$.
For each organism group and threshold, we computed true positives, false positives, false negatives, and true negatives, then derived sensitivity, specificity, positive predictive value, negative predictive value, F1 score, and Cohen’s kappa.
The threshold with the highest F1 score, breaking ties by kappa and then by smaller threshold, was retained as the optimal threshold for each organism group.

#### Adjusted rank-based mixed-effects concordance model

To adjust for technical nuisance structure while remaining nonparametric with respect to abundance scale, we fit a rank-based linear mixed model for each organism group.
These models were restricted to samples with at least 5,000 bacterial Bracken reads.
For each organism-group abundance $A_i^{(g)}$, we computed its within-dataset rank $r_i = \operatorname{rank}(A_i^{(g)})$, using average ranks for ties, and then transformed ranks to a normal-score scale $z_i = \Phi^{-1}\left(\frac{r_i - 0.5}{n}\right)$, where $\Phi^{-1}$ is the inverse standard normal cumulative distribution function and $n$ is the number of samples in the filtered dataset.
The adjusted model was

$$z_i = \beta_0 + \beta_1 I(\text{culture positive}_i) +
\beta_2 \,\text{host fraction}_i
+
\beta_3 \log_{10}(\text{bacterial reads}_i)
+
u_{\text{patient}(i)}
+
b_{\text{batch}(i)}
+
\varepsilon_i,
$$

with $u_{\text{patient}(i)} \sim N(0,\sigma^2_{\text{patient}}), \, b_{\text{batch}(i)} \sim N(0,\sigma^2_{\text{batch}}), \, \varepsilon_i \sim N(0,\sigma^2)$.

Here, culture positive was the fixed effect of interest, while host fraction and log10 bacterial reads were nuisance fixed effects.
Patient and batch were modeled as random intercepts to account for repeated sampling within patient and technical batch structure associated with culture date.
In this updated force-all-target analysis, we attempted one adjusted model for each of the nine culture organism groups regardless of class size, and only marked a target as non-estimable when there was no class variation or no variation in the rank-normalized abundance response.
Fixed-effect p values for culture status were taken from Wald tests and adjusted across all nine organism groups by Benjamini-Hochberg FDR.
Singular mixed-model fits were explicitly flagged as boundary random-effect solutions and interpreted as unstable sensitivity-level evidence rather than robust primary inference.

## Results

### Host genomic DNA content in wound skin swabs is associated with structured batch effects and acute-like wound status

Skin swabs are known to contain substantial host genomic DNA.
In our shotgun metagenomic sequencing, host fraction was high and variable across samples (range 0.003 to 1.000; median 0.947), consistent with pronounced host-carryover heterogeneity across swabs ([fig_14_01_host_fraction_overview.svg](figures/fig_14_01_host_fraction_overview.svg)).
Rather than focusing entirely on bacterial reads, we used this fraction as a complementary signal of wound microenvironment and host carryover.
To separate biology from nuisance structure, we modeled logit-transformed host fraction in a Gaussian mixed model with patient and culture-date batch as random effects and body region, chronicity group, culture positivity, and years since first patient sample as fixed effects.
Both random effects contributed materially to model fit: relative to a fixed-effects-only model, adding patient improved fit (LRT = 7.11, p = 0.00384) and adding batch improved fit (LRT = 3.42, p = 0.0323); when added sequentially, patient remained informative on top of batch (LRT = 5.31, p = 0.0106), whereas the additional batch contribution on top of patient was weaker (LRT = 1.62, p = 0.102), indicating stronger patient-level than incremental batch-date contribution in the fully adjusted setting ([fig_06_01_host_model_compare.svg](figures/fig_06_01_host_model_compare.svg)).
After controlling for these structured effects, acute-like wounds remained the clearest biological signal and were associated with higher host fraction (beta = 2.70, Wald p = 0.0262), with a consistent direction under adjusted model summaries ([fig_14_02_host_gaussian_mixed_summary.svg](figures/fig_14_02_host_gaussian_mixed_summary.svg)).
This direction remained supported under factor-specific post hoc correction across chronicity levels (q = 0.0785) and in the prespecified acute-like versus others contrast from the beta-binomial analysis (beta = 0.842, p = 0.0116, q = 0.0232).
Upper-extremity samples also trended toward higher host fraction (beta = 1.64, Wald p = 0.0728), whereas broad culture positivity, elapsed time since first patient sample, and the remaining chronicity and location categories did not show convincing independent association in the same adjusted model ([fig_14_03_host_gaussian_followup.svg](figures/fig_14_03_host_gaussian_followup.svg)).
The fitted random intercepts showed broad spread across both batch dates and patients, reinforcing that structured nuisance variation remained large even after fixed-effect adjustment ([fig_14_04_host_gaussian_random_intercepts.svg](figures/fig_14_04_host_gaussian_random_intercepts.svg)).
As an optional sensitivity view, a complementary beta-binomial fit reproduced the same acute-like directionality ([fig_11_01_host_beta_binomial.svg](figures/fig_11_01_host_beta_binomial.svg)).
As an additional optional context view, QC-level host-burden summaries showed the same high-host background pattern across samples ([fig_02_01_qc_host_burden.svg](figures/fig_02_01_qc_host_burden.svg)).

### Patient identity and wound chronicity remain associated with metagenomic similarity after technical adjustment

Raw Bray-Curtis distances showed that samples collected from the same patient on the same batch date were substantially more similar than samples from different patients (median distance 0.5897 vs 0.8777, BH-adjusted q = 0.000158), whereas this similarity weakened across different batch dates within the same patient (median distance 0.8660, q = 0.112) [table_03_02_pairwise_distance_summary.tsv](tables/table_03_02_pairwise_distance_summary.tsv), a contrast visible in the raw pairwise-distance distributions ([fig_03_01_pairwise_distance.svg](figures/fig_03_01_pairwise_distance.svg)).
However, the higher-resolution pairwise mixed-effects models showed that, after correcting for mean host fraction, mean bacterial read depth, batch matching, elapsed-time gap, and culture-positivity matching, similarity remained significantly associated with shared patient identity and shared chronicity.
In the body-region model, same patient was associated with lower Bray-Curtis distance (estimate = -0.0607, p = 0.0116, q = 0.0310) and same chronicity was even stronger (estimate = -0.0692, p = 1.22e-08, q = 9.76e-08); the same pattern held in the exact-location model (same patient: estimate = -0.0647, p = 0.00700, q = 0.0187; same chronicity: estimate = -0.0698, p = 9.28e-09, q = 7.42e-08) [table_12_02_pairwise_similarity_mixed_effects.tsv](tables/table_12_02_pairwise_similarity_mixed_effects.tsv), supporting patient and chronicity as robust adjusted similarity signals ([fig_12_02_pairwise_similarity_mixed.svg](figures/fig_12_02_pairwise_similarity_mixed.svg)).
Technical factors remained important in the same models, particularly mean host fraction (body-region model: estimate = -0.391, p = 1.36e-04, q = 5.43e-04; exact-location model: estimate = -0.393, p = 1.15e-04, q = 4.61e-04) and, more weakly, bacterial read depth (body-region model: estimate = -0.0660, p = 0.0562, q = 0.0903; exact-location model: estimate = -0.0670, p = 0.0518, q = 0.104), whereas same batch date itself was not independently associated with similarity (p = 0.647, q = 0.647 and p = 0.641, q = 0.720, respectively).
Broad body region showed only a modest, suggestive association with greater similarity (estimate = -0.0230, p = 0.0565, q = 0.0903), while same exact location was not supported after adjustment (estimate = +0.0138, p = 0.720, q = 0.720), consistent with adjusted marginal patterns that separate broad-site and exact-site effects ([fig_12_03_pairwise_adjusted_margins.svg](figures/fig_12_03_pairwise_adjusted_margins.svg)).
Together, these results indicate that patient identity and wound chronicity explain residual metagenomic similarity more consistently than exact anatomical site once technical variation is taken into account.

### Shotgun metagenomic detection agrees with culture in a nonparametric descriptive analysis and remains significant after rank-based technical adjustment

We evaluated agreement between wound culture and shotgun metagenomic sequencing at the organism-group level using two complementary layers of analysis: a descriptive nonparametric comparison of organism abundance between culture-positive and culture-negative samples, and a nuisance-adjusted rank-based mixed-effects model that accounted for patient, batch, host burden, and bacterial depth.
In the descriptive analysis across all 74 sequenced samples, the primary evidence came from one-sided Mann-Whitney U tests comparing abundance ranks between culture-positive and culture-negative samples, with Benjamini-Hochberg correction across organism groups.
By this primary U-test criterion, the strongest agreements were observed for P. aeruginosa (U = 839, q = 2.4e-05), S. aureus (U = 854, q = 5.5e-04), Klebsiella spp. (U = 314, q = 0.00210), Serratia (U = 275, q = 0.00146), and GAS (U = 274, q = 7.3e-05) [table_04_01_culture_concordance.tsv](tables/table_04_01_culture_concordance.tsv).
AUROC was used as a secondary descriptive confirmation of rank separation rather than the main inferential test, and showed the same ordering of strong concordance (P. aeruginosa 0.866, S. aureus 0.767, Klebsiella spp. 0.910, Serratia 0.982, GAS 0.979) ([fig_04_01_culture_concordance.svg](figures/fig_04_01_culture_concordance.svg)).
To test whether culture-versus-metagenomics agreement persisted after correcting for technical nuisance structure, we then fit rank-based mixed models on the subset of 68 samples with at least 5,000 bacterial Bracken reads.
In these models, the response was a normal-score transform of the rank of organism-group relative abundance, and fixed effects included culture status, host fraction, and bacterial read depth, with random intercepts for patient and batch.
Under this adjusted force-all-target analysis, culture-positive status remained strongly associated with higher metagenomic abundance for GAS (estimate = 2.114, q = 1.13e-07), P. aeruginosa (estimate = 1.420, q = 2.68e-06), Serratia (estimate = 1.369, q = 2.86e-04), S. aureus (estimate = 1.155, q = 1.47e-05), and Klebsiella spp. (estimate = 1.966, q = 2.04e-05) [table_13_03_culture_mixed_concordance.tsv](tables/table_13_03_culture_mixed_concordance.tsv), indicating persistence of concordance after technical adjustment ([fig_13_04_culture_adjusted_concordance.svg](figures/fig_13_04_culture_adjusted_concordance.svg)).
When all sparse targets were forced into the adjusted model set, Proteus (estimate = 1.308, q = 0.0158) and E. coli (estimate = 2.374, q = 0.00214) also appeared significant but both were singular fits, so these were treated as unstable signals rather than robust conclusions.
A. baumannii was borderline (estimate = 0.785, q = 0.0912) and E. faecalis was not supported after adjustment (estimate = 0.188, q = 0.723).
Threshold-dependent detection tradeoffs across organism groups are captured by the threshold-sweep profiles ([fig_13_01_culture_threshold_sweep.svg](figures/fig_13_01_culture_threshold_sweep.svg)).
Organism-wise overlap between culture and sequencing calls across abundance thresholds is summarized by the Venn-style panel grid ([fig_13_02_culture_venn_diagrams.svg](figures/fig_13_02_culture_venn_diagrams.svg)).
Culture-positive versus culture-negative abundance separation is further visualized in organism-level abundance boxplots with outliers shown ([fig_13_03_culture_abundance_density.svg](figures/fig_13_03_culture_abundance_density.svg)).
Together, these results indicate that shotgun metagenomic detection is concordant with culture for several major wound-associated organisms and that this agreement persists even after nonparametric abundance ranking and correction for patient-specific, batch, host, and depth effects.

## Third party packaged used (including both python and R packages)

### Python

- `pandas`: tabular data loading, cleaning, reshaping, and summary tables.
- `numpy`: numerical arrays, vectorized transformations, and matrix-style operations.
- `scipy`: nonparametric tests, distributions, and likelihood-ratio p-value calculations.
- `statsmodels`: Gaussian mixed models, regression, and multiple-testing correction.
- `matplotlib`: base figure generation and panel layout.
- `seaborn`: statistical plotting layers such as boxplots, violins, and distribution plots.
- `openpyxl`: reading Excel metadata workbooks.
- `nbformat`: notebook generation and structured notebook writing.

### R

- `readr`: TSV/CSV import and export.
- `dplyr`: data wrangling and grouped summaries.
- `tidyr`: reshaping data between wide and long formats.
- `ggplot2`: publication-style figure generation.
- `stringr`: string parsing and text cleaning.
- `vegan`: PERMANOVA and ecological distance-based community analysis.
- `lmerTest`: linear mixed-effects modeling with inferential summaries.
- `broom.mixed`: tidying mixed-model outputs into analysis tables.
- `emmeans`: estimated marginal means and planned contrasts.

## Reproducibility

The following notebooks generate the figures and tables for the 3 results sections above.

### Question 1: Host genomic DNA fraction

- `02_qc_and_host_burden.ipynb` -> `fig_02_01_qc_host_burden.svg`, `table_02_01_qc_metrics.tsv`, `table_02_02_host_model.tsv`
- `06_mixed_models_and_repeated_measures.ipynb` -> `fig_06_01_host_model_compare.svg`, `table_06_01_patient_structure_diagnostics.tsv`, `table_06_02_host_mixed_effects.tsv`, `table_06_03_host_mixed_status.tsv`
- `11_host_fraction_beta_binomial.ipynb` -> `fig_11_01_host_beta_binomial.svg`, `table_11_01_host_beta_binomial_effects.tsv`, `table_11_02_host_beta_binomial_status.tsv`
- `14_host_fraction_gaussian_story_figures.ipynb` -> `fig_14_01_host_fraction_overview.svg`, `fig_14_02_host_gaussian_mixed_summary.svg`, `fig_14_03_host_gaussian_followup.svg`, `fig_14_04_host_gaussian_random_intercepts.svg`, `table_14_01_host_gaussian_followup.tsv`, `table_14_02_host_gaussian_random_effects.tsv`

### Question 2: Metagenomic similarity by shared covariates

- `03_bracken_community_structure.ipynb` -> `fig_03_01_pairwise_distance.svg`, `table_03_01_pairwise_distances.tsv`, `table_03_02_pairwise_distance_summary.tsv`
- `12_adjusted_community_structure.ipynb` -> `fig_12_01_adjusted_permanova.svg`, `fig_12_02_pairwise_similarity_mixed.svg`, `fig_12_03_pairwise_adjusted_margins.svg`, `table_12_01_adjusted_community_permanova.tsv`, `table_12_02_pairwise_similarity_mixed_effects.tsv`, `table_12_03_pairwise_similarity_model_status.tsv`, `table_12_04_pairwise_adjusted_margins.tsv`

### Question 3: Culture versus shotgun metagenomics concordance

- `04_culture_concordance.ipynb` -> `fig_04_01_culture_concordance.svg`, `table_04_01_culture_concordance.tsv`
- `13_culture_threshold_and_concordance.ipynb` -> `fig_13_01_culture_threshold_sweep.svg`, `fig_13_02_culture_venn_diagrams.svg`, `fig_13_03_culture_abundance_density.svg`, `fig_13_04_culture_adjusted_concordance.svg`, `table_13_01_culture_threshold_sweep.tsv`, `table_13_02_culture_optimal_thresholds.tsv`, `table_13_03_culture_mixed_concordance.tsv`, `table_13_04_culture_venn_counts.tsv`

### Shared or Supporting Workflow Components

- `06_mixed_models_and_repeated_measures.ipynb` -> `fig_06_02_species_mixed_effects.svg`, `table_06_04_species_mixed_effects.tsv`, `table_06_05_species_mixed_status.tsv`, `table_06_06_mixed_vs_cluster_comparison.tsv`
- `07_bracken_vs_metaphlan_sensitivity.ipynb` -> `table_07_01_bracken_metaphlan_sample_depth.tsv`, `table_07_02_bracken_metaphlan_distance_summary.tsv`, `table_07_03_bracken_metaphlan_taxon_correlations.tsv`, `table_07_04_bracken_species_models_shared_samples.tsv`, `table_07_05_metaphlan_species_models.tsv`, `table_07_06_bracken_metaphlan_model_comparison.tsv`
- `08_halla_exploration.ipynb` -> `fig_08_01_halla_top25.svg`, `table_08_01_halla_method_status.tsv`, `table_08_02_halla_top_pairwise_associations.tsv`, `table_08_03_halla_significant_clusters.tsv`
- `09_maaslin2_multivariable.ipynb` -> `fig_09_01_maaslin2_summary.svg`, `table_09_01_maaslin2_model_summary.tsv`, `table_09_02_maaslin2_focus_results.tsv`
- `10_lefse_targeted_contrast.ipynb` -> `fig_10_01_lefse_summary.svg`, `table_10_01_lefse_comparison_summary.tsv`, `table_10_02_lefse_significant_features.tsv`
