# Novelty re-check (lit2)

Produced by a subagent on 2026-11-04, saved verbatim in substance by the orchestrating session (the subagent could not write files). Access legend: FULL = full text extracted and read; PAGE = page read via the fetch tool (summariser-extracted quotes); SNIP = search snippet only, not opened. "(norm.)" = maths notation normalised from extracted text. Scratch files in research/theory2/lit2/ (pdf/, raw/, cands.py) are not committed.

## 0. Bottom line

1. No paper found that scoops any headline result. The search could not reach citation indices (Semantic Scholar and OpenAlex returned 429), so this is weak evidence of absence.
2. Main threat to Thm 1.3/1.5/Cor 1.6: the Reisach et al. camera-ready. Weichwald's page lists arXiv 2605.06288 as "NeurIPS, 2026 (accepted)" (co-author self-report; no neurips.cc page found). On arXiv it is still v1 (7 May 2026), no venue.
   - Their v1 says "We do not analyze their properties beyond this", so as of v1 the necessity claim is ours.
   - Their App. A.3 proof already contains the absorption argument "for all t > p, all nodes i(t) in VT have the same relatives". Our results refine their proof idea.
   - Re-read the camera-ready the day it appears.
3. Four things in the draft are partly classical and must be credited:
   - F1 phenomenon: raw-n inconsistency and effective sample size are in Bartlett 1935, Clifford et al. 1989, Roy 1989, Runge et al. 2019 (Supplement S32), Afyouni-Smith-Nichols 2019. Afyouni's variance contains exactly our B_ab summand rho_XX,k rho_YY,k + rho_XY,k rho_XY,-k.
   - Aggregation algebra: block means converging to a static SEM with weights A_ji/(1-A_jj) is Gong et al. UAI 2017, Eqs. (5) and (10). Partial-to-full overlap with our Thm agg(b).
   - F3: residual-confounding / unreliable-covariate result of Westfall and Yarkoni 2016 and Brunner and Austin 2009.
   - Reliability identity rho_uZ rho_vZ / rho_uv = r: believed to be the classical three-indicator identity (source not checked); flag as classical.
4. Defensible as new:
   - Exact finite-T and stationary values for every summary graph, with matching necessary-and-sufficient conditions.
   - Sortability consequences of aggregation versus subsampling.
   - A non-asymptotic exact-recovery bound for the relatives set under autocorrelation.
   - Hidden-driver onset and proxy thresholds for relatives-based sorting, tested on real data.
   - Lean 4 for sortability. Causalean (Tan-Syrgkanis) exists for causal inference in general: do not write "first formal verification in causal inference".

## 1. Citers and follow-ups of arXiv 2605.06288

| Item | Finding | Access |
|---|---|---|
| arXiv | v1 only; stat.ME, cs.AI; no comments/journal-ref; /abs/2605.06288v2 returns 404 | PAGE |
| Venue | sweichwald.de/research: "NeurIPS, 2026 (accepted)", co-author self-report; no proceedings page found | PAGE (raw HTML) |
| Reisach's page | scriddie.github.io/publications still "arXiv preprint, not yet in a listed journal" (stale) | PAGE |
| Same group | "The Case for Time in Causal DAGs" (Philosophy of Science, arXiv 2501.19311); Kugler-Henckel-Weichwald (NeurIPS 2026 poster). Neither analyses unrolled-DAG sortability | PAGE / SNIP |
| Citing papers | None found. Workaround: arXiv API scans of post-7-May-2026 papers on sortability/benchmark artefacts, pdftotext + grep of 7 candidates (2609.03705, 2608.15332, 2609.31675, 2610.09782, 2606.18074, 2609.11897, 2606.17516). Only 2609.03705 cites Reisach, and only the 2021 paper | FULL (grep) |
| Follow-ups on unrolled-DAG sortability | none found | search |

What their v1 says (FULL): Thm 4 is sufficiency only ("every weakly connected component of the directed summary graph is strongly connected"); proof: fraction of edges with t <= p tends to zero; self-loops assumed everywhere, deterministic start at t=1; no stationarity, aggregation or hidden-variable analysis (they "assume causal sufficiency"); finite samples: "thresholding the empirical correlation matrix therefore yields noisy estimates of the number of relatives", no bound.

## 2. Per-result novelty tables

### 2.1 Thm 1.3/1.5/Cor 1.6 (exact value, limit 1/2 + |E_x|/(2|E_S|), O(1/T) rate, necessity)

| Closest prior | What it does | Access | Relation |
|---|---|---|---|
| Reisach et al. 2605.06288 | Thm 4 sufficiency; absorption after p steps | FULL | Partial. Our "if" half is theirs; exact per-edge value, limit for every graph, rate and necessity not in v1 |
| Lohse and Wahl, TMLR 2025 | "the presence of cycles would dilute the sortability signal and would naturally push it closer to 1/2" | FULL | Same intuition for var/R2, no theorem, no relatives. Absent |
| Herman et al., CLeaR 2025, Sec. 6 | Time-series SCM generation | FULL (grep) | Generator only. Absent |
| Loranchet-Assaad 2508.21742; Sanchez 2510.24639 | Summary-graph orientation; causal ordering | PAGE / FULL (grep) | Different question. Absent |

Verdict: partial (sufficiency only); rest absent. Risk: the camera-ready.

### 2.2 Stationary Thm 1.9/Cor 1.10

Reisach et al.: fixed start at t=1 only, no burn-in or infinite past (FULL): absent. DoTime 2607.27263, TimeGraph (KDD 2025), CauseMe: burn-in for stationarity, no sortability analysis (SNIP). Lohse and Wahl: stationary processes, var/R2 only (FULL): absent. Verdict: absent. Do not claim burn-in is new; claim the exact consequence.

### 2.3 Aggregation versus subsampling

Gong et al. UAI 2017: aggregated series follows an instantaneous causal model in the i.i.d. case; LiNGAM estimates (1-A_jj)^-1 A_ji (PAGE + raw PMC5995575): overlap with Thm agg(b); we add the Fejer-type rate (a) and the sortability consequences. Gong et al. ICML 2015: Granger fails under subsampling, non-Gaussian noise gives identifiability (FULL): identifiability only. Hyttinen et al.; Danks-Plis: subsampled constraint-based discovery (not opened). 2305.05276, DHT-CIT, 2506.14862: identifiability of summary/full-time graphs (SNIP). Lohse and Wahl: no aggregation/subsampling (FULL). Verdict: partial; sortability under aggregation/subsampling absent, long-run SEM algebra overlaps Gong 2017.

### 2.4 F1: raw-n inconsistency, effective sample size, recovery bound

| Closest prior | What it does | Access | Relation |
|---|---|---|---|
| Yule 1926; Granger-Newbold 1974 | Nonsense correlation | Crossref metadata | Classic |
| Bartlett 1935 | Effective d.o.f. for AR(1) correlation (via Afyouni) | metadata + Afyouni | Overlap (n_eff = n/B for null pairs) |
| Clifford-Richardson-Hemon 1989; Roy 1989 | Variance inflation of r; multivariate serial/cross-correlation covariance | metadata + Afyouni | Overlap |
| Afyouni-Smith-Nichols, NeuroImage 2019 | Variance with sum_k w_k(rho_XX,k rho_YY,k + rho_XY,k rho_XY,-k); ignoring it "will inflate Z-scores and produce excess false positives" | PAGE | Overlap: same quantity as our B_ab, W_ab; older corrections biased when true correlation is non-zero |
| Runge et al. 2019, Supp. S32 | "fewer degrees of freedom and will be typically wider than the assumed null distribution" | FULL | Overlap: qualitative inconsistency in PCMCI's paper |
| Reisach et al. | No autocorrelation analysis or bound | FULL | Absent |
| Non-asymptotic exact-recovery bound for the relatives set | none found | search | Absent: this is F1's novelty, not the inconsistency |

### 2.5 F2: hidden-driver saturation, n ~ z_c^2/l^2

Chandrasekaran-Parrilo-Willsky 2012 (sparse + low-rank, metadata): population fact behind Prop 2.1, related. Cevid-Buhlmann-Meinshausen 2020 (SNIP): related regime, no sorting. Westfall-Yarkoni 2016: error "highest ... when sample sizes are large and reliability is moderate" (PAGE): same power-driven logic, for regression. Runge 2019: common drivers as spurious-link source (FULL): qualitative. Hidden-driver onset n for relatives-based sorting: none found: absent. n_half = 3 + (z_c/atanh kappa)^2 is an elementary Fisher-z power calculation; keep the "textbook tools" tone.

### 2.6 F3: proxy reliability

Westfall-Yarkoni 2016: common strategies "exhibit extremely high Type I error rates" with imperfectly measured controls (PAGE): overlap in substance. Brunner-Austin 2009 (metadata only): overlap. Greenland 1980; Ogburn-VanderWeele 2012; Kuroki-Pearl 2014; Miao et al. 2018 (SNIP): related. Three-indicator identity (Spearman): not verified, flag as classical. Reisach et al.: absent. Verdict: partial; novelty is the closed form for relatives sorting, r_crit, the rank-one requirement, and the real-data test.

### 2.7 Other claims

- Real-data validation of sortability: Lohse and Wahl (FULL) measured var and R2 on river-flow data (Tran et al. 2024) and Causal Chambers: rivers "marginal variances tend to decrease along the causal order"; chambers variances increase. CausalRivers (FULL grep), CausalArena 2609.11897 (only cites varsortability), FoundCause 2606.17516 (uses Light Tunnel): no sortability analysis. A GitHub claim of a river-size shortcut (`causal-rivers-lab`) is SNIP, unreviewed. Verdict: partial; real-data var/R2 sortability exists and is comparable; pre-registered rel-sortability on real data is absent.
- Formal verification: Causalean (Tan-Syrgkanis 2607.22511, FULL; titled CausalSmith in the BibTeX stub): "a broad Lean 4 formalization of causal inference ... 8,179 machine-checked declarations", LLM-assisted. Makhija 2608.17634 (FULL): do-operator graph surgery in Lean 4. Chancelier et al. 2108.03018 (PAGE): d-separation with Coq-checked proof. Absent for sortability and causal discovery; present for causal inference generally.
- Ancestor testing in time series: Schultheiss-Ulmer-Buhlmann, J. Causal Inference 13 (2025) (metadata only): related; cite.

## 3. References the draft only read at abstract level

Reisach 2021 (2102.13647) FULL: confirmed. Reisach 2023 (2303.18211) FULL: confirmed. Ormaniec 2025 (2406.11601) FULL: "iSCMs are not Var-sortable"; R2 evidence empirical only (Sec. 5.1, Fig. 4): keep "empirically". NOTEARS (1803.01422) FULL: varsortability link from Reisach 2021. Runge 2019 (1702.07007) FULL: PCMCI confirmed; S32 has the autocorrelation false-positive statement to cite for F1. Gong 2015 FULL: identifiability under subsampling. Gong 2017: confirmed but overlaps Thm (b). Lohse-Wahl FULL: claim confirmed (Sec. 3); add their real-data results. Herman 2025: unchanged.

CLT for quadratic forms of stationary Gaussian processes: Ginovyan and Taqqu, arXiv 2102.00343 (FULL), Thm 3.1(C): "If 1/p+1/q <= 1/2, then the functional QT with an appropriate normalization obeys the CLT". Remark 3.1 attributes the discrete-time case to Avram, PTRF 79:37-45 (1988, doi 10.1007/BF00319101) and other conditions to Giraitis-Surgailis, PTRF 86:87-104 (1990, doi 10.1007/BF01207515); Grenander-Szego for p=q=infinity; Fox-Taqqu PTRF 74:213-240 (1987). Crossref metadata verified, texts not opened. Gap: these are scalar-process theorems; F1 needs a joint CLT for (var a, var b, cross-covariance) of a bivariate process. Options: cite Roy 1989 and/or Brockwell-Davis (Springer 1991) Ch. 11 (not readable; do not cite theorem numbers), or prove it. Suggested proof (the subagent's, not a checked source): g'Ag - tr A is a weighted sum of independent centred chi-square variables, asymptotically normal when ||A||_op/||A||_F -> 0 (Lindeberg); FINITE.md line 30 already gives ||A||_op <= kappa_2/(2n) with Var of order 1/n. Removes the unverified assumption; Lean candidate.

Bartlett's formula: Bartlett 1946, JRSS Suppl. 8:27-41 (Crossref verified, text not opened); Roy 1989, Biometrika 76:824-827 (metadata verified) for the multivariate extension; Afyouni et al. 2019 (PAGE) is the most accessible verified statement (their Eq. 2 "takes the same form as the estimator of Roy (1989)", citing Bartlett 1935/1946 and Clifford 1989). Our Lemma 1.2 derives the exact finite-n variance, so Bartlett is needed only for the limit and attribution.

## 4. Rewording of main.tex

1. Abstract, "exact ... condition ... also necessary": keep, put "(only sufficiency is shown in the arXiv version)" in the intro. Change "every non-strongly-connected summary graph ... stays strictly above 1/2" to "for the finite unrolled DAG", because stationary out-forests give 1/2.
2. Abstract, "raw correlation tests are inconsistent under autocorrelation": "As is classical, raw tests with nominal n are inconsistent under autocorrelation; we give the effective sample size for the relatives count and a non-asymptotic recovery bound." Add Bartlett, Clifford, Roy, Afyouni, Runge to the F1 text.
3. Aggregation (abstract and contribution 3): attribute the long-run SEM with weights A_ji/(1-lambda_j) to Gong et al. 2017; claim only the sortability consequences (var/R2 limits, rel invariance, R2 reversal) and the Fejer-type bound.
4. F3 sentence: add "in line with known residual-confounding results for unreliable covariates (Westfall and Yarkoni 2016)". Mark the reliability identity as the classical three-indicator identity.
5. Intro, "settle the question completely": "determine the value for every summary graph and horizon".
6. Lean contribution: "to our knowledge the first machine-checked development of sortability results; Lean formalisations of other parts of causal inference exist (Causalean)".
7. Real-data section and contribution 6: add Lohse and Wahl's river and chamber findings; claim the first pre-registered test of rel-sortability on these data. Avoid "first real-data validation of sortability".
8. Related work: replace with the paragraph below.
9. REFS_LOG: move Gong 2017, Bartlett/Roy, Afyouni, Westfall-Yarkoni to cited; replace the CLT "unverified" note with Ginovyan-Taqqu or the Lindeberg proof.
10. `reisach2026topological`: update to NeurIPS 2026 (accepted) once the proceedings entry exists; add a "re-check camera-ready" item to NOTES.md.

### Ready-to-paste related-work paragraph

```latex
\textbf{Related work.} Varsortability~\citep{reisach2021beware} and
$R^2$-sortability~\citep{reisach2023scale} have been analysed for standardised
SCMs~\citep{ormaniec2025standardizing}, for unit-free generators with a time-series
extension~\citep{herman2025unitless}, and, for time series, by
\citet{lohse2025sortability}, who adapt both criteria to summary graphs, note that
cyclic pairs push the signal towards $1/2$, and measure them on river-flow and Causal
Chambers data; they give no theorem and do not study relatives or observation
schemes. \citet{reisach2026topological} introduce rel-sortability and the
time-unrolled DAG, and prove that strong connectivity of every weak component of the
summary graph is sufficient for the limit $1/2$. We determine the value exactly for
every summary graph, horizon and start, and show that the condition is necessary for
the finite DAG. Block means of a VAR converge to a static linear SEM with coefficients
$A_{ji}/(1-A_{jj})$~\citep{gong2017aggregated}; that work and
\citet{gong2015subsampled} study identifiability of the causal-frequency model from
aggregated or subsampled data, not sortability. The inconsistency of a Pearson test
with the nominal sample size under autocorrelation is classical~\citep{bartlett1935,
clifford1989, roy1989} and is used in practice~\citep{runge2019detecting,
afyouni2019effective}; we add an effective sample size for the relatives count with a
non-asymptotic recovery bound, and quantify how a latent driver or an imperfect proxy
defeats it, in line with known residual-confounding results for unreliable
covariates~\citep{westfall2016confounding}. Formal libraries for causal inference
exist~\citep{tan2026causalsmith}; we machine-check the sortability results and the
deterministic core of the finite-sample theory. Identifiability from subsampled and
aggregated series is a separate question from sortability, and time-series discovery
methods such as PCMCI~\citep{runge2019detecting} test conditional independences,
whereas our finite-sample results concern only the marginal tests that define relatives.
```

## 5. Papers to cite that the draft does not (metadata verified via Crossref/arXiv API on 2026-11-04)

Must cite: Bartlett, JRSS 98(3):536-543 (1935), doi 10.2307/2342284; Clifford, Richardson, Hemon, Biometrics 45(1):123-134 (1989), doi 10.2307/2532039; Roy, Biometrika 76(4):824-827 (1989), doi 10.1093/biomet/76.4.824; Afyouni, Smith, Nichols, NeuroImage 199:609-625 (2019), doi 10.1016/j.neuroimage.2019.05.011; Westfall and Yarkoni, PLOS ONE 11(3):e0152719 (2016), doi 10.1371/journal.pone.0152719; Ginovyan and Taqqu, arXiv 2102.00343; Gong 2017 Eq. (10) and Lohse-Wahl real-data findings (already cited; use them).

Should cite: Avram 1988; Giraitis-Surgailis 1990; Tan-Syrgkanis arXiv 2607.22511; Makhija arXiv 2608.17634; Chancelier-de Lara-Heymann arXiv 2108.03018; Granger-Newbold 1974 (doi 10.1016/0304-4076(74)90034-7); Yule 1926 (doi 10.2307/2341482); Brunner-Austin 2009 (doi 10.1002/cjs.10004, metadata only); Chandrasekaran-Parrilo-Willsky 2012 (doi 10.1214/11-aos949); Brouillard et al., CLeaR 2025 (PMLR 275:834-873, arXiv 2412.01953).

Optional: Schultheiss-Ulmer-Buhlmann (doi 10.1515/jci-2024-0011); Seng et al. arXiv 2206.07195; Montagna et al. arXiv 2310.14246; Reisach-Suarez-Weichwald-Chambaz arXiv 2501.19311; Kugler-Henckel-Weichwald (NeurIPS 2026); Loranchet-Assaad arXiv 2508.21742.

BibTeX stubs (end pages for Bartlett 1935 and Clifford 1989 are from memory; verify before use):

```bibtex
@article{bartlett1935, author={Bartlett, M. S.}, title={Some Aspects of the Time-Correlation Problem in Regard to Tests of Significance}, journal={Journal of the Royal Statistical Society}, volume={98}, number={3}, pages={536--543}, year={1935}}
@article{clifford1989, author={Clifford, P. and Richardson, S. and H{\'e}mon, D.}, title={Assessing the Significance of the Correlation between Two Spatial Processes}, journal={Biometrics}, volume={45}, number={1}, pages={123--134}, year={1989}}
@article{roy1989, author={Roy, R.}, title={Asymptotic Covariance Structure of Serial Correlations in Multivariate Time Series}, journal={Biometrika}, volume={76}, number={4}, pages={824--827}, year={1989}}
@article{afyouni2019effective, author={Afyouni, Soroosh and Smith, Stephen M. and Nichols, Thomas E.}, title={Effective Degrees of Freedom of the {Pearson's} Correlation Coefficient under Autocorrelation}, journal={NeuroImage}, volume={199}, pages={609--625}, year={2019}}
@article{westfall2016confounding, author={Westfall, Jacob and Yarkoni, Tal}, title={Statistically Controlling for Confounding Constructs Is Harder than You Think}, journal={PLOS ONE}, volume={11}, number={3}, pages={e0152719}, year={2016}}
@misc{tan2026causalsmith, author={Tan, Jiyuan and Syrgkanis, Vasilis}, title={{CausalSmith}: A Formally Grounded, Self-Improving Agentic Framework for Automated Research in Causal Inference}, year={2026}, eprint={2607.22511}, archivePrefix={arXiv}}
@misc{ginovyan2021survey, author={Ginovyan, Mamikon S. and Taqqu, Murad S.}, title={A Survey on Limit Theorems for {Toeplitz} Type Quadratic Functionals of Stationary Processes and Applications}, year={2021}, eprint={2102.00343}, archivePrefix={arXiv}}
```

## 6. What could not be verified

- Citation indices (Semantic Scholar/OpenAlex rate-limited): "no citers" rests on arXiv scans and full-text grep of 7 papers.
- NeurIPS 2026 acceptance rests on one co-author web page.
- Texts not opened: Bartlett 1935/1946, Clifford 1989, Roy 1989, Yule, Granger-Newbold, Brunner-Austin, Avram, Giraitis-Surgailis, Fox-Taqqu, Brockwell-Davis, Chandrasekaran et al., Cevid et al., Danks-Plis, Hyttinen et al.
- Snippets only: arXiv 2605.04081, 2506.14862, 2305.05276, DHT-CIT, DoTime, TimeGraph, causal-rivers-lab, Statlib/StatsMLlib, Ogburn-VanderWeele, Kuroki-Pearl, Miao et al., Greenland 1980.
- The search engine is US-only and returns summaries. Repeat the manual arXiv search for "rel-sortability" and "relatives" with "unrolled" on the submission date.
