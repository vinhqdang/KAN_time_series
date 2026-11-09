# OSF Preregistration (OSF Standard Pre-Data Collection / Existing-Data template): B3

*Prepared for posting on osf.io by the authors. The agent that wrote it cannot post to OSF. If this text is posted, it must be posted **before** the analysis commit described in `PLAN.md`; otherwise the registration is retrospective and must be labelled as such. The authoritative, internally time-stamped version is `PLAN.md` in the project repository, at the commit that adds this file.*

## Study information

**Title.** Do estimated relatives counts on real physical data follow the source-SCC structure of the stationary relatives-sortability theorem, or the weak-component structure? A discriminating test on held-out Causal Chambers data.

**Description.** For a stationary process on a summary graph, the stationary relatives-sortability value of an edge (i, j) is 1 if the set of source strongly connected components (SCCs) reaching j differs from the set reaching i, and ½ otherwise. Stated on an observed node set S: the edge is strictly sorted iff S contains a relative of j that is not a relative of i. A previous test (E1) could not distinguish this rule from the trivial rule "½ iff same weak component", because every weak component in its graph had one source. Here we use held-out data whose weak components contain several independent sources, so that the two rules disagree on about half of the instances.

**Hypotheses.**
- **H1 (primary).** On the multivariate time series `lt_camera_walks_v1/ar_1_uniform_{ref,pol_1_90,pol_12_30}` (five independent AR(1) actuators driving seven light sensors), take the edge instances where the source-SCC rule predicts 1 and the weak-component rule predicts ½ (the set D\*). There, the observed edge values from a marginal-dependence relatives estimator agree better with the source-SCC rule than with the constant-½ rule and the weak-component rule.
- **H2 (co-primary).** The same holds on `lt_camera_v1/scm_5_red` and `scm_5_blue` (i.i.d. data from the actuator collider R → G ← B), against the constant-½ / weak-component rule and against the sensor-indicator rule.

Directional, one-sided.

## Design plan

**Study type.** Observational analysis of existing public data from a physical testbed. The interventions in the scm_* sets were made by the data creators.

**Blinding.** No values of the listed files had been loaded when this plan was written. Prior exposure to other data from the same device is disclosed (Existing data, below).

**Design.**
- Five actuators per ar_1 experiment and two or three source actuators per scm experiment.
- The published ground-truth graph of the light tunnel (causalchamber 0.2.8) plus the edges among the actuators given by the generator code.
- For H1, the four polariser → ir_3/vis_3 edges (Malus' law, an even function of the angle difference under a symmetric design) are removed from the relatives graph, as in a previously stated lemma. This choice has no effect on the H1 instance set or on its predictions; it affects secondary analyses only.

## Sampling plan

**Existing data.** Registration prior to analysis of the data. The data are public (CC BY 4.0) and were not downloaded by the authors before this plan.
- The authors have analysed other datasets from the same light tunnel: `lt_walks_v1/actuators_white`, `lt_crl_benchmark_v1/citris_1` and `buchholz_1_obs`, and `lt_camera_v1/scm_{1,2,4,5}_reference`.
- A development check on `actuators_white` (already analysed) showed that all 21 colour → sensor edges are detected at n = 3,000 and that independent colours are never declared dependent. The primary outcome was therefore expected.

**Explanation of existing data.** Integrity is supported by:
- CRC-32 values of each zip member, recorded from the archive's central directory before download: ar_1_uniform_ref b153930d, ar_1_uniform_pol_1_90 894684f7, ar_1_uniform_pol_12_30 dcab420f, scm_5_red a3d5a406, scm_5_blue 08eb6c50, scm_5_green 158bdeb3, scm_4_green de1c4f3d, scm_4_blue efc3eb05, scm_4_red 9fdbe1ed;
- SHA-256 values recorded at first download, before any statistic;
- regeneration of the ar_1 input protocols from the published seeded generator.

**Data collection procedures.** HTTP range download of the single csv members from the Causal Chambers S3 archive (`lt_camera_walks_v1.zip`, `lt_camera_v1.zip`).

**Sample size.**
- H1: 3 × 30,000 time steps, cut into 30 units of 3,000 consecutive steps.
- H2: 2 × 10,000 i.i.d. rows, cut into 20 units of 1,000 rows.
- Per unit: the full node set plus 10 random node subsets of size 5 and 10 of size 8 (fixed seeds).
- Expected D\* instances, graph-only: 2,361 (H1) and 2,078 (H2).

**Sample size rationale.** The number of units fixes the attainable sign-flip p (about 10⁻⁵ with 100,000 Monte Carlo flips). D\* covers 56% (H1) and 47% (H2) of all instances.

**Stopping rule.** The analysis is run once on all listed data.

## Variables

**Measured variables.**
- For each unit, the pairwise "declared dependent" decisions of an effective-sample-size Fisher-z test: Bartlett correction with K = 50 (H1) or 20 (H2), lag 0, α = 0.01, Bonferroni over the pairs of each node set.
- For node v in S: ρ̂(v) = 1 + the number of declared partners in S.
- For each induced true edge (u, v): o = 1, ½ or 0 according as ρ̂(u) <, =, > ρ̂(v).

**Predictors compared.**
- The source-SCC rule (the theorem).
- Constant ½.
- The weak-component-size rule.
- The theorem applied to the unmodified design graph.
- Eight naive rules: sensor indicator; ancestors (full graph, subgraph); in-degree (full, subgraph); longest-path depth (full, subgraph); every edge 1.

## Analysis plan

**Statistical models.**
- Loss |o − p| for predicted value p.
- For each competitor C, take the instances in the analysed set where the theorem and C disagree. Per unit, Δ = mean loss(theorem) − mean loss(C).
- One-sided sign-flip test of mean Δ < 0 across units: exact for ≤ 16 units, otherwise 100,000 Monte Carlo flips with seed 1.
- Unit-cluster bootstrap CI (2,000 resamples).
- Competitors with identical predictions on the analysed set are merged.

**Inference criteria.**
- Holm correction over the distinct testable competitors within each hypothesis, at family-wise α = 0.025 (Bonferroni over H1 and H2).
- A hypothesis passes iff, for every testable competitor on D\*, pooled Δ ≤ −0.10 and the Holm-adjusted p ≤ 0.025.
- On D\*, Δ against a tie rule equals 0.5 − P(o = 1), so the criterion requires strict sorting on at least 60% of D\* instances.

**Data exclusion.** Rows with missing values are dropped. A node that is constant within a unit is removed from that unit. Both are reported.

**Missing data.** As above.

**Other planned analyses (secondary; criteria fixed).**
- **Prerequisites.** Non-relative pairs declared dependent < 0.5 and relative pairs detected ≥ 0.8. These gate only the interpretation of a failure.
- **S1.** Instances where the theorem predicts ½ and the edge's own pair is declared dependent: the theorem against every disagreeing competitor (Δ < 0, Holm p ≤ 0.025).
- **S2.** All instances, E1-style.
- **S3 (H1).** Within-edge switch: per unit, the mean o on colour edges with a second colour source in S minus without; pass iff ≥ 0.25 and p ≤ 0.025.
- **S4.** Interventional multi-source sets scm_5_green, scm_4_green and scm_4_blue (pooled), plus scm_4_red as a single-source control.
- **S5.** The theorem on the refined graph against the theorem on the design graph.
- **S6.** A mechanism share on D\*.

**Kill rules.**
- **K1.** A failure with failed prerequisites is uninformative; a failure with passed prerequisites is a refutation.
- **K2.** A competitor significantly better on D\* must be reported as out-predicting the theorem.
- **K3.** A pass of H1 with S1 failing restricts the claim to "beats tie rules where it predicts strict order".
- **K4.** A mechanism share below 0.5 means the pass is not attributed to the source-SCC mechanism.

## Other

- Code (`graphs_b3.py`, `predict_b3.py`, `test_b3.py`, `fetch_b3.py`) and graph-only predictions are committed together with the plan.
- **Honest predictions:** H1 and H2 pass; S1 for H1 is uncertain (a possible Malus residual at ir_3/vis_3).
- This test cannot make the theorem a risky prediction on this device. The theorem is a deterministic function of the graph. What is tested is whether estimated counts on real held-out data follow the source-SCC structure rather than the weak-component structure.
