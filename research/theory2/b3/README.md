# B3: the source-SCC rule against tie rules, on held-out data with several sources per weak component

Analysis plan: `PLAN.md`, commit **827179d**, pushed to `origin/main` before any byte of the held-out csv members was read. The OSF-formatted text is `OSF_PREREG.md`. **It was not posted to OSF before data access** (the agent cannot post it), so the plan is internally time-stamped only.

Files:
- Code: `graphs_b3.py`, `predict_b3.py` (graph-only), `dev_b3.py` (development on study-3 data), `fetch_b3.py` (download, hashes, design check), `test_b3.py` (the pre-registered analysis, run once).
- Post hoc: `design_check2.py`, `explore_b3.py`.
- Outputs in `results/`:
  - `predict_b3_results.txt`, `dev_b3.log` (before the plan);
  - `fetch_report.json` (SHA-256, rows, design check);
  - `results_b3.json`, `test_b3.log` (the test);
  - `explore_b3.log` (post hoc).
- Raw data are not committed. Reproduce with `OMP_NUM_THREADS=1 python3 -I fetch_b3.py <dir> results/fetch_report.json`, then `OMP_NUM_THREADS=1 python3 -I test_b3.py <dir> results` (5 s on one core).

## Data
- **Primary H1, real time series: Causal Chambers `lt_camera_walks_v1`, experiments `ar_1_uniform_ref`, `ar_1_uniform_pol_1_90` and `ar_1_uniform_pol_12_30`.**
  - 3 × 30,000 steps. Five independent stationary AR(1) actuators (R, G, B, θ1, θ2) drive seven light sensors and two angle sensors.
  - In the relatives graph G_lin, the colour block {R, G, B, 7 sensors} is one weak component with **three source SCCs**.
  - 30 units of 3,000 steps; median n_eff per unit 1,222.
- **Co-primary H2, real snapshot data: `lt_camera_v1/scm_5_red` and `scm_5_blue`** (2 × 10,000 i.i.d. rows).
  - The actuator collider R → G ← B; the interventions on the sources leave the graph unchanged.
  - 20 units of 1,000 rows.
- **Secondary S4:** `scm_5_green`, `scm_4_green`, `scm_4_blue` (two or three sources after the intervention). **Control:** `scm_4_red` (chain, one source).
- None of these files had been downloaded by the project before. SHA-256 values are in `results/fetch_report.json`; the zip CRC-32 values matched the plan.

## Verdicts against the pre-registered rules

D\* is the set of edge instances where the theorem (Thm 1.9 / Cor 1.10, Section-6 rule) and the weak-component-size rule disagree. On D\* the theorem predicts 1 and both the weak-component rule and constant ½ predict ½. Δ is the theorem's mean loss minus the competitor's; the criterion is Δ ≤ −0.10 and Holm p ≤ 0.025.

| criterion | threshold | result | verdict |
|---|---|---|---|
| H1 prerequisites (G_lin) | non-relatives declared < 0.5, relatives detected ≥ 0.8 | 0.054 [0.026, 0.082]; 0.990 [0.985, 0.995] | pass |
| **H1** (ar_1, D\* = 2,361 instances, 30 units) vs constant ½ = weak-component rule (identical on D\*) | Δ ≤ −0.10, Holm p ≤ 0.025 | Δ = **−0.498** [−0.500, −0.495]; theorem better in 30/30 units; p ≈ 1e-5. Observed 1 on 2,356 instances, ½ on 5 | **pass** |
| H1: other competitors on D\* | – | the eight old predictors and theorem-on-G_D agree with the theorem on every D\* instance (untestable, as computed before the data) | – |
| H2 prerequisites | as above | 0.000; 1.000 | pass |
| **H2** (scm_5 collider, D\* = 2,078 instances, 20 units) vs constant ½ = weak-component rule | as above | Δ = **−0.500**, 20/20 units, Holm p = 2e-5. Observed 1 on 2,078/2,078 | **pass** |
| **H2** vs sensor indicator (264 instances: R → G, B → G with the other source in S) | as above | Δ = **−0.500**, 20/20, Holm p = 2e-5 | **pass** |
| H2 readout: collider edges into G | – | other source in S: observed 1 on 264/264. Other source absent: observed ½ on 123/123 | – |
| **S1 (H1)**: theorem-½ instances whose own pair is declared dependent (1,251), vs every disagreeing competitor | Δ < 0, Holm p ≤ 0.025 | sensor indicator and the full-graph rules: Δ = −0.428 [−0.469, −0.384], 30/30. Subgraph rules: −0.430, 30/30. Theorem on G_D: Δ = −0.208 [−0.363, −0.038], 21/30, Holm p = 0.021 | **pass** (the G_D comparison only narrowly) |
| S1 (H2) | as above | Δ = −0.500 for all, 20/20 | pass |
| **S3** (H1): within-edge switch, colour edge with minus without a second source in S | ≥ 0.25, p ≤ 0.025 | **+0.453**, 30/30 units, p ≈ 1e-5 | pass |
| **S6** mechanism share on D\* | ≥ 0.5 (K4) | H1 0.992; H2 1.000 | pass |
| K1 / K2 / K3 / K4 (H1, H2) | – | none triggered | – |

**Pre-registered claims:**
- **H1 claim earned.** Thm 1.9 / Cor 1.10 separated from the trivial tie rules on held-out real time series.
- **H1 full separation earned.** H1 and S1 pass, K2 and K4 are not triggered: the theorem beats the tie rules where it predicts strict order, and beats the eight naive sort rules where it predicts ties on detected edges.
- **H2 collider claim earned**, in the same two parts.

Exact-match rates on all H1 instances (S2, reported):

| predictor | exact match |
|---|---|
| theorem | 0.976 |
| theorem on G_D | 0.954 |
| eight naive rules | 0.707–0.723 |
| weak-component rule | 0.421 |
| constant ½ | 0.277 |

In H2: theorem 1.000, constant ½ 0.530.

## Secondary results that went against the theorem or against my predictions (reported plainly)

1. **scm_4_blue fails, and the tie rule is significantly better.** S4 per set; no verdict was attached, but this is a K2-type result.
   - On D\* (1,563 instances) the theorem matches 0.385 and the tie rule 0.615. Δ = +0.115 [0.103, 0.125], theorem better in 0/10 units, reverse p = 0.001 (the smallest attainable with 10 units).
   - Post hoc cause (`explore_b3.log`): under do(blue), B ranges only 1–13 on the 0–255 scale (sd 1.6). None of B's seven sensor edges is ever detected (|r| ≤ 0.03), so B is a source on paper but not in the data.
   - The theorem applied to the published graph is then wrong, and the tie rule is right. The pre-registered prerequisite (pooled relatives detection 0.837 ≥ 0.8) **did not catch this**: one source whose edges are all undetectable can hide inside a pooled detection rate.
   - The pooled S4 still passes (Δ = −0.295 [−0.390, −0.194], but only 20/30 units). scm_5_green (D\* match 0.977) and scm_4_green (1.000) pass individually.
2. **The Malus linear-zero refinement (G_lin) failed in one of the three H1 experiments.**
   - In `ar_1_uniform_pol_1_90` (θ2 around 90°, crossed polarisers), θ1 and θ2 are clearly correlated with ir_3 and vis_3 (median r = +0.28 and −0.21; declared in 90–100% of units). I had predicted these pairs would not be detected.
   - So 16% of G_lin non-relative pairs are declared there, against 0.000 in the other two experiments. red–vis_3 is never detected there (crossed polarisers darken the third sensor pair).
   - On the 74 instances of that experiment where G_lin and G_D disagree, the theorem on G_D matches 0.959 and on G_lin 0.041. In `ref` and `pol_12_30` it is the reverse (1.000 vs 0.000, 162 instances).
   - Pooled, S5 favours G_lin (Δ = −0.199). **The refinement is design-dependent and should not be presented as generally valid.** It does not affect H1: on D\* both graphs give the same prediction, as fixed in the plan.
3. **Mismatches inside H1:** 101 of 4,238 instances.
   - Mostly theorem ½, observed 1, at colour → ir_3/vis_3 edges with a polariser or angle node in S: the Malus residual of item 2.
   - 5 D\* instances observed ½.

## Deviations
- **D1 (design check, not gating).** The regenerated protocol matches the recorded inputs of `ar_1_uniform_ref` exactly (colour difference 0; angles within 1.2e-6). For `pol_1_90` and `pol_12_30` the published generator's seeds do not reproduce the recorded series (`fetch_report.json`).
  - A post hoc input-only check (`design_check2.py`) found that ranges and centres agree with the generator: θ2 in [60, 120] in `pol_1_90`; θ1 and θ2 in [0, 60] in `pol_12_30`; Δ = 30. The dataset README is wrong on both points.
  - These two experiments were evidently generated with other seeds, so independence of their five input series rests on the generator's structure, not on regeneration. The plan did not make this check gating, and nothing was changed.
- **D2.** `python3` on the PATH has no numpy, so all scripts were run with `/usr/bin/python3 -I` (3.13). No code change.
- **D3.** The repository's `.gitignore` ignores `*.log`. `results/dev_b3.log`, `test_b3.log` and `explore_b3.log` were force-added.
- No threshold, unit, seed, K, node set, graph or competitor changed after the data were read. `test_b3.py` ran once on the held-out data; its code is byte-identical to the plan commit.

## Assessment: does this answer DA CRITICAL 3 / C5?
- **What it removes.** The specific objection that the theorem's success is indistinguishable from "½ iff same weak component, else compare component sizes".
  - On held-out real time series from five independent sources, the two rules disagree on 56% of instances. There the estimated counts side with the theorem 99.8% of the time (2,356/2,361).
  - On the instances where the theorem predicts a tie, it beats the eight naive sort rules (92.8% exact against 7%).
  - The constant-½, weak-component and theorem-on-G_D competitors were fixed in advance.
  - The scm_5 collider replicates exactly on held-out data: the same edge R → G is sorted when B is observed and tied when it is not, 387/387.
  - Estimator failure (no power or saturation) pushes the observed value toward the tie rules on D\*, so the D\* pass cannot come from power-induced ties. That was E1's weakness.
  - There are now 30 units from 3 series (plus 20 snapshot units), not 10 units from one series.
- **What it does not remove.**
  - (a) The test is low-risk. A development check on the same device (disclosed in the plan) showed that every colour edge is detected and that independent colours never are. The outcome was expected, and the theorem is a deterministic function of the graph, so the test checks that the estimated counts follow the source-SCC structure of the published graph.
  - (b) Where the published graph is unfaithful, the theorem fails as plainly as it succeeds elsewhere: scm_4_blue, an almost-constant source, where the tie rule wins; and the Malus refinement in pol_1_90.
  - (c) There was no external registration before data access, and the plan and results were committed by the same agent within an hour.
  - (d) All data still come from one light tunnel.
- **My view.** B3 answers the "only ties" part of DA CRITICAL 3 cleanly. It does not turn the real-data evidence into a test of anything beyond "the relatives estimator recovers the published graph's source structure when the graph is faithful". The paper should say exactly that and report scm_4_blue and pol_1_90 next to the passes.

## Paragraph a paper could use

> *Discriminating test (B3).* E1 could not separate the source-SCC rule from the trivial rule "½ iff same weak component", because each weak component of its graph had one source. We therefore ran a further test, with a plan fixed and pushed to the project repository before data access (internally time-stamped; not externally registered). It used held-out Causal Chambers data whose components contain several independent sources: three light-tunnel time series driven by five independent AR(1) actuators (30 units of 3,000 steps), and two snapshot sets from the actuator collider R → G ← B (20 units). The competitors were constant ½, the weak-component-size rule, the theorem applied to the design graph, and the eight naive rules of E1. On the 2,361 time-series instances where the theorem and the weak-component rule disagree, the estimated relatives counts were strictly sorted as the theorem predicts in 99.8% of cases (Δ = −0.50 against both tie rules, 30/30 units, Holm p ≈ 10⁻⁵). On the 1,251 instances where the theorem predicts a tie and the edge itself is detected, the theorem beat the naive sort rules (Δ = −0.43, 30/30 units). In the collider data the edge R → G was sorted whenever B was observed (264/264) and tied whenever it was not (123/123). The test was low-risk: development data from the same device had shown every colour edge to be detectable. Where the published graph is unfaithful, the rule fails accordingly. In one interventional set the intervened source varied over only 1–13 of 255 intensity levels; its edges were undetectable and the tie rule out-predicted the theorem (Δ = +0.12, 0/10 units). In one of the three time series the polariser linear-zero refinement failed, and the design graph predicted better there.
