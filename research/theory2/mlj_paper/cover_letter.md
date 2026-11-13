Quang-Vinh Dang
British University Vietnam, Hung Yen, Vietnam
vinh.dq4@buv.edu.vn

13 November 2026

To the Editors of *Machine Learning*

Dear Editors,

Please consider the manuscript "Rel-Sortability of Time-Unrolled Causal Graphs" for publication in *Machine Learning*.

**What the paper is about.** Benchmarks for causal discovery are usually simulated, and recent work has shown that simulation choices can make a benchmark easier than it looks. Reisach, Chambaz, Blanchard and Weichwald (arXiv:2605.06288) proposed time-unrolled DAGs, built from a summary graph, as a way to generate benchmark graphs, and introduced rel-sortability, a count of how well the relatives of a variable (variables joined to it by an open path) reveal the order of the graph. We ask what happens to that quantity when the process is unrolled over a finite horizon, and what it means for methods that never see the time labels. The thesis is that hiding the labels does not hide the time order: the relatives structure of the unrolled graph carries a graph-determined footprint of time.

**What we show.**

1. We compute the rel-sortability of every edge exactly, for every summary graph with self-loops and every horizon, together with its limit, and a burn-in (the largest finite distance in the summary graph) after which the value no longer depends on the start. The graph statements are checked in Lean 4. The burn-in is minimal for directed chains and can be too long for other graphs.
2. A fixed start inflates the footprint, and strongly connected summary graphs are not protected at short horizons. In simulations the fixed start raised the order accuracy of a relatives-sorting baseline by about 0.13.
3. We give conditions under which relatives can be estimated from one autocorrelated record, and results on what is identifiable without labels.
4. In controlled tests the gain follows the correlations a test can detect, not the population relatives pattern. The share of the exact gap that is realised grows with sample size (0.18 at n = 200, 0.76 at n = 25,000). Tuned DAGMA, NOTEARS and GOLEM-NV overtook FLOP on in-trees under the fixed start and lost to it after a burn-in; variance is involved but does not explain this alone.

**What we do not claim.** The paper proposes no new causal-discovery algorithm. Support from real data (CausalRivers, Causal Chambers) is mostly negative or not discriminating, and we say so. The benchmark construction is hypothetical: we found no deployed generator that uses it as shipped. Results are for lag-one dynamics. The paper has a table that lists each claim with its evidence and status (proved on paper, checked in Lean, simulated, post hoc, or failed).

**Fit with the journal.** The paper combines exact theory, finite-sample analysis, designed simulation studies and real-data tests on how causal-discovery methods are evaluated, which falls within the scope of *Machine Learning*. Each simulation study was planned in a time-stamped plan committed to the public repository before its results (client commit times; server push times could be retrieved for only some plans, as the manuscript states), and post hoc analyses are labelled as such.

**Related work and other submissions.** The manuscript extends a preprint by other authors (cited above). The author's earlier paper on SPADE (Array 32, 101335, 2026) is a learner paper, is cited once, and shares no text or results with this manuscript. The manuscript is not published and is not under consideration elsewhere.

**Use of AI tools.** AI tools were used to support the work and the writing, as described in Section 6.4 of the manuscript. The author reviewed the content and takes full responsibility for it. AI systems are not listed as authors.

**Data and code.** No new data were collected. All code, including the Lean development, plans and result files, is at https://github.com/vinhqdang/KAN_time_series (directory research/theory2). Public data sets and their checksums are listed in DATA.md.

**Competing interests.** The author declares no competing interests, and received no funding for this work.

Thank you for considering the manuscript.

Sincerely,

Quang-Vinh Dang
