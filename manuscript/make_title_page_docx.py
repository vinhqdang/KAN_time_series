"""Generate the editable DOCX title page (with author details) for the
Decision Support Systems submission."""
from docx import Document
from docx.shared import Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

doc = Document()
# base font
style = doc.styles["Normal"]
style.font.name = "Times New Roman"
style.font.size = Pt(12)

def heading(text):
    p = doc.add_paragraph()
    r = p.add_run(text); r.bold = True; r.font.size = Pt(12)
    p.space_after = Pt(4)
    return p

# Title
t = doc.add_paragraph()
tr = t.add_run("SPADE: spline additive-noise DAG estimation for interpretable "
               "nonlinear causal discovery in managerial decision support")
tr.bold = True; tr.font.size = Pt(15)
t.alignment = WD_ALIGN_PARAGRAPH.CENTER

doc.add_paragraph()

# Authors with superscript affiliation markers
authors = doc.add_paragraph(); authors.alignment = WD_ALIGN_PARAGRAPH.CENTER
def author(name, sup, star=False):
    r = authors.add_run(name);
    s = authors.add_run(sup + ("," if not star else "")); s.font.superscript = True
    if star:
        a = authors.add_run("*"); a.font.superscript = True
    authors.add_run("  ")
authors.add_run("Quang-Vinh Dang")
r = authors.add_run("a,†"); r.font.superscript = True
authors.add_run(",  Dat Le")
r = authors.add_run("b"); r.font.superscript = True
authors.add_run(",  Minh Ngoc Dinh")
r = authors.add_run("c,*"); r.font.superscript = True

# Affiliations
aff = doc.add_paragraph(); aff.alignment = WD_ALIGN_PARAGRAPH.CENTER
for mark, text in [("a", "British University Vietnam, Hung Yen, Vietnam"),
                   ("b", "Ho Chi Minh City University of Economics and Finance, Ho Chi Minh City, Vietnam"),
                   ("c", "Millennia Education, Ho Chi Minh City, Vietnam")]:
    s = aff.add_run(mark); s.font.superscript = True; s.italic = True
    rr = aff.add_run(" " + text + "\n"); rr.italic = True

# Corresponding author
corr = doc.add_paragraph()
corr.add_run("* Corresponding author: Minh Ngoc Dinh. ").bold = True
corr.add_run("Email addresses: vinh.dq4@buv.edu.vn (Quang-Vinh Dang), "
             "datla@uef.edu.vn (Dat Le), minh.dinh@maeducation.com (Minh Ngoc Dinh).")
sub = doc.add_paragraph()
sub.add_run("† Submitting author. ").bold = True
sub.add_run("Manuscript submitted by Quang-Vinh Dang on behalf of all authors, for convenience "
            "of the submission process.")

doc.add_paragraph()

# Abstract
heading("Abstract")
abstract = 'Understanding causal relationships in organizational time-series data is critical for managerial decision-making, yet forecasting systems often trade interpretability for accuracy. We introduce SPADE (SPline Additive-noise DAG Estimation), an information-systems artifact built on Kolmogorov-Arnold networks that, from a single differentiable model, learns an interpretable non-linear causal graph and produces one-step forecasts. SPADE predicts each variable only through per-(cause, lag) B-spline edge functions of its candidate parents—an information bottleneck that makes the learned structure identifiable—with an edge-level group-lasso for sparsity and an optional acyclicity constraint on contemporaneous edges. We evaluate it against recent differentiable DAG learners (DAGMA, GraN-DAG, GOLEM, non-linear NOTEARS), score-matching ordering methods (SCORE, NoGAM), and temporal methods (PCMCI, VAR-LiNGAM), using threshold-free AUROC/AUPRC, paired significance tests, and hyperparameters tuned only on held-out seeds. On non-linear instantaneous DAGs SPADE attains AUROC ≈ 0.92-0.95 across widths din6,10,20, clearly ahead of every fixed-architecture gradient-based learner and of SCORE; against NoGAM the result is mixed (ahead at two of three widths, behind at d=10), while SPADE trains 275-930x faster and exposes the edge functions that neither score-matching method provides. A variant with a dense backbone collapses to chance, confirming that the bottleneck drives identifiability. On lagged structure SPADE is perfect on linear graphs and competitive on non-linear ones, trailing the best conditioning-based methods; forecasting is competitive but not best (MSE approx0.029 on an 8-asset financial panel). On three public non-financial datasets (bike-sharing demand, daily and hourly, and Beijing air quality), scored against domain-certain background knowledge, SPADE leads on daily data, ties linear NOTEARS on 2,000-row hourly samples, and leads on the full hourly series (n=17,303 and 41,543). On the financial panel, the learned graph is stable well above chance (permutation test, 4.7x), but a stationary (returns) re-analysis collapses it to an uninformative graph, so the levels-based graph reflects shared non-stationary co-movement rather than validated causal structure. SPADE\'s spline edge functions expose linear, saturating, and threshold-like dependencies, supporting a "glass-box" decision-support tool for moderate-dimensional, non-linear business systems whose boundary conditions we state explicitly.'
doc.add_paragraph(abstract)

# Keywords
kw = doc.add_paragraph()
kw.add_run("Keywords: ").bold = True
kw.add_run("Causal discovery; Kolmogorov–Arnold networks; Interpretable machine "
           "learning; Time-series forecasting; Decision support systems; Directed acyclic graphs")

doc.add_paragraph()

# Declarations
heading("Declarations")
d1 = doc.add_paragraph(); d1.add_run("Funding. ").bold = True
d1.add_run("This research received no specific grant from funding agencies in the public, "
           "commercial, or not-for-profit sectors. ")
d2 = doc.add_paragraph(); d2.add_run("Conflict of interest. ").bold = True
d2.add_run("The authors declare no competing interests.")
d3 = doc.add_paragraph(); d3.add_run("Data and code availability. ").bold = True
d3.add_run("Code and data to reproduce all results are available at https://github.com/vinhqdang/KAN_time_series.")

doc.save("title_page.docx")
print("wrote title_page.docx")
