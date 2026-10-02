"""Builds response_to_reviewers.tex and response_to_reviewers_plain.txt from one source (content.tex)."""
import re, sys
src = open(sys.argv[1]).read()
parts = dict(re.findall(r"%%(\w+)\n(.*?)(?=\n%%\w+\n|\Z)", src, re.S))
parts = {k: v.strip() for k, v in parts.items()}

def to_plain(t):
    t = re.sub(r"\\begin\{itemize\}\n?|\\end\{itemize\}\n?", "", t)
    t = re.sub(r"\\item\[(\w+)\] ", "", t)
    t = t.replace("\\item ", "  - ")
    for a, b in [(r"\\textbf\{([^}]*)\}", r"\1"), (r"\\emph\{([^}]*)\}", r"\1")]:
        t = re.sub(a, b, t)
    for a, b in [("$\\pm$", "+/-"), ("\\pm", "+/-"), ("\\times", "x"), ("$-0.77$", "-0.77"), ("$+0.51$", "+0.51"), ("$d = ", "d = "),
                 ("{,}", ","), ("~", " "), ("---", " - "), ("--", "-"), ("``", '"'), ("''", '"'), ("\\%", "%")]:
        t = t.replace(a, b)
    t = re.sub(r"\$([^$]*)\$", lambda m: m.group(1).replace("\\", ""), t)
    return t.replace("$", "")

def block_tex(h, body):
    return body

tex = r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{times}
\usepackage{enumitem}
\usepackage{hyperref}
\title{Response to Reviewers --- Revision 2\\
\large SPADE: spline additive-noise DAG estimation for interpretable nonlinear\\
causal discovery in managerial decision support\\
Manuscript ARRAY-D-26-04878, \emph{Array}}
\author{}
\date{}
\begin{document}
\maketitle
""" + parts["INTRO"] + "\n"
for name, hd, rev, ans in [("Reviewer 1", "R1", "R1", "R1A"), ("Reviewer 3", "R3", "R3", "R3A"),
                           ("Reviewer 4", "R4", "R4", "R4A"), ("Reviewer 2", "R2", "R2", "R2A")]:
    r = re.sub(r"^\\item\[\w+\] ", "", parts[rev])
    tex += f"\n\\section*{{{name}}}\n\\textbf{{Reviewer comments.}} {r}\n\n\\textbf{{Response.}} {parts[ans]}\n"
tex += "\n\\end{document}\n"
open("response_to_reviewers.tex", "w").write(tex)

txt = "RESPONSE TO REVIEWERS - REVISION 2\nSPADE: spline additive-noise DAG estimation for interpretable nonlinear causal discovery in managerial decision support\nManuscript ARRAY-D-26-04878, Array\n\n" + to_plain(parts["INTRO"]) + "\n"
for name, rev, ans in [("REVIEWER 1", "R1", "R1A"), ("REVIEWER 3", "R3", "R3A"), ("REVIEWER 4", "R4", "R4A"), ("REVIEWER 2", "R2", "R2A")]:
    r = re.sub(r"^\\item\[\w+\] ", "", parts[rev])
    txt += f"\n==========\n{name}  (paste the text below into the reply box for this reviewer's comment)\n==========\n\nREVIEWER COMMENT (summary): {to_plain(r)}\n\nRESPONSE:\n{to_plain(parts[ans])}\n"
open("response_to_reviewers_plain.txt", "w").write(txt)
print(len(tex), len(txt))
