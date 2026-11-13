# Build: make   (pdflatex + bibtex; checks the log for undefined references)
main.pdf: main.tex appendix.tex refs.bib sn-jnl.cls figs/tikzstyle.tex $(wildcard figs/tikz/*.tex) $(wildcard figs/fig_*.pdf)
	pdflatex -interaction=nonstopmode main.tex
	bibtex main
	pdflatex -interaction=nonstopmode main.tex
	pdflatex -interaction=nonstopmode main.tex
	@! grep -E "(Reference|Citation) .* undefined|undefined references|^!" main.log || (echo "UNDEFINED REFERENCES" && false)

# Regenerate the figures (not needed for `make`: the generated PDFs and tikz/*.tex are committed).
# gen_tikz.py is pure Python 3; the data figures need numpy, pandas, scipy, matplotlib (PYV, e.g. the b1y virtualenv).
PYV ?= /home/user/venv312/bin/python
figs:
	cd figs && python3 gen_tikz.py
	cd figs && $(PYV) fig_theory.py && $(PYV) fig_q1.py && $(PYV) fig_learners.py && $(PYV) fig_finite.py && $(PYV) make_fig_oracle.py

.PHONY: figs clean
clean:
	rm -f main.aux main.bbl main.blg main.log main.out
