# Build: make   (pdflatex + bibtex; checks the log for undefined references)
main.pdf: main.tex appendix.tex refs.bib icml2026.sty icml2026.bst
	pdflatex -interaction=nonstopmode main.tex
	bibtex main
	pdflatex -interaction=nonstopmode main.tex
	pdflatex -interaction=nonstopmode main.tex
	@! grep -i "undefined" main.log || (echo "UNDEFINED REFERENCES" && false)

clean:
	rm -f main.aux main.bbl main.blg main.log main.out
