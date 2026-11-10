# Build: make   (pdflatex + bibtex; checks the log for undefined references)
main.pdf: main.tex appendix.tex refs.bib sn-jnl.cls
	pdflatex -interaction=nonstopmode main.tex
	bibtex main
	pdflatex -interaction=nonstopmode main.tex
	pdflatex -interaction=nonstopmode main.tex
	@! grep -E "(Reference|Citation) .* undefined|undefined references|^!" main.log || (echo "UNDEFINED REFERENCES" && false)

clean:
	rm -f main.aux main.bbl main.blg main.log main.out
