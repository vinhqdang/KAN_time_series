#!/bin/sh
# Rebuild everything: example-curve data (slow, optional), figures, tables, PDF.
cd "$(dirname "$0")"
[ -f data/example_curves.npz ] || python3.13 make_example_curves.py
python3.13 make_figures.py && python3.13 make_tables.py || exit 1
pdflatex -interaction=nonstopmode main.tex >/dev/null && bibtex main >/dev/null && pdflatex -interaction=nonstopmode main.tex >/dev/null && pdflatex -interaction=nonstopmode main.tex | grep -i "undefined\|error" ; pdfinfo main.pdf | grep Pages
