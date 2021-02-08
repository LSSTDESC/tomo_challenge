ROOT = paper

# for bibliography tests
BIB = $(ROOT).bib
TEX := $(ROOT).tex
PDF := $(ROOT).pdf

.PHONY: default plots all

all: plots $(TEX) $(BIB)
	pdflatex $(ROOT)
	pdflatex $(ROOT)
	bibtex $(ROOT)
	bibtex $(ROOT)
	pdflatex $(ROOT)
	pdflatex $(ROOT)

plots: results/make_figures.py
	cd results; python make_figures.py

clean:
	rm -f $(ROOT).bbl $(ROOT).aux $(ROOT).blg $(ROOT).out $(ROOT).log $(ROOT).pdf texput.log