ROOT = paper

# for bibliography tests
BIB = $(ROOT).bib
TEX := $(ROOT).tex
PDF := $(ROOT).pdf

.PHONY: default

all: $(TEX) $(BIB)
	pdflatex $(ROOT)
	pdflatex $(ROOT)
	bibtex $(ROOT)
	bibtex $(ROOT)
	pdflatex $(ROOT)
	pdflatex $(ROOT)
