export TEXINPUTS := desc-tex/styles/:desc-tex/logos/:

# for bibliography tests
BIB := desc-tex/bib/lsstdesc.bib
ROOT := note
TEX := $(ROOT).tex
PDF := $(ROOT).pdf

.PHONY: default

default: $(TEX) $(BBL)
	pdflatex $(TEX)

%.bbl: %.aux $(BIB) $(NOTES)
	bibtex $<
