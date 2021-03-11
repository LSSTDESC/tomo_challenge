ROOT = paper

PLOTS = results/edge_comparison.pdf \
		results/g_band_loss.pdf \
		results/metric_comparisons.pdf \
		results/tables/table_FOM_DETF_3x2_dc2.tex \
		results/tables/table_FOM_ww_dc2.tex \
		results/tables/table_FOM_ww_buzzard.tex

# for bibliography tests
BIB = $(ROOT).bib
TEX := $(ROOT).tex
PDF := $(ROOT).pdf
AUTH = authors.tex


all: $(PDF)

$(PDF): $(PLOTS) $(TEX) $(BIB) $(AUTH)
	pdflatex $(ROOT)
	bibtex $(ROOT)
	bibtex $(ROOT)
	pdflatex $(ROOT)
	pdflatex $(ROOT)
	pdflatex $(ROOT)

plots: $(PLOTS)

results/%.pdf: results/make_figures.py
	cd results; python make_figures.py

results/tables/%.tex: results/make_figures.py
	cd results; python make_figures.py

clean:
	rm -f $(ROOT).bbl $(ROOT).aux $(ROOT).blg $(ROOT).out $(ROOT).log $(ROOT).pdf texput.log

.PHONY: plots all
