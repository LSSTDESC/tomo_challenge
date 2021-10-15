ROOT = paper

PLOTS = results/edge_comparison.pdf \
		results/g_band_loss.pdf \
		results/metric_comparisons.pdf \
		results/tables/table_FOM_DETF_3x2_dc2.tex \
		results/initial_data.pdf \
		results/colour_colour.pdf \
		results/nzbinned_dc2_funbins.pdf \
		results/nzoverlap_dc2_funbins.pdf \
		results/metric_grid_dc2_griz.pdf \
		results/metric_grid_dc2_riz.pdf \
		results/metric_grid_buzzard_griz.pdf \
		results/metric_grid_buzzard_riz.pdf \
		results/rank_grid.pdf \
		results/9bin_dc2.tex \
		results/9bin_buzzard.tex

# for bibliography tests
BIB = $(ROOT).bib
TEX := $(ROOT).tex
PDF := $(ROOT).pdf
AUTH = authors2.tex


all: $(PDF)

$(PDF): $(TEX) $(BIB) $(AUTH)
	pdflatex $(ROOT)
	bibtex $(ROOT)
	bibtex $(ROOT)
	pdflatex $(ROOT)
	pdflatex $(ROOT)
	pdflatex $(ROOT)

plots: $(PLOTS)

arxiv.tar.gz:  $(TEX)
	tar -zvcf arxiv.tar.gz $(PLOTS) $(AUTH) $(TEX) $(ROOT).bbl apjfonts.sty aasjournal.bst openjournal.cls

results/%.pdf: results/make_figures.py
	cd results; python3 make_figures.py

results/tables/%.tex: results/make_figures.py
	cd results; python3 make_figures.py

clean:
	rm -f $(ROOT).bbl $(ROOT).aux $(ROOT).blg $(ROOT).out $(ROOT).log $(ROOT).pdf texput.log

.PHONY: plots all
