PAPER_DIR := paper
FIGURE_DIR := $(PAPER_DIR)/figures
FIGURE_SCRIPT := scripts/make_figures.py
VERIFY_SCRIPT := scripts/verify_math.py
PDF := $(PAPER_DIR)/main.pdf
ARXIV_DIR := build/arxiv-source
ARXIV_TARBALL := arxiv-submission.tar.gz

.PHONY: all figures paper verify data clean distclean arxiv

all: figures paper

verify:
	uv run python $(VERIFY_SCRIPT)

# Regenerate the empirical summaries from the public logs (downloads + grades;
# cached afterward so figures rebuild without re-downloading). Optional: the
# cached JSON summaries are committed, so `make figures` works without this.
data:
	uv run python scripts/analyze_brown.py
	uv run python scripts/analyze_rhow.py

figures:
	uv run python $(FIGURE_SCRIPT)
	# README raster, extracted from the paper's own lead figure (not a separate export)
	@if command -v pdftocairo >/dev/null 2>&1; then \
		pdftocairo -png -r 200 -singlefile $(FIGURE_DIR)/two_ceilings.pdf $(FIGURE_DIR)/two_ceilings; \
	elif command -v pdftoppm >/dev/null 2>&1; then \
		pdftoppm -png -r 200 -singlefile $(FIGURE_DIR)/two_ceilings.pdf $(FIGURE_DIR)/two_ceilings; \
	else echo "no pdftocairo/pdftoppm on PATH; skipping README raster"; fi

paper:
	cd $(PAPER_DIR) && if command -v tectonic >/dev/null 2>&1; then tectonic main.tex; else latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex; fi

clean:
	find $(PAPER_DIR) -maxdepth 1 -type f \( \
		-name '*.aux' -o -name '*.bbl' -o -name '*.bcf' -o -name '*.blg' -o \
		-name '*.dvi' -o -name '*.fdb_latexmk' -o -name '*.fls' -o \
		-name '*.lof' -o -name '*.log' -o -name '*.lot' -o -name '*.nav' -o \
		-name '*.out' -o -name '*.run.xml' -o -name '*.snm' -o \
		-name '*.synctex.gz' -o -name '*.toc' -o -name '*.vrb' -o \
		-name '*.xdv' \
	\) -delete
	find . -type d \( -name '__pycache__' -o -name '.pytest_cache' -o -name '.mypy_cache' \) -prune -exec rm -rf {} +
	rm -f .DS_Store $(PAPER_DIR)/.DS_Store

distclean: clean
	rm -rf .venv build
	rm -f $(PDF) $(ARXIV_TARBALL)
	rm -f $(FIGURE_DIR)/*.pdf $(FIGURE_DIR)/*.png

arxiv: figures paper
	rm -rf $(ARXIV_DIR)
	mkdir -p $(ARXIV_DIR)/figures
	cp $(PAPER_DIR)/main.tex $(PAPER_DIR)/references.bib $(ARXIV_DIR)/
	if [ -f $(PAPER_DIR)/main.bbl ]; then cp $(PAPER_DIR)/main.bbl $(ARXIV_DIR)/; fi
	cp $(FIGURE_DIR)/*.pdf $(ARXIV_DIR)/figures/
	tar -czf $(ARXIV_TARBALL) -C $(ARXIV_DIR) main.tex references.bib main.bbl figures
	@echo "Wrote $(ARXIV_TARBALL)"
	@tar -tzf $(ARXIV_TARBALL) | sort
