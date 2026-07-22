# agent-memory-ablation, reproduce every number in paper/README.md.
#
# `make eval` is the canonical entrypoint named in the paper. On Windows without
# GNU make, use the identical shim:  python scripts/run.py eval
#
# PY selects the interpreter; override with `make PY=python3 eval`.
PY ?= python

.PHONY: all setup corpus queries eval figures paper pdf test clean help

help:
	@echo "make setup    - install pinned dependencies"
	@echo "make corpus   - regenerate the frozen synthetic corpus (seed 42)"
	@echo "make queries  - regenerate the frozen query sets (seed 1234)"
	@echo "make eval     - build all arms, run 200 queries, write results/ + traces/"
	@echo "make figures  - regenerate paper figures from results/"
	@echo "make paper    - eval + figures (everything the paper cites)"
	@echo "make pdf      - render paper/agent-memory-ablation.pdf (reportlab; no TeX needed)"
	@echo "make pdf-latex- compile paper/paper.tex -> paper.pdf (arXiv source; needs pdflatex)"
	@echo "make test     - run the test suite"
	@echo "make all      - corpus + queries + eval + figures"
	@echo ""
	@echo "offline / CI:  AMA_EMBEDDER=hash make eval   (no model download)"

setup:
	$(PY) -m pip install -r requirements.txt

corpus:
	$(PY) -m corpus.generate --seed 42

queries:
	$(PY) -m harness.build_queries --seed 1234

eval:
	$(PY) -m harness.run_eval

figures:
	$(PY) -m paper.generate_figures

paper: eval figures

# Render the PDF with reportlab (pure Python, no TeX toolchain). Tables are
# built from results/summary.json + corpus/manifest.json, so run after eval.
pdf: figures
	$(PY) -m paper.build_pdf

# Compile the LaTeX (arXiv-ready) source -> paper/paper.pdf, matching the house
# style of the companion papers. Prefers tectonic (self-contained, auto-fetches
# packages); falls back to pdflatex; prints guidance if neither is present.
pdf-latex: figures
	@cd paper && if command -v tectonic >/dev/null 2>&1; then \
	  tectonic -X compile paper.tex; \
	elif command -v pdflatex >/dev/null 2>&1; then \
	  pdflatex -interaction=nonstopmode paper.tex && pdflatex -interaction=nonstopmode paper.tex; \
	else \
	  echo "No TeX engine found. Install tectonic (winget install TectonicProject.Tectonic)"; \
	  echo "or MiKTeX/TeX Live, or upload paper.tex + figures/ to Overleaf/arXiv."; \
	fi

test:
	$(PY) -m pytest tests/ -q

all: corpus queries eval figures

clean:
	rm -rf results/indexes .hf_cache .cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name .pytest_cache -prune -exec rm -rf {} +
