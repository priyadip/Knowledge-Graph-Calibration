#!/usr/bin/env bash
# Every number, table and figure of the paper from the released per-fact results (CPU only, a few minutes).
#   bash scripts/reproduce_paper.sh            # writes outputs/tables, outputs/figures, outputs/reports
#   bash scripts/reproduce_paper.sh --figure1  # also renders Figure 1 (needs Chrome, Ghostscript, Linux Libertine)
# PopQA is read through the Hugging Face cache (scripts/data/download_data.py); nothing else is downloaded.
set -euo pipefail
cd "$(dirname "$0")/.."
run() { echo "== $*"; python "$@"; }

run scripts/data/check_data.py                 # released inputs unchanged
run scripts/analysis/merge_results.py          # results/merged from results/grid and results/paired
run scripts/analysis/paper_models.py           # Qwen2.5 and OLMo-2: numbers.tex, Figure 2
run scripts/analysis/families.py               # all seven models: summary.json
run scripts/analysis/stats.py                  # format by query type, truth over samples, Holm counts
run scripts/analysis/entityquestions.py        # transfer and replication on EntityQuestions
run scripts/analysis/mixture_rule.py           # share of KG statements and the rule fixed in advance
run scripts/analysis/overlap_coverage.py       # EntityQuestions name overlap, covered facts per format
run scripts/analysis/validation.py             # subject-clustered intervals, local repair error
run scripts/analysis/controls.py               # matched format, multiple choice, truth variants
run scripts/paper/make_numbers.py              # numbers08.tex, Tables 2 to 5, Figure 3
if [[ "${1:-}" == "--figure1" ]]; then
  run scripts/paper/figure1.py --render        # Figure 1 (draw.io source and PDF)
fi
echo "done: outputs/tables (numbers.tex, numbers08.tex, table*.tex), outputs/figures, outputs/reports"
