# Facts or Formats? How Knowledge-Graph Calibration Shapes What Pruned Language Models Remember

Code, data and per-fact results for the paper. The code prunes the SwiGLU feed-forward units of a language model by
select-and-repair (activation-norm selection and a ridge repair of the down projection, at GRIDE widths), with the
calibration text taken from generic Web text (C4) or written from Wikidata statements, and measures which held-out
PopQA facts the pruned model still answers.

Every number, table and figure of the paper can be regenerated on a CPU from the released results
(`scripts/reproduce_paper.sh`). Every result can be regenerated on GPUs with the commands of
`scripts/run_experiments.sh`.

## Settings

One model at one budget is a setting. The budget is the share of feed-forward units kept: 70% or 50%. Qwen2.5 and
OLMo-2 run every condition (eight settings); the three further families run the main conditions (six settings).

| # | Model | Budget | Hugging Face repository (revision) |
|---|---|---|---|
| 1, 2 | Qwen2.5-1.5B | 70%, 50% | [Qwen/Qwen2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B) (`8faed761d45a263340a0528343f099c05c9a4323`) |
| 3, 4 | OLMo-2-1B | 70%, 50% | [allenai/OLMo-2-0425-1B](https://huggingface.co/allenai/OLMo-2-0425-1B) (`a1847dff35000b4271fa70afc5db10fd29fedbdf`) |
| 5, 6 | Qwen2.5-7B | 70%, 50% | [Qwen/Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B) (`d149729398750b98c0af14eb82c78cfe92750796`) |
| 7, 8 | OLMo-2-7B | 70%, 50% | [allenai/OLMo-2-1124-7B](https://huggingface.co/allenai/OLMo-2-1124-7B) (`7df9a82518afdecae4e8c026b27adccc8c1f0032`) |
| 9, 10 | Llama-3.1-8B | 70%, 50% | [unsloth/Meta-Llama-3.1-8B](https://huggingface.co/unsloth/Meta-Llama-3.1-8B) (`e9a141a2091ea561b96483212645a2a05e6f99fc`), an ungated copy of [meta-llama/Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B) |
| 11, 12 | Mistral-7B-v0.3 | 70%, 50% | [mistralai/Mistral-7B-v0.3](https://huggingface.co/mistralai/Mistral-7B-v0.3) (`caa1feb0e54d415e2df31207e5f4e273e33509b1`) |
| 13, 14 | Phi-4 (14B) | 70%, 50% | [microsoft/phi-4](https://huggingface.co/microsoft/phi-4) (`2db69c1c3e91a05d2c64a3185acfbaf36f744e25`) |

The Llama copy has the same SHA-256 per shard as an independent copy
([NousResearch/Meta-Llama-3.1-8B](https://huggingface.co/NousResearch/Meta-Llama-3.1-8B)) and the byte sizes of the
official repository. Two LLM judges score the saved answers of Qwen2.5 and OLMo-2: Phi-4 (above) and
[unsloth/Meta-Llama-3.1-8B-Instruct](https://huggingface.co/unsloth/Meta-Llama-3.1-8B-Instruct)
(`a2856192dd7c25b842431f39c179a6c2c2f627d1`), a copy of
[meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct).

## External data

| Source | Use | Link |
|---|---|---|
| PopQA, test split (revision `098765c79ea10a2cb19c828324e33281b8336ec0`) | facts, questions, answer aliases, Wikidata identifiers | [akariasai/PopQA](https://huggingface.co/datasets/akariasai/PopQA) |
| Wikidata, SPARQL endpoint (retrieved October 2026, cached in `data/wikidata`) | statements, property labels, entity types | [query.wikidata.org](https://query.wikidata.org/) |
| Wikipedia page summaries (cached in `data/wikidata/wiki_summaries.json`) | lead paragraphs of calibration subjects | [REST API](https://en.wikipedia.org/api/rest_v1/) |
| C4, `en`, validation (revision `1588ec454efa1a09f29cd18ddd04fe05fc8653a2`, streamed) | generic calibration text, perplexity | [allenai/c4](https://huggingface.co/datasets/allenai/c4) |
| WikiText-2, `wikitext-2-raw-v1`, test (revision `b08601e04326c79dfdd32d625aee71d232d685c3`) | perplexity | [Salesforce/wikitext](https://huggingface.co/datasets/Salesforce/wikitext) |
| ARC-Easy, test (revision `210d026faf9955653af8916fad021475a3f00453`) | zero-shot accuracy | [allenai/ai2_arc](https://huggingface.co/datasets/allenai/ai2_arc) |
| HellaSwag, validation (revision `218ec52e09a7e7462a5400043bb9a69a41d06b76`) | zero-shot accuracy | [Rowan/hellaswag](https://huggingface.co/datasets/Rowan/hellaswag) |
| EntityQuestions (Sciavolino et al., 2021) | transfer and replication on 18 further relations | [princeton-nlp/EntityQuestions](https://github.com/princeton-nlp/EntityQuestions), [dataset.zip](https://nlp.cs.princeton.edu/projects/entity-questions/dataset.zip) |

The derived inputs (split, Wikidata and Wikipedia caches, every calibration text, the EntityQuestions sample and the
control data) are released in `data/` (see `data/README.md`). Wikidata statements are CC0, Wikipedia text is
CC BY-SA 4.0, and C4 is ODC-BY; every other dataset keeps the license of its source.

## Setup

Python 3.13 with the pinned packages (PyTorch 2.7.1 with CUDA 12.6, Transformers 5.15.0, Datasets 4.8.4):

```bash
pip install -r requirements.txt
python scripts/data/download_data.py      # PopQA and the evaluation sets into the Hugging Face cache; EntityQuestions
python scripts/data/download_models.py    # the eight checkpoints at their revisions (GPU runs only)
python scripts/data/check_data.py         # every released input against data/MANIFEST.json
```

The runs used NVIDIA L40S GPUs (48 GB) and bfloat16 weights. Statistics are accumulated in float32 and every repair
is solved in float64. Phi-4 needs 44 GB free on one GPU, or two GPUs with `KGX_SPLIT=1` (`kgx/models.py`).

## Reproducing the paper from the released results (CPU)

```bash
bash scripts/reproduce_paper.sh             # six minutes on a 96-core server; add --figure1 to render Figure 1
```

| Paper | File in `outputs/` | Written by |
|---|---|---|
| Numbers in the text | `tables/numbers.tex`, `tables/numbers08.tex` | `scripts/analysis/paper_models.py`, `scripts/paper/make_numbers.py` |
| Figure 1 | `figures/figure1.drawio`, rendered to `figures/figure1.pdf` | `scripts/paper/figure1.py --render` |
| Figure 2 | `figures/fig3_ppl_vs_retention.pdf` | `scripts/analysis/paper_models.py` |
| Figure 3 | `figures/fig4_facts_or_formats7.pdf` | `scripts/paper/make_numbers.py` |
| Table 2 | `tables/table2_main7.tex` | `scripts/paper/make_numbers.py` |
| Table 3 | `tables/table3_summary.tex` | `scripts/paper/make_numbers.py` |
| Table 4 | `tables/tableA_validation.tex` | `scripts/paper/make_numbers.py` |
| Table 5 | `tables/tableB_controls.tex` | `scripts/paper/make_numbers.py` |
| Tables 1 and 6 | written in LaTeX; Table 6 takes its numbers from the two macro files | |

The scripts of `scripts/analysis` write the statistics (`outputs/tables/*.json`) that `scripts/paper/make_numbers.py`
turns into macros and tables, and a readable summary of each analysis (`outputs/reports/*.md`). Intervals come from
paired bootstraps with fixed seeds, so the outputs are identical from run to run. Every sentence of the paper that
states a count in words has a guard in `scripts/paper/make_numbers.py` that stops the script if the data no longer
support it, and `outputs/reports/pending.md` lists any number whose run is missing. The paper's Figure 1 is the
draw.io source exported from the draw.io editor; `--render` draws the same source with the draw.io viewer, so its
layout differs slightly while every text and number is the same.

## Rerunning the experiments (GPU)

```bash
bash scripts/run_experiments.sh data grid paired families widths controls entityquestions methods answers
```

| Section | What it runs | Results |
|---|---|---|
| `grid` | main grid of Qwen2.5 and OLMo-2: every condition at both budgets, three calibration samples, ridge strength, 4x calibration text, zero-shot scores, Lemma 1 terms (`kgcal.run_grid`) | `results/grid` |
| `paired` | counterfactual and made-up-subject conditions with line-paired variants, then `merge_results.py` | `results/paired`, `results/merged` |
| `families` | the same protocol on Llama-3.1-8B, Mistral-7B-v0.3 and Phi-4 (`kgx.run_grid`) | `results/families` |
| `widths` | width estimators and runs at GRIDE-MLE and resampled widths | `results/width_check` |
| `controls` | matched format, likelihood multiple choice, truth variants | `results/controls` |
| `entityquestions` | transfer of the PopQA calibration, and calibration on the EntityQuestions graph | `results/entityquestions` |
| `methods` | share of KG statements, sequential statistics, structured OBS selection, SparseGPT, moments, inference cost | `results/{mixture,sequential,obs,sparsegpt,mechanism,cost}` |
| `answers` | saved answers, four scoring rules, two LLM judges, the sample for the human check | `results/paired_cloze`, `results/answers` |

The main grid of one 7B model takes about 2.3 hours on one L40S. Runs that rebuild a stored configuration (for
example C4/C4 inside the EntityQuestions, control and mixture runs) check that its answers reproduce the stored ones
bit for bit. The generation batch size is part of the configuration, because padding changes a few answers; the
commands give the batch size of every stored run.

## Repository layout

```
kgcal/                 method and protocol on Qwen2.5 and OLMo-2
  paths.py             every input and output location, relative to the repository root (KGCAL_ROOT overrides it)
  data.py              PopQA split, Wikidata statements, verbalizers, calibration texts, C4
  model_ops.py         loading at pinned revisions, feed-forward statistics, GRIDE widths
  compress.py          selection rules, ridge repair, bias-only repair, Lemma 1 terms
  evaluate.py          question and cloze recall with alias matching, perplexity, zero-shot accuracy
  run_grid.py          one model, its budgets and a grid of calibration conditions
  estimators.py        intrinsic-dimension estimators of the width check
kgx/                   the same protocol on Llama-3.1, Mistral and Phi-4 (start-of-text token, fused FFN of Phi-4)
scripts/
  data/                downloads and the builders of every file in data/
  experiments/         GPU runs beyond the main grids
  analysis/            statistics from results/ (CPU)
  paper/               macros, tables and figures of the paper (CPU)
  reproduce_paper.sh   the CPU pipeline
  run_experiments.sh   the GPU runs
data/                  inputs (data/README.md)
results/               per-fact results of every run (results/README.md)
outputs/               written by scripts/reproduce_paper.sh
```

## License

The code is released under the MIT license (`LICENSE`). Data in `data/` and `results/` derived from external sources
keep the licenses of those sources.
