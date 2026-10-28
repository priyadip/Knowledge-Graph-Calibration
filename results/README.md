# Per-fact results

Every run writes three kinds of files under a tag (`qwen15b`, `olmo1b`, `qwen7b`, `olmo7b`, `llama8b`, `mistral7b`,
`phi4`, with a suffix for a variant run):

- `<tag>.jsonl`: one record per configuration with its budget (`target`: share of feed-forward units kept, 1.0 for
  the dense model), condition (`cond`), calibration sample (`cseed`), kept shares (`ffn_keep`, `param_keep`),
  accuracies (`val_q_acc`, `test_q_acc`, `test_c_acc`), perplexities (`ppl_wikitext2`, `ppl_c4`) and, where scored,
  `arc_easy` and `hellaswag`.
- `<tag>_bits.json`: per-fact correctness, `{"<target>|<cond>": {"val_q": "0101...", "test_q": ..., "test_c": ...}}`,
  one character per fact in the order of `kgcal.data.facts_by_split` (1,435 validation and 5,690 test facts;
  `q` question queries, `c` cloze queries). Every statistic of the paper is computed from these strings.
- `<tag>_gride.json`: the layer dimensions d_l from which the widths of both budgets follow; `<tag>_theory.jsonl`: the
  Lemma 1 terms of the measured blocks.

A condition is named `<selection text>|<repair text>`: `C4|MIX:KG-D` selects units on C4 and repairs on MIX(KG-D),
where MIX(X) gives C4 and X equal weight in the moments. `none|C4` selects on C4 without repair and `bias|C4` repairs
the bias only. A text without `MIX:` is used alone, and `mag` or `rand` in place of a selection text selects by
weight magnitude or at random.

| Folder | Runs | Written by |
|---|---|---|
| `grid/` | main grid of Qwen2.5 and OLMo-2 (`_cs1`, `_cs2`: calibration samples; `_zs`: zero-shot scores of the 1B models; `_lam1e-1`, `_lam1e-2`: ridge strength; `_cal4x`: 4x calibration text) | `kgcal.run_grid` |
| `paired/` | counterfactual conditions with paired files (`batch_matched/`, `a1_seed0/`, `a2_seeds/`), made-up subjects (`a4_entities/`), Lemma 1 terms of the paired selections (`a3_theory/`) | `scripts/experiments/paired_*.py` |
| `merged/` | the Qwen2.5 and OLMo-2 results the paper reports: `grid/` with the paired runs in place of the unpaired variants; `MANIFEST.json` records the source and batch size of every replaced entry | `scripts/analysis/merge_results.py` |
| `families/` | Llama-3.1-8B, Mistral-7B-v0.3 and Phi-4, the suffixes as in `grid/`; `_dims.json`, `_gmle*`: width check | `kgx.run_grid`, `kgx.widths` |
| `width_check/` | estimators on synthetic and real clouds (`e1_synthetic.json`, `<tag>_dims.json`, `widths_<tag>.json`), runs at GRIDE-MLE widths (`_gmle`) and at widths of another unit subsample (`_seed1`) | `scripts/experiments/width_estimators*.py`, `grid_alt_widths.py`, `scripts/analysis/widths.py` |
| `controls/` | matched format (`_fmt`), likelihood multiple choice (`_mc`), truth variants (`_truth`, `_truth_cs1`, `_truth_cs2`); `_info.json` records the facts, token counts and replay checks of a run | `scripts/experiments/controls*.py` |
| `entityquestions/` | transfer of the PopQA calibration (`_eq`) and calibration on the EntityQuestions graph (`_eqb`, three samples), with the decoded answers | `scripts/experiments/entityquestions*.py` |
| `mixture/` | share of KG statements in the calibration budget | `scripts/experiments/mixture_share.py` |
| `sequential/`, `obs/`, `sparsegpt/` | sequential statistics, structured OBS selection, SparseGPT at 50% of the weights | `scripts/experiments/{sequential,obs_selection,sparsegpt}.py` |
| `paired_cloze/` | paired variants with cloze queries, and every decoded answer (`_pc_answers.json`) | `scripts/experiments/paired_cloze.py` |
| `answers/` | four scoring rules and two judges per answer (`_scores.json`); the blind sample (`human_check.csv`), its key, and the labels of one author (`human_labels.csv`) | `scripts/experiments/score_answers.py`, `judge.py`, `scripts/data/make_human_sample.py` |
| `mechanism/`, `cost/` | moment distances and kept-unit overlaps of paired texts; inference and compression cost | `scripts/experiments/moments_overlap.py`, `measure_cost.py` |

Replays: a run that rebuilds a stored configuration checks that its answers match the stored ones bit for bit, with
two documented cases. Reruns of Llama-3.1-8B with selection on C4 at 70% differ on at most 25 of 5,690 answers, so
the reference for that configuration is the EntityQuestions rerun (`entityquestions/llama8b_eq_bits.json`). The
stored sample-2 runs of Qwen2.5-7B and OLMo-2-7B used generation batch 48, and their truth controls use the same
batch.
