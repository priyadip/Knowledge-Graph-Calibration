#!/usr/bin/env bash
# The GPU runs behind results/, with the options they were run with. One section per argument:
#   bash scripts/run_experiments.sh data grid paired families widths controls entityquestions methods answers
# Each command uses the GPUs in CUDA_VISIBLE_DEVICES (one 48 GB GPU per run; Phi-4 needs 44 GB free, or two GPUs, see
# "families"). kgcal.run_grid and kgx.run_grid resume a run: configurations already in its result files are skipped.
# Generation batch sizes matter: answers can change with the batch size, and every replay check compares bits.
set -euo pipefail
cd "$(dirname "$0")/.."
ZS="none|C4,C4|C4,C4|MIX:KG-D,MIX:KG-D|MIX:KG-D"   # conditions also scored on ARC-Easy and HellaSwag
G=results/grid
MODEL=(); MODEL[1]=Qwen/Qwen2.5-1.5B; MODEL[2]=allenai/OLMo-2-0425-1B; MODEL[3]=Qwen/Qwen2.5-7B; MODEL[4]=allenai/OLMo-2-1124-7B
TAG=(); TAG[1]=qwen15b; TAG[2]=olmo1b; TAG[3]=qwen7b; TAG[4]=olmo7b
grid() { python -m kgcal.run_grid "$@"; }
fam() { python -m kgx.run_grid "$@"; }

data() {          # inputs (data/ is released: only the downloads are needed to rerun anything)
  python scripts/data/download_models.py
  python scripts/data/download_data.py
  # to rebuild the released inputs instead (identical files except KG-O and KG-A, see build_popqa_kg.py):
  # python scripts/data/build_popqa_kg.py && python scripts/data/build_paired_variants.py
  # python scripts/data/build_entityquestions.py && python scripts/data/build_entityquestions_calib.py
  # python scripts/data/build_controls.py
}

grid_runs() {     # main grid of Qwen2.5 and OLMo-2 -> results/grid (default batch 96, group 4)
  for i in 1 2; do
    grid --model ${MODEL[$i]} --tag ${TAG[$i]} --fractions 0.7,0.5 --grid all --theory
    grid --model ${MODEL[$i]} --tag ${TAG[$i]} --fractions 0.7,0.5 --grid followup          # KG-O2 and KG-Ac
    cp $G/${TAG[$i]}_gride.json $G/${TAG[$i]}_zs_gride.json                                  # same widths
    grid --model ${MODEL[$i]} --tag ${TAG[$i]}_zs --fractions 0.7,0.5 --grid key --zeroshot "$ZS" --no_cloze
    for cs in 1 2; do grid --model ${MODEL[$i]} --tag ${TAG[$i]} --fractions 0.7,0.5 --grid key --cseed $cs --no_cloze; done
  done
  grid --model ${MODEL[3]} --tag qwen7b --fractions 0.7,0.5 --grid core7b --group 2 --batch 64 --zeroshot "$ZS"
  grid --model ${MODEL[4]} --tag olmo7b --fractions 0.7,0.5 --grid core7b --group 4 --batch 64 --zeroshot "$ZS"
  for i in 3 4; do
    grid --model ${MODEL[$i]} --tag ${TAG[$i]} --fractions 0.7,0.5 --grid key --cseed 1 --group 2 --batch 64 --no_cloze
    grid --model ${MODEL[$i]} --tag ${TAG[$i]} --fractions 0.7,0.5 --grid key --cseed 2 --group 2 --batch 48 --no_cloze
    for lam in 1e-2 1e-1; do                                                                  # ridge strength
      cp $G/${TAG[$i]}_gride.json $G/${TAG[$i]}_lam${lam}_gride.json
      grid --model ${MODEL[$i]} --tag ${TAG[$i]}_lam$lam --fractions 0.7 --grid lam --group 2 --batch 64 --lam $lam --no_cloze
    done
    cp $G/${TAG[$i]}_gride.json $G/${TAG[$i]}_cal4x_gride.json                               # 4x calibration text
    grid --model ${MODEL[$i]} --tag ${TAG[$i]}_cal4x --fractions 0.7 --grid c4only --group 2 --batch 64 --half 256 --no_cloze
  done
  grid --model ${MODEL[4]} --tag olmo7b --fractions 0.7,0.5 --grid theory --theory_only --group 2   # Lemma 1 terms, 7B
}

paired() {        # paired-variant reruns at the generation batch of the main runs -> results/paired
  for cs in 0 1 2; do
    python scripts/experiments/paired_variants.py qwen15b $cs --fractions 0.7,0.5
    python scripts/experiments/paired_variants.py olmo1b $cs --fractions 0.7,0.5
  done
  python scripts/experiments/paired_variants.py qwen7b 0 --with_true --fractions 0.7,0.5 --group 1 --batch 64
  python scripts/experiments/paired_variants.py olmo7b 0 --with_true --fractions 0.7,0.5 --group 2 --batch 64
  for t in qwen7b olmo7b; do
    python scripts/experiments/paired_variants.py $t 1 --fractions 0.7,0.5 --group 2 --batch 64
    python scripts/experiments/paired_variants.py $t 2 --fractions 0.7,0.5 --group 2 --batch 48
  done
  for t in qwen15b olmo1b; do python scripts/experiments/paired_subjects.py $t --fractions 0.7,0.5; done
  for t in qwen7b olmo7b; do python scripts/experiments/paired_subjects.py $t --fractions 0.7,0.5 --group 2 --batch 64; done
  for t in qwen15b olmo1b; do python scripts/experiments/paired_theory.py $t --fractions 0.7,0.5; done
  python scripts/experiments/paired_theory.py olmo7b --fractions 0.7,0.5 --group 2
  python scripts/analysis/merge_results.py                                                    # -> results/merged
}

families() {      # Llama-3.1-8B, Mistral-7B-v0.3, Phi-4 -> results/families (batch 64 for every run)
  # Phi-4 on two GPUs: CUDA_VISIBLE_DEVICES=0,1 KGX_SPLIT=1 KGX_SPLIT_FIRST=15 before each Phi-4 command
  # (blocks 0-14 on the first GPU; answers identical to one GPU, checked on 256 test questions)
  for m in llama8b mistral7b phi4; do
    g=2; [ $m = phi4 ] && g=1
    if [ $m = phi4 ]; then                                                                    # main grid, one budget per run
      fam --model $m --tag $m --group $g --batch 64 --grid core7b --zeroshot "$ZS" --fractions 0.7
      fam --model $m --tag $m --group $g --batch 64 --grid core7b --zeroshot "$ZS" --fractions 0.5
    else
      fam --model $m --tag $m --group $g --batch 64 --grid core7b --zeroshot "$ZS" --fractions 0.7,0.5
    fi
    for cs in 1 2; do fam --model $m --tag $m --group $g --batch 64 --grid key --cseed $cs --fractions 0.7,0.5 --no_cloze; done
    fam --model $m --tag $m --group $g --batch 64 --grid theory --theory_only --fractions 0.7,0.5
    for lam in 1e-2 1e-1; do
      fam --model $m --tag ${m}_lam$lam --group $g --batch 64 --grid lam --fractions 0.7 --lam $lam --no_cloze --gride_from $m
    done
    fam --model $m --tag ${m}_cal4x --group $g --batch 64 --grid c4only --fractions 0.7 --half 256 --no_cloze --gride_from $m
    python -m kgx.widths $m                                                                   # width check
    fam --model $m --tag ${m}_gmle --group $g --batch 64 --grid key --fractions 0.7,0.5 --no_cloze \
        --dims_file results/families/${m}_gmle_dims.json
  done
}

widths() {        # width check of Qwen2.5 and OLMo-2 -> results/width_check
  python scripts/experiments/width_estimators_synthetic.py                                    # CPU
  for t in qwen15b olmo1b qwen7b olmo7b; do python scripts/experiments/width_estimators.py $t; done
  python scripts/analysis/widths.py                                                           # CPU: widths_<tag>.json
  for t in qwen15b olmo1b; do
    for v in gmle seed1; do python scripts/experiments/grid_alt_widths.py $t $v --fractions 0.7,0.5 --grid key --no_cloze; done
  done
  python scripts/experiments/grid_alt_widths.py qwen7b gmle --fractions 0.7,0.5 --grid key --group 1 --batch 32 --no_cloze
  python scripts/experiments/grid_alt_widths.py olmo7b gmle --fractions 0.7,0.5 --grid key --group 2 --batch 32 --no_cloze
}

controls() {      # matched format, multiple choice, truth variants -> results/controls
  python scripts/data/build_controls.py
  for exp in fmt mc truth; do
    for t in qwen15b olmo1b qwen7b olmo7b; do python scripts/experiments/controls.py $t $exp; done
    for t in llama8b mistral7b phi4; do python scripts/experiments/controls_families.py $t $exp; done
  done
  for cs in 1 2; do
    for t in llama8b mistral7b phi4; do python scripts/experiments/controls_families.py $t truth $cs; done
    for t in qwen15b olmo1b; do python scripts/experiments/controls.py $t truth $cs; done
  done
  python scripts/experiments/controls.py qwen7b truth 1
  python scripts/experiments/controls.py olmo7b truth 1
  KGCAL_GEN_BATCH=48 python scripts/experiments/controls.py qwen7b truth 2                    # the stored sample-2 runs used 48
  KGCAL_GEN_BATCH=48 python scripts/experiments/controls.py olmo7b truth 2
}

entityquestions() {   # transfer (mode a) and replication (mode b, three samples) -> results/entityquestions
  for t in qwen15b olmo1b qwen7b olmo7b; do
    python scripts/experiments/entityquestions.py $t
    python scripts/experiments/entityquestions.py $t b
    for cs in 1 2; do python scripts/experiments/entityquestions.py $t b $cs; done
  done
  for t in llama8b mistral7b phi4; do
    python scripts/experiments/entityquestions_families.py $t
    python scripts/experiments/entityquestions_families.py $t b
    for cs in 1 2; do python scripts/experiments/entityquestions_families.py $t b $cs; done
  done
}

methods() {       # mixture share, sequential statistics, structured OBS, SparseGPT, moments, inference cost
  for t in qwen15b olmo1b; do
    python scripts/experiments/mixture_share.py $t
    for cs in 1 2; do python scripts/experiments/mixture_share.py $t $cs; done
  done
  for t in qwen7b olmo7b; do python scripts/experiments/mixture_share.py $t; python scripts/experiments/mixture_share.py $t --extra; done
  for t in qwen15b olmo1b qwen7b olmo7b; do
    python scripts/experiments/sequential.py $t
    python scripts/experiments/obs_selection.py $t
    python scripts/experiments/sparsegpt.py $t
    python scripts/experiments/moments_overlap.py $t
    python scripts/experiments/measure_cost.py $t                                             # one GPU to itself
  done
}

answers() {       # answers kept for rescoring, four scoring rules, two LLM judges, the human-check sample
  for t in qwen15b olmo1b qwen7b olmo7b; do python scripts/experiments/paired_cloze.py $t; done
  for t in qwen15b olmo1b qwen7b olmo7b; do python scripts/experiments/score_answers.py $t; python scripts/experiments/judge.py $t; done
  python scripts/data/make_human_sample.py       # results/answers/human_check.csv; human_labels.csv holds the labels
}

for section in "$@"; do
  case $section in
    data) data ;; grid) grid_runs ;; paired) paired ;; families) families ;; widths) widths ;;
    controls) controls ;; entityquestions) entityquestions ;; methods) methods ;; answers) answers ;;
    *) echo "unknown section: $section"; exit 1 ;;
  esac
done
