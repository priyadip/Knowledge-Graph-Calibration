"""Counterfactual calibration conditions with paired files, at the generation batch size of the original true runs.

Usage:
  python scripts/experiments/paired_variants.py <tag> <cseed> [--with_true] [run_grid options, e.g. --batch 64 --group 2]
- Calibration files: KG-D and KG-QA from data/calib, KG-D~ and KG-QA~ from data/calib_paired (line i is the
  counterfactual of the true line i).
- cseed 0: no shuffle, so the counterfactual half-sets contain exactly the facts of the true half-sets.
- cseed 1, 2: run_grid shuffles every source from random.Random(cseed) in the order KG-D, KG-D~, KG-QA, KG-QA~. The
  replacement Random below recomputes the four original permutations and gives each counterfactual source the
  permutation of its true partner. Only run_grid's own name `random` is replaced.
- --with_true (cseed 0 only) also evaluates MIX:KG-D|MIX:KG-D, to check that a rerun reproduces the original run.
Outputs go to results/paired/batch_matched. The made-up-subject condition and the Lemma 1 terms of the paired
selections are run by paired_subjects.py and paired_theory.py.
"""
import json
import random
import shutil
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from kgcal import paths as P            # noqa: E402
from kgcal import data as D            # noqa: E402
from kgcal import run_grid as R        # noqa: E402

MODELS = {"qwen15b": "Qwen/Qwen2.5-1.5B", "olmo1b": "allenai/OLMo-2-0425-1B",
          "qwen7b": "Qwen/Qwen2.5-7B", "olmo7b": "allenai/OLMo-2-1124-7B"}
tag, cseed, rest = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
with_true = "--with_true" in rest
extra = [a for a in rest if a != "--with_true"]
assert not (with_true and cseed), "--with_true is only defined for cseed 0"

CF = ("MIX:KG-D~|MIX:KG-D~", "MIX:KG-QA~|MIX:KG-QA~")
keep = CF + (("MIX:KG-D|MIX:KG-D",) if with_true else ())
orig_grid = R.grid
R.grid = lambda name: [c for c in orig_grid("key") if c[0] in keep]

ORDER = ["KG-D", "KG-D~", "KG-QA", "KG-QA~"]
PARTNER = {"KG-D~": "KG-D", "KG-QA~": "KG-QA"}


class PairedCFRandom(random.Random):
    """Shuffles the counterfactual sources with the permutation their true partner received in the original run."""

    def __init__(self, seed):
        super().__init__(seed)
        n = {k: sum(1 for _ in open(P.CALIB / f"{k}.jsonl")) for k in ORDER}
        ref, self.perm = random.Random(seed), {}
        for k in ORDER:                                  # the original sequence of four shuffles
            idx = list(range(n[k]))
            ref.shuffle(idx)
            self.perm[k] = idx
        self.calls = 0

    def shuffle(self, x):
        k = ["KG-D~", "KG-QA~"][self.calls]              # sorted(used) = [KG-D~, KG-QA~] in this grid
        p = self.perm[PARTNER[k]]
        assert len(p) == len(x)
        x[:] = [x[i] for i in p]
        self.calls += 1


if cseed:
    R.random = types.SimpleNamespace(Random=PairedCFRandom)
out = P.PAIRED / "batch_matched"                             # run_grid writes to D.RESULTS
new_tag = f"{tag}_paired"
out.mkdir(parents=True, exist_ok=True)
shutil.copy(P.GRID / f"{tag}_gride.json", out / f"{new_tag}_gride.json")
D.PAIRED = True
D.RESULTS = out
sys.argv = ["run_grid", "--model", MODELS[tag], "--tag", new_tag, "--grid", "key", "--cseed", str(cseed), "--no_cloze"] + extra
R.main()
