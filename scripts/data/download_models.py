"""Download every checkpoint of the paper at its pinned revision into the Hugging Face cache.

python scripts/data/download_models.py [tag ...]      (no tag: all eight)

Compressed models: Qwen2.5-1.5B, OLMo-2-1B, Qwen2.5-7B and OLMo-2-7B (kgcal.model_ops.REVISIONS), and Llama-3.1-8B,
Mistral-7B-v0.3 and Phi-4 (kgx.models.REGISTRY). Answer judges: Phi-4 and Llama-3.1-8B-Instruct.
Llama 3.1 is gated on its official repository, so ungated copies are used:
- unsloth/Meta-Llama-3.1-8B: the four shards have the same SHA-256 as NousResearch/Meta-Llama-3.1-8B (an independent
  copy) and the same byte sizes as meta-llama/Llama-3.1-8B.
- unsloth/Meta-Llama-3.1-8B-Instruct, for the second answer judge.
Mistral's duplicate consolidated.safetensors is skipped.
"""
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from kgcal.model_ops import REVISIONS  # noqa: E402
from kgx.models import REGISTRY  # noqa: E402

PAPER = {"qwen15b": "Qwen/Qwen2.5-1.5B", "olmo1b": "allenai/OLMo-2-0425-1B", "qwen7b": "Qwen/Qwen2.5-7B",
         "olmo7b": "allenai/OLMo-2-1124-7B"}
REPOS = {tag: (repo, REVISIONS[repo]) for tag, repo in PAPER.items()}
REPOS.update({tag: (spec["repo"], spec["rev"]) for tag, spec in REGISTRY.items()})
REPOS["judge_llama"] = ("unsloth/Meta-Llama-3.1-8B-Instruct", "a2856192dd7c25b842431f39c179a6c2c2f627d1")

for tag in sys.argv[1:] or list(REPOS):
    repo, rev = REPOS[tag]
    path = snapshot_download(repo, revision=rev, ignore_patterns=["consolidated*", "original/*", "*.pth", "*.gguf"])
    print(tag, repo, rev, "->", path, flush=True)
