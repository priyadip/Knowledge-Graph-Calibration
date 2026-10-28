"""Download the external datasets.

python scripts/data/download_data.py

- EntityQuestions (Sciavolino et al., 2021) into data/entityquestions/raw/dataset.zip, checked against the SHA-256 of
  the copy the paper used. Only scripts/data/build_entityquestions*.py read it; their outputs are released in data/.
- PopQA, WikiText-2, ARC-Easy and HellaSwag into the Hugging Face cache, at the revisions of kgcal.data.HF_REVISIONS.
  C4 (allenai/c4, en, validation) is streamed whenever calibration text is packed, so it is not downloaded here.
"""
import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from kgcal import paths as P  # noqa: E402
from kgcal.data import HF_REVISIONS  # noqa: E402

EQ_URL = "https://nlp.cs.princeton.edu/projects/entity-questions/dataset.zip"
EQ_SHA256 = "015eb64a12fa2f9b2f95d2c78bba69779e0d200dbbd4b0772f3ce57bc00da9c5"

raw = P.EQ / "raw" / "dataset.zip"
if not raw.exists():
    raw.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(EQ_URL, raw)
digest = hashlib.sha256(raw.read_bytes()).hexdigest()
print("EntityQuestions", raw, "sha256", "matches" if digest == EQ_SHA256 else f"DIFFERS ({digest})")

from datasets import load_dataset  # noqa: E402

for repo, config, split in (("akariasai/PopQA", None, "test"), ("Salesforce/wikitext", "wikitext-2-raw-v1", "test"),
                            ("allenai/ai2_arc", "ARC-Easy", "test"), ("Rowan/hellaswag", None, "validation")):
    ds = load_dataset(repo, config, split=split, revision=HF_REVISIONS[repo])
    print(repo, config or "", split, len(ds), "rows")
