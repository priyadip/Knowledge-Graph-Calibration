"""Check every released input file against data/MANIFEST.json (SHA-256 and line count).

python scripts/data/check_data.py            check (exit status 1 if a file differs or is missing)
python scripts/data/check_data.py --write    rewrite the manifest from the files present
The downloaded EntityQuestions archive (data/entityquestions/raw) is checked by download_data.py instead.
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from kgcal import paths as P  # noqa: E402

MAN = P.DATA / "MANIFEST.json"


def entry(p: Path) -> dict:
    b = p.read_bytes()
    return {"sha256": hashlib.sha256(b).hexdigest(), "lines": b.count(b"\n"), "bytes": len(b)}


files = sorted(p for p in P.DATA.rglob("*") if p.is_file() and p != MAN and "raw" not in p.relative_to(P.DATA).parts
               and p.suffix in (".json", ".jsonl"))
now = {str(p.relative_to(P.DATA)): entry(p) for p in files}
if "--write" in sys.argv:
    MAN.write_text(json.dumps(now, indent=1) + "\n")
    print(f"wrote {MAN} ({len(now)} files)")
    sys.exit(0)
ref = json.loads(MAN.read_text())
bad = [k for k in ref if now.get(k) != ref[k]]
for k in bad:
    print("differs" if k in now else "missing", k)
print(f"{len(ref) - len(bad)} of {len(ref)} files match data/MANIFEST.json")
sys.exit(1 if bad else 0)
