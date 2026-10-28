"""Knowledge-graph data layer: PopQA splits, Wikidata statements, verbalizers and calibration text builders.

Every calibration source is a list of text lines. Packing into token sequences happens in pack().
"""
from __future__ import annotations

import json
import random
import re
import time
import urllib.parse
import urllib.request
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import paths as P

ROOT = P.ROOT
SPLITS, KG, CALIB = P.SPLITS, P.KG, P.CALIB
RESULTS = P.GRID                     # output folder of run_grid; scripts that reuse run_grid point it elsewhere
PAIRED = False                       # True: calib_path() gives the paired variants (paths.calib_file)
# Wikimedia asks every client for a user agent with contact details: set KGCAL_USER_AGENT before fetching data
UA = os.environ.get("KGCAL_USER_AGENT", "kgcal-research-code/1.0 (set KGCAL_USER_AGENT to your contact address)")
# Hugging Face dataset revisions (the snapshots every run of the paper read)
HF_REVISIONS = {"akariasai/PopQA": "098765c79ea10a2cb19c828324e33281b8336ec0",
                "allenai/c4": "1588ec454efa1a09f29cd18ddd04fe05fc8653a2",
                "Salesforce/wikitext": "b08601e04326c79dfdd32d625aee71d232d685c3",
                "allenai/ai2_arc": "210d026faf9955653af8916fad021475a3f00453",
                "Rowan/hellaswag": "218ec52e09a7e7462a5400043bb9a69a41d06b76"}


def calib_path(name: str) -> Path:
    """Calibration text `name` as run_grid reads it: data/calib, or the paired variant when PAIRED is set."""
    return P.calib_file(name, paired=PAIRED)

# PopQA relation -> Wikidata property, declarative template, cloze template (declarative without the object)
REL = {
    "occupation": ("P106", "{s}'s occupation is {o}."),
    "place of birth": ("P19", "{s} was born in {o}."),
    "genre": ("P136", "The genre of {s} is {o}."),
    "father": ("P22", "The father of {s} is {o}."),
    "country": ("P17", "{s} is located in {o}."),
    "producer": ("P162", "{s} was produced by {o}."),
    "director": ("P57", "{s} was directed by {o}."),
    "capital of": ("P1376", "{s} is the capital of {o}."),
    "screenwriter": ("P58", "The screenplay of {s} was written by {o}."),
    "composer": ("P86", "The music of {s} was composed by {o}."),
    "color": ("P462", "The color of {s} is {o}."),
    "religion": ("P140", "The religion of {s} is {o}."),
    "sport": ("P641", "{s} plays {o}."),
    "author": ("P50", "{s} was written by {o}."),
    "mother": ("P25", "The mother of {s} is {o}."),
    "capital": ("P36", "The capital of {s} is {o}."),
}
TESTED_PROPS = {v[0] for v in REL.values()}
# Wikimedia-internal or catalogue properties that carry no world knowledge
SKIP_PROPS = {"P910", "P1343", "P5008", "P1424", "P7084", "P8989", "P3876", "P2184", "P1151", "P1792", "P6104",
              "P373", "P935", "P1753", "P2354", "P1740", "P1465", "P1464", "P2959", "P5125", "P8402", "P972"}
PREP_END = re.compile(r"\b(of|in|at|for|by|to|from|with|on|as)$")


def ans(r: dict) -> list[str]:
    return json.loads(r["possible_answers"])


def qid(uri: str) -> str:
    return uri.rsplit("/", 1)[1]


def declarative(rel: str, s: str, o: str) -> str:
    return REL[rel][1].format(s=s, o=o)


def cloze(rel: str, s: str) -> str:
    return REL[rel][1].split(" {o}")[0].format(s=s)


VERB_START = {"contains", "has", "follows", "followed", "replaces", "replaced", "shares", "located", "part", "member",
              "owned", "operated", "published", "produced", "named", "said", "applies", "uses", "used", "depicts",
              "described", "based", "derived", "influenced", "connects", "crosses", "participant", "different",
              "significant", "performer", "nominated", "affiliated", "headquarters", "is", "was", "lies", "flows"}


def generic_statement(s: str, plabel: str, o: str) -> str:
    """Schema-derived verbalization from the Wikidata property label.

    Noun labels ("occupation") give "The occupation of S is O."; relational labels ("member of", "contains the
    administrative territorial entity") give "S member of O." / "S contains the administrative territorial entity O."
    """
    first = plabel.split()[0].lower() if plabel else ""
    if plabel == "instance of":
        return f"{s} is an instance of {o}."
    if plabel == "subclass of":
        return f"{s} is a subclass of {o}."
    if PREP_END.search(plabel) or first in VERB_START:
        return f"{s} {plabel} {o}."
    return f"The {plabel} of {s} is {o}."


# --------------------------------------------------------------------------- PopQA and splits
def load_popqa() -> list[dict]:
    """PopQA test split; the 3 rows without a subject name are dropped (14,264 facts remain)."""
    from datasets import load_dataset
    rows = [dict(r) for r in load_dataset("akariasai/PopQA", split="test", revision=HF_REVISIONS["akariasai/PopQA"])]
    return [r for r in rows if r["subj"] and r["question"] and r["possible_answers"]]


def make_split(rows: list[dict], seed: int = 0) -> dict:
    """Subject-disjoint split: E_cal 50%; E_eval 50% (validation 20%, test 80% of E_eval subjects)."""
    path = SPLITS / "popqa_split.json"
    if path.exists():
        return json.loads(path.read_text())
    subj = sorted({qid(r["s_uri"]) for r in rows})
    rng = random.Random(seed)
    rng.shuffle(subj)
    half = len(subj) // 2
    cal, ev = subj[:half], subj[half:]
    nval = len(ev) // 5
    split = {"seed": seed, "cal": sorted(cal), "val": sorted(ev[:nval]), "test": sorted(ev[nval:])}
    path.write_text(json.dumps(split))
    return split


def facts_by_split(rows: list[dict], split: dict) -> dict[str, list[dict]]:
    where = {q: k for k in ("cal", "val", "test") for q in split[k]}
    out = {"cal": [], "val": [], "test": []}
    for r in rows:
        out[where[qid(r["s_uri"])]].append(r)
    return out


# --------------------------------------------------------------------------- Wikidata (SPARQL)
def sparql(query: str, retries: int = 5) -> list[dict]:
    url = "https://query.wikidata.org/sparql?" + urllib.parse.urlencode({"query": query, "format": "json"})
    for k in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/sparql-results+json"})
            with urllib.request.urlopen(req, timeout=90) as resp:
                return json.load(resp)["results"]["bindings"]
        except Exception as e:  # noqa: BLE001
            wait = 5 * (k + 1)
            print(f"sparql retry {k + 1} after error {e!r}; sleeping {wait}s", flush=True)
            time.sleep(wait)
    raise RuntimeError("SPARQL failed repeatedly")


def fetch_statements(qids: list[str], batch: int = 80) -> dict[str, list[list[str]]]:
    """Truthy item-valued statements (subject -> [[property, object QID, object label], ...]), cached."""
    path = KG / "statements.json"
    have = json.loads(path.read_text()) if path.exists() else {}
    todo = [q for q in qids if q not in have]
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        vals = " ".join(f"wd:{q}" for q in chunk)
        rows = sparql(f"""SELECT ?s ?p ?o ?oLabel WHERE {{
          VALUES ?s {{ {vals} }} ?s ?wdt ?o . ?prop wikibase:directClaim ?wdt .
          FILTER(STRSTARTS(STR(?o), "http://www.wikidata.org/entity/Q"))
          BIND(STRAFTER(STR(?prop), "entity/") AS ?p)
          ?o rdfs:label ?oLabel . FILTER(LANG(?oLabel) = "en") }}""")
        for q in chunk:
            have.setdefault(q, [])
        for b in rows:
            have[qid(b["s"]["value"])].append([b["p"]["value"], qid(b["o"]["value"]), b["oLabel"]["value"]])
        if (i // batch) % 10 == 0:
            path.write_text(json.dumps(have))
            print(f"statements: {min(i + batch, len(todo))}/{len(todo)} subjects fetched", flush=True)
        time.sleep(0.5)
    path.write_text(json.dumps(have))
    return have


def fetch_property_labels(pids: list[str]) -> dict[str, str]:
    path = KG / "property_labels.json"
    have = json.loads(path.read_text()) if path.exists() else {}
    todo = [p for p in pids if p not in have]
    for i in range(0, len(todo), 200):
        vals = " ".join(f"wd:{p}" for p in todo[i:i + 200])
        for b in sparql(f"""SELECT ?p ?l WHERE {{ VALUES ?p {{ {vals} }} ?p rdfs:label ?l .
                            FILTER(LANG(?l) = "en") }}"""):
            have[qid(b["p"]["value"])] = b["l"]["value"]
        time.sleep(0.5)
    path.write_text(json.dumps(have))
    return have


def fetch_types(qids: list[str]) -> dict[str, list[str]]:
    """P31 (instance of) of items, for type-consistency checks of counterfactual objects."""
    path = KG / "types.json"
    have = json.loads(path.read_text()) if path.exists() else {}
    todo = [q for q in qids if q not in have]
    for i in range(0, len(todo), 250):
        chunk = todo[i:i + 250]
        for q in chunk:
            have.setdefault(q, [])
        vals = " ".join(f"wd:{q}" for q in chunk)
        for b in sparql(f"SELECT ?o ?t WHERE {{ VALUES ?o {{ {vals} }} ?o wdt:P31 ?t . }}"):
            have[qid(b["o"]["value"])].append(qid(b["t"]["value"]))
        time.sleep(0.5)
    path.write_text(json.dumps(have))
    return have


def fetch_wiki_summaries(titles: list[str], workers: int = 8) -> dict[str, str]:
    path = KG / "wiki_summaries.json"
    have = json.loads(path.read_text()) if path.exists() else {}
    todo = [t for t in titles if t not in have]

    def one(t: str) -> tuple[str, str]:
        url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + urllib.parse.quote(t.replace(" ", "_"))
        for k in range(3):
            try:
                with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=30) as r:
                    return t, json.load(r).get("extract", "")
            except Exception:  # noqa: BLE001
                time.sleep(2 * (k + 1))
        return t, ""

    with ThreadPoolExecutor(workers) as ex:
        for n, (t, x) in enumerate(ex.map(one, todo)):
            have[t] = x
            if n % 1000 == 0:
                path.write_text(json.dumps(have))
                print(f"wiki summaries: {n}/{len(todo)}", flush=True)
    path.write_text(json.dumps(have))
    return have


# --------------------------------------------------------------------------- calibration text builders
def build_calibration_sources(rows: list[dict], split: dict, seed: int = 0) -> dict[str, list[str]]:
    """All calibration text sources (except C4), as shuffled lists of lines. Writes calib/<name>.jsonl."""
    rng = random.Random(seed)
    F = facts_by_split(rows, split)
    st = json.loads((KG / "statements.json").read_text())
    plab = json.loads((KG / "property_labels.json").read_text())
    types = json.loads((KG / "types.json").read_text()) if (KG / "types.json").exists() else {}
    wiki = json.loads((KG / "wiki_summaries.json").read_text()) if (KG / "wiki_summaries.json").exists() else {}
    cal = F["cal"]

    # counterfactual objects: another object of the same relation, never a gold answer of this subject,
    # preferring an object that shares a P31 type with the gold object
    pool: dict[str, list[tuple[str, str]]] = {}
    for r in cal:
        pool.setdefault(r["prop"], []).append((ans(r)[0], qid(r["o_uri"])))
    def counterfactual(r: dict) -> str:
        gold = {a.lower() for a in ans(r)}
        gt = set(types.get(qid(r["o_uri"]), []))
        cands = [(o, q) for o, q in pool[r["prop"]] if o.lower() not in gold]
        typed = [(o, q) for o, q in cands if gt & set(types.get(q, []))]
        return rng.choice(typed or cands)[0]

    # schema-only: synthetic subject names recombined from name parts of other subjects of the same relation
    real = {r["subj"].lower() for r in rows}
    parts: dict[str, list[list[str]]] = {}
    for r in cal:
        parts.setdefault(r["prop"], []).append(r["subj"].split())
    def synthetic(r: dict) -> str:
        for _ in range(50):
            a, b = rng.choice(parts[r["prop"]]), rng.choice(parts[r["prop"]])
            name = " ".join(a[: max(1, len(a) // 2)] + b[max(1, len(b) // 2):] if len(b) > 1 else a[:1] + b)
            if name.lower() not in real:
                return name
        return r["subj"] + " " + rng.choice(["Varn", "Oltes", "Brask"])

    src: dict[str, list[str]] = {}
    src["KG-D"] = [declarative(r["prop"], r["subj"], ans(r)[0]) for r in cal]
    src["KG-D~"] = [declarative(r["prop"], r["subj"], counterfactual(r)) for r in cal]
    src["KG-QA"] = [f"Q: {r['question']} A: {ans(r)[0]}" for r in cal]
    src["KG-QA~"] = [f"Q: {r['question']} A: {counterfactual(r)}" for r in cal]
    src["KG-S"] = [declarative(r["prop"], synthetic(r), ans(r)[0]) for r in cal]
    # relation-disjoint: statements of calibration subjects with properties outside the tested relations
    other = []
    for r in cal:
        for p, oq, ol in st.get(qid(r["s_uri"]), []):
            if p in TESTED_PROPS or p in SKIP_PROPS or p not in plab:
                continue
            other.append(generic_statement(r["subj"], plab[p], ol))
    src["KG-O"] = other
    # anchored: statements of evaluation subjects; drop the tested property and any statement naming a gold answer
    anchor = []
    for r in F["val"] + F["test"]:
        gold = {a.lower() for a in ans(r)}
        gold_q = qid(r["o_uri"])
        tested = REL[r["prop"]][0]
        for p, oq, ol in st.get(qid(r["s_uri"]), []):
            if p == tested or p in SKIP_PROPS or p not in plab or oq == gold_q or ol.lower() in gold:
                continue
            anchor.append(generic_statement(r["subj"], plab[p], ol))
    src["KG-A"] = anchor
    src["WIKI"] = [wiki[r["s_wiki_title"]] for r in cal if wiki.get(r["s_wiki_title"])]
    for k in src:
        rng.shuffle(src[k])
        with open(CALIB / f"{k}.jsonl", "w") as f:
            for line in src[k]:
                f.write(json.dumps(line) + "\n")
    return src


def leakage_report(rows: list[dict], split: dict, src: dict[str, list[str]]) -> dict:
    """Share of test facts whose gold answer string appears in a calibration source (as a whole word)."""
    F = facts_by_split(rows, split)
    out = {}
    for name, lines in src.items():
        blob = "\n".join(lines).lower()
        hit = 0
        for r in F["test"]:
            a = ans(r)[0].lower()
            if len(a) > 3 and re.search(r"\b" + re.escape(a) + r"\b", blob):
                hit += 1
        out[name] = round(hit / max(1, len(F["test"])), 4)
    (SPLITS / "leakage.json").write_text(json.dumps(out, indent=1))
    return out


def c4_lines(n_docs: int = 3000, skip: int = 0) -> list[str]:
    from datasets import load_dataset
    ds = load_dataset("allenai/c4", "en", split="validation", streaming=True,
                      revision=HF_REVISIONS["allenai/c4"])
    out = []
    for i, r in enumerate(ds):
        if i < skip:
            continue
        out.append(r["text"])
        if len(out) >= n_docs:
            break
    return out


def pack(tok, lines: list[str], n_seq: int, seq_len: int = 512):
    """Concatenate lines (newline separated) and cut into n_seq sequences of seq_len tokens."""
    import torch
    ids: list[int] = []
    for t in lines:
        ids.extend(tok(t + "\n", add_special_tokens=False)["input_ids"])
        if len(ids) >= n_seq * seq_len:
            break
    if len(ids) < n_seq * seq_len:
        raise ValueError(f"not enough text: {len(ids)} tokens for {n_seq} x {seq_len}")
    return torch.tensor(ids[: n_seq * seq_len]).view(n_seq, seq_len)


def leakage_strict(rows: list[dict], split: dict) -> dict:
    """Fact-level leakage checks (the coarse answer-string overlap counts common objects such as 'politician').

    1. Homonyms: test subjects whose surface name equals a calibration subject's name (different QID). For these,
       a calibration statement could state the same relation and answer.
    2. Anchored statements (KG-A): test facts for which a statement of the same subject has an object whose label
       contains, or is contained in, a gold answer alias (the build removed exact matches only).
    3. Wikipedia text (WIKI): test facts whose subject name and a gold answer co-occur in one summary.
    """
    F = facts_by_split(rows, split)
    st = json.loads((KG / "statements.json").read_text())
    plab = json.loads((KG / "property_labels.json").read_text())
    wiki = json.loads((KG / "wiki_summaries.json").read_text())
    cal_by_name: dict[str, list[dict]] = {}
    for r in F["cal"]:
        cal_by_name.setdefault(r["subj"].lower(), []).append(r)
    homonym = same_fact = 0
    for r in F["test"]:
        hits = cal_by_name.get(r["subj"].lower(), [])
        if hits:
            homonym += 1
            gold = {a.lower() for a in ans(r)}
            if any(h["prop"] == r["prop"] and gold & {a.lower() for a in ans(h)} for h in hits):
                same_fact += 1
    anchor_fuzzy = 0
    for r in F["test"]:
        gold = [a.lower() for a in ans(r) if len(a) > 3]
        tested = REL[r["prop"]][0]
        for p, oq, ol in st.get(qid(r["s_uri"]), []):
            if p == tested or p in SKIP_PROPS or p not in plab or oq == qid(r["o_uri"]):
                continue
            o = ol.lower()
            if o in {a.lower() for a in ans(r)}:
                continue
            if any(g in o or (len(o) > 3 and o in g) for g in gold):
                anchor_fuzzy += 1
                break
    summaries = [x.lower() for x in (wiki[r["s_wiki_title"]] for r in F["cal"]) if x]
    wiki_co = 0
    for r in F["test"]:
        s = r["subj"].lower()
        gold = [a.lower() for a in ans(r) if len(a) > 3]
        if len(s) > 3 and any(s in x and any(g in x for g in gold) for x in summaries):
            wiki_co += 1
    n = len(F["test"])
    out = {"test_facts": n,
           "homonym_subjects": round(homonym / n, 4), "homonym_same_fact": round(same_fact / n, 4),
           "anchor_fuzzy_answer": round(anchor_fuzzy / n, 4), "wiki_subject_answer_cooccur": round(wiki_co / n, 4)}
    (SPLITS / "leakage_strict.json").write_text(json.dumps(out, indent=1))
    return out


def fuzzy_hit(label: str, gold: list[str]) -> bool:
    o = label.lower()
    return any(o == g or (len(g) > 3 and g in o) or (len(o) > 3 and o in g) for g in gold)


def build_anchor_clean(rows: list[dict], split: dict, seed: int = 0) -> list[str]:
    """KG-Ac: anchored statements of evaluation subjects with the tested property removed and every statement whose
    object label equals, contains, or is contained in a gold answer alias removed (KG-A removed exact matches only)."""
    rng = random.Random(seed)
    F = facts_by_split(rows, split)
    st = json.loads((KG / "statements.json").read_text())
    plab = json.loads((KG / "property_labels.json").read_text())
    by_subj: dict[str, list[dict]] = {}
    for r in F["val"] + F["test"]:
        by_subj.setdefault(qid(r["s_uri"]), []).append(r)
    out = []
    for sq, facts in by_subj.items():
        gold = [a.lower() for r in facts for a in ans(r)]
        gold_q = {qid(r["o_uri"]) for r in facts}
        tested = {REL[r["prop"]][0] for r in facts}
        subj = facts[0]["subj"]
        for p, oq, ol in st.get(sq, []):
            if p in tested or p in SKIP_PROPS or p not in plab or oq in gold_q or fuzzy_hit(ol, gold):
                continue
            out.append(generic_statement(subj, plab[p], ol))
    rng.shuffle(out)
    with open(CALIB / "KG-Ac.jsonl", "w") as f:
        for line in out:
            f.write(json.dumps(line) + "\n")
    return out


def leakage_flags(rows: list[dict], split: dict) -> dict:
    """Per test fact (order of facts_by_split(...)['test']): homonym same fact, fuzzy answer in KG-A, Wikipedia
    co-occurrence. Used to rerun contrasts without the affected facts."""
    F = facts_by_split(rows, split)
    st = json.loads((KG / "statements.json").read_text())
    plab = json.loads((KG / "property_labels.json").read_text())
    wiki = json.loads((KG / "wiki_summaries.json").read_text())
    cal_by_name: dict[str, list[dict]] = {}
    for r in F["cal"]:
        cal_by_name.setdefault(r["subj"].lower(), []).append(r)
    summaries = [x.lower() for x in (wiki[r["s_wiki_title"]] for r in F["cal"]) if x]
    flags = {"homonym_same_fact": [], "anchor_fuzzy": [], "wiki_cooccur": []}
    for r in F["test"]:
        gold = [a.lower() for a in ans(r)]
        hits = cal_by_name.get(r["subj"].lower(), [])
        flags["homonym_same_fact"].append(any(h["prop"] == r["prop"] and set(gold) & {a.lower() for a in ans(h)}
                                              for h in hits))
        tested = REL[r["prop"]][0]
        flags["anchor_fuzzy"].append(any(p != tested and p not in SKIP_PROPS and p in plab and fuzzy_hit(ol, gold)
                                         for p, oq, ol in st.get(qid(r["s_uri"]), [])))
        s = r["subj"].lower()
        g4 = [g for g in gold if len(g) > 3]
        flags["wiki_cooccur"].append(len(s) > 3 and any(s in x and any(g in x for g in g4) for x in summaries))
    (SPLITS / "leakage_flags.json").write_text(json.dumps(flags))
    return {k: round(sum(v) / len(v), 4) for k, v in flags.items()}
