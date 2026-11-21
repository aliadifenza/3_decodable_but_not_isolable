"""Deterministic audit for the Kantian dataset (17-field JSONL + 6-column CSV mapping), any split.
Usage: python audit_dataset.py <out.json> <split.jsonl>:<mapping.csv> [<split2.jsonl>:<mapping2.csv> ...]
Checks: schema/types, ID uniqueness, JSONL<->CSV consistency, bundle composition, label consistency, role labels vs
dependency parse, member/non-member head-noun shortcut per stratum, kind-noun leakage, causal-prompt format
(patient mention, bystander mention, target continuation), exact/near duplicates within and across splits, template
disjointness across splits, G4 probe overlap (prompts/templates exact+near, probe nouns)."""
import json, csv, re, sys, os, collections, hashlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kinds import ALIASES
import spacy
NLP = spacy.load(os.environ.get("SPACY_MODEL", "en_core_web_trf"))
_cache = {}
def parse(text):
    if text not in _cache: _cache[text] = NLP(text)
    return _cache[text]
FIELDS = ["unique_id", "family", "proposed_split", "bundle_id", "record_type", "entity_kind", "entity_description", "comparison_entity_kind",
          "material", "relevant_properties", "grammatical_role", "sentence", "candidate_target_word", "alternative_target_words",
          "matching_rationale", "expected_category_membership", "potential_ambiguity"]
MAP_FIELDS = ["unique_id", "bundle_id", "split", "focal_kind", "property_group", "template_id"]
ROLE_CANON = {"subject": "subject", "passive subject": "passive_subject", "passive_subject": "passive_subject",
              "direct object": "direct_object", "direct_object": "direct_object", "indirect object": "indirect_object",
              "prepositional object": "prepositional_object", "prepositional_object": "prepositional_object",
              "object_of_preposition": "prepositional_object", "genitive complement (of-phrase)": "genitive_complement",
              "predicate nominative": "predicate_nominative", "indirect_object": "indirect_object", "genitive_complement": "genitive_complement", "predicate_nominative": "predicate_nominative"}
DEP2ROLE = {"nsubj": "subject", "nsubjpass": "passive_subject", "dobj": "direct_object", "dative": "indirect_object",
            "pobj": "prepositional_object", "attr": "predicate_nominative", "appos": None, "conj": None}
NEAR = 0.8
G4_ROOT = os.environ.get("G4_ROOT", "/home/claude/3_kantian_positions/g4_probe")


def toks(s): return set(re.findall(r"[a-z]+", (s or "").lower()))
def jacc(a, b): return len(a & b) / max(1, len(a | b))


def g4():
    sys.path.insert(0, G4_ROOT); import g4_probe as G
    m = G.probe_only_manifest(); items = list(m["items"])
    try:
        sys.path.insert(0, os.path.join(G4_ROOT, "..", "g5_verification")); import g5v_probe as V; items += V.derived_manifest()["items"]
    except Exception: pass
    return items, set(m["nouns"])


def ref_head(doc, r):
    """Head token of the referent noun phrase. Recognition: the NP whose subtree best overlaps entity_description
    (falls back to the first nominal argument). Causal: first mention of a kind alias."""
    if r["record_type"] == "causal_prediction":
        for a in sorted(ALIASES.get(r["entity_kind"], []), key=len, reverse=True):
            for m in re.finditer(r"\b%s\b" % re.escape(a), doc.text, re.I):
                tk = [t for t in doc if m.start() <= t.idx < m.end()]
                if tk and tk[-1].pos_ in ("NOUN", "PROPN"): return tk[-1]
        return None
    dt = toks(r.get("entity_description"))
    best, bs = None, -1
    for t in doc:
        if t.pos_ not in ("NOUN", "PROPN") or t.dep_ in ("compound",): continue
        sub = toks(" ".join(x.text for x in t.subtree))
        s = jacc(sub, dt) + (0.01 if t.dep_ in DEP2ROLE else 0)
        if s > bs: best, bs = t, s
    return best


def parse_role(t):
    if t is None: return None
    x = t
    while x.dep_ in ("conj", "appos") and x.head is not x: x = x.head
    if x.dep_ == "pobj" and x.head.dep_ == "prep" and x.head.text.lower() == "of" and x.head.head.pos_ in ("NOUN", "PRON"): return "genitive_complement"
    if x.dep_ == "pobj" and x.head.dep_ == "dative": return "indirect_object"
    return DEP2ROLE.get(x.dep_, x.dep_)


def count_alias(text, kind):
    """Non-overlapping mentions of the kind (longest alias first); 'can'/'corn'/'wax' only when parsed as nouns."""
    d = parse(text); taken = set(); n = 0
    for a in sorted(ALIASES.get(kind, []), key=len, reverse=True):
        for m in re.finditer(r"\b%s\b" % re.escape(a), text, re.I):
            span = set(range(m.start(), m.end()))
            if span & taken: continue
            tk = [t for t in d if m.start() <= t.idx < m.end()]
            if a in ("can", "cans", "corn", "wax") and tk and tk[-1].pos_ not in ("NOUN", "PROPN"): continue
            taken |= span; n += 1
    return n


def audit(pairs, out):
    A = dict(files=[], checks={}, issues=collections.defaultdict(list), stats={}); I = A["issues"]
    allrecs = []
    for jp, cp in pairs:
        R = [json.loads(l) for l in open(jp) if l.strip()]; M = list(csv.DictReader(open(cp)))
        A["files"].append(dict(jsonl=os.path.basename(jp), sha256=hashlib.sha256(open(jp, "rb").read()).hexdigest(), n=len(R),
                               csv=os.path.basename(cp), csv_sha256=hashlib.sha256(open(cp, "rb").read()).hexdigest()))
        mm = {m["unique_id"]: m for m in M}
        if [m for m in M if list(m.keys()) != MAP_FIELDS]: I["CSV_COLUMNS"].append(cp)
        for r in R:
            if list(r.keys()) != FIELDS: I["SCHEMA_FIELDS"].append(r.get("unique_id"))
            m = mm.get(r["unique_id"])
            if m is None: I["MISSING_IN_CSV"].append(r["unique_id"]); continue
            if (m["bundle_id"], m["split"], m["focal_kind"]) != (r["bundle_id"], r["proposed_split"], r["entity_kind"]): I["CSV_MISMATCH"].append(r["unique_id"])
            r["_pg"], r["_tpl"] = m["property_group"], m["template_id"]
        if set(mm) - {r["unique_id"] for r in R}: I["EXTRA_IN_CSV"] += sorted(set(mm) - {r["unique_id"] for r in R})
        allrecs += R
    ids = collections.Counter(r["unique_id"] for r in allrecs); I["DUP_ID"] = [k for k, v in ids.items() if v > 1]
    for r in allrecs:
        rt, mem = r["record_type"], r["expected_category_membership"]
        if (rt, mem) not in (("identity_preserving", True), ("matched_nonmember", False), ("causal_prediction", None)): I["LABEL_INCONSISTENT"].append(r["unique_id"])
        if rt == "causal_prediction":
            if not r["candidate_target_word"] or len(r["candidate_target_word"].split()) != 1: I["TARGET_NOT_SINGLE_WORD"].append(r["unique_id"])
            if r["candidate_target_word"] in r["alternative_target_words"]: I["TARGET_IN_ALTERNATIVES"].append(r["unique_id"])
        elif r["candidate_target_word"] or r["alternative_target_words"]: I["TARGET_ON_RECOGNITION"].append(r["unique_id"])
        if r["grammatical_role"] not in ROLE_CANON: I["ROLE_LABEL_UNKNOWN"].append(r["unique_id"])
    # roles vs parse; head nouns; leakage
    heads = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
    role_mis = []
    for r in allrecs:
        d = parse(r["sentence"]); h = ref_head(d, r); r["_head"] = h.lemma_.lower() if h is not None else None
        pr = parse_role(h); lab = ROLE_CANON.get(r["grammatical_role"])
        r["_parsed_role"] = pr
        if h is None: I["REFERENT_NOT_FOUND"].append(r["unique_id"])
        elif lab and pr != lab: role_mis.append(dict(id=r["unique_id"], label=r["grammatical_role"], parsed=pr, head=h.text))
        if r["record_type"] != "causal_prediction":
            heads[(r["proposed_split"], r["family"], r["_pg"])]["member" if r["expected_category_membership"] else "non"][r["_head"]] += 1
            if count_alias(r["sentence"], r["entity_kind"]): I["KIND_NOUN_IN_RECOGNITION_SENTENCE"].append(r["unique_id"])
    A["role_mismatches"] = role_mis
    sc = {}
    for k, v in heads.items():
        mh, nh = set(v["member"]), set(v["non"]); shared = mh & nh
        frac_m = sum(c for h, c in v["member"].items() if h in shared) / max(1, sum(v["member"].values()))
        frac_n = sum(c for h, c in v["non"].items() if h in shared) / max(1, sum(v["non"].values()))
        sc["|".join(k)] = dict(member_heads=dict(v["member"].most_common(6)), non_heads=dict(v["non"].most_common(6)),
                               shared=sorted(shared), member_frac_shared_head=round(frac_m, 2), non_frac_shared_head=round(frac_n, 2))
    A["head_noun_shortcut"] = sc
    # causal format
    cf = []
    for r in allrecs:
        if r["record_type"] != "causal_prediction": continue
        s = r["sentence"]; pm = count_alias(s, r["entity_kind"]); bys = r["comparison_entity_kind"]
        bm = len(re.findall(r"\b%s\b" % re.escape(bys.split()[-1]), s, re.I)) if bys else None
        cf.append(dict(id=r["unique_id"], family=r["family"], patient_mentions=pm, bystander_field=bys, bystander_head_mentions=bm,
                       ends=" ".join(s.split()[-3:]), target=r["candidate_target_word"]))
        if pm != 1: I["CAUSAL_PATIENT_MENTIONS_NE_1"].append(r["unique_id"])
        if bys and bm == 0: I["CAUSAL_BYSTANDER_FIELD_NOT_IN_PROMPT"].append(r["unique_id"])
    A["causal_format"] = cf
    # duplicates and splits
    T = [(r["unique_id"], r["proposed_split"], toks(r["sentence"]), r["sentence"]) for r in allrecs]
    ex = collections.defaultdict(list)
    for x in T: ex[x[3].strip().lower()].append(x[0])
    I["EXACT_DUP_SENTENCE"] = [v for v in ex.values() if len(v) > 1]
    near_x, near_w = [], 0
    for i in range(len(T)):
        for j in range(i + 1, len(T)):
            if jacc(T[i][2], T[j][2]) >= NEAR:
                if T[i][1] != T[j][1]: near_x.append((T[i][0], T[j][0]))
                else: near_w += 1
    I["NEAR_DUP_ACROSS_SPLITS"] = near_x; A["stats"]["near_dup_pairs_within_split"] = near_w
    tpl = collections.defaultdict(set); bun = collections.defaultdict(set)
    for r in allrecs: tpl[r["_tpl"]].add(r["proposed_split"]); bun[r["bundle_id"]].add(r["proposed_split"])
    I["TEMPLATE_ACROSS_SPLITS"] = sorted(t for t, s in tpl.items() if len(s) > 1)
    I["BUNDLE_ACROSS_SPLITS"] = sorted(b for b, s in bun.items() if len(s) > 1)
    items, nouns = g4()
    gt = [(p, toks(p.replace("{}", ""))) for p in items]
    for r in allrecs:
        st = toks(r["sentence"])
        if r["sentence"] in items: I["G4_EXACT"].append(r["unique_id"])
        elif any(jacc(st, t) >= NEAR for _, t in gt): I["G4_NEAR_DUP"].append(r["unique_id"])
        hit = sorted(n for n in nouns if re.search(r"\b%s\b" % re.escape(n), r["sentence"], re.I))
        if hit: A["issues"]["G4_NOUN_PRESENT:" + r["proposed_split"]].append(dict(id=r["unique_id"], nouns=hit))
    A["stats"]["counts"] = {f"{s}|{f}|{t}": n for (s, f, t), n in collections.Counter((r["proposed_split"], r["family"], r["record_type"]) for r in allrecs).items()}
    A["stats"]["role_counts"] = {f"{s}|{f}|{ROLE_CANON.get(x, x)}": n for (s, f, x), n in collections.Counter((r["proposed_split"], r["family"], r["grammatical_role"]) for r in allrecs).items()}
    A["issues"] = {k: v for k, v in A["issues"].items() if v}
    A["summary"] = {k: len(v) for k, v in A["issues"].items()} | dict(role_mismatches=len(role_mis))
    json.dump(A, open(out, "w"), indent=1, default=str)
    return A


if __name__ == "__main__":
    a = audit([tuple(p.split(":")) for p in sys.argv[2:]], sys.argv[1]); print(json.dumps(a["summary"], indent=1))
