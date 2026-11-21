"""Dataset v1 (discovery v1.2 + validation v1.1) -> g5r_probe input schema. Engineering adapter for the FROZEN G5 spec.

Refuses any evaluation file. Deterministic. Usage (local, needs spaCy en_core_web_trf for 129 paraphrased discovery spans):
  python3 g5r_from_dataset.py <discovery.jsonl> <discovery_mapping.csv> <validation.jsonl> <validation_mapping.csv> <out.jsonl>

Recognition records -> ref_type "description" (the noun-free referent phrase; its LAST token is measured).
Noun-reference items (needed by frozen G5 §4 noun AUC / lexical gap) are DERIVED: the same sentence with the referent phrase
replaced by "the <kind noun>" (members) or "the <comparison noun>" (non-members; first alternative when the field lists several).
Causal records (discovery only; validation causal items are not used by G5r): prompt_c2 = dataset prompt (bystander present),
prompt = same prompt with the bystander clause removed (bystander-shift eligibility, Decision 4). P3 feasibility uses discovery
causal items (frozen §6; no separate P3-discovery split exists)."""
import csv, json, re, sys, hashlib, collections

KIND_NOUN = {"raw hen's egg": "egg", "aluminium beverage can": "beverage can", "car windshield": "windshield",
             "corn kernel": "corn kernel", "raw egg white": "egg white", "beeswax block": "beeswax block",
             "parchment": "parchment", "balsa strip": "balsa strip", "acrylic sheet": "acrylic sheet",
             "umbrella": "umbrella", "wristwatch": "wristwatch", "teapot": "teapot"}
CUT_NONMEMBER_FALLBACK = {"parchment": "synthetic writing film", "balsa strip": "foam strip", "acrylic sheet": "glass panel"}
LEAD = re.compile(r"^(While|With|As|Beside|Near)\b[^,]*,\s*")
MID = re.compile(r"\s(while|as) the (teapot|umbrella|wristwatch)\b[^,]*? and ")
OVERRIDE = {   # unique_id -> exact referent phrase in the sentence, for spans the parse heuristic gets wrong (reviewed 2026-11-21)
    "HEAT_DISC_B001_A2": "the deep crimson seed with an unbroken hull and moist starchy center",
    "HEAT_DISC_B001_B": "The smooth ochre pebble worn oval by river water",
    "HEAT_DISC_B006_A2": "the flattened dehydrated grain with a chalky starchy center",
    "HEAT_DISC_B008_A1": "the parched golden grain with a dense starchy interior",
    "HEAT_DISC_B010_A2": "the dense, slightly opalescent protein liquid",
    "HEAT_DISC_B014_A2": "the glazed coat of aqueous protein",
    "HEAT_DISC_B016_A1": "the wet glistening film of aqueous protein",
    "HEAT_DISC_SUP_PG1_sealed_moisture_hull_S06": "the desiccated red seed with two cotyledons",
    "HEAT_DISC_SUP_PG2_aqueous_protein_film_S06": "the delicate aqueous guar-gum film",
    "HEAT_DISC_SUP_PG1_coagulable_albumin_fluid_S08": "the pale golden gelatin-rich liquid",
    "CUT_D033": "A delicate timber length with open grain",
}
BYST = {"physical_impact": "umbrella", "heating": "wristwatch", "cutting": "teapot"}


def toks(s): return re.findall(r"[a-z]+", s.lower())
def jacc(a, b): a, b = set(a), set(b); return len(a & b) / max(1, len(a | b))
def first_alt(s): return re.split(r"\s*/\s*|\s+or\s+", s.strip())[0]


_nlp = None
def nlp():
    global _nlp
    if _nlp is None:
        import spacy; _nlp = spacy.load("en_core_web_trf")
    return _nlp


LOC = {"on", "in", "inside", "beside", "near", "onto", "across", "into", "at", "under", "over", "by", "along"}
ROLE_DEPS = {"subject": {"nsubj"}, "passive_subject": {"nsubjpass"}, "direct_object": {"dobj"}, "prepositional_object": {"pobj"},
             "indirect_object": {"dative"}, "genitive_complement": {"pobj"}}


def ref_span(sent, desc, role, override=None):
    """Verbatim match if possible. Otherwise (paraphrased discovery descriptions): dependency parse; candidate referent heads
    with the record's grammatical role (fallback: any noun); pick the subtree covering most description words (ties:
    shortest); drop trailing locative PPs attached to the head. Measured position = last token."""
    if override: i = sent.find(override); assert i >= 0, override; return [i, i + len(override)], "manual_override"
    i = sent.lower().find(desc.lower().rstrip("."))
    if i >= 0: return [i, i + len(desc.rstrip("."))], "verbatim"
    d = set(toks(desc)); doc = nlp()(sent)
    nouns = [t for t in doc if t.pos_ in ("NOUN", "PROPN") and t.dep_ != "ROOT"]
    cands = [t for t in nouns if t.dep_ in ROLE_DEPS.get(role, set())] or nouns
    best = None
    for t in cands:
        drop = set()
        for ch in t.children:                       # trailing locative PPs attached to the head are not part of the referent
            if ch.dep_ == "prep" and ch.lower_ in LOC and ch.i > t.i: drop |= {x.i for x in ch.subtree}
        sub = [x for x in t.subtree if x.i not in drop]
        sub = sub[:next((k for k in range(1, len(sub)) if sub[k].i != sub[k - 1].i + 1), len(sub))]   # contiguous prefix
        while len(sub) > 1 and sub[-1].is_punct: sub = sub[:-1]
        a, b = sub[0].idx, sub[-1].idx + len(sub[-1].text)
        sc = (len(set(toks(sent[a:b])) & d) / max(1, len(d)), -(b - a))
        if best is None or sc > best[0]: best = (sc, [a, b])
    return best[1], "parsed_subtree"


def noun_variant(sent, span, noun):
    a, b = span; art = "The" if a == 0 else "the"
    pre = sent[:a]
    if a > 0 and re.search(r"\b(the|a|an)\s$", pre, re.I): pre = re.sub(r"\b(the|a|an)\s$", "", pre, flags=re.I)
    post = sent[b:]
    if "," in sent[a:b] and post.startswith(","): post = post[1:]     # drop the comma that closed an appositive description
    s = pre + f"{art} {noun}" + post; na = len(pre); return s, [na, na + len(art) + 1 + len(noun)]


def strip_bystander(s, fam):
    m = LEAD.match(s)
    if m and BYST[fam] in m.group(0):
        r = s[m.end():]; return r[0].upper() + r[1:]
    m = MID.search(s)
    if m: return s[:m.start()] + " and " + s[m.end():]
    raise ValueError("bystander clause not found: " + s)


def patient_word(s, kind):
    from_ = sorted(ALIASES[kind], key=len, reverse=True)
    for a in from_:
        if re.search(r"\b%s\b" % re.escape(a), s.lower()): return a.split()[-1]
    raise ValueError(f"patient not found ({kind}): {s}")


def main(dj, dm, vj, vm, out):
    global ALIASES
    sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__file__), "..", "datasets", "kantian_v1", "scripts"))
    from kinds import ALIASES
    recs, audit = [], collections.Counter()
    for jp, mp in ((dj, dm), (vj, vm)):
        if "evaluation" in jp: raise SystemExit("REFUSED: evaluation file")
        M = {m["unique_id"]: m for m in csv.DictReader(open(mp))}
        for r in (json.loads(l) for l in open(jp) if l.strip()):
            if r["proposed_split"] == "evaluation": raise SystemExit("REFUSED: evaluation record")
            m = M[r["unique_id"]]
            if r["record_type"] == "causal_prediction":
                if r["proposed_split"] != "discovery": audit["causal_validation_skipped"] += 1; continue
                c2 = r["sentence"]; p = strip_bystander(c2, r["family"]); pw = patient_word(p, r["entity_kind"])
                bw = r["comparison_entity_kind"]
                assert len(re.findall(r"\b%s\b" % re.escape(pw), p.lower())) == 1 and len(re.findall(r"\b%s\b" % bw, c2.lower())) == 1, r["unique_id"]
                recs.append(dict(id=r["unique_id"], bundle_id=r["bundle_id"], template_id=m["template_id"], split="discovery", type="causal",
                                 family=r["family"], kind=r["entity_kind"], patient=pw, bystander=bw, target=r["candidate_target_word"],
                                 prompt=p, prompt_c2=c2)); audit["causal"] += 1; continue
            mem = bool(r["expected_category_membership"])
            span, how = ref_span(r["sentence"], r["entity_description"], r["grammatical_role"], OVERRIDE.get(r["unique_id"])); audit["span_" + how] += 1
            base = dict(bundle_id=r["bundle_id"], split=r["proposed_split"], type="recognition", target_kind=r["entity_kind"], is_member=mem,
                        stratum="member:identity_variant" if mem else "property_matched:" + m["property_group"], role=r["grammatical_role"])
            recs.append(dict(base, id=r["unique_id"], template_id=m["template_id"], ref_type="description", text=r["sentence"], ref_span=span,
                             span_method=how))
            noun = KIND_NOUN[r["entity_kind"]] if mem else first_alt(r["comparison_entity_kind"] or CUT_NONMEMBER_FALLBACK.get(r["entity_kind"], ""))
            assert noun, r["unique_id"]
            s2, sp2 = noun_variant(r["sentence"], span, noun)
            recs.append(dict(base, id=r["unique_id"] + "_NOUN", template_id=m["template_id"] + "_NOUN", ref_type="noun", text=s2, ref_span=sp2,
                             derived_from=r["unique_id"])); audit["noun_derived"] += 1
    with open(out, "w") as f:
        for r in recs: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    b = open(out, "rb").read(); print(dict(audit), "records", len(recs), "sha256", hashlib.sha256(b).hexdigest())


if __name__ == "__main__":
    main(*sys.argv[1:6])
