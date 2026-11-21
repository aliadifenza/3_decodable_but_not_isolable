"""Adaption export -> g5r schema converter + dataset audit (kind-agnostic; no G4 kinds assumed).

Usage
  python3 adaption_convert.py inspect  <export>                      # list fields, example and distinct values (to write the mapping)
  python3 adaption_convert.py convert  <export> <mapping.json> <outdir>

Outputs (convert)
  outdir/g5r_input.jsonl   discovery / validation / p3_discovery records only (evaluation records are NEVER written or altered)
  outdir/audit.json        machine-readable audit;  outdir/audit.md  human summary (missing fields, counts, exclusions, checks)

Mapping file (JSON): {"format": "jsonl|json|csv", "records_key": null|"key",
  "fields": {schema_field: "dotted.source.path" | {"const": value} | null},
  "value_maps": {schema_field: {source_value: schema_value}},
  "type_values": {"recognition": [...], "causal": [...]}}
Schema fields: id bundle_id template_id split type target_kind is_member stratum ref_type role text ref_phrase ref_span
               mentioned_attribute family patient bystander target prompt prompt_c2"""
import csv, json, re, sys, os, hashlib, collections

SPLITS = {"discovery", "validation", "evaluation", "p3_discovery"}
REQ = dict(recognition=["id", "bundle_id", "template_id", "split", "target_kind", "is_member", "stratum", "ref_type", "role", "text"],
           causal=["id", "bundle_id", "split", "family", "patient", "bystander", "target", "prompt", "prompt_c2"])
DOMAIN = dict(ref_type={"noun", "description", "pronoun"})
MIN_V_PER_STRATUM, TARGET_EVAL_PER_FAMILY, NEAR_DUP = 10, 48, 0.8


def read_export(path, fmt=None, key=None):
    fmt = fmt or ("csv" if path.endswith(".csv") else "jsonl" if path.endswith(".jsonl") else "json")
    if fmt == "csv": return list(csv.DictReader(open(path, newline="")))
    if fmt == "jsonl": return [json.loads(l) for l in open(path) if l.strip()]
    d = json.load(open(path)); return d[key] if key else d


def get(rec, path):
    for p in path.split("."):
        if isinstance(rec, dict) and p in rec: rec = rec[p]
        else: return None
    return rec


def norm_tokens(s): return set(re.findall(r"[a-z]+", (s or "").lower()))
def jacc(a, b): return len(a & b) / max(1, len(a | b))
def count_word(text, w): return len(re.findall(r"\b%s\b" % re.escape(w.lower()), (text or "").lower())) if w else 0


def g4_exclusions():
    here = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [here, os.path.join(here, "..", "g4_probe")]
    import g4_probe as G4
    m = G4.probe_only_manifest()
    try:
        import g5v_probe as V; derived = V.derived_manifest()["items"]
    except Exception: derived = []
    return dict(items=set(m["items"]) | set(derived), nouns=set(m["nouns"]), templates=[t for t in m["items"] if "{}" in t])


def inspect(path):
    recs = read_export(path); keys = collections.Counter(); vals = collections.defaultdict(collections.Counter)
    def walk(r, pre=""):
        for k, v in r.items():
            p = pre + k
            if isinstance(v, dict): walk(v, p + ".")
            else:
                keys[p] += 1
                if not isinstance(v, (list, dict)) and len(str(v)) <= 60: vals[p][str(v)] += 1
    for r in recs: walk(r)
    out = dict(n_records=len(recs), fields={k: dict(present=c, distinct=len(vals[k]), top=vals[k].most_common(8)) for k, c in keys.items()})
    print(json.dumps(out, indent=1)); return out


def convert(path, mapping_path, outdir):
    M = json.load(open(mapping_path)); raw = read_export(path, M.get("format"), M.get("records_key"))
    X = g4_exclusions(); audit = dict(source=dict(path=os.path.basename(path), sha256=hashlib.sha256(open(path, "rb").read()).hexdigest(), n_raw=len(raw)),
                                      issues=collections.Counter(), examples=collections.defaultdict(list))
    def issue(code, rid):
        audit["issues"][code] += 1
        if len(audit["examples"][code]) < 5: audit["examples"][code].append(rid)
    tv = {v: t for t, vs in M.get("type_values", {}).items() for v in vs}
    recs = []
    for i, r in enumerate(raw):
        o = {}
        for f, src in M["fields"].items():
            if src is None: continue
            v = src["const"] if isinstance(src, dict) and "const" in src else get(r, src)
            vm = M.get("value_maps", {}).get(f)
            if vm is not None and v is not None: v = vm.get(str(v), v)
            o[f] = v
        o.setdefault("id", f"row{i}"); o["type"] = tv.get(str(o.get("type")), o.get("type"))
        if isinstance(o.get("is_member"), str): o["is_member"] = o["is_member"].strip().lower() in ("1", "true", "yes", "member")
        for f in ("target_kind", "patient", "bystander", "target"):
            if isinstance(o.get(f), str): o[f] = o[f].strip()
        recs.append(o)
    valid = []
    for o in recs:
        t = o.get("type")
        if t not in REQ: issue("UNKNOWN_TYPE", o["id"]); continue
        miss = [f for f in REQ[t] if o.get(f) in (None, "")]
        if miss:
            for f in miss: issue(f"MISSING:{t}.{f}", o["id"])
            continue
        if o["split"] not in SPLITS: issue("BAD_SPLIT", o["id"]); continue
        if t == "recognition":
            if o["ref_type"] not in DOMAIN["ref_type"]: issue("BAD_REF_TYPE", o["id"]); continue
            if not o.get("ref_span"):
                ph = o.get("ref_phrase")
                if not ph: issue("MISSING:recognition.ref_phrase_or_span", o["id"]); continue
                k = o["text"].lower().rfind(ph.lower())
                if k < 0: issue("REF_PHRASE_NOT_IN_TEXT", o["id"]); continue
                o["ref_span"] = [k, k + len(ph)]
            if o["ref_type"] == "description" and count_word(o["text"], o["target_kind"]) > 0:
                issue("DESCRIPTION_LEAKS_KIND_NOUN", o["id"]); o["_leak"] = True
            if o["ref_type"] == "pronoun": o["_pronoun"] = True
        else:
            if count_word(o["prompt"], o["patient"]) != 1: issue("CAUSAL_PATIENT_NOT_NAMED_ONCE", o["id"]); o["_fmt"] = False
            if count_word(o["prompt"], o["bystander"]) != 0 or count_word(o["prompt_c2"], o["bystander"]) != 1: issue("CAUSAL_BYSTANDER_NOT_ONCE_IN_C2_ONLY", o["id"]); o["_fmt"] = False
            if count_word(o["prompt_c2"], o["patient"]) != 1: issue("CAUSAL_C2_PATIENT_NOT_ONCE", o["id"]); o["_fmt"] = False
            if len(o["target"].split()) != 1: issue("TARGET_NOT_SINGLE_WORD", o["id"]); o["_fmt"] = False
        valid.append(o)
    # ---- G4 exclusions
    for o in valid:
        txt = [o.get("text"), o.get("prompt"), o.get("prompt_c2")]
        exact = any(s in X["items"] for s in txt if s)
        near = any(jacc(norm_tokens(s), norm_tokens(p)) >= NEAR_DUP for s in txt if s for p in X["items"])
        nouns = sorted({n for n in X["nouns"] for s in txt if s and count_word(s, n)})
        o["_g4"] = dict(exact=exact, near=near, nouns=nouns)
        if exact: issue("G4_PROBE_EXACT", o["id"])
        elif near: issue("G4_PROBE_NEAR_DUP", o["id"])
        if o["split"] == "evaluation" and (exact or near or nouns): issue("EVAL_EXCLUDED_G4", o["id"])
    # ---- leakage across splits
    by_bundle = collections.defaultdict(set); tpl_split = collections.defaultdict(set)
    for o in valid:
        by_bundle[o["bundle_id"]].add(o["split"])
        if o.get("template_id"): tpl_split[o["template_id"]].add(o["split"])
    cross_b = [b for b, s in by_bundle.items() if len(s) > 1]; cross_t = [t for t, s in tpl_split.items() if len(s) > 1]
    for b in cross_b: issue("BUNDLE_CROSSES_SPLITS", b)
    for t in cross_t: issue("TEMPLATE_SHARED_ACROSS_SPLITS", t)
    ev = [o for o in valid if o["split"] == "evaluation"]; nonev = [o for o in valid if o["split"] != "evaluation"]
    ev_tok = [(o["id"], norm_tokens(o.get("text") or o.get("prompt"))) for o in ev]
    for o in nonev:
        s = norm_tokens(o.get("text") or o.get("prompt"))
        if any(jacc(s, t) >= NEAR_DUP for _, t in ev_tok): issue("NEAR_DUP_OF_EVAL_IN_NONEVAL", o["id"]); o["_evaldup"] = True
    # ---- counts and requirement checks
    rec = [o for o in valid if o["type"] == "recognition"]; cau = [o for o in valid if o["type"] == "causal"]
    patients = sorted({o["patient"] for o in cau}); bystanders = sorted({o["bystander"] for o in cau}); kinds = sorted({o["target_kind"] for o in rec})
    counts = {}
    for K in kinds:
        rk = [o for o in rec if o["target_kind"] == K]; c = collections.Counter()
        for o in rk: c[(o["split"], "member" if o["is_member"] else "non", o["stratum"], o["ref_type"], o["role"])] += 1
        counts[K] = {"|".join(k): v for k, v in sorted(c.items())}
    checks = {}
    for K in sorted(set(patients) | set(bystanders)):
        rk = [o for o in rec if o["target_kind"] == K and not o.get("_leak") and not o["_g4"]["exact"]]
        V = [o for o in rk if o["split"] == "validation"]; D = [o for o in rk if o["split"] == "discovery"]
        strata_V = collections.Counter(o["stratum"] for o in V if not o["is_member"] and str(o["stratum"]).startswith("property_matched"))
        role_set = sorted({o["role"] for o in rk})
        checks[K] = dict(
            role="patient" if K in patients else "bystander", recognition_records=len(rk),
            has_discovery=bool(D), has_validation=bool(V),
            property_matched_strata_V=dict(strata_V),
            strata_V_ge_min=bool(strata_V) and all(v >= MIN_V_PER_STRATUM for v in strata_V.values()),
            members_V=sum(o["is_member"] for o in V),
            noun_free_descriptions=dict(member=sum(o["is_member"] and o["ref_type"] == "description" for o in rk),
                                        non_member=sum((not o["is_member"]) and o["ref_type"] == "description" for o in rk)),
            roles=role_set, roles_ge_2=len(role_set) >= 2,
            mentioned_attribute_present=any(o.get("mentioned_attribute") for o in rk if o["ref_type"] == "description"))
    fam = collections.defaultdict(collections.Counter)
    for o in cau:
        ok = o.get("_fmt", True) and not (o["split"] == "evaluation" and (o["_g4"]["exact"] or o["_g4"]["near"] or o["_g4"]["nouns"]))
        fam[o["family"]][(o["split"], "format_ok_after_G4_exclusion" if ok else "excluded")] += 1
    causal_counts = {f: {"|".join(k): v for k, v in c.items()} for f, c in fam.items()}
    eval_ok = {f: c.get(("evaluation", "format_ok_after_G4_exclusion"), 0) for f, c in fam.items()}
    audit.update(n_mapped=len(recs), n_valid=len(valid), patients=patients, bystanders=bystanders, recognition_kinds=kinds,
                 recognition_counts=counts, kind_checks=checks, causal_counts=causal_counts,
                 evaluation_causal_items_per_family_pre_tokenization=eval_ok,
                 eval_target_met={f: n >= TARGET_EVAL_PER_FAMILY for f, n in eval_ok.items()},
                 cross_split_bundles=len(cross_b), shared_templates=len(cross_t),
                 pending=["single-token check of targets under the Gemma tokenizer (needs the gated tokenizer; Modal CPU or HF-authenticated machine)",
                          "baseline eligibility (p>=0.10, rank<=5, bystander shift<=1.0) needs the model"])
    audit["issues"] = dict(audit["issues"]); audit["examples"] = dict(audit["examples"])
    os.makedirs(outdir, exist_ok=True)
    out = [{k: v for k, v in o.items() if not k.startswith("_")} for o in nonev
           if not o["_g4"]["exact"] and not o.get("_leak") and not o.get("_evaldup") and o.get("_fmt", True)]
    with open(os.path.join(outdir, "g5r_input.jsonl"), "w") as f:
        for o in out: f.write(json.dumps(o) + "\n")
    audit["written_g5r_records"] = len(out)
    json.dump(audit, open(os.path.join(outdir, "audit.json"), "w"), indent=1, default=str)
    md = [f"# Adaption dataset audit\n\nSource `{audit['source']['path']}` sha256 `{audit['source']['sha256'][:16]}…`: {len(raw)} raw, {len(valid)} valid; "
          f"{len(out)} non-evaluation records written for G5r. Evaluation records were read for auditing only and not written.\n",
          "## Issues\n"] + [f"- {k}: {v} (e.g. {audit['examples'].get(k, [])[:3]})" for k, v in sorted(audit["issues"].items())] + \
         ["\n## Kind checks (patients and bystanders)\n"] + [f"- **{K}** ({c['role']}): {json.dumps({k: v for k, v in c.items() if k != 'role'})}" for K, c in checks.items()] + \
         ["\n## Causal items\n"] + [f"- {f}: {json.dumps(c)}" for f, c in causal_counts.items()] + \
         [f"\nEvaluation items per family before tokenization and baseline checks: {eval_ok} (target ≥ {TARGET_EVAL_PER_FAMILY})\n",
          "## Pending (needs the model or tokenizer)\n"] + [f"- {p}" for p in audit["pending"]]
    open(os.path.join(outdir, "audit.md"), "w").write("\n".join(md))
    return audit


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "inspect": inspect(sys.argv[2])
    elif cmd == "convert": a = convert(sys.argv[2], sys.argv[3], sys.argv[4]); print(open(os.path.join(sys.argv[4], "audit.md")).read())
