"""C2 bystander fix (PI decision 2026-11-21): one fixed bystander kind per family in every causal prompt, plus bystander
recognition records (10 members + 10 property-matched non-members per kind, discovery and validation).
Usage: python3 bystander_fix.py <dataset_root>   (reads discovery/v1.1, validation/v1.0, evaluation/v1.0; writes
discovery/v1.2, validation/v1.1, evaluation/v1.1 with changelogs). Deterministic; no randomness."""
import csv, json, os, sys
from banks import BYSTANDERS, FRAMES, AGENTS
from bystander_bank import BYST, FRAMES_D, AGENTS_D, ROLES, DISC_SUB
from generate_splits import toks, jacc, cap

FAMS = ["physical_impact", "heating", "cutting"]
PFX = {"physical_impact": "IMP", "heating": "HEAT", "cutting": "CUT"}
MAPF = ["unique_id", "bundle_id", "split", "focal_kind", "property_group", "template_id"]
NEAR = 0.8


def load(d, name):
    R = [json.loads(l) for l in open(os.path.join(d, name + ".jsonl")) if l.strip()]
    M = list(csv.DictReader(open(os.path.join(d, name + "_mapping.csv"))))
    return R, M


def write(R, M, d, name):
    os.makedirs(d, exist_ok=True)
    def w(recs, maps, n):
        with open(os.path.join(d, n + ".jsonl"), "w") as f:
            for r in recs: f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with open(os.path.join(d, n + "_mapping.csv"), "w", newline="") as f:
            x = csv.DictWriter(f, fieldnames=MAPF); x.writeheader(); x.writerows(maps)
    w(R, M, name)
    mm = {m["unique_id"]: m for m in M}
    for fam in FAMS:
        rr = [r for r in R if r["family"] == fam]; w(rr, [mm[r["unique_id"]] for r in rr], f"{name}_{fam}")


def fix_rationale(r, old, new):
    s = r["matching_rationale"]
    for o in sorted({old, old.split()[-1]}, key=len, reverse=True):
        s = s.replace(f"'{o}'", f"'{new}'")
    return s + f" [Bystander set to the fixed C2 kind '{new}' for this family (PI decision 2026-11-21).]"


def np_(b, adj, core): return f"the {adj} {b['head']} {core}"


def sentence(fr, npx, agent):
    s = fr.replace("{Ag}", agent); i = s.index("{NP}"); rest = s[i + 4:]
    if "," in npx and not rest.startswith("."): npx += ","
    return cap(s[:i] + npx + rest)


def recog(split, fam, b, used, log):
    code = "DISC" if split == "discovery" else "VAL"
    off = 0 if split == "discovery" else 10
    recs, maps = [], []
    for j in range(10):
        role = ROLES[j % 4]
        if split == "discovery":
            fl = FRAMES_D[role]; fi = (j // 4) % len(fl); fr = fl[fi]; ag = AGENTS_D[j % len(AGENTS_D)]
        else:
            fl = FRAMES[role]["V"]; fi = (j // 4) % len(fl); fr = fl[fi]; ag = AGENTS["V"][j % len(AGENTS["V"])]
        adj = b["adj"][(j + (0 if split == "discovery" else 3)) % len(b["adj"])]
        tpl = f"BYS_{code}_T_{role.upper()}_{fi:02d}"
        pair = None
        for member in (True, False):
            core = b["m" if member else "n"][off + j]; npx = np_(b, adj, core); s = sentence(fr, npx, ag)
            t = toks(s)
            clash = [u for u, ut in used if jacc(t, ut) >= NEAR and u != pair]
            if clash: raise RuntimeError(f"near-duplicate {split} {b['kind']} {j} {member}: {clash[:3]}")
            uid = f"{PFX[fam]}_{code}_BYS_{'M' if member else 'N'}{j + 1:02d}"
            r = {"unique_id": uid, "family": fam, "proposed_split": split, "bundle_id": f"{PFX[fam]}_{code}_BYS_SUPP",
                 "record_type": "identity_preserving" if member else "matched_nonmember", "entity_kind": b["kind"],
                 "entity_description": npx, "comparison_entity_kind": b["cmp"],
                 "material": b["mat_m"] if member else b["mat_n"], "relevant_properties": b["props_m"] if member else b["props_n"],
                 "grammatical_role": role, "sentence": s, "candidate_target_word": "", "alternative_target_words": [],
                 "matching_rationale": (f"C2 bystander recognition: member of the fixed bystander kind '{b['kind']}', identity carried by the defining configuration without naming the kind."
                                        if member else f"C2 bystander recognition: matched non-member ({b['cmp']}) sharing head noun '{b['head']}', surface properties and frame with member {j + 1:02d}; kind not named."),
                 "expected_category_membership": member,
                 "potential_ambiguity": "Noun-free reference: recognisability depends on the defining configuration, not on a kind noun."}
            recs.append(r); maps.append(dict(unique_id=uid, bundle_id=r["bundle_id"], split=split, focal_kind=b["kind"],
                                             property_group="BYS_FORM", template_id=tpl))
            used.append((uid, t)); pair = uid
            log.append(dict(unique_id=uid, field="(new record)", old="", new=s, reason="C2 bystander recognition record added"))
    return recs, maps


def main(root):
    D, DM = load(os.path.join(root, "discovery/v1.1"), "discovery_v1.1")
    V, VM = load(os.path.join(root, "validation/v1.0"), "validation_v1.0")
    E, EM = load(os.path.join(root, "evaluation/v1.0"), "evaluation_v1.0")
    logs = {"discovery": [], "validation": [], "evaluation": []}
    # 1. discovery causal: per-item phrase substitution
    for r in D:
        if r["record_type"] != "causal_prediction": continue
        b = BYST[r["family"]]; sub = DISC_SUB[r["unique_id"]]; old_s, old_k = r["sentence"], r["comparison_entity_kind"]
        if sub is not None:
            assert r["sentence"].count(sub[0]) == 1, r["unique_id"]
            r["sentence"] = r["sentence"].replace(sub[0], sub[1])
        r["comparison_entity_kind"] = b["kind"]; r["matching_rationale"] = fix_rationale(r, old_k, b["kind"])
        if r["sentence"] != old_s:
            logs["discovery"].append(dict(unique_id=r["unique_id"], field="sentence", old=old_s, new=r["sentence"], reason="bystander phrase replaced by fixed C2 bystander"))
        logs["discovery"].append(dict(unique_id=r["unique_id"], field="comparison_entity_kind", old=old_k, new=b["kind"], reason="fixed C2 bystander"))
    # 2. validation/evaluation causal: replace the leading bystander clause, deterministic cycling with near-dup avoidance
    allsent = {r["unique_id"]: toks(r["sentence"]) for R in (D, V, E) for r in R if r["record_type"] != "causal_prediction"}
    allsent.update({r["unique_id"]: toks(r["sentence"]) for r in D if r["record_type"] == "causal_prediction"})
    sent_txt = set()
    clus = {m["unique_id"]: (m["split"], m["template_id"]) for m in VM + EM}   # event cluster = causal template (event) within a split
    for split, R, key in (("validation", V, "V"), ("evaluation", E, "E")):
        clauses = {bc: bh for bc, bh in BYSTANDERS[key]}; k = {f: 0 for f in FAMS}
        for r in R:
            if r["record_type"] != "causal_prediction": continue
            old_s, old_k = r["sentence"], r["comparison_entity_kind"]
            bc = next(c for c in clauses if old_s.startswith(c)); assert clauses[bc] == old_k
            body = old_s[len(bc):]; opts = BYST[r["family"]][key]; t = None
            # v1.0 already contains items sharing one causal event (same template_id) that differ only in patient phrasing and bystander;
            # these stay a disclosed event cluster: they must get distinct clauses (no exact duplicate) and must not be
            # near-duplicates of any item outside their event cluster or of any sentence in another split.
            for i in range(len(opts)):
                cand = opts[(k[r["family"]] + i) % len(opts)] + body; ct = toks(cand)
                if cand in sent_txt: continue
                if not any(jacc(ct, u) >= NEAR for uid, u in allsent.items() if clus.get(uid) != clus[r["unique_id"]]): t = cand; break
            if t is None:
                ct = toks(opts[0] + body)
                raise RuntimeError(f"no clause for {r['unique_id']} {clus[r['unique_id']]}: " + str([(u, clus.get(u)) for u, x in allsent.items() if jacc(ct, x) >= NEAR][:4]))
            k[r["family"]] += 1
            r["sentence"] = t; allsent[r["unique_id"]] = toks(t); sent_txt.add(t)
            r["comparison_entity_kind"] = BYST[r["family"]]["kind"]
            r["matching_rationale"] = fix_rationale(r, old_k, r["comparison_entity_kind"])
            logs[split] += [dict(unique_id=r["unique_id"], field="sentence", old=old_s, new=t, reason="bystander clause replaced by fixed C2 bystander"),
                            dict(unique_id=r["unique_id"], field="comparison_entity_kind", old=old_k, new=r["comparison_entity_kind"], reason="fixed C2 bystander")]
    # 3. bystander recognition records (discovery, validation)
    used = [(u, t) for u, t in allsent.items()]
    for split, R, M in (("discovery", D, DM), ("validation", V, VM)):
        for fam in FAMS:
            rr, mm = recog(split, fam, BYST[fam], used, logs[split]); R += rr; M += mm
    # 4. write
    out = {"discovery": ("discovery/v1.2", "discovery_v1.2", D, DM), "validation": ("validation/v1.1", "validation_v1.1", V, VM),
           "evaluation": ("evaluation/v1.1", "evaluation_v1.1", E, EM)}
    for split, (d, name, R, M) in out.items():
        write(R, M, os.path.join(root, d), name)
        with open(os.path.join(root, d, name + "_changelog.csv"), "w", newline="") as f:
            x = csv.DictWriter(f, fieldnames=["unique_id", "field", "old", "new", "reason"]); x.writeheader(); x.writerows(logs[split])
        print(split, len(R), "changes", len(logs[split]))


if __name__ == "__main__":
    main(sys.argv[1])
