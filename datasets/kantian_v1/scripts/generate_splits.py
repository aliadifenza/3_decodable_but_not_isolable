"""Generate validation (432) and evaluation (1,152) splits in the discovery 17-field JSONL + 6-column CSV format.
Allocation per family: validation = 6 strata x (4 standard bundles x 4 records + 8 recognition supplements) = 144;
evaluation = 6 strata x 16 standard bundles x 4 records = 384. Standard bundle = A1 member, A2 member, B matched
non-member (same frame/role/agent/adjective as A1: a minimal pair), C causal prompt.
Phrase banks are partitioned so no member/non-member phrase or causal event is shared across splits; recognition frames
and bystander clauses are partitioned too (templates never cross splits). Deterministic (seeded). Near-duplicates
(token Jaccard >= 0.8) against discovery and the other split are rejected and resampled.
Usage: python generate_splits.py <discovery_v1.1.jsonl> <outdir>"""
import json, csv, random, re, sys, os, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from banks import FRAMES, FAMILY_ROLES, AGENTS, BYSTANDERS, S, CAUSAL, CAUSAL_LIST, ALTS_OVERRIDE

SEED = {"validation": 20261121, "evaluation": 20261122}
CODE = {"validation": ("VAL", "V"), "evaluation": ("EVAL", "E")}
NB = {"validation": 4, "evaluation": 16}
PART_CORE = {"V": [7, 8, 9], "E": [0, 1, 2, 3, 4, 5, 6]}
FAM_STRATA = collections.OrderedDict()
for (f, pg) in S: FAM_STRATA.setdefault(f, []).append(pg)
PREFIX = {"physical_impact": "IMP", "heating": "HEAT", "cutting": "CUT"}
NEAR = 0.8


def toks(s): return set(re.findall(r"[a-z]+", s.lower()))
def jacc(a, b): return len(a & b) / max(1, len(a | b))
def cap(s): return s[0].upper() + s[1:]


class Gen:
    def __init__(self, split, banned):
        self.split = split; self.code, self.key = CODE[split]; self.rng = random.Random(SEED[split]); self.banned = banned
        self.recs, self.maps, self.used = [], [], []

    def ok(self, sent, exempt=None):
        """Reject near-duplicates of discovery/other split AND of earlier items in this split (independence of items);
        only the intended within-bundle minimal pair (B vs its A1) is exempt."""
        t = toks(sent)
        return all(jacc(t, b) < NEAR for b in self.banned) and all(jacc(t, u) < NEAR for u in self.used if u is not exempt)

    def np(self, st, member, adj=None, core=None):
        adj = adj if adj is not None else self.rng.choice(st["adj"])
        cores = [st["m" if member else "n"][i] for i in PART_CORE[self.key]]
        core = core if core is not None else self.rng.choice(cores)
        head = st["heads"][0]
        return f"the {adj} {head} {core}", adj, core

    def frame(self, role):
        fs = FRAMES[role][self.key]; i = self.rng.randrange(len(fs)); return i, fs[i]

    def sentence(self, fr, role, npx, agent):
        s = fr.replace("{Ag}", agent)
        i = s.index("{NP}"); rest = s[i + 4:]
        if "," in npx and not rest.startswith("."): npx = npx + ","     # close the appositive before the rest of the clause
        s = s[:i] + npx + rest
        return cap(s)

    def rec(self, fam, pg, uid, bundle, rtype, st, role, sent, npx, tpl, member, target="", alts=None, cmpk=None, rationale="", amb=""):
        r = {"unique_id": uid, "family": fam, "proposed_split": self.split, "bundle_id": bundle, "record_type": rtype,
             "entity_kind": st["kind"], "entity_description": npx,
             "comparison_entity_kind": cmpk if cmpk is not None else (st["cmp"] if fam != "cutting" or not member else ""),
             "material": st["mat_m"] if member in (True, None) else st["mat_n"],
             "relevant_properties": st["props_m"] if member in (True, None) else st["props_n"],
             "grammatical_role": role, "sentence": sent, "candidate_target_word": target, "alternative_target_words": alts or [],
             "matching_rationale": rationale, "expected_category_membership": member, "potential_ambiguity": amb}
        self.recs.append(r); self.used.append(toks(sent))
        self.maps.append({"unique_id": uid, "bundle_id": bundle, "split": self.split, "focal_kind": st["kind"], "property_group": pg, "template_id": tpl})

    def tpl_rec(self, fam, role, fi): return f"{PREFIX[fam]}_{self.code}_T_{role.upper()}_{fi:02d}"

    def recog(self, fam, pg, st, member, role, uid, bundle, adj=None, agent=None, fixed=None, exempt=None):
        for _ in range(2000):
            fi, fr = fixed or self.frame(role)
            npx, a, core = self.np(st, member, adj=adj)
            if adj is None and toks(a) & toks(core): continue            # no word repeated between adjective and description
            ag = agent or self.rng.choice(AGENTS[self.key])
            sent = self.sentence(fr, role, npx, ag)
            if self.ok(sent, exempt) and toks(sent) not in self.used: break
        else: raise RuntimeError(f"no non-duplicate sentence for {uid}")
        why = (f"Member of the focal kind in stratum {pg}; identity carried by the defining configuration without naming the kind; "
               f"non-defining surface properties vary." if member else
               f"Matched non-member ({st['cmp']}) in stratum {pg}: shares head noun '{st['heads'][0]}' and surface properties with the focal kind, "
               f"differs in the defining configuration; kind not named.")
        amb = "Noun-free reference: recognisability depends on the defining configuration, not on a kind noun."
        self.rec(fam, pg, uid, bundle, "identity_preserving" if member else "matched_nonmember", st, role, sent, npx,
                 self.tpl_rec(fam, role, fi), member, rationale=why, amb=amb)
        return (fi, fr), a, ag

    def causal(self, fam, pg, st, uid, bundle, k):
        C = CAUSAL[st["kind"]]; evs = C[CAUSAL_LIST.get(pg, "ev")]
        n = len(evs); vi = [n - 1] if n <= 6 else [n - 2, n - 1]
        idx = vi if self.key == "V" else [i for i in range(n) if i not in vi]
        for _ in range(3000):
            ei = idx[k % len(idx)] if self.rng.random() < 0.5 else self.rng.choice(idx)
            role, text, target = evs[ei]
            P = self.rng.choice(C["P"]); bc, bh = self.rng.choice(BYSTANDERS[self.key])
            sent = bc + text.replace("{P}", P)
            if self.ok(sent) and toks(sent) not in self.used: break
        else: raise RuntimeError(f"no non-duplicate causal prompt for {uid}")
        why = (f"Causal prompt for stratum {pg}: patient named once ('{P}'); bystander '{bh}' is inert and causally irrelevant "
               f"(V1.1 C2); ending cues the single-word target.")
        amb = "Target-token eligibility and baseline strength require the model check; bystander is not an independently validated C2 feature."
        tpl = f"{PREFIX[fam]}_{self.code}_T_CAUSAL_{st['kind'].split()[-1].upper()}_{ei:02d}"
        alts = [a for a in ALTS_OVERRIDE.get(CAUSAL_LIST.get(pg, "ev"), C["alts"]) if a != target][:4]   # target never among the wrong outcomes
        self.rec(fam, pg, uid, bundle, "causal_prediction", st, role, sent, "", tpl, None, target=target, alts=alts,
                 cmpk=bh, rationale=why, amb=amb)

    def run(self):
        for fam, pgs in FAM_STRATA.items():
            roles = FAMILY_ROLES[fam]; bno = 0; seq = 0; rc = 0
            for pg in pgs:
                st = S[(fam, pg)]
                for b in range(NB[self.split]):
                    bno += 1
                    if fam == "physical_impact":
                        bid = f"IMP_{self.code}_{pg}_B{bno:02d}"; ids = [f"{bid}_A1", f"{bid}_A2", f"{bid}_B", f"{bid}_C"]
                    elif fam == "heating":
                        bid = f"HEAT_{self.code}_B{bno:03d}"; ids = [f"{bid}_A1", f"{bid}_A2", f"{bid}_B", f"{bid}_C"]
                    else:
                        bid = f"CUT_{self.code}_B{bno:02d}"; ids = [f"CUT_{self.key}{seq + i + 1:03d}" for i in range(4)]; seq += 4
                    r1, r2 = roles[rc % len(roles)], roles[(rc + 1) % len(roles)]; rc += 2
                    fr1, adj1, ag1 = self.recog(fam, pg, st, True, r1, ids[0], bid); a1 = self.used[-1]
                    self.recog(fam, pg, st, True, r2, ids[1], bid)
                    self.recog(fam, pg, st, False, r1, ids[2], bid, adj=adj1, agent=ag1, fixed=fr1, exempt=a1)
                    self.causal(fam, pg, st, ids[3], bid, b)
                if self.split == "validation":
                    for j in range(8):
                        member = j < 2; role = roles[rc % len(roles)]; rc += 1
                        if fam == "physical_impact":
                            uid = f"IMP_{self.code}_{pg}_S_{'M' if member else 'N'}{(j + 1) if member else (j - 1):02d}"; bid = f"IMP_{self.code}_{pg}_SUPP"
                        elif fam == "heating":
                            uid = f"HEAT_{self.code}_SUP_{pg}_S{j + 1:02d}"; bid = f"HEAT_{self.code}_SUP_{pg}"
                        else:
                            seq += 1; uid = f"CUT_{self.key}{seq:03d}"; bid = f"CUT_{self.code}_SUPP_{pg}_{j + 1:02d}"
                        self.recog(fam, pg, st, member, role, uid, bid)
        return self.recs, self.maps


def write(recs, maps, d, name):
    with open(os.path.join(d, name + ".jsonl"), "w") as f:
        for r in recs: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(d, name + "_mapping.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["unique_id", "bundle_id", "split", "focal_kind", "property_group", "template_id"]); w.writeheader(); w.writerows(maps)


def main(disc, out):
    os.makedirs(out, exist_ok=True)
    D = [json.loads(l) for l in open(disc) if l.strip()]
    banned = [toks(r["sentence"]) for r in D]
    V = Gen("validation", banned); vr, vm = V.run()
    E = Gen("evaluation", banned + [toks(r["sentence"]) for r in vr]); er, em = E.run()
    for split, recs, maps in (("validation", vr, vm), ("evaluation", er, em)):
        write(recs, maps, out, f"{split}_v1.0")
        for fam in FAM_STRATA:
            ix = [i for i, r in enumerate(recs) if r["family"] == fam]
            write([recs[i] for i in ix], [maps[i] for i in ix], out, f"{split}_v1.0_{fam}")
    print("validation", len(vr), collections.Counter(r["family"] for r in vr))
    print("evaluation", len(er), collections.Counter(r["family"] for r in er))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
