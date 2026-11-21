"""Phase A repairs for the discovery split (v1 -> v1.1). Deterministic; every change logged with reason.
Only demonstrable errors are repaired: (1) role-label spelling harmonised to one vocabulary; (2) role labels that
contradict the sentence (verified by two dependency parsers AND manual reading); (3) cutting causal prompts whose
bystander takes part in the cut (V1.1 C2 requires a causally irrelevant bystander): bystander clause rewritten to an
inert placement, keeping ID, patient, verb, ending and target. IDs, allocation and all other fields are untouched.
Usage: python repair_discovery.py <in.jsonl> <in.csv> <out.jsonl> <out.csv> <changelog.csv>"""
import json, csv, sys

CANON = {"subject": "subject", "passive subject": "passive_subject", "passive_subject": "passive_subject",
         "direct object": "direct_object", "direct_object": "direct_object", "indirect object": "indirect_object",
         "prepositional object": "prepositional_object", "prepositional_object": "prepositional_object",
         "object_of_preposition": "prepositional_object", "genitive complement (of-phrase)": "genitive_complement",
         "predicate nominative": "predicate_nominative"}
PASSIVE = ["CUT_D013", "CUT_D015", "CUT_D017", "CUT_D021", "CUT_D023", "CUT_D027", "CUT_D029", "CUT_D031", "CUT_D039", "CUT_D043",
           "CUT_D047", "CUT_D049", "CUT_D051", "CUT_D053", "CUT_D055", "CUT_D057", "CUT_D059", "CUT_D061", "CUT_D063", "CUT_D069",
           "CUT_D071", "CUT_D073", "CUT_D075", "CUT_D077", "CUT_D079", "CUT_D081", "CUT_D083", "CUT_D085", "CUT_D087", "CUT_D089",
           "CUT_D091", "CUT_D093", "CUT_D095", "CUT_D105", "CUT_D107", "CUT_D113", "CUT_D115", "CUT_D121", "CUT_D123", "CUT_D129",
           "CUT_D131", "CUT_D137", "CUT_D139", "HEAT_DISC_B001_C", "HEAT_DISC_B003_C", "HEAT_DISC_B010_C", "HEAT_DISC_B011_C",
           "HEAT_DISC_B013_C", "HEAT_DISC_B015_C", "HEAT_DISC_B020_C", "HEAT_DISC_B021_C", "HEAT_DISC_B022_C", "HEAT_DISC_B023_C",
           "IMP_DISC_K03_CTX_B22_B", "IMP_DISC_K03_CTX_B24_A2", "IMP_DISC_K01_FORM_B02_C", "IMP_DISC_K01_MAT_B06_C"]
DIRECT = ["IMP_DISC_K01_MAT_B08_C", "IMP_DISC_K02_MAT_B16_C", "IMP_DISC_K03_MAT_B17_C", "IMP_DISC_K03_MAT_B19_C", "IMP_DISC_K03_MAT_B20_C",
          "IMP_DISC_K03_CTX_B21_C", "IMP_DISC_K03_CTX_B23_C", "IMP_DISC_K03_CTX_B24_C", "HEAT_DISC_B016_A1",
          "HEAT_DISC_SUP_PG2_aqueous_protein_film_S08", "HEAT_DISC_SUP_PG1_low_temp_softening_mass_S02",
          "HEAT_DISC_SUP_PG2_molded_apiary_slab_S08", "HEAT_DISC_SUP_PG2_aqueous_protein_film_S06", "CUT_D020",
          "CUT_D068", "CUT_D072", "CUT_D076", "CUT_D080", "CUT_D084", "CUT_D088", "CUT_D092", "CUT_D096"]
SUBJECT = ["IMP_DISC_K01_FORM_B03_C", "IMP_DISC_K02_FORM_B12_C"]
PREP = ["IMP_DISC_K02_FORM_B10_A2", "IMP_DISC_K03_MAT_B20_A1", "CUT_D122"]
ROLE_FIX = {**{i: "passive_subject" for i in PASSIVE}, **{i: "direct_object" for i in DIRECT},
            **{i: "subject" for i in SUBJECT}, **{i: "prepositional_object" for i in PREP}}
BYSTANDER_FIX = {
    "CUT_D008": ("while the weighted silk kept the margin flat and", "while the weighted silk lay folded on a side table and"),
    "CUT_D016": ("while the glass weight held the far edge and", "while the glass weight sat on a nearby shelf and"),
    "CUT_D024": ("as the silk cord held the opposite edge and", "as the silk cord hung coiled on a hook and"),
    "CUT_D028": ("while the cotton weight stayed in place and", "while the cotton weight sat in a drawer across the room and"),
    "CUT_D032": ("while the felt pad protected the underlying surface and", "while the felt pad lay unused in a drawer and"),
    "CUT_D040": ("while the steel square held the edge true and", "while the steel square hung on the wall and"),
    "CUT_D044": ("while the wooden block braced the far end and", "while the wooden block sat on a shelf across the room and"),
    "CUT_D048": ("while the plastic clamp held the free end and", "while the plastic clamp lay in a tool tray and"),
    "CUT_D052": ("while the rubber mat protected the table and", "while the rubber mat lay rolled up in a corner and"),
    "CUT_D056": ("while the cork stop kept the piece from sliding and", "while the cork stop lay in a parts drawer and"),
    "CUT_D060": ("while the metal weight stabilized the free end and", "while the metal weight sat on a distant shelf and"),
    "CUT_D064": ("while the silicone pad cushioned the underside and", "while the silicone pad lay folded on a side cart and"),
    "CUT_D076": ("While the foam underlay protected the table,", "While the foam underlay lay rolled up in a corner,"),
    "CUT_D092": ("While the alignment fence stayed in place,", "While the alignment fence leaned against the far wall,"),
}


def main(ij, ic, oj, oc, log):
    R = [json.loads(l) for l in open(ij) if l.strip()]; L = []
    ids = {r["unique_id"] for r in R}
    for k in list(ROLE_FIX) + list(BYSTANDER_FIX): assert k in ids, k
    for r in R:
        i = r["unique_id"]; old = r["grammatical_role"]; new = ROLE_FIX.get(i, CANON[old])
        if new != old:
            why = ("role label contradicts sentence (two parsers + manual reading)" if i in ROLE_FIX and CANON[old] != new
                   else "role-label spelling harmonised to dataset-wide vocabulary")
            L.append(dict(unique_id=i, field="grammatical_role", old=old, new=new, reason=why)); r["grammatical_role"] = new
        if i in BYSTANDER_FIX:
            a, b = BYSTANDER_FIX[i]; assert r["sentence"].count(a) == 1, i
            s2 = r["sentence"].replace(a, b)
            L.append(dict(unique_id=i, field="sentence", old=r["sentence"], new=s2,
                          reason="bystander took part in the cut; V1.1 C2 requires a causally irrelevant bystander; clause made inert"))
            r["sentence"] = s2
    with open(oj, "w") as f:
        for r in R: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    rows = list(csv.DictReader(open(ic)))
    with open(oc, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with open(log, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["unique_id", "field", "old", "new", "reason"]); w.writeheader(); w.writerows(L)
    print(f"{len(L)} changes ({sum(x['field']=='sentence' for x in L)} sentences, {sum(x['field']=='grammatical_role' for x in L)} role labels)")


if __name__ == "__main__":
    main(*sys.argv[1:6])
