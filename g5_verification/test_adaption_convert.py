"""Deterministic test for adaption_convert: a SYNTHETIC export with fresh kinds (lantern, kettle; bystander: radio),
foreign field names, and planted defects. Checks conversion, audit counts, G4 exclusions, and g5r loader compatibility."""
import json, os, sys, tempfile, collections
here = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [here, os.path.join(here, "..", "g4_probe")]
import adaption_convert as AC, g5r_probe as GR, g4_probe as G4

rows, n = [], 0
def R(**kw):
    global n; n += 1; rows.append(dict(uid=f"u{n}", **kw))
KINDS = {"lantern": (["paper lantern", "red lantern"], {"material": ["paper fan", "paper cup"], "function": ["desk lamp", "torch light"]}),
         "kettle": (["steel kettle", "old kettle"], {"material": ["steel pan", "steel bowl"], "function": ["coffee pot", "tea urn"]}),
         "radio": (["small radio", "old radio"], {"material": ["small speaker", "old phone"], "function": ["music box", "loud speaker"]})}
TPL = {"train": [("T1", "subj", "The {} sat on the bench ."), ("T2", "obj", "She lifted the {} carefully .")],
       "dev": [("T3", "subj", "The {} rested near the stove ."), ("T4", "obj", "He moved the {} again .")]}
for K, (mem, prop) in KINDS.items():
    for sp, tpls in TPL.items():
        for tid, role, t in tpls:
            for i in range(5):                                    # 5 bundles per template to reach >=10 per stratum in dev
                for ph in mem: R(meta=dict(split=sp, bundle=f"{K}-{sp}-{tid}-{i}", template=tid), kind="recog", entity=K, member="yes",
                                 contrast="identity_variant", ref="noun", gram=role, sentence=t.format(ph) + f" {i}", phrase=ph)
                for pk, phs in prop.items():
                    for ph in phs: R(meta=dict(split=sp, bundle=f"{K}-{sp}-{tid}-{i}", template=tid), kind="recog", entity=K, member="no",
                                     contrast=f"property_matched:{pk}", ref="noun", gram=role, sentence=t.format(ph) + f" {i}", phrase=ph)
                R(meta=dict(split=sp, bundle=f"{K}-{sp}-{tid}-{i}", template=tid), kind="recog", entity=K, member="yes", contrast="identity_variant",
                  ref="desc", gram=role, sentence=t.format("glowing thing you carry") + f" {i}", phrase="glowing thing you carry", attr="function")
for fam, (pat, tgt) in {"impact": ("lantern", "crack"), "heating": ("kettle", "whistle"), "cutting": ("rope2", "split")}.items():
    for i in range(6):
        sp = ["train", "p3", "test"][i % 3]
        p = f"The hammer hit the {pat} hard {i}, causing it to"
        R(meta=dict(split=sp, bundle=f"c-{fam}-{i}", template=f"C{fam}"), kind="cause", family=fam, patient=pat, bystander="radio", outcome=tgt,
          prompt=p, prompt_c2="As a radio played nearby, " + p[0].lower() + p[1:])
# planted defects
R(meta=dict(split="test", bundle="evx", template="TX"), kind="cause", family="impact", patient="vase", bystander="radio", outcome="shatter",
  prompt="The hammer hit the vase, causing it to", prompt_c2="As a radio played nearby, the hammer hit the vase, causing it to")      # eval uses G4 noun
R(meta=dict(split="train", bundle="g4x", template="TG"), kind="cause", family="impact", patient="vase", bystander="clock", outcome="shatter",
  prompt=G4.FAMILIES["impact"]["prompts"][0], prompt_c2=G4.C2_PREFIX + "x")                                                          # exact G4 probe
R(meta=dict(split="train", bundle="lantern-dev-T3-0", template="T1"), kind="recog", entity="lantern", member="yes", contrast="identity_variant",
  ref="noun", gram="subj", sentence="The paper lantern sat on the bench .", phrase="paper lantern")                                   # bundle crosses splits
R(meta=dict(split="train", bundle="leak", template="T1"), kind="recog", entity="lantern", member="yes", contrast="identity_variant",
  ref="desc", gram="subj", sentence="The lantern you carry sat on the bench .", phrase="lantern you carry")                          # description leaks noun
R(meta=dict(split="train", bundle="twice", template="Cx"), kind="cause", family="impact", patient="lantern", bystander="radio", outcome="crack",
  prompt="The lantern hit the lantern, causing it to", prompt_c2="As a radio played nearby, the lantern hit the lantern, causing it to")  # patient twice

mapping = dict(format="jsonl", fields=dict(id="uid", bundle_id="meta.bundle", template_id="meta.template", split="meta.split", type="kind",
                                           target_kind="entity", is_member="member", stratum="contrast", ref_type="ref", role="gram", text="sentence",
                                           ref_phrase="phrase", mentioned_attribute="attr", family="family", patient="patient", bystander="bystander",
                                           target="outcome", prompt="prompt", prompt_c2="prompt_c2"),
               value_maps=dict(split=dict(train="discovery", dev="validation", test="evaluation", p3="p3_discovery"),
                               ref_type=dict(desc="description"), stratum=dict(identity_variant="member:identity_variant"),
                               role=dict(subj="subject", obj="object")),
               type_values=dict(recognition=["recog"], causal=["cause"]))
d = tempfile.mkdtemp(); exp = os.path.join(d, "export.jsonl"); mp = os.path.join(d, "map.json"); out = os.path.join(d, "out")
open(exp, "w").write("\n".join(json.dumps(r) for r in rows)); json.dump(mapping, open(mp, "w"))
A = AC.convert(exp, mp, out)
I = A["issues"]
assert I.get("EVAL_EXCLUDED_G4", 0) >= 1, I
assert I.get("G4_PROBE_EXACT", 0) == 1, I
assert I.get("BUNDLE_CROSSES_SPLITS", 0) >= 1 and I.get("TEMPLATE_SHARED_ACROSS_SPLITS", 0) >= 1, I
assert I.get("DESCRIPTION_LEAKS_KIND_NOUN", 0) == 1, I
assert I.get("CAUSAL_PATIENT_NOT_NAMED_ONCE", 0) == 2, I   # planted item + the G4 probe prompt ("causing the vase to")
assert I.get("NEAR_DUP_OF_EVAL_IN_NONEVAL", 0) >= 1, I     # synthetic causal prompts differ only by an index: correctly flagged
assert A["patients"] == ["kettle", "lantern", "rope2", "vase"] and A["bystanders"] == ["clock", "radio"], (A["patients"], A["bystanders"])
assert A["kind_checks"]["lantern"]["strata_V_ge_min"] and A["kind_checks"]["lantern"]["roles_ge_2"]
assert A["kind_checks"]["rope2"]["recognition_records"] == 0                 # missing recognition data for a patient is surfaced
assert A["kind_checks"]["lantern"]["noun_free_descriptions"]["non_member"] == 0   # missing non-member descriptions surfaced
recs = GR.load(os.path.join(out, "g5r_input.jsonl"))                         # g5r loader accepts output (no eval, no G4 probe items)
assert all(r["split"] != "evaluation" for r in recs) and len(recs) == A["written_g5r_records"]
print(open(os.path.join(out, "audit.md")).read()[:1500])
print("ADAPTION_CONVERT_TEST PASS")
