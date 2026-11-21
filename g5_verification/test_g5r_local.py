"""Deterministic local tests for g5r_probe (tiny random Gemma2, toy tokenizer, random JumpReLU SAE, SYNTHETIC schema fixture).
The fixture only exercises the code paths and schema; it is not a dataset and says nothing about the real model."""
import json, re, sys, os, numpy as np, torch, tempfile, copy
here = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [here, os.path.join(here, "..", "g4_probe")]
import g4_probe as G4, g5r_probe as GR
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast, Gemma2Config, Gemma2ForCausalLM

KINDS = {"vase": dict(mem=["glass vase", "tall vase", "steel vase", "blue vase"], prop={"material": ["glass jar", "glass bowl"], "function": ["flower pot", "water jug"]},
                      desc_m=["thing for cut flowers", "vessel holding tulips"], desc_n=["thing for storing jam", "dish for serving soup"]),
         "ice": dict(mem=["ice", "clear ice", "solid ice", "old ice"], prop={"material": ["cold glass", "frozen soup"], "function": ["cold stone", "chill pack"]},
                     desc_m=["frozen water block", "cube of frozen water"], desc_n=["cold metal block", "chilled stone cube"]),
         "rope": dict(mem=["rope", "thick rope", "old rope", "long rope"], prop={"material": ["thick thread", "old cable"], "function": ["long chain", "steel wire"]},
                      desc_m=["twisted fibre cord", "braided hemp line"], desc_n=["linked metal line", "twisted steel line"]),
         "clock": dict(mem=["clock", "wall clock", "old clock", "small clock"], prop={"material": ["wall mirror", "old radio"], "function": ["small timer", "old watch"]},
                       desc_m=["thing that tells the hours", "ticking time keeper"], desc_n=["thing that plays songs", "ticking metal box"])}
TPL = {"D": [("t1", "subject", "The {} fell off the table ."), ("t2", "object", "She picked up the {} slowly .")],
       "V": [("t3", "subject", "The {} stood near the door ."), ("t4", "object", "He carried the {} outside .")]}
SPLIT = {"D": "discovery", "V": "validation"}

def fixture():
    recs, n = [], 0
    for K, d in KINDS.items():
        for sp, tpls in TPL.items():
            for tid, role, t in tpls:
                def add(phrase, member, stratum, ref_type):
                    nonlocal n
                    text = t.format(phrase); a = t.index("{}")
                    recs.append(dict(id=f"r{n}", bundle_id=f"b{n}", template_id=tid, split=SPLIT[sp], type="recognition", target_kind=K,
                                     is_member=member, stratum=stratum, ref_type=ref_type, role=role, text=text, ref_span=[a, a + len(phrase)])); n += 1
                for ph in d["mem"]: add(ph, True, "member:identity_variant", "noun")
                for pk, phs in d["prop"].items():
                    for ph in phs: add(ph, False, f"property_matched:{pk}", "noun")
                for ph in d["desc_m"]: add(ph, True, "member:identity_variant", "description")
                for ph in d["desc_n"]: add(ph, False, "property_matched:function", "description")
    for fam, F in G4.FAMILIES.items():
        for k, p in enumerate(F["prompts"]):
            q = p.replace(f"causing the {F['patient']} to", "causing it to").replace("The ", "Then the ", 1)   # perturb: avoid probe-only exact text
            recs.append(dict(id=f"c_{fam}_{k}", bundle_id=f"cb_{fam}_{k}", split="discovery" if k % 2 else "p3_discovery", type="causal",
                             family=fam, patient=F["patient"], bystander="clock", target=F["target"], prompt=q,
                             prompt_c2=G4.C2_PREFIX + q[0].lower() + q[1:]))
    return recs

recs = fixture()
texts = [r.get("text") or "" for r in recs] + [r.get("prompt", "") + " " + r.get("prompt_c2", "") for r in recs]
words = sorted({w for t in texts for w in re.findall(r"\w+|[^\w\s]+", t)} | {f["target"] for f in G4.FAMILIES.values()})
vocab = {"<pad>": 0, "<eos>": 1, "<bos>": 2, "<unk>": 3, **{w: i + 4 for i, w in enumerate(words)}}
tk = Tokenizer(models.WordLevel(vocab, unk_token="<unk>")); tk.pre_tokenizer = pre_tokenizers.Whitespace()
tk.post_processor = processors.TemplateProcessing(single="<bos> $A", special_tokens=[("<bos>", 2)])
tok = PreTrainedTokenizerFast(tokenizer_object=tk, bos_token="<bos>", eos_token="<eos>", unk_token="<unk>", pad_token="<pad>")
torch.manual_seed(0)
cfg = Gemma2Config(vocab_size=len(vocab), hidden_size=64, intermediate_size=128, num_hidden_layers=10, num_attention_heads=4,
                   num_key_value_heads=2, head_dim=16, query_pre_attn_scalar=16, sliding_window=64, max_position_embeddings=128, attn_implementation="eager")
model = Gemma2ForCausalLM(cfg).eval(); model.config._attn_implementation = "eager"
rng = np.random.default_rng(0); d, Fw = 64, 256
SAES = {L: dict(W_enc=rng.normal(size=(d, Fw)) * 0.5, b_enc=np.zeros(Fw), threshold=np.full(Fw, 0.3), W_dec=rng.normal(size=(Fw, d)) * .1, b_dec=np.zeros(d)) for L in range(10)}

# T1: loader refuses evaluation split and G4 probe-only items
p = tempfile.mktemp(); open(p, "w").write("\n".join(json.dumps(r) for r in recs + [dict(id="x", split="evaluation", type="recognition")]))
try: GR.load(p); raise AssertionError("evaluation not refused")
except SystemExit as e: assert "evaluation" in str(e)
p = tempfile.mktemp(); open(p, "w").write("\n".join(json.dumps(r) for r in recs + [dict(id="y", split="discovery", type="causal", prompt=G4.FAMILIES["impact"]["prompts"][0])]))
try: GR.load(p); raise AssertionError("probe item not refused")
except SystemExit as e: assert "probe-only" in str(e)
p = tempfile.mktemp(); open(p, "w").write("\n".join(json.dumps(r) for r in recs)); recs_ok = GR.load(p)
print("T1 loader guards PASS")

GR.P_MIN, GR.RANK_MAX, GR.BYSTANDER_SHIFT_MAX = 0.0, 10**9, 1e9     # random model: relax eligibility so matching code paths run
GR.T["min_per_stratum"] = 1
def run():
    return GR.run_repair(model, tok, "cpu", copy.deepcopy(recs_ok), lambda L: SAES[L], log=lambda s: None, layers=[2, 3, 4], ks=[1, 2, 4, 8])
R1 = run(); R2 = run()
dump = lambda R: json.dumps(R["layers"], sort_keys=True, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
assert dump(R1) == dump(R2); print("T2 determinism (two runs identical) PASS")
assert not {k: v for k, v in R1["errors"].items() if k != "bad_span"}, R1["errors"]
for L, LO in R1["layers"].items():
    for K, i in LO["kinds"].items():
        o = i["greedy_order"]; assert len(o) == 8 and len(set(o)) == 8
        for k in (1, 2, 4, 8): assert i["per_k"][k]["latents"] == o[:k]                      # nested sets
        for ro, lo in i["loro_orders"].items(): assert lo == [] or len(lo) == 8               # LORO reselection ran
        if "R3" in i:
            r3 = i["R3"]
            for b, s in r3["c2_sets"].items(): assert s == LO["kinds"][b]["greedy_order"][:r3["k_used"]]   # C2 bugfix semantics
            assert not set(r3["c1_latents"]) & set(o[:r3["k_used"]])                          # C1 disjoint from Substance
            for row in r3["rows"]:
                assert set(row["codes"]) <= {"MATCH_INFEASIBLE"}                              # active ratio NOT an exclusion
                assert "active_ratio_C1" in row and "active_ratio_C2" in row                  # ...but measured
                if not row["codes"]:
                    assert row["N_star"] <= min(row["n_S"], row["n_S2"]) + 1e-6
                    assert row["N_star"] <= GR.A_MAX * min(row["n_C1"], row["n_C2"]) + 1e-6
                    assert row["N_star"] >= GR.RHO_MIN * max(row["n_S"], row["n_S2"]) - 1e-6
print("T3 nested k-sets, LORO, C2 = bystander greedy prefix, C1 disjoint, active-ratio measured-not-excluded, N* bounds PASS")
# T4: force the P3 + realized-collateral code paths to execute (relax rho_min only for this plumbing test)
GR.RHO_MIN = 0.0
R4 = GR.run_repair(model, tok, "cpu", copy.deepcopy(recs_ok), lambda L: SAES[L], log=lambda s: None, layers=[3], ks=[1, 2, 4, 8])
p3 = [i["P3"] for i in R4["layers"][3]["kinds"].values() if "R3" in i]
assert any(x is not None and x["n_prompts"] > 0 and len(x["top_groups"]) > 0 for x in p3), p3   # kinds whose P3 prompts all fail matching legitimately return None
kl = [i["per_k"][8]["R1"]["kl_mean"] for i in R4["layers"][3]["kinds"].values()]
assert all(v > 0 for v in kl), kl
print("T4 P3 feasibility path + KL>0 PASS", [round(v, 6) for v in kl])
GR.RHO_MIN = 0.5
print("layer verdicts", {L: {K: v["ok"] for K, v in vv.items()} for L, vv in R1["layer_verdicts"].items()}, "| L*", R1["layer_selection"]["L_star"])
print("LOCAL_TESTS PASS")
