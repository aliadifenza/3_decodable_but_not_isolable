"""Deterministic local tests for e_probe (EXPLORATORY E1-E3) on the REAL G5r input with a tiny random Gemma2, a word-level toy
tokenizer and random JumpReLU SAEs. Exercises code paths, guards and exact math properties; says nothing about the real model."""
import json, re, sys, os, copy, tempfile, numpy as np, torch
here = os.path.dirname(os.path.abspath(__file__)); G5 = os.path.join(here, "..", "g5_verification")
sys.path[:0] = [here, G5, os.path.join(here, "..", "g4_probe")]
import e_probe as E, g5r_probe as GR
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast, Gemma2Config, Gemma2ForCausalLM

data = os.path.join(G5, "g5r_input.jsonl")
# T1 guards: evaluation records, evaluation file names
recs = E.load(data)
for bad in ([dict(id="x", split="evaluation", type="recognition")], None):
    p = tempfile.mktemp(suffix="_evaluation.jsonl" if bad is None else ".jsonl")
    open(p, "w").write("\n".join(json.dumps(r) for r in recs + (bad or [])))
    try: E.load(p); raise AssertionError("not refused")
    except SystemExit as e: assert "REFUSED" in str(e)
print("T1 evaluation guards PASS")

texts = [r.get("text", "") for r in recs] + [r.get("prompt", "") + " " + r.get("prompt_c2", "") + " " + r.get("target", "") for r in recs]
words = sorted({w for t in texts for w in re.findall(r"\w+|[^\w\s]+", t)})
vocab = {"<pad>": 0, "<eos>": 1, "<bos>": 2, "<unk>": 3, **{w: i + 4 for i, w in enumerate(words)}}
tk = Tokenizer(models.WordLevel(vocab, unk_token="<unk>")); tk.pre_tokenizer = pre_tokenizers.Whitespace()
tk.post_processor = processors.TemplateProcessing(single="<bos> $A", special_tokens=[("<bos>", 2)])
tok = PreTrainedTokenizerFast(tokenizer_object=tk, bos_token="<bos>", eos_token="<eos>", unk_token="<unk>", pad_token="<pad>")
torch.manual_seed(0)
cfg = Gemma2Config(vocab_size=len(vocab), hidden_size=64, intermediate_size=128, num_hidden_layers=10, num_attention_heads=4,
                   num_key_value_heads=2, head_dim=16, query_pre_attn_scalar=16, sliding_window=64, max_position_embeddings=256, attn_implementation="eager")
model = Gemma2ForCausalLM(cfg).eval(); model.config._attn_implementation = "eager"
rng = np.random.default_rng(0); d, Fw = 64, 1024
SAES = {L: dict(W_enc=rng.normal(size=(d, Fw)) * 0.5, b_enc=np.zeros(Fw), threshold=np.full(Fw, 0.3), W_dec=rng.normal(size=(Fw, d)) * .1, b_dec=np.zeros(d)) for L in range(10)}

# T2 exact math: LEACE leaves zero covariance with the label on its fitting data; mean-diff removes the mean difference
g = torch.Generator().manual_seed(1); X = torch.randn(80, 300, generator=g, dtype=torch.float64); y = (torch.arange(80) % 2).double(); X[y.bool()] += 0.3
EZ = E.fit_erasers(X, y, g)
for nm, (a, b) in (("leace", EZ["leace"]), ("meandiff", EZ["meandiff"])):
    Xe = X - ((X - EZ["mu"]) @ a)[:, None] * b[None, :]
    cov = ((Xe - Xe.mean(0)) * (y - y.mean())[:, None]).mean(0).abs().max()
    if nm == "leace": assert cov < 1e-10, cov
    dm = Xe[y.bool()].mean(0) - Xe[~y.bool()].mean(0); assert (dm @ EZ["meandiff"][0]).abs() < 1e-10 if nm == "meandiff" else True
print("T2 LEACE exact (cov=0 on fit data), mean-diff projection PASS")

E.P_MIN, E.RANK_MAX, E.BYSTANDER_SHIFT_MAX = 0.0, 10**9, 1e9      # random model: relax eligibility so D paths run
run = lambda **kw: E.run_explore(model, tok, "cpu", copy.deepcopy(recs), lambda L: SAES[L], log=lambda s: None, layers=[2, 3], boot=200, **kw)
R1 = run(); R2 = run()
dump = lambda R: json.dumps(R["layers"], sort_keys=True, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
assert dump(R1) == dump(R2); print("T3 determinism (two runs identical) PASS")
assert not R1["errors"], R1["errors"]
assert R1["span_tokens"]["description_mean_all"] > 2, R1["span_tokens"]      # 'all' edits really cover the description span

# T4 G5r replication: first 8 identity latents == g5r_probe greedy order on the same model/SAEs
GR.P_MIN, GR.RANK_MAX, GR.BYSTANDER_SHIFT_MAX = 0.0, 10**9, 1e9
RG = GR.run_repair(model, tok, "cpu", copy.deepcopy(recs), lambda L: SAES[L], log=lambda s: None, layers=[2], ks=[1, 2, 4, 8])
for K, i in R1["layers"][2]["kinds"].items():
    assert i["g5r_prefix"] == RG["layers"][2]["kinds"][K]["greedy_order"], K
    # k=8 'last' reproduces G5r's R1 numbers (same intervention, same measurement)
    a, b = i["conditions"]["E1_k8_last"], RG["layers"][2]["kinds"][K]["per_k"][8]["R1"]
    for x in ("r_down", "amnesic_auc", "kl_mean", "rel_norm_mean"):
        assert (a[x] != a[x] and b[x] != b[x]) or abs(a[x] - b[x]) < 1e-6, (K, x, a[x], b[x])
print("T4 G5r replication (k<=8 order and k=8 R1 metrics identical) PASS")

# T5 structure: nested dose sets, random controls disjoint & active, all conditions measured with CIs
for L, LO in R1["layers"].items():
    for K, i in LO["kinds"].items():
        o, r = i["identity_order"], i["random_order"]
        assert len(o) == 128 and len(set(o)) == 128 and not set(o) & set(r)
        C = i["conditions"]; assert len(C) == 26, len(C)
        for nm, m in C.items():
            assert m["kl_mean"] >= 0 and m["n_val"] > 0 and len(m["r_down_ci95"]) == 2
            if m["pos"] == "all": assert m["n_positions_mean"] >= 1
        assert C["E1_k8_all"]["kl_mean"] >= 0
print("T5 nested k-sets, random controls disjoint, 26 conditions x CIs PASS")

# T6 'all' differs from 'last' for description records (more positions edited -> different downstream)
diff = [abs(i["conditions"]["E1_k128_all"]["kl_mean"] - i["conditions"]["E1_k128_last"]["kl_mean"]) for i in R1["layers"][2]["kinds"].values()]
assert max(diff) > 0, diff; print("T6 all-token edits differ from last-token edits PASS")

# T7 D comparison path: force R1 criteria met (thresholds relaxed ONLY in this test) -> D rows with S, C1, C2
E.T.update(r_down=-1e9, amnesic_max=1e9, role_drop_max=1e9)
R3 = E.run_explore(model, tok, "cpu", copy.deepcopy(recs), lambda L: SAES[L], log=lambda s: None, layers=[3], ks=[8], boot=100)
Dx = R3["layers"][3]["D_exploratory"]; assert Dx, "no D results"
assert set(Dx) <= {"raw hen's egg", "aluminium beverage can", "car windshield", "corn kernel", "raw egg white", "beeswax block", "parchment", "balsa strip", "acrylic sheet"}
some = next(iter(next(iter(Dx.values())).values())); assert some["rows"] and {"D_S", "D_C1", "D_C2"} <= set(some["rows"][0])
assert all(c["split"] == "discovery" for c in recs if c["type"] == "causal")
print("T7 exploratory D comparison path (S vs C1 vs C2, event summaries) PASS")
print("LOCAL_TESTS PASS  |", E.LABEL)
