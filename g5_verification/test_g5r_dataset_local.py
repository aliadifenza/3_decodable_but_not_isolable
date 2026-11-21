"""Local plumbing test of g5r_probe on the REAL G5r input (dataset v1: discovery v1.2 + validation v1.1 via g5r_from_dataset.py)
with a tiny random Gemma2 + word-level toy tokenizer + random SAEs. Checks schema compatibility, span/position resolution,
causal C2 spans, per-kind outputs and the frozen impact-only layer-selection rule. Says nothing about the real model."""
import json, re, sys, os, numpy as np, torch, copy, collections
here = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [here, os.path.join(here, "..", "g4_probe")]
import g5r_probe as GR
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast, Gemma2Config, Gemma2ForCausalLM

recs = GR.load(os.path.join(here, "g5r_input.jsonl"))
assert not any(r["split"] == "evaluation" for r in recs)
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
rng = np.random.default_rng(0); d, Fw = 64, 256
SAES = {L: dict(W_enc=rng.normal(size=(d, Fw)) * 0.5, b_enc=np.zeros(Fw), threshold=np.full(Fw, 0.3), W_dec=rng.normal(size=(Fw, d)) * .1, b_dec=np.zeros(d)) for L in range(10)}
GR.P_MIN, GR.RANK_MAX, GR.BYSTANDER_SHIFT_MAX = 0.0, 10**9, 1e9     # random model: relax eligibility so matching code paths run
R = GR.run_repair(model, tok, "cpu", copy.deepcopy(recs), lambda L: SAES[L], log=lambda s: None, layers=[2, 3], ks=[1, 2, 4, 8])
assert "bad_span" not in R["errors"], R["errors"]["bad_span"][:5]
assert not {k: v for k, v in R["errors"].items()}, R["errors"]
kinds = set(R["layers"][2]["kinds"]); exp = {r["target_kind"] for r in recs if r["type"] == "recognition"}
assert kinds == exp and len(kinds) == 12, kinds
for L in (2, 3):
    for K, i in R["layers"][L]["kinds"].items():
        if K in ("umbrella", "wristwatch", "teapot"): continue
        assert "R3" in i and i["R3"]["eligible_items"] == 8, (L, K, i.get("R3", {}).get("eligible_items"))   # 8 discovery causal items per kind, all spans OK
        assert set(i["R3"]["c2_sets"]) <= {"umbrella", "wristwatch", "teapot"}
sel = R["layer_selection"]
assert sel["confirmatory_kinds"] == ["aluminium beverage can", "car windshield", "raw hen's egg"], sel["confirmatory_kinds"]
assert sel["all_layers_completed"] and "IMPACT" in sel["rule"]
print("dataset schema run PASS:", len(recs), "records,", len(kinds), "kinds; L_star on random model =", sel["L_star"], "(expected None)")
