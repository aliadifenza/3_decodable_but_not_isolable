"""Local code-path test for g5v_probe.run_verify: tiny random Gemma2, toy word tokenizer, random JumpReLU SAE.
Plumbing only; says nothing about the real model."""
import json, re, sys, os, numpy as np, torch
here = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [here, os.path.join(here, "..", "g4_probe")]
import g4_probe as G4, g5v_probe as V
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast, Gemma2Config, Gemma2ForCausalLM

texts = G4.probe_only_manifest()["items"] + V.derived_manifest()["items"] + [t.format(w) for t in G4.TEMPLATES_TRAIN + G4.TEMPLATES_TEST + G4.PRONOUN_TEMPLATES
                                       for ws in list(G4.CONCEPTS.values()) + [G4.POOL] for w in ws]
words = sorted({w for t in texts for w in re.findall(r"\w+|[^\w\s]+", t)} | {f["target"] for f in G4.FAMILIES.values()})
vocab = {"<pad>": 0, "<eos>": 1, "<bos>": 2, "<unk>": 3, **{w: i + 4 for i, w in enumerate(words)}}
tk = Tokenizer(models.WordLevel(vocab, unk_token="<unk>")); tk.pre_tokenizer = pre_tokenizers.Whitespace()
tk.post_processor = processors.TemplateProcessing(single="<bos> $A", special_tokens=[("<bos>", 2)])
tok = PreTrainedTokenizerFast(tokenizer_object=tk, bos_token="<bos>", eos_token="<eos>", unk_token="<unk>", pad_token="<pad>")
torch.manual_seed(0)
cfg = Gemma2Config(vocab_size=len(vocab), hidden_size=64, intermediate_size=128, num_hidden_layers=8, num_attention_heads=4,
                   num_key_value_heads=2, head_dim=16, query_pre_attn_scalar=16, sliding_window=64, max_position_embeddings=128,
                   attn_implementation="eager")
model = Gemma2ForCausalLM(cfg).eval(); model.config._attn_implementation = "eager"
rng = np.random.default_rng(0); d, F = 64, 256
SAES = {L: dict(W_enc=rng.normal(size=(d, F)) * 0.5, b_enc=np.zeros(F), threshold=np.full(F, 0.5), W_dec=rng.normal(size=(F, d)) * .1, b_dec=np.zeros(d)) for L in range(8)}
V.P_MIN, V.RANK_MAX = 0.0, 10**9          # random model: relax eligibility so the matching path is exercised
R = V.run_verify(model, tok, "cpu", print, lambda L: SAES[L])
json.dumps(R, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
assert not R["errors"], R["errors"]
assert R["layers_scanned"] == [2, 3, 4, 5]
assert all(i["codes"] == [] or set(i["codes"]) <= {"BYSTANDER_DESTABILIZES"} for i in R["items"]), [i["codes"] for i in R["items"]]
assert all(len(i["P"]) == 1 and len(i["Pc"]) == 1 for i in R["items"])        # Decision-2 format: one mention each
for r in R["realized_at_L_star"]:
    for n in ("S", "C1", "S2", "C2"):
        assert r[n]["tol_ok"], (n, r[n]); assert r[n]["alpha"] <= V.A_MAX + 1e-9
print("L*", R["layer_selection"]["L_star"], "| realized rows", len(R["realized_at_L_star"]), "| tol_all", R["tolerance_all_ok"],
      "| excl L*", R["exclusions_by_layer"][R["layer_selection"]["L_star"]])
print("LOCAL_TEST PASS")
