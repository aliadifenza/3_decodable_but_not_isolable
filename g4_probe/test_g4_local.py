"""Local code-path test for g4_probe.run_probe on a tiny random-weight Gemma2 + toy word tokenizer.
Verifies plumbing only; says nothing about the real model."""
import json, re, sys, os, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g4_probe as G
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast, Gemma2Config, Gemma2ForCausalLM

texts = G.probe_only_manifest()["items"] + [t.format(w) for t in G.TEMPLATES_TRAIN + G.TEMPLATES_TEST + G.PRONOUN_TEMPLATES
                                             for ws in list(G.CONCEPTS.values()) + [G.POOL, [G.INTERCHANGE_DONOR]] for w in ws]
words = sorted({w for t in texts for w in re.findall(r"\w+|[^\w\s]+", t)} | {f["target"] for f in G.FAMILIES.values()}
               | {a for f in G.FAMILIES.values() for a in f["alts"]} | set(G.STEEL_DIAG["targets"]))
vocab = {"<pad>": 0, "<eos>": 1, "<bos>": 2, "<unk>": 3, **{w: i + 4 for i, w in enumerate(words)}}
tk = Tokenizer(models.WordLevel(vocab, unk_token="<unk>")); tk.pre_tokenizer = pre_tokenizers.Whitespace()
tk.post_processor = processors.TemplateProcessing(single="<bos> $A", special_tokens=[("<bos>", 2)])
tok = PreTrainedTokenizerFast(tokenizer_object=tk, bos_token="<bos>", eos_token="<eos>", unk_token="<unk>", pad_token="<pad>")
torch.manual_seed(0)
cfg = Gemma2Config(vocab_size=len(vocab), hidden_size=64, intermediate_size=128, num_hidden_layers=8, num_attention_heads=4,
                   num_key_value_heads=2, head_dim=16, query_pre_attn_scalar=16, sliding_window=64, max_position_embeddings=128,
                   attn_implementation="eager")
model = Gemma2ForCausalLM(cfg).eval()
model.config._attn_implementation = "eager"
d = 64; rng = np.random.default_rng(0)
sae = lambda L: dict(W_enc=rng.normal(size=(d, 256)), b_enc=np.zeros(256), threshold=np.full(256, 0.1), W_dec=rng.normal(size=(256, d)) * .05,
                     b_dec=np.zeros(d), _path=f"toy/layer_{L}")
R = G.run_probe(model, tok, "cpu", print, sae_loader=sae)
s = json.dumps(R, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
assert not R["errors"], R["errors"]
assert R["check1"]["deterministic_repeat_bitwise"]
assert all(r["target"]["single_token"] for r in R["check2"])
assert all(p["isolation_layer_input_unchanged"] for p in R["p3"])
for row in R["interventions"]:
    S = row["recs"]["S_patient"]["interventions"][0]["l2_total"]
    for k in ("C1_unrelated_matched", "C1_random_matched"):
        assert abs(row["recs"][k]["interventions"][0]["l2_total"] - S) < 1e-3 * max(S, 1), (k, S)
    if "C2_bystander_matched" in row["recs"]:
        S2 = row["recs"]["S_patient_c2"]["interventions"][0]["l2_total"]
        assert abs(row["recs"]["C2_bystander_matched"]["interventions"][0]["l2_total"] - S2) < 1e-3 * max(S2, 1)
# full restoration of all KV groups must not exceed exact intact output only via that layer; sanity: fields present
print("L*", R["check4"]["L_star"], "| rows", len(R["interventions"]), "| p3", [(p["chosen"]["layer"], p["chosen"]["head"]) for p in R["p3"]],
      "| sae", "sae" in R, "| json bytes", len(s))
print("LOCAL_TEST PASS")
