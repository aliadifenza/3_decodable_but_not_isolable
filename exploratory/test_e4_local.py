"""Local plumbing test for e4_probe on the real G5r input with a tiny random Gemma2 (says nothing about the real model)."""
import json, re, sys, os, copy, torch
here = os.path.dirname(os.path.abspath(__file__)); G5 = os.path.join(here, "..", "g5_verification")
sys.path[:0] = [here, G5, os.path.join(here, "..", "g4_probe")]
import e_probe as E, e4_probe as E4
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast, Gemma2Config, Gemma2ForCausalLM
recs = E.load(os.path.join(G5, "g5r_input.jsonl"))
texts = [r.get("text", "") for r in recs] + [r.get("prompt", "") + " " + r.get("prompt_c2", "") + " " + r.get("target", "") for r in recs]
words = sorted({w for t in texts for w in re.findall(r"\w+|[^\w\s]+", t)})
vocab = {"<pad>": 0, "<eos>": 1, "<bos>": 2, "<unk>": 3, **{w: i + 4 for i, w in enumerate(words)}}
tk = Tokenizer(models.WordLevel(vocab, unk_token="<unk>")); tk.pre_tokenizer = pre_tokenizers.Whitespace()
tk.post_processor = processors.TemplateProcessing(single="<bos> $A", special_tokens=[("<bos>", 2)])
tok = PreTrainedTokenizerFast(tokenizer_object=tk, bos_token="<bos>", eos_token="<eos>", unk_token="<unk>", pad_token="<pad>")
torch.manual_seed(0)
cfg = Gemma2Config(vocab_size=len(vocab), hidden_size=64, intermediate_size=128, num_hidden_layers=10, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                   query_pre_attn_scalar=16, sliding_window=64, max_position_embeddings=256, attn_implementation="eager")
model = Gemma2ForCausalLM(cfg).eval(); model.config._attn_implementation = "eager"
run = lambda: E4.run_e4(model, tok, "cpu", copy.deepcopy(recs), log=lambda s: None, layers=[2, 4], boot=100)
R1, R2 = run(), run()
dump = lambda R: json.dumps(R["layers"], sort_keys=True, default=str)
assert dump(R1) == dump(R2) and not R1["errors"]; print("T1 determinism PASS")
for L, LO in R1["layers"].items():
    assert len(LO["kinds"]) == 12
    for K, res in LO["kinds"].items():
        assert len(res) == 6
        for nm, m in res.items():
            assert m["kl_mean"] >= 0 and m["rel_norm_final"] >= 0
            if m["method"] in ("meandiff", "leace") and m["mode"] == "spanfinal" and m["clean_auc_now"] == m["clean_auc_now"] and m["clean_auc_now"] > 0.6:
                pass
print("T2 structure (12 kinds x 6 conditions) PASS")
# T3 final-token eraser bites at the read position: r_now for LEACE spanfinal well above the random direction on average
import statistics as st
ln = [m["r_now"] for LO in R1["layers"].values() for res in LO["kinds"].values() for nm, m in res.items() if nm == "E4a_spanfinal_leace" and m["r_now"] == m["r_now"]]
rn = [m["r_now"] for LO in R1["layers"].values() for res in LO["kinds"].values() for nm, m in res.items() if nm == "E4a_spanfinal_randdir" and m["r_now"] == m["r_now"]]
print("median r_now leace", round(st.median(ln), 3), "randdir", round(st.median(rn), 3)); assert st.median(ln) > st.median(rn)
print("T3 final-token erasure removes identity at the read position (tiny model) PASS")
print("LOCAL_TESTS PASS |", E4.LABEL)
