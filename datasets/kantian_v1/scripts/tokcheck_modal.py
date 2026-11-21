"""Tokenization check with the approved Gemma tokenizer (google/gemma-2-2b @ pinned revision). CPU-only Modal job:
downloads tokenizer files only (no model weights, no GPU). For every causal_prediction record: target is a single token
when appended to the prompt (prompt + " " + target == prompt tokens + 1), each alternative likewise, prompt prefix stable.
For every recognition record: the reference phrase (entity_description) is located in the sentence and its last token
exists. Writes tokcheck_results.json.
Run from the Mac (folder containing this file and the three JSONL files):
  MODAL_IMAGE_BUILDER_VERSION=2025.06 python3 -m modal run tokcheck_modal.py
"""
import json, os, collections

MODEL_ID, REV = "google/gemma-2-2b", "c5ebcd40d208330abc697524c919956e692655cf"
FILES = ["discovery_v1.1.jsonl", "validation_v1.0.jsonl", "evaluation_v1.0.jsonl"]


def check(records, tok):
    out = dict(causal={}, recognition={}, summary={})
    enc = lambda s: tok(s, add_special_tokens=True)["input_ids"]
    def single(prompt, w):
        a, b = enc(prompt), enc(prompt + " " + w)
        return dict(ok=len(b) == len(a) + 1 and b[:len(a)] == a, pieces=tok.convert_ids_to_tokens(b[len(a):]) if b[:len(a)] == a else "prefix-mismatch")
    S = collections.Counter()
    for r in records:
        key = (r["proposed_split"], r["family"])
        if r["record_type"] == "causal_prediction":
            t = single(r["sentence"], r["candidate_target_word"]); alts = {a: single(r["sentence"], a) for a in r["alternative_target_words"]}
            out["causal"][r["unique_id"]] = dict(target=r["candidate_target_word"], target_single=t["ok"], target_pieces=t["pieces"],
                                                 alts_single={a: v["ok"] for a, v in alts.items()})
            S[key + ("causal",)] += 1; S[key + ("causal_target_single",)] += t["ok"]
            S[key + ("causal_all_alts_single",)] += all(v["ok"] for v in alts.values())
        else:
            ph = r.get("entity_description") or ""; i = r["sentence"].lower().find(ph.lower()) if ph else -1
            e = tok(r["sentence"], return_offsets_mapping=True, add_special_tokens=True)
            last = None
            if i >= 0:
                cand = [k for k, (s, f) in enumerate(e["offset_mapping"]) if f > s and s < i + len(ph) and f > i]
                last = max(cand) if cand else None
            out["recognition"][r["unique_id"]] = dict(phrase_found=i >= 0, last_token=last, n_tokens=len(e["input_ids"]))
            S[key + ("recognition",)] += 1; S[key + ("recognition_span_found",)] += i >= 0 and last is not None
    out["summary"] = {"|".join(k): v for k, v in sorted(S.items())}
    out["multi_token_targets"] = sorted({(v["target"], " ".join(map(str, v["target_pieces"]))) for v in out["causal"].values() if not v["target_single"]})
    return out


try:
    import modal
    app = modal.App("alignedlab-exp3-ws1-tokcheck")
    image = modal.Image.debian_slim(python_version="3.12").pip_install("transformers==4.46.3", "tokenizers==0.20.3", "huggingface_hub==0.26.5")
    vol = modal.Volume.from_name("kant-hf-cache", create_if_missing=True)

    @app.function(image=image, cpu=1.0, memory=2048, timeout=600, volumes={"/root/.cache/huggingface": vol},
                  secrets=[modal.Secret.from_name("huggingface-secret")])
    def run(blobs: dict):
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REV)
        recs = [json.loads(l) for b in blobs.values() for l in b.decode().splitlines() if l.strip()]
        out = check(recs, tok)
        out["tokenizer"] = dict(model=MODEL_ID, revision=REV, cls=type(tok).__name__, vocab=len(tok))
        return json.dumps(out, default=str).encode()

    @app.local_entrypoint()
    def main():
        blobs = {f: open(f, "rb").read() for f in FILES}
        b = run.remote(blobs); open("tokcheck_results.json", "wb").write(b); d = json.loads(b)
        print(json.dumps(d["summary"], indent=1)); print("multi-token targets:", d["multi_token_targets"]); print("wrote tokcheck_results.json")
except ImportError:
    pass
