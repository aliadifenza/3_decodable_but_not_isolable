"""Merge partial G5r runs (same data, model, revision) and apply the FROZEN §7 layer rule over the full range 2-14.
Per-layer results are independent of other layers (same seed, same clean captures), so a union of per-layer results is
equivalent to one run. Usage: python3 merge_g5r.py <out.json> <run1.json> <run2.json> ..."""
import json, sys, hashlib
LAYERS = list(range(2, 15))

def main(out, *paths):
    runs = [json.load(open(p)) for p in paths]
    key = lambda d: (d["data_sha256"], d["identity"]["model"], d["identity"]["revision"])
    assert len({key(d) for d in runs}) == 1, "runs differ in data/model/revision"
    layers, verdict, sae, conf = {}, {}, {}, None
    for d in runs:
        R = d["results"]; sae.update(d["identity"]["sae_files"])
        c = R["layer_selection"]["confirmatory_kinds"]; assert conf in (None, c); conf = c
        for L, LO in R["layers"].items():
            L = int(L); assert L not in layers, f"layer {L} in two runs"; layers[L] = LO; verdict[L] = R["layer_verdicts"][str(L)]
    complete = sorted(layers) == LAYERS
    ok = [L for L in sorted(verdict) if all(verdict[L][K]["ok"] for K in conf)] if complete else []
    mm = lambda L: min(verdict[L][K]["R2_margin"] for K in conf)
    L_star = max(ok, key=lambda x: (mm(x), -x)) if ok else None
    M = dict(status="G5r merged result (frozen §7 rule)", sources=[dict(path=p, sha256=hashlib.sha256(open(p, "rb").read()).hexdigest(),
             runtime_s=d["runtime_s"], cost_est=d["gpu_cost_usd_estimate"], layers=sorted(int(x) for x in d["results"]["layers"]),
             errors=d["results"].get("errors")) for p, d in zip(paths, runs)],
             layers_covered=sorted(layers), complete=complete, confirmatory_kinds=conf, eligible=ok, L_star=L_star,
             k_at_L_star=({K: layers[L_star]["kinds"][K]["k_selected"] for K in layers[L_star]["kinds"]} if L_star is not None else None),
             verdicts={L: {K: {x: v[x] for x in ("R1", "R2", "R3", "P3", "ok", "R2_margin")} for K, v in verdict[L].items()} for L in sorted(verdict)},
             sae_files=sae, total_cost_est=round(sum(d["gpu_cost_usd_estimate"] for d in runs), 3))
    json.dump(M, open(out, "w"), indent=1, default=str)
    print(json.dumps({k: M[k] for k in ("layers_covered", "complete", "eligible", "L_star", "k_at_L_star", "total_cost_est")}, default=str))

if __name__ == "__main__":
    main(*sys.argv[1:])
