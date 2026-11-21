"""Kantian Positional Logic — EXPLORATORY E4 "already travelled?" (2026-11-21 ~20:50 CET; PI standing rule: spend within €30 total).

EXPLORATORY. NOT CONFIRMATORY. Discovery + validation only; evaluation refused (e_probe.load).

Question: E3 erased the identity direction only at the reference span at layer L, yet ~75% of identity was still read
downstream (layer L+4, final token). Did identity simply travel ahead — copied by attention in layers < L into the final
token / other context tokens — or does the model reconstitute it from what remains?
  E4a span+final : site eraser (fit on discovery reference-site reps, as E3) on the span tokens + a FINAL-token eraser
                   (fit on discovery final-token reps at layer L) on the final token.
  E4b sequence   : one pooled eraser fit on ALL non-BOS token reps of discovery sentences at layer L (label = sentence's
                   membership), applied at every non-BOS token.
  Methods        : mean-difference projection, LEACE (10% shrinkage) — e_probe.fit_erasers; control: covariance-matched
                   random direction at the same positions.
Measures (validation): r_now = identity removal at layer L, final token (clean-trained probe at L) — checks the erasure
  bites where the downstream probe reads; r_down = removal at L+4, final token (as R1/E3); 95% item-bootstrap CIs;
  ‖Δ‖/‖h‖ at the final token; next-token KL.
Reading: if r_down → high under E4a/E4b, the E3 "return" was mostly travel (copies outside the span). If r_now is high
  but r_down stays low, identity is re-derived downstream from non-linear / distributed residue.

Run: MODAL_IMAGE_BUILDER_VERSION=2025.06 python3 -m modal run e4_probe.py --data g5r_input.jsonl --layers 2,4,6,8,10,12,14
"""
import json, os, re, time, hashlib
import g4_probe as G4
import e_probe as E

MODEL_ID, REV, SAE_REPO = E.MODEL_ID, E.REV, E.SAE_REPO
GPU, GPU_USD_PER_H, TIMEOUT_S, SEED = "A10G", 1.10, 40 * 60, E.SEED
LAYERS = [2, 4, 6, 8, 10, 12, 14]; READ_OFFSET, BATCH, BOOT = 4, 32, 1000
LABEL = "EXPLORATORY E4 (already travelled?) — not confirmatory; discovery/validation only; evaluation sealed"


def run_e4(model, tok, device, recs, log=print, time_left=lambda: 1e9, layers=LAYERS, boot=BOOT):
    import torch
    torch.manual_seed(SEED); F = torch.nn.functional
    t0 = time.time(); R = dict(label=LABEL, settings=dict(layers=layers, READ_OFFSET=READ_OFFSET, BOOT=boot, SHRINK=E.SHRINK), stages={}, errors={})
    def stage(n): R["stages"][n] = round(time.time() - t0, 1); log(f"[{R['stages'][n]:7.1f}s] {n}")
    nL = model.config.num_hidden_layers; Vv = len(tok); L_mod = model.model.layers
    read_layers = sorted({min(L + READ_OFFSET, nL - 1) for L in layers})
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    def enc(text):
        e = tok(text, return_offsets_mapping=True, add_special_tokens=True); return e["input_ids"], e["offset_mapping"]
    def toks_in(offs, a, b): return [i for i, (s, e) in enumerate(offs) if e > s and s < b and e > a]
    def out_t(o): return o[0] if isinstance(o, tuple) else o
    def run_batch(ids_list, layer_edits=(), capture=None, capture_all=None, logits_at=None):
        n = len(ids_list); m = max(map(len, ids_list))
        ids = torch.full((n, m), pad, dtype=torch.long, device=device); am = torch.zeros((n, m), dtype=torch.long, device=device)
        for i, s in enumerate(ids_list): ids[i, :len(s)] = torch.tensor(s, device=device); am[i, :len(s)] = 1
        store, hs = {}, []
        for L, f in layer_edits: hs.append(L_mod[L].register_forward_hook(f))
        for L, (rows, pos) in (capture or {}).items():
            def mk(L, rows, pos):
                def f(mod, i, o): store[L] = out_t(o)[rows, pos, :].detach().float().clone()
                return f
            hs.append(L_mod[L].register_forward_hook(mk(L, rows, pos)))
        for L in (capture_all or []):
            def mk2(L):
                def f(mod, i, o): store[("all", L)] = out_t(o).detach().float().cpu().clone()
                return f
            hs.append(L_mod[L].register_forward_hook(mk2(L)))
        try:
            with torch.no_grad(): o = model(ids, attention_mask=am, use_cache=False)
        finally:
            for h in hs: h.remove()
        if logits_at is not None: store["logits"] = o.logits[logits_at[0], logits_at[1], :Vv].double().clone()
        return store
    def edit_hook(rows, pos, fns):                      # fns: list (one per (row,pos)) of erasers grouped by identity
        groups = {}
        for i, f_ in enumerate(fns): groups.setdefault(id(f_), (f_, []))[1].append(i)
        def f(mod, i, o):
            h = out_t(o)
            for fn, idx in groups.values():
                rr = [rows[j] for j in idx]; pp = [pos[j] for j in idx]; h[rr, pp, :] = fn(h[rr, pp, :])
            return o
        return f
    def auc1(p, n):
        if len(p) == 0 or len(n) == 0: return float("nan")
        a, b = p.double()[:, None], n.double()[None, :]; return float((a > b).double().mean() + 0.5 * (a == b).double().mean())
    def fit_probe(X, y, lam=1e-2):
        X = X.double().cpu(); y = y.double().cpu(); mu, sd = X.mean(0), X.std(0) + 1e-6; Z = (X - mu) / sd
        w = torch.zeros(Z.shape[1] + 1, dtype=torch.float64, requires_grad=True)
        opt = torch.optim.LBFGS([w], max_iter=200, line_search_fn="strong_wolfe")
        def closure():
            opt.zero_grad(); z = Z @ w[:-1] + w[-1]
            loss = F.binary_cross_entropy_with_logits(z, y) + lam * (w[:-1] ** 2).sum() / len(y); loss.backward(); return loss
        opt.step(closure); w = w.detach()
        return lambda Xn: ((Xn.double().cpu() - mu) / sd) @ w[:-1] + w[-1]
    def boot_r(sc, sa, yb, g):
        P_ = torch.nonzero(yb).flatten(); N_ = torch.nonzero(~yb).flatten()
        ip = P_[torch.randint(len(P_), (boot, len(P_)), generator=g)]; iN = N_[torch.randint(len(N_), (boot, len(N_)), generator=g)]
        def A(s): s = s.double(); a, b = s[ip][:, :, None], s[iN][:, None, :]; return (a > b).double().mean((1, 2)) + 0.5 * (a == b).double().mean((1, 2))
        bc, ba = A(sc), A(sa); r = torch.where(bc > 0.5, 1 - (ba - 0.5) / (bc - 0.5).clamp_min(1e-9), torch.full_like(bc, float("nan")))
        r = r[~torch.isnan(r)]; return [float(torch.quantile(r, .025)), float(torch.quantile(r, .975))] if len(r) else [float("nan")] * 2
    def removal(probe, Xc, Xa, yb, g):
        sc, sa = probe(Xc), probe(Xa); c, a = auc1(sc[yb], sc[~yb]), auc1(sa[yb], sa[~yb])
        if not c > 0.5: return float("nan"), [float("nan")] * 2, c
        return 1 - (a - 0.5) / (c - 0.5), boot_r(sc, sa, yb, g), c

    rec_r = [r for r in recs if r["type"] == "recognition"]
    for r in rec_r:
        ids, offs = enc(r["text"]); r["_ids"] = ids; r["_all"] = toks_in(offs, *r["ref_span"]); r["_last"] = len(ids) - 1
        r["_pos"] = max(r["_all"]) if r["_all"] else None
    rec_r = [r for r in rec_r if r["_pos"] is not None]
    kinds = sorted({r["target_kind"] for r in rec_r})
    batches = [grp[i:i + BATCH] for sp in ("discovery", "validation") for grp in [[r for r in rec_r if r["split"] == sp]] for i in range(0, len(grp), BATCH)]
    for b in batches:
        rows = list(range(len(b)))
        st = run_batch([r["_ids"] for r in b], capture={L: (rows, [r["_pos"] for r in b]) for L in layers}, logits_at=(rows, [r["_last"] for r in b]),
                       capture_all=layers if b[0]["split"] == "discovery" else None)
        st2 = run_batch([r["_ids"] for r in b], capture={L: (rows, [r["_last"] for r in b]) for L in sorted(set(read_layers) | set(layers))})
        for j, r in enumerate(b):
            r["_m"] = {L: st[L][j].cpu() for L in layers}; r["_lp"] = torch.log_softmax(st["logits"][j], -1).cpu()
            r["_fL"] = {L: st2[L][j].cpu() for L in layers}; r["_f"] = {L: st2[L][j].cpu() for L in read_layers}
            if r["split"] == "discovery":   # every non-BOS token (pooled eraser); batches are split-pure
                r["_seq"] = {L: st[("all", L)][j, 1:len(r["_ids"])].half() for L in layers}
    stage("capture")

    out = {}
    for L in layers:
        if time_left() < 200: R["errors"]["layers"] = f"stopped before layer {L}: time budget"; break
        Lr = min(L + READ_OFFSET, nL - 1); LO = dict(read_layer=Lr, kinds={})
        for ki, K in enumerate(kinds):
            D = [r for r in rec_r if r["split"] == "discovery" and r["target_kind"] == K]; V = [r for r in rec_r if r["split"] == "validation" and r["target_kind"] == K]
            if not D or not V: continue
            g = torch.Generator().manual_seed(SEED + 7000 + 1000 * L + ki)
            yD = torch.tensor([float(r["is_member"]) for r in D], dtype=torch.float64); yV = torch.tensor([float(r["is_member"]) for r in V]); yb = yV.bool()
            EZ_site = E.fit_erasers(torch.stack([r["_m"][L] for r in D]).double(), yD, g)
            EZ_fin = E.fit_erasers(torch.stack([r["_fL"][L] for r in D]).double(), yD, g)
            Xs = torch.cat([r["_seq"][L].double() for r in D]); ys = torch.cat([torch.full((r["_seq"][L].shape[0],), float(r["is_member"]), dtype=torch.float64) for r in D])
            EZ_seq = E.fit_erasers(Xs, ys, g); del Xs
            def mk(EZ, nm):
                a_, b_, m_ = EZ[nm][0].float().to(device), EZ[nm][1].float().to(device), EZ["mu"].float().to(device)
                return lambda x: x - ((x.float() - m_) @ a_)[:, None] * b_[None, :]
            conds = {}
            for nm in ("meandiff", "leace", "randdir"):
                conds[f"E4a_spanfinal_{nm}"] = ("spanfinal", mk(EZ_site, nm), mk(EZ_fin, nm))
                conds[f"E4b_sequence_{nm}"] = ("sequence", mk(EZ_seq, nm), None)
            pnow = fit_probe(torch.stack([r["_fL"][L] for r in D]), yD.float()); pdown = fit_probe(torch.stack([r["_f"][Lr] for r in D]), yD.float())
            res = {}
            for name, (mode, f_span, f_fin) in conds.items():
                fnow, fdown, kls = [], [], []
                for i in range(0, len(V), BATCH):
                    b = V[i:i + BATCH]; rows, pos, fns = [], [], []
                    for j, r in enumerate(b):
                        if mode == "spanfinal":
                            ps = sorted(set(r["_all"]) - {r["_last"]}); rows += [j] * len(ps); pos += ps; fns += [f_span] * len(ps)
                            rows.append(j); pos.append(r["_last"]); fns.append(f_fin)
                        else:
                            ps = list(range(1, len(r["_ids"]))); rows += [j] * len(ps); pos += ps; fns += [f_span] * len(ps)
                    br = list(range(len(b))); last = [r["_last"] for r in b]
                    st = run_batch([r["_ids"] for r in b], layer_edits=[(L, edit_hook(rows, pos, fns))], capture={L: (br, last), Lr: (br, last)}, logits_at=(br, last))
                    fnow.append(st[L].cpu()); fdown.append(st[Lr].cpu()); lpa = torch.log_softmax(st["logits"], -1).cpu()
                    for j, r in enumerate(b): kls.append(float((r["_lp"].exp() * (r["_lp"] - lpa[j])).sum()))
                fnow = torch.cat(fnow); fdown = torch.cat(fdown)
                Fc_now = torch.stack([r["_fL"][L] for r in V]); Fc_down = torch.stack([r["_f"][Lr] for r in V])
                r_now, ci_now, c_now = removal(pnow, Fc_now, fnow, yb, g); r_down, ci_down, c_down = removal(pdown, Fc_down, fdown, yb, g)
                res[name] = dict(mode=mode, method=name.split("_")[-1], r_now=r_now, r_now_ci95=ci_now, clean_auc_now=c_now, r_down=r_down, r_down_ci95=ci_down,
                                 clean_auc_down=c_down, rel_norm_final=float(((fnow - Fc_now).norm(dim=-1) / Fc_now.norm(dim=-1)).mean()), kl_mean=sum(kls) / len(kls))
            LO["kinds"][K] = res
        out[L] = LO; stage(f"layer_{L}")
    for r in rec_r: r.pop("_seq", None)
    R["layers"] = out; stage("done"); return R


try:
    import modal
    app = modal.App("alignedlab-exp3-exploratory-e4")
    image = (modal.Image.debian_slim(python_version="3.12").pip_install(*G4.PINS)
             .env({"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}).add_local_python_source("g4_probe", "e_probe"))
    vol = modal.Volume.from_name("kant-hf-cache", create_if_missing=True)

    @app.function(image=image, gpu=GPU, timeout=TIMEOUT_S, memory=32768, volumes={"/root/.cache/huggingface": vol},
                  secrets=[modal.Secret.from_name("huggingface-secret")])
    def run(data: bytes, layers=None):
        import torch, tempfile
        t0 = time.time(); logs = []
        def log(s): print(s, flush=True); logs.append(s)
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True; torch.use_deterministic_algorithms(True, warn_only=True)
        p = tempfile.mktemp(suffix=".jsonl"); open(p, "wb").write(data); recs = E.load(p)
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer, AutoModelForCausalLM
        path = snapshot_download(MODEL_ID, revision=REV); tok = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32, attn_implementation="eager").to("cuda").eval()
        R = run_e4(model, tok, "cuda", recs, log, time_left=lambda: TIMEOUT_S - 90 - (time.time() - t0), layers=layers or LAYERS)
        rt = time.time() - t0
        return json.dumps(dict(status=LABEL, identity=dict(model=MODEL_ID, revision=REV, commit_hash=getattr(model.config, "_commit_hash", None)),
                               data_sha256=hashlib.sha256(data).hexdigest(), results=R, runtime_s=round(rt, 1),
                               gpu_cost_usd_estimate=round(rt / 3600 * GPU_USD_PER_H, 3), logs=logs),
                          default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)).encode()

    @app.local_entrypoint()
    def main(data: str, layers: str = ""):
        Ls = [int(x) for x in layers.split(",") if x] or None
        print(f"{LABEL}\nlayers {Ls or LAYERS}; worst-case cost ${TIMEOUT_S / 3600 * GPU_USD_PER_H:.2f} (hard timeout {TIMEOUT_S // 60} min)")
        b = run.remote(open(data, "rb").read(), Ls); os.makedirs("modal_out/e", exist_ok=True)
        name = "e4_results.json" if not Ls else "e4_results_layers_%s.json" % "-".join(map(str, Ls))
        open("modal_out/e/" + name, "wb").write(b); d = json.loads(b); print("wrote modal_out/e/" + name)
        print(f"DONE runtime {d['runtime_s']}s est ${d['gpu_cost_usd_estimate']} errors {d['results'].get('errors')}")
except ImportError:
    pass
