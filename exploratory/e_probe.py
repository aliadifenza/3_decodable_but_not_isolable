"""Kantian Positional Logic — EXPLORATORY follow-up E1-E3 (PI-approved 2026-11-21 19:38-19:39 CET; spend cap $10 hard, target ~$2.5).

EXPLORATORY. NOT CONFIRMATORY. Never feeds P2/P3 of the frozen protocol (G5r: no valid layer -> P2/P3 inconclusive).
Discovery + validation only; the loader refuses any evaluation-split record and any path naming the evaluation file.

Question: why did G5r's sparse SAE ablation (<= 8 latents, one position) remove ~0 identity information downstream?
  E1 Dose   : ablate k in {8,16,32,64,128} identity latents. First 8 = exact G5r greedy order (top-64 pool); continued
              greedily (same worst-stratum-AUC criterion) from a top-512 pool.                     [exploratory extension]
  E2 Place  : ablate at the last token of the reference ("last", = G5r) vs at ALL tokens of the reference span ("all").
  E3 Whole  : erase one linear identity direction from the FULL residual (incl. SAE reconstruction error), fit on
              discovery site representations: (a) mean-difference projection, (b) LEACE (Belrose et al. 2023) with
              10% covariance shrinkage. Applied at last / all reference tokens.
  Controls  : same-k random ACTIVE latents (activation freq >= 0.10 on discovery members, disjoint from identity set);
              a random covariance-matched direction (Xc^T g, g ~ N(0, I)) erased by the same projection.
Measures (= G5r R1, on validation): downstream removal r (probe trained on clean discovery at L+4, final token) with
  item-bootstrap 95% CI; same-site amnesic AUC (fresh probe, ablated discovery -> ablated validation) with CI over test
  items; role-probe AUC drop (collateral); ||d||/||h|| at the site; next-token KL. "R1 criteria met" uses the FROZEN R1
  thresholds (r >= 0.67, amnesic <= 0.70, role drop <= 0.05) only as a descriptive bar.
Optional D comparison (discovery causal items only, exploratory): for every condition meeting the R1 criteria on an
  impact/heating/cutting patient kind: D = L_intact - L_intervened (target log-odds); C1 = the condition's random
  control at the patient; C2 = the same condition fitted for the family bystander, applied to the bystander in prompt_c2.
  No magnitude matching (deviation from Decision 2, stated). Eligibility codes as frozen §2; event-cluster means.

Run (PI-authorized):  MODAL_IMAGE_BUILDER_VERSION=2025.06 python3 -m modal run e_probe.py --data g5r_input.jsonl --layers 2,3,4,5,6,7,8
"""
import json, os, re, time, math, hashlib
import g4_probe as G4

MODEL_ID, REV, SAE_REPO = G4.MODEL_ID, G4.REV, "google/gemma-scope-2b-pt-res"
GPU, GPU_USD_PER_H, TIMEOUT_S, SEED = "A10G", 1.10, 45 * 60, 20261121
SPEND_CAP_USD = 10.0                                   # PI hard cap for ALL exploratory runs together
assert TIMEOUT_S / 3600 * GPU_USD_PER_H * 3 < SPEND_CAP_USD   # even 3 runs that hit the timeout stay far below the cap
LAYERS = list(range(2, 15)); KS = [8, 16, 32, 64, 128]; READ_OFFSET = 4; BATCH = 32
POOL_G5R, POOL_EXT, SHRINK, RAND_FREQ_MIN, BOOT = 64, 512, 0.10, 0.10, 1000
P_MIN, RANK_MAX, BYSTANDER_SHIFT_MAX = 0.10, 5, 1.0            # frozen §2 eligibility (reused for D)
T = dict(r_down=0.67, amnesic_max=0.70, role_drop_max=0.05)       # frozen R1 thresholds, used descriptively
LABEL = "EXPLORATORY E1-E3 — not confirmatory; discovery/validation only; evaluation sealed"


def load(path):
    if "evaluation" in os.path.basename(path).lower(): raise SystemExit("REFUSED: evaluation file")
    recs = [json.loads(l) for l in open(path) if l.strip()]
    if any(r.get("split") == "evaluation" for r in recs): raise SystemExit("REFUSED: evaluation-split records present")
    probe = set(G4.probe_only_manifest()["items"])
    bad = [r["id"] for r in recs if r.get("text") in probe or r.get("prompt") in probe]
    if bad: raise SystemExit(f"REFUSED: G4 probe-only items present: {bad[:5]}")
    return recs


def fit_erasers(XD, yD, g):
    """Linear erasers on site representations (float64). Returns (a, b) per method for x -> x - ((x - mu).a) b.
    meandiff: orthogonal projection off the class-mean difference. leace: LEACE for a binary concept with shrunk covariance
    Sig = (1-s) Xc'Xc/n + s (tr/d) I (exact: Cov(r(x), z) = 0 on the fitting data). randdir: covariance-matched random direction."""
    import torch
    mu = XD.mean(0); Xc = XD - mu; dm = XD[yD.bool()].mean(0) - XD[~yD.bool()].mean(0); u_md = dm / dm.norm()
    _, sv, Vt = torch.linalg.svd(Xc, full_matrices=False); lam_s = sv ** 2 / len(XD); cI = SHRINK * lam_s.sum() / Xc.shape[1]
    lam = (1 - SHRINK) * lam_s + cI
    def spow(x, p): return cI ** p * x + Vt.T @ ((lam ** p - cI ** p) * (Vt @ x))
    sxz = Xc.T @ (yD - yD.mean()) / len(XD); v = spow(sxz, -0.5); u = v / v.norm()
    a_le, b_le = spow(u, -0.5), spow(u, 0.5)
    gr = torch.randn(len(XD), generator=g, dtype=torch.float64); u_rd = Xc.T @ gr; u_rd = u_rd / u_rd.norm()
    return dict(mu=mu, meandiff=(u_md, u_md), leace=(a_le, b_le), randdir=(u_rd, u_rd),
                cos_meandiff_leace=float((u_md @ b_le) / b_le.norm()), rank=int(len(sv)))


def run_explore(model, tok, device, recs, sae_loader, log=print, time_left=lambda: 1e9, layers=LAYERS, ks=KS, boot=BOOT, do_D=True):
    import torch
    torch.manual_seed(SEED)
    F = torch.nn.functional
    t0 = time.time(); R = dict(label=LABEL, settings=dict(layers=layers, ks=ks, READ_OFFSET=READ_OFFSET, POOL_G5R=POOL_G5R, POOL_EXT=POOL_EXT,
                     SHRINK=SHRINK, RAND_FREQ_MIN=RAND_FREQ_MIN, BOOT=boot, R1_thresholds_descriptive=T), stages={}, errors={})
    def stage(n): R["stages"][n] = round(time.time() - t0, 1); log(f"[{R['stages'][n]:7.1f}s] {n}")
    nL = model.config.num_hidden_layers; Vv = len(tok); L_mod = model.model.layers
    read_layers = sorted({min(L + READ_OFFSET, nL - 1) for L in layers})
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0

    def enc(text):
        e = tok(text, return_offsets_mapping=True, add_special_tokens=True); return e["input_ids"], e["offset_mapping"]
    def toks_in(offs, a, b): return [i for i, (s, e) in enumerate(offs) if e > s and s < b and e > a]
    def word_pos(text, offs, w):
        out = []
        for m in re.finditer(r"\b%s\b" % re.escape(w), text): out += toks_in(offs, *m.span())
        return sorted(set(out))
    def out_t(o): return o[0] if isinstance(o, tuple) else o

    def run_batch(ids_list, layer_edits=(), capture=None, logits_at=None):
        n = len(ids_list); m = max(map(len, ids_list))
        ids = torch.full((n, m), pad, dtype=torch.long, device=device); am = torch.zeros((n, m), dtype=torch.long, device=device)
        for i, s in enumerate(ids_list): ids[i, :len(s)] = torch.tensor(s, device=device); am[i, :len(s)] = 1
        store, hs = {}, []
        for L, f in layer_edits: hs.append(L_mod[L].register_forward_hook(f))
        if capture:
            for L, (rows, pos) in capture.items():
                def mk(L, rows, pos):
                    def f(mod, i, o): store[L] = out_t(o)[rows, pos, :].detach().float().clone()
                    return f
                hs.append(L_mod[L].register_forward_hook(mk(L, rows, pos)))
        try:
            with torch.no_grad(): o = model(ids, attention_mask=am, use_cache=False)
        finally:
            for h in hs: h.remove()
        if logits_at is not None: store["logits"] = o.logits[logits_at[0], logits_at[1], :Vv].double().clone()
        return store

    def edit_hook(rows, pos, fn):                       # fn: [n, d] -> [n, d], applied to the residual stream (layer-L output)
        def f(mod, i, o):
            h = out_t(o); h[rows, pos, :] = fn(h[rows, pos, :]); return o
        return f

    # ---------------- AUC + probes (identical to g5r_probe)
    def auc_cols(P, N):
        out = torch.zeros(P.shape[1], device=P.device)
        for s in range(0, P.shape[1], 2048):
            a, b = P[:, None, s:s + 2048], N[None, :, s:s + 2048]
            out[s:s + 2048] = (a > b).float().mean((0, 1)) + 0.5 * (a == b).float().mean((0, 1))
        return out
    def auc1(p, n):
        if len(p) == 0 or len(n) == 0: return float("nan")
        return float(auc_cols(p[:, None].float(), n[:, None].float())[0])
    def fit_probe(X, y, lam=1e-2):
        X = X.double().cpu(); y = y.double().cpu(); mu, sd = X.mean(0), X.std(0) + 1e-6; Z = (X - mu) / sd
        w = torch.zeros(Z.shape[1] + 1, dtype=torch.float64, requires_grad=True)
        opt = torch.optim.LBFGS([w], max_iter=200, line_search_fn="strong_wolfe")
        def closure():
            opt.zero_grad(); z = Z @ w[:-1] + w[-1]
            loss = F.binary_cross_entropy_with_logits(z, y) + lam * (w[:-1] ** 2).sum() / len(y); loss.backward(); return loss
        opt.step(closure); w = w.detach()
        return lambda Xn: ((Xn.double().cpu() - mu) / sd) @ w[:-1] + w[-1]
    def probe_auc(Xtr, ytr, Xte, yte, ret_scores=False):
        if ytr.sum() == 0 or ytr.sum() == len(ytr) or yte.sum() == 0 or yte.sum() == len(yte): return (float("nan"), None) if ret_scores else float("nan")
        s = fit_probe(Xtr, ytr)(Xte); a = auc1(s[yte.bool()], s[~yte.bool()])
        return (a, s) if ret_scores else a
    def boot_aucs(scores, yb, g):                       # stratified item bootstrap; scores: list of [n] tensors -> [len, B] AUCs
        P_ = torch.nonzero(yb).flatten(); N_ = torch.nonzero(~yb).flatten()
        ip = P_[torch.randint(len(P_), (boot, len(P_)), generator=g)]; iN = N_[torch.randint(len(N_), (boot, len(N_)), generator=g)]
        out = []
        for s in scores:
            s = s.double(); a, b = s[ip][:, :, None], s[iN][:, None, :]
            out.append((a > b).double().mean((1, 2)) + 0.5 * (a == b).double().mean((1, 2)))
        return out
    def q(x): x = x[~torch.isnan(x)]; return [float(torch.quantile(x, .025)), float(torch.quantile(x, .975))] if len(x) else [float("nan")] * 2
    def logodds(lp, t): l = lp[t]; return float(l - torch.log1p(-torch.exp(l)))

    # ---------------- recognition records: site = last token of ref_span (G5r); "all" = every token of ref_span
    rec_r = [r for r in recs if r["type"] == "recognition"]
    for r in rec_r:
        ids, offs = enc(r["text"]); r["_ids"] = ids; r["_all"] = toks_in(offs, *r["ref_span"])
        r["_pos"] = max(r["_all"]) if r["_all"] else None; r["_last"] = len(ids) - 1
        if r["_pos"] is None: R["errors"].setdefault("bad_span", []).append(r["id"])
    rec_r = [r for r in rec_r if r["_pos"] is not None]
    kinds = sorted({r["target_kind"] for r in rec_r})
    for i in range(0, len(rec_r), BATCH):
        b = rec_r[i:i + BATCH]; rows = list(range(len(b)))
        st = run_batch([r["_ids"] for r in b], capture={L: (rows, [r["_pos"] for r in b]) for L in layers}, logits_at=(rows, [r["_last"] for r in b]))
        st2 = run_batch([r["_ids"] for r in b], capture={L: (rows, [r["_last"] for r in b]) for L in read_layers})
        for j, r in enumerate(b):
            r["_m"] = {L: st[L][j].cpu() for L in layers}; r["_f"] = {L: st2[L][j].cpu() for L in read_layers}
            r["_lp"] = torch.log_softmax(st["logits"][j], -1).cpu()
    R["span_tokens"] = dict(mean_all=sum(len(r["_all"]) for r in rec_r) / len(rec_r),
                            description_mean_all=(lambda d: sum(len(r["_all"]) for r in d) / max(1, len(d)))([r for r in rec_r if r["ref_type"] == "description"]))
    stage("recognition_capture")

    # ---------------- causal records (discovery only) — frozen §2 eligibility codes, intact log-odds
    caus = [c for c in recs if c["type"] == "causal" and c["split"] == "discovery"] if do_D else []
    for c in caus:
        ids, offs = enc(c["prompt"]); ids2, offs2 = enc(c["prompt_c2"]); c["_ids"], c["_ids2"] = ids, ids2
        c["P"], c["P2"], c["Pc"] = word_pos(c["prompt"], offs, c["patient"]), word_pos(c["prompt_c2"], offs2, c["patient"]), word_pos(c["prompt_c2"], offs2, c["bystander"])
        a, _ = enc(c["prompt"] + " " + c["target"]); a2, _ = enc(c["prompt_c2"] + " " + c["target"])
        c["tid"] = a[-1] if len(a) == len(ids) + 1 and a[:len(ids)] == ids else None
        c["tid2"] = a2[-1] if len(a2) == len(ids2) + 1 and a2[:len(ids2)] == ids2 else None
        c["codes"] = []
        if c["tid"] is None or c["tid2"] is None: c["codes"].append("TOKENIZATION")
        if not c["P"] or not c["P2"] or not c["Pc"]: c["codes"].append("C2_SPAN")
        if "TOKENIZATION" not in c["codes"]:
            st = run_batch([ids, ids2], logits_at=([0, 1], [len(ids) - 1, len(ids2) - 1]))
            lp0, lp1 = torch.log_softmax(st["logits"][0], -1), torch.log_softmax(st["logits"][1], -1)
            c["L"], c["L2"] = logodds(lp0, c["tid"]), logodds(lp1, c["tid2"])
            c["p"], c["p2"] = float(lp0[c["tid"]].exp()), float(lp1[c["tid2"]].exp())
            c["rank"], c["rank2"] = int((lp0 > lp0[c["tid"]]).sum()) + 1, int((lp1 > lp1[c["tid2"]]).sum()) + 1
            if not (c["p"] >= P_MIN and c["rank"] <= RANK_MAX and c["p2"] >= P_MIN and c["rank2"] <= RANK_MAX): c["codes"].append("BASELINE_INELIGIBLE")
            if abs(c["L"] - c["L2"]) > BYSTANDER_SHIFT_MAX: c["codes"].append("BYSTANDER_DESTABILIZES")
        if len(c["P"]) != 1 or len(c["P2"]) != 1 or len(c["Pc"]) != 1: c["codes"].append("NAMED_ONCE")
    stage("causal_capture")

    def sel(**kw): return [r for r in rec_r if all(r.get(k) == v for k, v in kw.items())]

    out_layers = {}
    for L in layers:
        if time_left() < 240: R["errors"]["layers"] = f"stopped before layer {L}: time budget"; break
        Wnp = sae_loader(L); W = tuple(torch.tensor(Wnp[k], device=device, dtype=torch.float32) for k in ("W_enc", "b_enc", "threshold", "W_dec"))
        We, be, th, Wd = W
        def acts(X): pre = X.to(device) @ We + be; return pre * (pre > th)
        Lr = min(L + READ_OFFSET, nL - 1); LO = dict(read_layer=Lr, kinds={}); fitted = {}
        for ki, K in enumerate(kinds):
            D = sel(split="discovery", target_kind=K); V = sel(split="validation", target_kind=K)
            if not D or not V: continue
            g = torch.Generator().manual_seed(SEED + 1000 * L + ki)
            # ---- identity latents: strata + greedy exactly as g5r_probe (worst-stratum AUC), then extended
            S_ = {}
            for r in D:
                if r["is_member"]: continue
                key = "description" if r["ref_type"] == "description" else (r["stratum"] if r["stratum"].startswith("property_matched") else None)
                if key: S_.setdefault(key, []).append(r)
            Aall = {}
            for key, nm in S_.items():
                mm = [r for r in D if r["is_member"] and (r["ref_type"] == "description") == (key == "description")]
                if mm: Aall[key] = (acts(torch.stack([r["_m"][L] for r in mm])), acts(torch.stack([r["_m"][L] for r in nm])))
            if not Aall: continue
            score = torch.stack([auc_cols(a, b) for a, b in Aall.values()]).min(0).values
            cand = torch.topk(score, min(POOL_G5R, score.numel())).indices.tolist(); order = []
            while len(order) < 8:                                         # verbatim G5r selection (k <= 8)
                best = None
                for c_ in cand:
                    if c_ in order: continue
                    v = min(auc1(a[:, order + [c_]].sum(1), b[:, order + [c_]].sum(1)) for a, b in Aall.values())
                    key_ = (v, c_ * -1)
                    if best is None or key_ > best[0]: best = (key_, c_)
                if best is None: break
                order.append(best[1])
            pool = torch.topk(score, min(POOL_EXT, score.numel())).indices
            while len(order) < max(ks):                                   # vectorized continuation, same criterion and tie rule
                cset = torch.tensor([c_ for c_ in pool.tolist() if c_ not in order], device=score.device)
                if len(cset) == 0: break
                vals = torch.stack([auc_cols(a[:, order].sum(1, keepdim=True) + a[:, cset], b[:, order].sum(1, keepdim=True) + b[:, cset]) for a, b in Aall.values()]).min(0).values
                mx = vals.max(); order.append(int(cset[vals == mx].min()))
            # ---- random active latents (control), nested prefix
            A_mem = acts(torch.stack([r["_m"][L] for r in D if r["is_member"]]))
            freq = (A_mem > 0).float().mean(0).cpu(); ok = freq >= RAND_FREQ_MIN; ok[order] = False
            cand_r = torch.nonzero(ok).flatten(); rand = cand_r[torch.randperm(len(cand_r), generator=g)].tolist()[:max(ks)]
            # ---- linear erasers on the full residual at the site (fit on discovery)
            XD = torch.stack([r["_m"][L] for r in D]).double(); yD = torch.tensor([float(r["is_member"]) for r in D], dtype=torch.float64)
            EZ = fit_erasers(XD, yD, g); mu = EZ["mu"]
            def mk_erase(a_vec, b_vec):                     # x -> x - ((x - mu) . a) b
                a_, b_, m_ = a_vec.float().to(device), b_vec.float().to(device), mu.float().to(device)
                return lambda x: x - ((x.float() - m_) @ a_)[:, None] * b_[None, :]
            erasers = {nm_: mk_erase(*EZ[nm_]) for nm_ in ("meandiff", "leace", "randdir")}
            def mk_sae(lat):
                lt = torch.tensor(lat, device=device, dtype=torch.long)
                return lambda x: x - (acts(x.float())[:, lt] @ Wd[lt])
            conds = {}
            for k in ks:
                for pm in ("last", "all"):
                    conds[f"E1_k{k}_{pm}"] = dict(fn=mk_sae(order[:k]), pos=pm, ctrl=f"RAND_k{k}_{pm}", family="sae", k=k)
                    conds[f"RAND_k{k}_{pm}"] = dict(fn=mk_sae(rand[:k]), pos=pm, ctrl=None, family="sae_random", k=min(k, len(rand)))
            for pm in ("last", "all"):
                for nm_ in ("meandiff", "leace"): conds[f"E3_{nm_}_{pm}"] = dict(fn=erasers[nm_], pos=pm, ctrl=f"E3_randdir_{pm}", family="erase", k=None)
                conds[f"E3_randdir_{pm}"] = dict(fn=erasers["randdir"], pos=pm, ctrl=None, family="erase_random", k=None)
            fitted[K] = conds
            # ---- measurement on validation (same as G5r R1 + bootstrap CIs)
            yV = torch.tensor([float(r["is_member"]) for r in V]); yb = yV.bool(); yDf = yD.float()
            XDc = torch.stack([r["_m"][L] for r in D]); XVc = torch.stack([r["_m"][L] for r in V])
            probe_f = fit_probe(torch.stack([r["_f"][Lr] for r in D]), yDf) if 0 < yDf.sum() < len(yDf) else None
            fV_cl = torch.stack([r["_f"][Lr] for r in V]); sc = probe_f(fV_cl) if probe_f else None
            same_clean, s_clean = probe_auc(XDc, yDf, XVc, yV, ret_scores=True)
            roles = sorted({r["role"] for r in D + V}); rl = roles[:2]
            Dm = [r for r in D if r["role"] in rl]; Vm_idx = [i for i, r in enumerate(V) if r["role"] in rl]
            pr = fit_probe(torch.stack([r["_m"][L] for r in Dm]), torch.tensor([float(r["role"] == rl[0]) for r in Dm])) if len(rl) == 2 and Dm else None
            yr = torch.tensor([V[i]["role"] == rl[0] for i in Vm_idx]) if Vm_idx else None
            down_clean = auc1(sc[yb], sc[~yb]) if sc is not None else float("nan")
            res = {}
            for name, cd in conds.items():
                fn = cd["fn"]
                with torch.no_grad():
                    XDa = fn(XDc.to(device)).cpu(); XVa = fn(XVc.to(device)).cpu()
                dn = (XVa - XVc).norm(dim=-1); hn = XVc.norm(dim=-1)
                amnesic, s_am = probe_auc(XDa, yDf, XVa, yV, ret_scores=True)
                fa, kls, npos = [], [], []
                for i in range(0, len(V), BATCH):
                    b = V[i:i + BATCH]
                    if cd["pos"] == "last": rows, pos = list(range(len(b))), [r["_pos"] for r in b]
                    else: rows = [j for j, r in enumerate(b) for _ in r["_all"]]; pos = [p_ for r in b for p_ in r["_all"]]
                    npos += [1 if cd["pos"] == "last" else len(r["_all"]) for r in b]
                    st = run_batch([r["_ids"] for r in b], layer_edits=[(L, edit_hook(rows, pos, fn))], capture={Lr: (list(range(len(b))), [r["_last"] for r in b])},
                                   logits_at=(list(range(len(b))), [r["_last"] for r in b]))
                    fa.append(st[Lr].cpu()); lpa = torch.log_softmax(st["logits"], -1).cpu()
                    for j, r in enumerate(b): kls.append(float((r["_lp"].exp() * (r["_lp"] - lpa[j])).sum()))
                fa = torch.cat(fa)
                if probe_f is not None and down_clean > 0.5:
                    sa = probe_f(fa); down_abl = auc1(sa[yb], sa[~yb]); r_down = 1 - (down_abl - 0.5) / (down_clean - 0.5)
                    bc, ba = boot_aucs([sc, sa], yb, g)
                    r_ci = q(torch.where(bc > 0.5, 1 - (ba - 0.5) / (bc - 0.5).clamp_min(1e-9), torch.full_like(bc, float("nan"))))
                else: down_abl = r_down = float("nan"); r_ci = [float("nan")] * 2
                am_ci = q(boot_aucs([s_am], yb, g)[0]) if s_am is not None else [float("nan")] * 2
                role_drop = float("nan")
                if pr is not None and yr is not None and 0 < yr.sum() < len(yr):
                    s1, s2 = pr(XVc[Vm_idx]), pr(XVa[Vm_idx]); role_drop = auc1(s1[yr], s1[~yr]) - auc1(s2[yr], s2[~yr])
                m = dict(family=cd["family"], k=cd["k"], pos=cd["pos"], control=cd["ctrl"], down_clean_auc=down_clean, down_ablated_auc=down_abl,
                         r_down=r_down, r_down_ci95=r_ci, same_site_clean_auc=same_clean, amnesic_auc=amnesic, amnesic_ci95=am_ci,
                         role_auc_drop=role_drop, kl_mean=sum(kls) / len(kls), rel_norm_mean=float((dn / hn).mean()), d_norm_mean=float(dn.mean()),
                         n_positions_mean=sum(npos) / len(npos), n_val=len(V))
                m["R1_criteria_met"] = bool(r_down == r_down and r_down >= T["r_down"] and amnesic == amnesic and amnesic <= T["amnesic_max"]
                                            and (role_drop != role_drop or role_drop <= T["role_drop_max"]))
                res[name] = m
            LO["kinds"][K] = dict(identity_order=order, g5r_prefix=order[:8], random_order=rand, random_pool=int(len(cand_r)),
                                  erasers=dict(cos_meandiff_leace=EZ["cos_meandiff_leace"], shrink=SHRINK, rank=EZ["rank"]), conditions=res)
        # ---------------- optional exploratory D comparison (discovery causal items, conditions meeting R1 criteria)
        Dres = {}
        for K in sorted({c.get("kind", c["patient"]) for c in caus}):
            if K not in LO["kinds"]: continue
            for name, m in LO["kinds"][K]["conditions"].items():
                if not m["R1_criteria_met"] or m["control"] is None: continue
                items = [c for c in caus if c.get("kind", c["patient"]) == K]
                rows_ = []
                for c in items:
                    if "TOKENIZATION" in c["codes"] or "C2_SPAN" in c["codes"]: continue
                    by = c["bystander"]
                    if by not in fitted or name not in fitted[by]: continue
                    pm = m["pos"]; P = c["P"] if pm == "all" else [max(c["P"])]; Pc = c["Pc"] if pm == "all" else [max(c["Pc"])]
                    def L_after(ids, posl, fn, t):
                        st = run_batch([ids], layer_edits=[(L, edit_hook([0] * len(posl), posl, fn))], logits_at=([0], [len(ids) - 1]))
                        return logodds(torch.log_softmax(st["logits"][0], -1), t)
                    dS = c["L"] - L_after(c["_ids"], P, fitted[K][name]["fn"], c["tid"])
                    dC1 = c["L"] - L_after(c["_ids"], P, fitted[K][m["control"]]["fn"], c["tid"])
                    dC2 = c["L2"] - L_after(c["_ids2"], Pc, fitted[by][name]["fn"], c["tid2"])
                    rows_.append(dict(id=c["id"], event=c.get("template_id"), codes=c["codes"], D_S=dS, D_C1=dC1, D_C2=dC2, d1=dS - dC1, d2=dS - dC2))
                el = [r_ for r_ in rows_ if not r_["codes"]]
                def ev_summary(rs):
                    ev = {}
                    for r_ in rs: ev.setdefault(r_["event"], []).append(r_)
                    if not ev: return dict(n_items=0, n_events=0)
                    e1 = torch.tensor([sum(x["d1"] for x in v) / len(v) for v in ev.values()], dtype=torch.float64)
                    e2 = torch.tensor([sum(x["d2"] for x in v) / len(v) for v in ev.values()], dtype=torch.float64)
                    gg = torch.Generator().manual_seed(SEED); ix = torch.randint(len(e1), (boot, len(e1)), generator=gg)
                    return dict(n_items=len(rs), n_events=len(ev), delta1_mean=float(e1.mean()), delta1_ci95=q(e1[ix].mean(1)),
                                delta2_mean=float(e2.mean()), delta2_ci95=q(e2[ix].mean(1)))
                Dres.setdefault(K, {})[name] = dict(eligible=ev_summary(el), all_items_incl_ineligible=ev_summary(rows_), rows=rows_)
        LO["D_exploratory"] = Dres
        out_layers[L] = LO
        del W, We, be, th, Wd
        stage(f"layer_{L}")
    R["layers"] = out_layers
    R["causal_eligibility"] = {c["id"]: c["codes"] for c in caus}
    stage("done")
    return R


try:
    import modal
    app = modal.App("alignedlab-exp3-exploratory-e1-e3")
    image = (modal.Image.debian_slim(python_version="3.12").pip_install(*G4.PINS)
             .env({"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}).add_local_python_source("g4_probe"))
    vol = modal.Volume.from_name("kant-hf-cache", create_if_missing=True)

    @app.function(image=image, gpu=GPU, timeout=TIMEOUT_S, volumes={"/root/.cache/huggingface": vol},
                  secrets=[modal.Secret.from_name("huggingface-secret")])
    def explore(data: bytes, layers=None):
        import torch, numpy as np, tempfile
        t0 = time.time(); logs = []
        def log(s): print(s, flush=True); logs.append(s)
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True; torch.use_deterministic_algorithms(True, warn_only=True)
        p = tempfile.mktemp(suffix=".jsonl"); open(p, "wb").write(data); recs = load(p)
        from huggingface_hub import snapshot_download, hf_hub_download, list_repo_files
        from transformers import AutoTokenizer, AutoModelForCausalLM
        path = snapshot_download(MODEL_ID, revision=REV); tok = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32, attn_implementation="eager").to("cuda").eval()
        files = list_repo_files(SAE_REPO); chosen = {}
        def sae_loader(L):
            fs = [f for f in files if f.startswith(f"layer_{L}/width_16k/") and f.endswith("params.npz")]
            chosen[L] = min(fs, key=lambda s: (abs(int(re.search(r"average_l0_(\d+)", s).group(1)) - 70), s))
            return dict(np.load(hf_hub_download(SAE_REPO, chosen[L])))
        R = run_explore(model, tok, "cuda", recs, sae_loader, log, time_left=lambda: TIMEOUT_S - 90 - (time.time() - t0), layers=layers or LAYERS)
        rt = time.time() - t0
        return json.dumps(dict(status=LABEL, identity=dict(model=MODEL_ID, revision=REV, commit_hash=getattr(model.config, "_commit_hash", None), sae_files=chosen),
                               data_sha256=hashlib.sha256(data).hexdigest(), env=dict(device=torch.cuda.get_device_name(0)),
                               results=R, runtime_s=round(rt, 1), gpu_cost_usd_estimate=round(rt / 3600 * GPU_USD_PER_H, 3), logs=logs),
                          default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)).encode()

    @app.local_entrypoint()
    def main(data: str, layers: str = ""):
        Ls = [int(x) for x in layers.split(",") if x] or None
        print(f"{LABEL}\nlayers {Ls or LAYERS}; worst-case cost of this run ${TIMEOUT_S / 3600 * GPU_USD_PER_H:.2f} (hard timeout {TIMEOUT_S // 60} min); PI cap ${SPEND_CAP_USD}")
        b = explore.remote(open(data, "rb").read(), Ls); os.makedirs("modal_out/e", exist_ok=True)
        name = "e_results.json" if not Ls else "e_results_layers_%s.json" % "-".join(map(str, Ls))
        open("modal_out/e/" + name, "wb").write(b); d = json.loads(b); print("wrote modal_out/e/" + name)
        print(f"DONE runtime {d['runtime_s']}s est ${d['gpu_cost_usd_estimate']} errors {d['results'].get('errors')}")
except ImportError:
    pass
