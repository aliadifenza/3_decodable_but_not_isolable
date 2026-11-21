"""Kantian Positional Logic Test — G5r VERIFICATION under the FROZEN G5 spec (protocol/G5_SPEC_FREEZE.md, PI-approved 2026-11-21).
Discovery/validation only; run authorized by PI 2026-11-21 (hard cap: 20-min timeout, ~$0.37).

Repairs verified (no causal-target log-odds is used for ANY selection):
  R1 intervention efficacy   : nested SAE sets k in {1,2,4,8} chosen on DISCOVERY identity criteria; on VALIDATION measure
                               identity-information removal (same-site amnesic probe + downstream probe), unrelated-information
                               preservation (role probe), magnitude (||d||, ||d||/||h||), next-token KL (reported only).
  R2 non-lexical identity    : worst-stratum selection over property-matched non-members, noun-free descriptions and roles;
                               out-of-template validation; leave-one-role-out reselection; lexical-shortcut latent flag.
  R3 controls                : C1 (active, identity-unselective latents at the same site) and C2 (bystander's own greedy set),
                               matched per Decision 2 EXCEPT active-unit ratio = measured only; valid-pair retention; collateral.
  L  layers 2..14            : identity validity (R2), efficacy (R1), retention (R3) and P3 feasibility (attention + information
                               use under S vs C1 on P3-discovery prompts; no target log-odds).

Input: Adaption data converted to JSONL (schema in SCHEMA below). Records with split "evaluation" are REFUSED.
Run (only after PI authorization):  MODAL_IMAGE_BUILDER_VERSION=2025.06 python3 -m modal run g5r_probe.py --data g5r_input.jsonl"""
import json, os, re, time, math, hashlib
import g4_probe as G4

MODEL_ID, REV, SAE_REPO = G4.MODEL_ID, G4.REV, "google/gemma-scope-2b-pt-res"
GPU, GPU_USD_PER_H, TIMEOUT_S, SEED = "A10G", 1.10, 20 * 60, 20261121
LAYERS = list(range(2, 15)); KS = [1, 2, 4, 8]; READ_OFFSET = 4; BATCH = 32
A_MAX, RHO_MIN, TOL = 2.0, 0.5, 0.10                         # Decision 2 (active-unit ratio: measured only, R3)
P_MIN, RANK_MAX, BYSTANDER_SHIFT_MAX = 0.10, 5, 1.0           # Decision 4 eligibility
# thresholds FROZEN by G5_SPEC_FREEZE.md (§4 R2, §7 R1, §8 R3, §6 P3) — identical values to the G5r proposal
T = dict(r_down=0.67, amnesic_max=0.70, role_drop_max=0.05,                       # R1
         recog_noun=0.85, worst_property=0.75, description=0.75, loro=0.75, lexical_gap_max=0.20, min_per_stratum=10,  # R2
         retention_min=0.75, c1_identity_removal_max=0.10, c2_r_min=0.67, collateral_role_drop_max=0.05,             # R3
         p3_attn_min=0.10, p3_info_ratio_min=1.5, p3_gap=2)                                                           # P3
SCHEMA = dict(
    recognition=dict(id=str, bundle_id=str, template_id=str, split="discovery|validation", type="recognition",
                     target_kind="vase|ice|rope|<bystander kind>", is_member=bool,
                     stratum="member:identity_variant | property_matched:<material|fragility|shape|function|...> | unmatched",
                     ref_type="noun|description|pronoun", role="subject|object|oblique|possessor", text=str,
                     ref_span="[start,end) char span of the reference (head noun / description) whose LAST token is measured"),
    causal=dict(id=str, bundle_id=str, split="discovery|p3_discovery", type="causal", family="impact|heating|cutting",
                patient=str, bystander=str, target=str, prompt="Decision-2 format (patient named once, '..., causing it to')",
                prompt_c2="bystander-inserted variant, bystander named once"))


def load(path):
    recs = [json.loads(l) for l in open(path) if l.strip()]
    if any(r.get("split") == "evaluation" for r in recs): raise SystemExit("REFUSED: evaluation-split records present")
    probe = set(G4.probe_only_manifest()["items"])
    bad = [r["id"] for r in recs if r.get("text") in probe or r.get("prompt") in probe]
    if bad: raise SystemExit(f"REFUSED: G4 probe-only items present: {bad[:5]}")
    return recs


def run_repair(model, tok, device, recs, sae_loader, log=print, time_left=lambda: 1e9, layers=LAYERS, ks=KS):
    import torch
    torch.manual_seed(SEED)
    F = torch.nn.functional
    t0 = time.time(); R = dict(settings=dict(layers=layers, ks=ks, READ_OFFSET=READ_OFFSET, A_MAX=A_MAX, RHO_MIN=RHO_MIN, TOL=TOL,
                                             thresholds_FROZEN=T), stages={}, errors={})
    def stage(n): R["stages"][n] = round(time.time() - t0, 1); log(f"[{R['stages'][n]:7.1f}s] {n}")
    nL, nH = model.config.num_hidden_layers, model.config.num_attention_heads
    nKV = getattr(model.config, "num_key_value_heads", nH); hd = getattr(model.config, "head_dim", model.config.hidden_size // nH)
    Vv = len(tok); L_mod = model.model.layers
    read_layers = sorted({min(L + READ_OFFSET, nL - 1) for L in layers})
    cap_layers = sorted(set(layers) | set(read_layers))
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0

    # ---------------- tokenization helpers (right padding; positions unaffected)
    def enc(text):
        e = tok(text, return_offsets_mapping=True, add_special_tokens=True); return e["input_ids"], e["offset_mapping"]
    def last_tok_in(offs, a, b):
        c = [i for i, (s, e) in enumerate(offs) if e > s and s < b and e > a]; return max(c) if c else None
    def word_pos(text, offs, w):
        out = []
        for m in re.finditer(r"\b%s\b" % re.escape(w), text):
            a, b = m.span(); out += [i for i, (s, e) in enumerate(offs) if e > s and s < b and e > a]
        return sorted(set(out))
    def out_t(o): return o[0] if isinstance(o, tuple) else o

    def run_batch(ids_list, layer_edits=(), capture=None, logits_at=None, oproj=None, attn=False):
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
        if oproj:
            rows, pos, Ts = oproj
            for Tl in Ts:
                def mk2(Tl):
                    def f(mod, args, kwargs):
                        x = args[0] if args else kwargs["input"]; store[("o", Tl)] = x[rows, pos, :].detach().float().clone()
                    return f
                hs.append(L_mod[Tl].self_attn.o_proj.register_forward_pre_hook(mk2(Tl), with_kwargs=True))
        try:
            with torch.no_grad(): o = model(ids, attention_mask=am, output_attentions=attn, use_cache=False)
        finally:
            for h in hs: h.remove()
        if logits_at is not None: store["logits"] = o.logits[logits_at[0], logits_at[1], :Vv].double().clone()
        if attn: store["attn"] = o.attentions
        return store

    def sae_edit(L, rows, pos, lat, alpha, W, rec=None):
        We, be, th, Wd = W
        def f(mod, i, o):
            h = out_t(o); x = h[rows, pos, :]; pre = x @ We[:, lat] + be[lat]; a = pre * (pre > th[lat])
            d = -(a @ Wd[lat]) * alpha.view(-1, 1); h[rows, pos, :] = x + d
            if rec is not None: rec["d"] = d.norm(dim=-1).cpu(); rec["h"] = x.norm(dim=-1).cpu(); rec["active"] = (a > 0).sum(-1).cpu()
            return o
        return f

    # ---------------- AUC + logistic probe (deterministic, CPU double)
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
    def probe_auc(Xtr, ytr, Xte, yte):
        if ytr.sum() == 0 or ytr.sum() == len(ytr) or yte.sum() == 0 or yte.sum() == len(yte): return float("nan")
        s = fit_probe(Xtr, ytr)(Xte); return auc1(s[yte.bool()], s[~yte.bool()])

    # ---------------- prepare recognition records
    rec_r = [r for r in recs if r["type"] == "recognition"]
    for r in rec_r:
        ids, offs = enc(r["text"]); r["_ids"] = ids; r["_pos"] = last_tok_in(offs, *r["ref_span"]); r["_last"] = len(ids) - 1
        if r["_pos"] is None: R["errors"].setdefault("bad_span", []).append(r["id"])
    rec_r = [r for r in rec_r if r["_pos"] is not None]
    kinds = sorted({r["target_kind"] for r in rec_r})
    # capture clean reps: measure pos (all cap_layers) and final pos (read layers), plus clean final logits
    for i in range(0, len(rec_r), BATCH):
        b = rec_r[i:i + BATCH]; rows = list(range(len(b)))
        st = run_batch([r["_ids"] for r in b], capture={**{L: (rows, [r["_pos"] for r in b]) for L in cap_layers}},
                       logits_at=(rows, [r["_last"] for r in b]))
        st2 = run_batch([r["_ids"] for r in b], capture={L: (rows, [r["_last"] for r in b]) for L in read_layers})
        for j, r in enumerate(b):
            r["_m"] = {L: st[L][j].cpu() for L in cap_layers}; r["_f"] = {L: st2[L][j].cpu() for L in read_layers}
            r["_lp"] = torch.log_softmax(st["logits"][j], -1).cpu()
    stage("recognition_capture")

    # ---------------- causal records (discovery + p3_discovery only)
    caus = [r for r in recs if r["type"] == "causal"]
    for c in caus:
        ids, offs = enc(c["prompt"]); ids2, offs2 = enc(c["prompt_c2"])
        c["_ids"], c["_ids2"] = ids, ids2
        c["P"], c["P2"], c["Pc"] = word_pos(c["prompt"], offs, c["patient"]), word_pos(c["prompt_c2"], offs2, c["patient"]), word_pos(c["prompt_c2"], offs2, c["bystander"])
        a, _ = enc(c["prompt"] + " " + c["target"]); a2, _ = enc(c["prompt_c2"] + " " + c["target"])
        c["tid"] = a[-1] if len(a) == len(ids) + 1 and a[:len(ids)] == ids else None
        c["tid2"] = a2[-1] if len(a2) == len(ids2) + 1 and a2[:len(ids2)] == ids2 else None
        c["codes"] = []
        if c["tid"] is None or c["tid2"] is None: c["codes"].append("TOKENIZATION")
        if len(c["P"]) != 1 or len(c["P2"]) != 1 or len(c["Pc"]) != 1: c["codes"].append("C2_SPAN")
        if "TOKENIZATION" not in c["codes"]:
            st = run_batch([ids, ids2], logits_at=([0, 1], [len(ids) - 1, len(ids2) - 1]))
            vals = []
            for j, t in ((0, c["tid"]), (1, c["tid2"])):
                lp = torch.log_softmax(st["logits"][j], -1); l = lp[t]
                vals.append((float(l - torch.log1p(-torch.exp(l))), float(torch.exp(l)), int((lp > l).sum()) + 1))
            (c["L"], c["p"], c["rank"]), (c["L2"], c["p2"], c["rank2"]) = vals
            if not (c["p"] >= P_MIN and c["rank"] <= RANK_MAX and c["p2"] >= P_MIN and c["rank2"] <= RANK_MAX): c["codes"].append("BASELINE_INELIGIBLE")
            if abs(c["L"] - c["L2"]) > BYSTANDER_SHIFT_MAX: c["codes"].append("BYSTANDER_DESTABILIZES")
        if "C2_SPAN" not in c["codes"] and "TOKENIZATION" not in c["codes"]:
            st = run_batch([ids], capture={L: ([0], c["P"]) for L in layers}); c["_hP"] = {L: st[L].cpu() for L in layers}
            st = run_batch([ids2], capture={L: ([0, 0], c["P2"] + c["Pc"]) for L in layers}); c["_hP2"] = {L: st[L].cpu() for L in layers}
    stage("causal_capture")

    def sel(**kw):
        return [r for r in rec_r if all((r.get(k) == v) if not callable(v) else v(r.get(k)) for k, v in kw.items())]

    out_layers = {}
    for L in layers:
        if time_left() < 180: R["errors"]["layers"] = f"stopped before layer {L}: time budget"; break
        Wnp = sae_loader(L); W = tuple(torch.tensor(Wnp[k], device=device, dtype=torch.float32) for k in ("W_enc", "b_enc", "threshold", "W_dec"))
        We, be, th, Wd = W
        def acts(X): pre = X.to(device) @ We + be; return pre * (pre > th)
        Lr = min(L + READ_OFFSET, nL - 1); LO = dict(read_layer=Lr, kinds={})
        for K in kinds:
            D = sel(split="discovery", target_kind=K); V = sel(split="validation", target_kind=K)
            if not D or not V: continue
            def strata(rs, role_out=None):
                S_ = {}
                mem = [r for r in rs if r["is_member"] and r["role"] != role_out]
                for r in rs:
                    if r["is_member"] or r["role"] == role_out: continue
                    if r["ref_type"] == "description": key = "description"
                    elif r["stratum"].startswith("property_matched"): key = r["stratum"]
                    else: continue
                    S_.setdefault(key, []).append(r)
                out = {}
                for key, nm in S_.items():
                    mm = [r for r in mem if (r["ref_type"] == "description") == (key == "description")]
                    if mm: out[key] = (mm, nm)
                return out
            def greedy(rs, role_out=None):
                st = strata(rs, role_out)
                if not st: return [], {}
                Aall = {key: (acts(torch.stack([r["_m"][L] for r in mm])), acts(torch.stack([r["_m"][L] for r in nm]))) for key, (mm, nm) in st.items()}
                score = torch.stack([auc_cols(a, b) for a, b in Aall.values()]).min(0).values
                cand = torch.topk(score, min(64, score.numel())).indices.tolist(); order = []
                while len(order) < max(ks):
                    best = None
                    for c_ in cand:
                        if c_ in order: continue
                        v = min(auc1(a[:, order + [c_]].sum(1), b[:, order + [c_]].sum(1)) for a, b in Aall.values())
                        key_ = (v, c_ * -1)
                        if best is None or key_ > best[0]: best = (key_, c_)
                    if best is None: break
                    order.append(best[1])
                lexical = {}
                if "description" in Aall:
                    mem_n = [r for r in rs if r["is_member"] and r["ref_type"] == "noun"]
                    non_n = [r for r in rs if not r["is_member"] and r["ref_type"] == "noun"]
                    if mem_n and non_n:
                        an = auc_cols(acts(torch.stack([r["_m"][L] for r in mem_n])), acts(torch.stack([r["_m"][L] for r in non_n])))
                        ad = auc_cols(*Aall["description"])
                        lexical = {c_: dict(noun=float(an[c_]), desc=float(ad[c_]), flag=bool(an[c_] >= 0.95 and ad[c_] <= 0.60)) for c_ in order}
                return order, lexical
            order, lexical = greedy(D)
            if not order: continue
            roles = sorted({r["role"] for r in D + V})
            loro_orders = {ro: greedy(D, role_out=ro)[0] for ro in roles}
            def ident_metrics(lat, rs, lat_by_role=None):
                st = strata(rs); res = {}
                for key, (mm, nm) in st.items():
                    a = acts(torch.stack([r["_m"][L] for r in mm]))[:, lat].sum(1); b = acts(torch.stack([r["_m"][L] for r in nm]))[:, lat].sum(1)
                    res[key] = dict(auc=auc1(a, b), n_mem=len(mm), n_non=len(nm))
                mn = [r for r in rs if r["is_member"] and r["ref_type"] == "noun"]; nn = [r for r in rs if not r["is_member"] and r["ref_type"] == "noun"]
                if mn and nn:
                    res["noun_all"] = dict(auc=auc1(acts(torch.stack([r["_m"][L] for r in mn]))[:, lat].sum(1), acts(torch.stack([r["_m"][L] for r in nn]))[:, lat].sum(1)))
                return res
            per_k = {}
            for k in ks:
                lat = order[:k]; iv = ident_metrics(lat, V)
                loro = {}
                for ro in roles:
                    lo_lat = loro_orders[ro][:k]
                    mr = [r for r in V if r["role"] == ro and r["is_member"]]; nr = [r for r in V if r["role"] == ro and not r["is_member"]]
                    if lo_lat and mr and nr:
                        loro[ro] = auc1(acts(torch.stack([r["_m"][L] for r in mr]))[:, lo_lat].sum(1), acts(torch.stack([r["_m"][L] for r in nr]))[:, lo_lat].sum(1))
                prop = [v["auc"] for kk, v in iv.items() if kk.startswith("property_matched")]
                counts_ok = all(v["n_mem"] >= T["min_per_stratum"] and v["n_non"] >= T["min_per_stratum"] for kk, v in iv.items() if kk != "noun_all")
                noun = iv.get("noun_all", {}).get("auc", float("nan")); desc = iv.get("description", {}).get("auc", float("nan"))
                r2 = dict(strata=iv, loro=loro, worst_property=min(prop) if prop else float("nan"), description=desc, noun=noun,
                          lexical_gap=noun - desc, counts_ok=counts_ok)
                crit = [(noun - T["recog_noun"]) / (1 - T["recog_noun"]), (r2["worst_property"] - T["worst_property"]) / (1 - T["worst_property"]),
                        (desc - T["description"]) / (1 - T["description"]), (min(loro.values()) - T["loro"]) / (1 - T["loro"]) if loro else float("nan"),
                        (T["lexical_gap_max"] - r2["lexical_gap"]) / T["lexical_gap_max"]]
                r2["margin"] = min(crit) if not any(math.isnan(x) for x in crit) else float("nan")
                r2["pass"] = bool(counts_ok and r2["margin"] == r2["margin"] and r2["margin"] >= 0)
                # ---- R1: efficacy (validation; ablation at the reference position, layer L)
                yD = torch.tensor([float(r["is_member"]) for r in D]); yV = torch.tensor([float(r["is_member"]) for r in V])
                def ablate_rep(rs):
                    X = torch.stack([r["_m"][L] for r in rs]).to(device); a = acts(X)[:, lat]
                    d = -(a @ Wd[lat]); return (X + d).cpu(), d.norm(dim=-1).cpu(), X.norm(dim=-1).cpu(), (a > 0).sum(-1).cpu()
                XDa, _, _, _ = ablate_rep(D); XVa, dn, hn, act = ablate_rep(V)
                XDc = torch.stack([r["_m"][L] for r in D]); XVc = torch.stack([r["_m"][L] for r in V])
                same_clean = probe_auc(XDc, yD, XVc, yV); amnesic = probe_auc(XDa, yD, XVa, yV)
                probe_f = fit_probe(torch.stack([r["_f"][Lr] for r in D]), yD) if 0 < yD.sum() < len(yD) else None
                fV_abl, kls = [], []
                for i in range(0, len(V), BATCH):
                    b = V[i:i + BATCH]; rows = list(range(len(b)))
                    st = run_batch([r["_ids"] for r in b], layer_edits=[(L, sae_edit(L, rows, [r["_pos"] for r in b], lat, torch.ones(len(b), device=device), W))],
                                   capture={Lr: (rows, [r["_last"] for r in b])}, logits_at=(rows, [r["_last"] for r in b]))
                    fV_abl.append(st[Lr].cpu())
                    lpa = torch.log_softmax(st["logits"], -1).cpu()
                    for j, r in enumerate(b): kls.append(float((r["_lp"].exp() * (r["_lp"] - lpa[j])).sum()))
                fV_abl = torch.cat(fV_abl); fV_cl = torch.stack([r["_f"][Lr] for r in V])
                if probe_f is not None:
                    sc, sa = probe_f(fV_cl), probe_f(fV_abl); yb = yV.bool()
                    down_clean, down_abl = auc1(sc[yb], sc[~yb]), auc1(sa[yb], sa[~yb])
                else: down_clean = down_abl = float("nan")
                r_down = 1 - (down_abl - 0.5) / (down_clean - 0.5) if down_clean == down_clean and down_clean > 0.5 else float("nan")
                # preservation: role probe (first two roles), same site, clean-trained
                role_drop = float("nan"); rl = roles[:2]
                if len(rl) == 2:
                    Dm = [r for r in D if r["role"] in rl]; Vm_idx = [i for i, r in enumerate(V) if r["role"] in rl]
                    if Dm and Vm_idx:
                        pr = fit_probe(torch.stack([r["_m"][L] for r in Dm]), torch.tensor([float(r["role"] == rl[0]) for r in Dm]))
                        yr = torch.tensor([V[i]["role"] == rl[0] for i in Vm_idx])
                        if 0 < yr.sum() < len(yr):
                            s1, s2 = pr(XVc[Vm_idx]), pr(XVa[Vm_idx]); role_drop = auc1(s1[yr], s1[~yr]) - auc1(s2[yr], s2[~yr])
                r1 = dict(same_site_clean_auc=same_clean, amnesic_auc=amnesic, down_clean_auc=down_clean, down_ablated_auc=down_abl,
                          r_down=r_down, role_auc_drop=role_drop, kl_mean=sum(kls) / len(kls), d_norm_mean=float(dn.mean()),
                          rel_norm_mean=float((dn / hn).mean()), active_mean=float(act.float().mean()))
                r1["pass"] = bool(r_down == r_down and r_down >= T["r_down"] and amnesic == amnesic and amnesic <= T["amnesic_max"]
                                  and (role_drop != role_drop or role_drop <= T["role_drop_max"]))
                per_k[k] = dict(latents=lat, R2=r2, R1=r1)
            kL = next((k for k in ks if per_k[k]["R1"]["pass"]), None)
            LO["kinds"][K] = dict(greedy_order=order, lexical_flags=lexical, loro_orders=loro_orders, per_k=per_k, k_selected=kL,
                                  k_rule="smallest k in KS passing R1 (validation identity-removal + role preservation); None = efficacy FAIL")
        # ---------------- R3 controls + P3 feasibility per patient kind, at k_selected (or max(KS) if None, flagged)
        byst = sorted({c["bystander"] for c in caus})
        for K in [k_ for k_ in LO["kinds"] if k_ not in byst]:
            info = LO["kinds"][K]; k = info["k_selected"] or max(ks); lat = info["greedy_order"][:k]
            D = sel(split="discovery", target_kind=K)
            memD = [r for r in D if r["is_member"] and r["ref_type"] == "noun"] or [r for r in D if r["is_member"]]
            nonD = [r for r in D if not r["is_member"]]
            A_m = acts(torch.stack([r["_m"][L] for r in memD])); A_n = acts(torch.stack([r["_m"][L] for r in nonD]))
            au = auc_cols(A_m, A_n); actfrac = (A_m > 0).float().mean(0); contrib = (A_m * Wd.norm(dim=-1)).mean(0)
            ok = (actfrac >= 0.5) & ((au - 0.5).abs() < 0.05); ok[lat] = False
            cS = float(contrib[lat].mean()); c1 = sorted(torch.nonzero(ok).flatten().tolist(), key=lambda j: (abs(float(contrib[j]) - cS), j))[:k]
            c2_sets = {b: LO["kinds"].get(b, {}).get("greedy_order", [])[:k] for b in byst}
            ccs = [c for c in caus if c.get("kind", c["patient"]) == K and c["split"] in ("discovery", "p3_discovery")]
            rows, retained = [], 0; elig = [c for c in ccs if not c["codes"]]
            for c in elig:
                c2 = c2_sets.get(c["bystander"], [])
                def nat(h, lt):
                    if not lt: return 0.0, 0
                    a = acts(h)[:, lt]; return float((a @ Wd[lt]).norm()), int((a > 0).sum())
                nS, aS = nat(c["_hP"][L], lat); nC1, aC1 = nat(c["_hP"][L], c1)
                nS2, aS2 = nat(c["_hP2"][L][:1], lat); nC2, aC2 = nat(c["_hP2"][L][1:], c2)
                N = min(nS, nS2, A_MAX * min(nC1, nC2)) if min(nC1, nC2) > 0 else 0.0
                codes = [] if (N > 0 and N >= RHO_MIN * max(nS, nS2)) else ["MATCH_INFEASIBLE"]
                retained += not codes
                rows.append(dict(id=c["id"], family=c["family"], split=c["split"], n_S=nS, n_C1=nC1, n_S2=nS2, n_C2=nC2, N_star=N,
                                 active_S=aS, active_C1=aC1, active_S2=aS2, active_C2=aC2,
                                 active_ratio_C1=(aC1 / aS) if aS else None, active_ratio_C2=(aC2 / aS2) if aS2 else None, codes=codes))
            # collateral of controls on validation recognition data (same pipeline as R1, but C1 latents)
            V = sel(split="validation", target_kind=K); coll = {}
            if c1 and V:
                yD = torch.tensor([float(r["is_member"]) for r in D]); yV = torch.tensor([float(r["is_member"]) for r in V])
                pf = fit_probe(torch.stack([r["_f"][min(L + READ_OFFSET, nL - 1)] for r in D]), yD) if 0 < yD.sum() < len(yD) else None
                if pf is not None:
                    fa = []
                    for i in range(0, len(V), BATCH):
                        b = V[i:i + BATCH]; rr = list(range(len(b)))
                        st = run_batch([r["_ids"] for r in b], layer_edits=[(L, sae_edit(L, rr, [r["_pos"] for r in b], c1, torch.ones(len(b), device=device), W))],
                                       capture={min(L + READ_OFFSET, nL - 1): (rr, [r["_last"] for r in b])})
                        fa.append(st[min(L + READ_OFFSET, nL - 1)].cpu())
                    fa = torch.cat(fa); fc = torch.stack([r["_f"][min(L + READ_OFFSET, nL - 1)] for r in V]); yb = yV.bool()
                    sc, sa = pf(fc), pf(fa); dc, da = auc1(sc[yb], sc[~yb]), auc1(sa[yb], sa[~yb])
                    coll["C1_identity_removal"] = 1 - (da - 0.5) / (dc - 0.5) if dc > 0.5 else float("nan")
            for b in byst:
                kb = LO["kinds"].get(b, {}).get("per_k", {})
                kk = max([x for x in kb if x <= k] or [None]) if kb else None
                if kk: coll[f"C2_{b}_r_down_at_k{kk}"] = kb[kk]["R1"]["r_down"]
            # ---- P3 feasibility on discovery causal prompts (frozen §6; no separate p3_discovery split) — attention + information use S vs C1; NO target log-odds
            p3 = [c for c in elig if c["split"] in ("discovery", "p3_discovery")]
            Ts = [t for t in range(L + T["p3_gap"], nL)]; p3res = None
            if p3 and Ts and c1:
                att = torch.zeros(len(Ts), nKV); ratio_num = torch.zeros(len(Ts), nKV); ratio_den = torch.zeros(len(Ts), nKV); used = 0
                for c in p3:
                    row = next(r for r in rows if r["id"] == c["id"])
                    if row["codes"]: continue
                    last = len(c["_ids"]) - 1; used += 1
                    st0 = run_batch([c["_ids"]], oproj=([0], [last], Ts), attn=True)
                    for ti, Tl in enumerate(Ts):
                        a = st0["attn"][Tl][0, :, last, c["P"]].sum(-1).cpu()          # [nH]
                        att[ti] += a.view(nKV, -1).mean(-1)
                    outs = {}
                    for name, lt, n in (("S", lat, row["n_S"]), ("C1", c1, row["n_C1"])):
                        alpha = torch.tensor([row["N_star"] / n], device=device)
                        outs[name] = run_batch([c["_ids"]], layer_edits=[(L, sae_edit(L, [0], c["P"], lt, alpha, W))], oproj=([0], [last], Ts))
                    for ti, Tl in enumerate(Ts):
                        dS = (outs["S"][("o", Tl)] - st0[("o", Tl)]).view(nKV, -1).norm(dim=-1).cpu()
                        dC = (outs["C1"][("o", Tl)] - st0[("o", Tl)]).view(nKV, -1).norm(dim=-1).cpu()
                        ratio_num[ti] += dS; ratio_den[ti] += dC
                if used:
                    att /= used; ratio = ratio_num / ratio_den.clamp_min(1e-9)
                    feas = [(Ts[ti], g, float(att[ti, g]), float(ratio[ti, g])) for ti in range(len(Ts)) for g in range(nKV)
                            if att[ti, g] >= T["p3_attn_min"] and ratio[ti, g] >= T["p3_info_ratio_min"]]
                    top = sorted([(Ts[ti], g, float(att[ti, g]), float(ratio[ti, g])) for ti in range(len(Ts)) for g in range(nKV)], key=lambda x: -x[2] * min(x[3], 10))[:5]
                    p3res = dict(n_prompts=used, feasible_groups=feas, top_groups=top, feasible=bool(feas))
            info["R3"] = dict(k_used=k, k_fallback=info["k_selected"] is None, c1_latents=c1, c1_pool=int(ok.sum()), c2_sets=c2_sets,
                              eligible_items=len(elig), retained=retained, retention=(retained / len(elig)) if elig else float("nan"),
                              rows=rows, collateral=coll)
            info["R3"]["pass"] = bool(elig and info["R3"]["retention"] >= T["retention_min"]
                                      and coll.get("C1_identity_removal", 1.0) <= T["c1_identity_removal_max"]
                                      and all(v >= T["c2_r_min"] for kk_, v in coll.items() if kk_.startswith("C2_") and v == v))
            info["P3"] = p3res
        out_layers[L] = LO
        del W, We, be, th, Wd
        stage(f"layer_{L}")
    R["layers"] = out_layers
    # ---------------- layer verdicts (no causal target log-odds anywhere)
    patients = sorted({c.get("kind", c["patient"]) for c in caus})
    confirm = sorted({c.get("kind", c["patient"]) for c in caus if c.get("family") == "physical_impact"}) or patients   # frozen §7: impact kinds
    verdict = {}
    for L, LO in out_layers.items():
        v = {}
        for K in patients:
            i = LO["kinds"].get(K)
            if not i: v[K] = dict(ok=False, why="no data"); continue
            k = i["k_selected"]; r2 = i["per_k"][k]["R2"] if k else i["per_k"][max(ks)]["R2"]
            v[K] = dict(R1=k is not None, R2=r2["pass"], R2_margin=r2["margin"], R3=i.get("R3", {}).get("pass", False),
                        P3=(i.get("P3") or {}).get("feasible", False))
            v[K]["ok"] = all(v[K][x] for x in ("R1", "R2", "R3", "P3"))
        verdict[L] = v
    complete = sorted(out_layers) == sorted(LAYERS)   # selection only over the full frozen range 2-14; partial runs are merged by merge_g5r.py
    ok_layers = [L for L, v in verdict.items() if all(v[K]["ok"] for K in confirm)] if complete else []
    ok_all = [L for L, v in verdict.items() if all(x["ok"] for x in v.values())]
    def mm(L): return min(verdict[L][K]["R2_margin"] for K in confirm)
    R["layer_verdicts"] = verdict
    L_star = max(ok_layers, key=lambda x: (mm(x), -x)) if ok_layers else None
    R["layer_selection"] = dict(eligible=ok_layers, L_star=L_star, confirmatory_kinds=confirm, all_layers_completed=complete,
                                eligible_for_all_kinds_reported=ok_all,
                                k_at_L_star=({K: out_layers[L_star]["kinds"][K]["k_selected"] for K in out_layers[L_star]["kinds"]} if L_star is not None else None),
                                rule="frozen §7: among layers 2-14 passing R1,R2,R3,P3 for every IMPACT patient kind: argmax min R2 margin; ties shallower; None = no valid layer (or run incomplete) -> P2/P3 inconclusive")
    stage("verdicts")
    return R


try:
    import modal
    app = modal.App("alignedlab-exp3-g5-repair-verification")
    image = (modal.Image.debian_slim(python_version="3.12").pip_install(*G4.PINS)
             .env({"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}).add_local_python_source("g4_probe"))
    vol = modal.Volume.from_name("kant-hf-cache", create_if_missing=True)

    @app.function(image=image, gpu=GPU, timeout=TIMEOUT_S, volumes={"/root/.cache/huggingface": vol},
                  secrets=[modal.Secret.from_name("huggingface-secret")])
    def verify(data: bytes, layers=None):
        import torch, numpy as np, subprocess, tempfile
        t0 = time.time(); logs = []
        def log(s): print(s, flush=True); logs.append(s)
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True; torch.use_deterministic_algorithms(True, warn_only=True)
        p = tempfile.mktemp(); open(p, "wb").write(data); recs = load(p)
        from huggingface_hub import snapshot_download, hf_hub_download, list_repo_files
        from transformers import AutoTokenizer, AutoModelForCausalLM
        path = snapshot_download(MODEL_ID, revision=REV); tok = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32, attn_implementation="eager").to("cuda").eval()
        files = list_repo_files(SAE_REPO); chosen = {}
        def sae_loader(L):
            fs = [f for f in files if f.startswith(f"layer_{L}/width_16k/") and f.endswith("params.npz")]
            chosen[L] = min(fs, key=lambda s: (abs(int(re.search(r"average_l0_(\d+)", s).group(1)) - 70), s))
            return dict(np.load(hf_hub_download(SAE_REPO, chosen[L])))
        R = run_repair(model, tok, "cuda", recs, sae_loader, log, time_left=lambda: TIMEOUT_S - 90 - (time.time() - t0), layers=layers or LAYERS)
        rt = time.time() - t0
        return json.dumps(dict(status="EXPLORATORY G5 REPAIR VERIFICATION — discovery/validation only",
                               identity=dict(model=MODEL_ID, revision=REV, commit_hash=getattr(model.config, "_commit_hash", None), sae_files=chosen),
                               data_sha256=hashlib.sha256(data).hexdigest(), env=dict(device=torch.cuda.get_device_name(0)),
                               results=R, runtime_s=round(rt, 1), gpu_cost_usd_estimate=round(rt / 3600 * GPU_USD_PER_H, 3), logs=logs),
                          default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)).encode()

    @app.local_entrypoint()
    def main(data: str, layers: str = ""):
        Ls = [int(x) for x in layers.split(",") if x] or None
        b = verify.remote(open(data, "rb").read(), Ls); os.makedirs("modal_out/g5r", exist_ok=True)
        name = "g5r_results.json" if not Ls else "g5r_results_layers_%s.json" % "-".join(map(str, Ls))
        open("modal_out/g5r/" + name, "wb").write(b); d = json.loads(b); print("wrote modal_out/g5r/" + name)
        print(f"DONE runtime {d['runtime_s']}s est ${d['gpu_cost_usd_estimate']} errors {d['results'].get('errors')}")
except ImportError:
    pass
