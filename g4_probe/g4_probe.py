"""AlignedLab Experiment 3 (Research Agenda Ch.2, Exp.2) — G4 FEASIBILITY PROBE.
EXPLORATORY ONLY. Every prompt, template and noun in this file is PROBE-ONLY and must be excluded from any G5+
discovery/held-out/confirmatory set. Nothing here validates P1, P2 or P3.

Run from the Mac (weights, caches and activations stay on Modal):
    python3 -m modal run g4_probe.py
Writes ./modal_out/g4/g4_results.json (small JSON only)."""
import json, os, re, time, math, hashlib

MODEL_ID = "google/gemma-2-2b"
REV = "c5ebcd40d208330abc697524c919956e692655cf"
GPU = "A10G"
GPU_USD_PER_H = 1.10            # public Modal list price (A10); billed figure is on the dashboard
TIMEOUT_S = 45 * 60             # hard stop, well inside the 90-min G4 cap
SEED = 20261120
PINS = ["torch==2.5.1", "transformers==4.46.3", "tokenizers==0.20.3", "huggingface_hub==0.26.5",
        "safetensors==0.4.5", "numpy==1.26.4"]

# ----------------------------------------------------------------------------------------------- probe-only data
TEMPLATES_TRAIN = ["The {} was on the shelf.", "She carefully picked up the {}.", "The {} stood in the corner of the room.",
                   "He moved the {} to the hallway.", "Someone left the {} near the window."]
TEMPLATES_TEST = ["I noticed the {} on the kitchen table.", "The old {} had been there for years.",
                  "They looked at the {} for a while."]
PRONOUN_TEMPLATES = ["There was the {} on the shelf, and it", "He saw the {} in the room, and it",
                     "She cleaned the {} yesterday, and it"]
CONCEPTS = {"vase": ["vase", "glass vase", "steel vase", "ceramic vase"],
            "ball": ["ball", "bowling ball", "heavy ball"],
            "ice": ["ice", "clear ice", "solid ice"],
            "flame": ["flame", "candle flame", "hot flame"],
            "rope": ["rope", "thick rope", "old rope"],
            "knife": ["knife", "sharp knife", "kitchen knife"],
            "clock": ["clock", "wall clock", "old clock"],
            "guitar": ["guitar", "old guitar", "acoustic guitar"]}       # guitar = C1 unrelated concept
POOL = ["bowl", "jar", "lamp", "book", "box", "cup", "plate", "bottle", "chair", "candle", "basket", "mirror",
        "shoe", "hat", "pillow"]
FAMILIES = {
    "impact": dict(patient="vase", agent="ball", target="shatter", alts=["break", "crack", "fall", "dent"], prompts=[
        "The bowling ball struck the fragile glass vase, causing the vase to",
        "The heavy ball hit the delicate glass vase, causing the vase to",
        "The bowling ball smashed into the thin glass vase, causing the vase to",
        "The ball crashed into the fragile crystal vase, causing the vase to",
        "A bowling ball struck the fragile glass vase, causing the vase to",
        "The bowling ball struck the fragile porcelain vase, causing the vase to",
        "The falling ball struck the fragile glass vase, causing the vase to",
        "The bowling ball knocked over the fragile glass vase, causing the vase to"]),
    "heating": dict(patient="ice", agent="flame", target="melt", alts=["break", "crack", "boil", "evaporate"], prompts=[
        "The flame heated the ice cube, causing the ice to",
        "The hot flame warmed the ice cube, causing the ice to",
        "The candle flame heated the ice, causing the ice to",
        "The flame touched the block of ice, causing the ice to"]),
    "cutting": dict(patient="rope", agent="knife", target="snap", alts=["break", "fall", "split", "tear"], prompts=[
        "The knife sliced through the rope, causing the rope to",
        "The sharp knife cut through the rope, causing the rope to",
        "The knife sliced through the taut rope, causing the rope to"]),
}
C2_PREFIX = "As a clock ticked on the shelf, "
STEEL_DIAG = dict(prompt="The bowling ball struck the steel vase, causing the vase to", targets=["shatter", "dent"])
INTERCHANGE_DONOR = "pillow"     # diagnostic only: full-residual swap of the patient noun


def probe_only_manifest():
    items = TEMPLATES_TRAIN + TEMPLATES_TEST + PRONOUN_TEMPLATES + [p for f in FAMILIES.values() for p in f["prompts"]]
    items += [C2_PREFIX + p[0].lower() + p[1:] for f in FAMILIES.values() for p in f["prompts"]] + [STEEL_DIAG["prompt"]]
    return dict(rule="ALL items below are G4 probe-only: exclude from any G5+ held-out evaluation set",
                n=len(items), sha256=[hashlib.sha256(s.encode()).hexdigest()[:16] for s in items], items=items,
                nouns=sorted({w for v in CONCEPTS.values() for w in v} | set(POOL)))


# ----------------------------------------------------------------------------------------------- core probe
def run_probe(model, tok, device, log, sae_loader=None, time_left=lambda: 1e9):
    import torch, numpy as np
    F = torch.nn.functional
    torch.manual_seed(SEED)
    R = dict(stages={}, errors={})
    layers = model.model.layers
    cfg = model.config
    nL, nH = cfg.num_hidden_layers, cfg.num_attention_heads
    nKV = getattr(cfg, "num_key_value_heads", nH)
    hd = getattr(cfg, "head_dim", cfg.hidden_size // nH)
    Vv = len(tok)
    R["arch"] = dict(n_layers=nL, n_heads=nH, n_kv_heads=nKV, head_dim=hd, d_model=cfg.hidden_size, vocab_tokenizer=Vv,
                     lm_head_rows=model.get_output_embeddings().weight.shape[0], heads_per_kv_group=nH // nKV,
                     attn_impl=getattr(cfg, "_attn_implementation", None),
                     final_logit_softcapping=getattr(cfg, "final_logit_softcapping", None),
                     attn_logit_softcapping=getattr(cfg, "attn_logit_softcapping", None))

    def stage(name):
        R["stages"][name] = round(time.time() - T0, 2); log(f"[{R['stages'][name]:7.1f}s] {name}")
    T0 = time.time()

    def enc(text):
        e = tok(text, return_offsets_mapping=True, add_special_tokens=True)
        return torch.tensor([e["input_ids"]], device=device), e["offset_mapping"], e["input_ids"]

    def word_positions(text, offs, word):
        pos = []
        for m in re.finditer(r"\b%s\b" % re.escape(word), text):
            a, b = m.span()
            pos += [i for i, (s, e) in enumerate(offs) if e > s and s < b and e > a]
        return sorted(set(pos))

    def target_token(prompt, word):
        _, _, ip = enc(prompt); _, _, ifull = enc(prompt + " " + word)
        ok = len(ifull) == len(ip) + 1 and ifull[:len(ip)] == ip
        return dict(word=word, single_token=ok, id=ifull[-1] if ok else None,
                    pieces=tok.convert_ids_to_tokens(ifull[len(ip):]) if ifull[:len(ip)] == ip else "prefix-mismatch")

    def out_t(o): return o[0] if isinstance(o, tuple) else o

    def fwd(ids, layer_hooks=(), extra=(), attn=False):
        hs = [layers[L].register_forward_hook(f) for L, f in layer_hooks]
        for mod, kind, f in extra:
            hs.append(mod.register_forward_hook(f) if kind == "fwd" else mod.register_forward_pre_hook(f, with_kwargs=True))
        try:
            with torch.no_grad():
                o = model(ids, attention_mask=torch.ones_like(ids), output_attentions=attn, use_cache=False)
        finally:
            for h in hs: h.remove()
        return o

    def lo_stats(logits_last, tid):
        lp = torch.log_softmax(logits_last[:Vv].double(), -1)
        l = lp[tid]
        return float(l - torch.log1p(-torch.exp(l))), float(torch.exp(l)), int((lp > l).sum()) + 1

    def capture_resid(ids, positions):
        store = {}
        def mk(L):
            def f(m, i, o): store[L] = out_t(o)[0, positions, :].detach().double().cpu().clone()
            return f
        o = fwd(ids, [(L, mk(L)) for L in range(nL)])
        return store, o

    # ---- 1. forward pass + activations + determinism
    ids0, offs0, _ = enc(FAMILIES["impact"]["prompts"][0])
    st, o1 = capture_resid(ids0, list(range(ids0.shape[1])))
    o2 = fwd(ids0)
    R["check1"] = dict(seq_len=int(ids0.shape[1]), resid_shape=list(st[0].shape), finite=bool(torch.isfinite(o1.logits).all()),
                       deterministic_repeat_bitwise=bool(torch.equal(o1.logits, o2.logits)),
                       tokens=tok.convert_ids_to_tokens(ids0[0].tolist()))
    stage("check1_forward")

    # ---- 2. targets: tokenization + baseline log-odds for every probe prompt
    base = {}
    for fam, F_ in FAMILIES.items():
        for k, p in enumerate(F_["prompts"]):
            tt = target_token(p, F_["target"])
            rec = dict(family=fam, idx=k, prompt=p, target=tt, alts=[target_token(p, a) for a in F_["alts"]])
            if tt["single_token"]:
                ids, offs, _ = enc(p); o = fwd(ids)
                rec["L"], rec["p"], rec["rank"] = lo_stats(o.logits[0, -1], tt["id"])
                rec["alt_logprob"] = {a["word"]: float(torch.log_softmax(o.logits[0, -1, :Vv].double(), -1)[a["id"]])
                                      for a in rec["alts"] if a["single_token"]}
                top = torch.topk(o.logits[0, -1, :Vv], 5).indices.tolist()
                rec["top5"] = tok.convert_ids_to_tokens(top)
            base[(fam, k)] = rec
    sd = STEEL_DIAG; ids, _, _ = enc(sd["prompt"]); o = fwd(ids); lpv = torch.log_softmax(o.logits[0, -1, :Vv].double(), -1)
    R["steel_vase_diag"] = {t: (float(lpv[target_token(sd["prompt"], t)["id"]]) if target_token(sd["prompt"], t)["single_token"] else None) for t in sd["targets"]}
    R["check2"] = [dict({k: v for k, v in r.items()}) for r in base.values()]
    stage("check2_targets")

    # ---- 4. candidate entity representations: difference-of-means directions per layer (+ held-out & pronoun AUC)
    def auc(pos, neg):
        s = np.concatenate([pos, neg]); r = s.argsort().argsort() + 1.0
        return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))

    def noun_reps(templates, phrase, at_end=False):
        out = []
        for t in templates:
            text = t.format(phrase); ids, offs, _ = enc(text)
            if at_end: pos = [ids.shape[1] - 1]
            else:
                start = t.index("{}"); a, b = start, start + len(phrase)
                head_a = a + phrase.rfind(" ") + 1 if " " in phrase else a
                pos = [max(i for i, (s, e) in enumerate(offs) if e > s and s < b and e > head_a)]
            st, _ = capture_resid(ids, pos)
            out.append(torch.stack([st[L][0] for L in range(nL)]))   # [nL, d]
        return out
    reps = {}
    for c, phrases in CONCEPTS.items():
        reps[c] = dict(train=[r for ph in phrases for r in noun_reps(TEMPLATES_TRAIN, ph)],
                       test=[r for ph in phrases for r in noun_reps(TEMPLATES_TEST, ph)],
                       pron=[r for ph in phrases[:1] for r in noun_reps(PRONOUN_TEMPLATES, ph, at_end=True)])
    pool = dict(train=[r for w in POOL for r in noun_reps(TEMPLATES_TRAIN, w)],
                test=[r for w in POOL for r in noun_reps(TEMPLATES_TEST, w)],
                pron=[r for w in POOL for r in noun_reps(PRONOUN_TEMPLATES, w, at_end=True)])
    P_tr, P_te, P_pr = (torch.stack(pool[k]) for k in ("train", "test", "pron"))
    dirs, sel = {}, {}
    for c in CONCEPTS:
        C_tr, C_te, C_pr = (torch.stack(reps[c][k]) for k in ("train", "test", "pron"))
        dom = C_tr.mean(0) - P_tr.mean(0)                                   # [nL, d]
        u = dom / dom.norm(dim=-1, keepdim=True)
        mu = (P_tr * u[None]).sum(-1).mean(0)                               # reference projection (non-entity mean) [nL]
        dirs[c] = dict(u=u, mu=mu)
        sel[c] = dict(heldout_auc=[auc((C_te * u[None]).sum(-1)[:, L].numpy(), (P_te * u[None]).sum(-1)[:, L].numpy()) for L in range(nL)],
                      pronoun_auc=[auc((C_pr * u[None]).sum(-1)[:, L].numpy(), (P_pr * u[None]).sum(-1)[:, L].numpy()) for L in range(nL)],
                      n_train=len(C_tr), n_test=len(C_te), n_pron=len(C_pr))
    g = torch.Generator().manual_seed(SEED)
    ur = torch.randn(nL, cfg.hidden_size, generator=g, dtype=torch.float64); ur = ur / ur.norm(dim=-1, keepdim=True)
    dirs["random"] = dict(u=ur, mu=(P_tr * ur[None]).sum(-1).mean(0))
    patients = [f["patient"] for f in FAMILIES.values()]
    lo, hi = 1, nL - 3
    score = [np.mean([sel[c]["pronoun_auc"][L] for c in patients]) for L in range(nL)]
    Lstar = int(max(range(lo, hi + 1), key=lambda L: (score[L], -L)))
    R["check4"] = dict(method="difference-of-means direction (concept vs neutral pool) at head-noun token, residual stream output of each layer",
                       selectivity=sel, L_star=Lstar, L_star_rule="argmax over layers 1..nL-3 of mean pronoun-position AUC (vase, ice, rope)",
                       mean_pronoun_auc_by_layer=score, pool=POOL, caveat="exploratory; held-out AUC at the noun token is partly lexical")
    stage("check4_directions")

    # ---- 3/5/6. interventions: Substance vs C1 (unrelated + random direction) vs C2 (bystander clock)
    def proj_hook(P, c, L, rec, target_norm=None):
        u = dirs[c]["u"][L].to(device=device, dtype=torch.float32); mu = float(dirs[c]["mu"][L])
        def f(m, i, o):
            h = out_t(o); x = h[0, P, :]
            delta = -((x @ u) - mu)[:, None] * u[None, :]
            raw = float(delta.norm()); scale = 1.0
            if target_norm is not None and raw > 0: scale = target_norm / raw; delta = delta * scale
            h[0, P, :] = x + delta
            rec.update(layer=L, hook=f"model.layers[{L}] output (resid_post)", operator="rank-1 projection-to-reference (non-entity mean)",
                       direction=c, dim=1, positions=P, l2_raw=raw, scale=scale, l2_total=float(delta.norm()),
                       l2_per_pos=[float(v) for v in delta.norm(dim=-1)], matched_to_norm=target_norm)
            return o
        return f

    def swap_hook(P, donor_rows, L, rec):
        def f(m, i, o):
            h = out_t(o); x = h[0, P, :]; d = donor_rows.to(device=device, dtype=h.dtype)
            rec.update(layer=L, hook=f"model.layers[{L}] output (resid_post)", operator="full-residual interchange (diagnostic)",
                       dim=int(h.shape[-1]), positions=P, l2_total=float((d - x).norm())); h[0, P, :] = d
            return o
        return f

    def measure(ids, tid, Lint, hooks):
        o = fwd(ids, hooks); return lo_stats(o.logits[0, -1], tid)[0]

    layers_test = sorted({Lstar, nL // 4, nL // 2, (3 * nL) // 4})
    R["interventions"] = []
    for (fam, k), b in base.items():
        if "L" not in b: continue
        F_ = FAMILIES[fam]; p = b["prompt"]; tid = b["target"]["id"]
        ids, offs, _ = enc(p)
        Ppat, Pag = word_positions(p, offs, F_["patient"]), word_positions(p, offs, F_["agent"])
        p2 = C2_PREFIX + p[0].lower() + p[1:]; ids2, offs2, _ = enc(p2)
        Pclk, Ppat2 = word_positions(p2, offs2, "clock"), word_positions(p2, offs2, F_["patient"])
        tt2 = target_token(p2, F_["target"])
        L2i = measure(ids2, tt2["id"], None, []) if tt2["single_token"] else None
        # donor for interchange diagnostic (only if token counts align)
        pd = re.sub(r"\b%s\b" % F_["patient"], INTERCHANGE_DONOR, p); idsd, offsd, _ = enc(pd)
        for L in layers_test:
            row = dict(family=fam, idx=k, layer=L, L_intact=b["L"], patient_pos=Ppat, agent_pos=Pag,
                       c2_prompt=p2, clock_pos=Pclk, patient_pos_c2=Ppat2, L_intact_c2=L2i, recs={})
            def run(name, hooks_fn, ids_=ids, tid_=tid, Lref=b["L"]):
                recs = [dict() for _ in hooks_fn]
                Lv = measure(ids_, tid_, L, [(L, hf(r)) for hf, r in zip(hooks_fn, recs)])
                row["recs"][name] = dict(D=Lref - Lv, L=Lv, interventions=recs)
                return recs
            sS = run("S_patient", [lambda r: proj_hook(Ppat, F_["patient"], L, r)])
            nS = math.sqrt(sum(x["l2_total"] ** 2 for x in sS))
            if Pag: run("S_agent", [lambda r: proj_hook(Pag, F_["agent"], L, r)])
            if Pag: run("S_both", [lambda r: proj_hook(Ppat, F_["patient"], L, r), lambda r: proj_hook(Pag, F_["agent"], L, r)])
            run("C1_unrelated_raw", [lambda r: proj_hook(Ppat, "guitar", L, r)])
            run("C1_unrelated_matched", [lambda r: proj_hook(Ppat, "guitar", L, r, target_norm=nS)])
            run("C1_random_raw", [lambda r: proj_hook(Ppat, "random", L, r)])
            run("C1_random_matched", [lambda r: proj_hook(Ppat, "random", L, r, target_norm=nS)])
            if L2i is not None and Pclk:
                sS2 = run("S_patient_c2", [lambda r: proj_hook(Ppat2, F_["patient"], L, r)], ids2, tt2["id"], L2i)
                nS2 = math.sqrt(sum(x["l2_total"] ** 2 for x in sS2))
                run("C2_bystander_raw", [lambda r: proj_hook(Pclk, "clock", L, r)], ids2, tt2["id"], L2i)
                run("C2_bystander_matched", [lambda r: proj_hook(Pclk, "clock", L, r, target_norm=nS2)], ids2, tt2["id"], L2i)
            if idsd.shape[1] == ids.shape[1]:
                std, _ = capture_resid(idsd, Ppat)
                run("DIAG_interchange_patient", [lambda r: swap_hook(Ppat, std[L], L, r)])
            R["interventions"].append(row)
        if time_left() < 600: R["errors"]["interventions"] = "stopped early: time budget"; break
    stage("check5_6_interventions")

    # ---- P3 feasibility: downstream heads + KV-group path restoration (first prompt of each family, layer L*)
    R["p3"] = []
    for fam, F_ in FAMILIES.items():
        b = base[(fam, 0)]
        if "L" not in b: continue
        p = b["prompt"]; tid = b["target"]["id"]; ids, offs, _ = enc(p); Ppat = word_positions(p, offs, F_["patient"])
        last = ids.shape[1] - 1; L = Lstar
        oi = fwd(ids, attn=True); Lc = lo_stats(oi.logits[0, -1], tid)[0]
        attn_to = {(T, h): float(oi.attentions[T][0, h, last, Ppat].sum()) for T in range(L + 1, nL) for h in range(nH)}
        def ohook(T, cap=None, zero_h=None):
            def f(m, args, kwargs):
                x = args[0] if args else kwargs["input"]
                if cap is not None: cap[T] = x[0, last].detach().double().cpu().clone()
                if zero_h is not None:
                    x = x.clone(); x[0, last, zero_h * hd:(zero_h + 1) * hd] = 0
                    return ((x,) + tuple(args[1:]), kwargs) if args else (args, dict(kwargs, input=x))
                return None
            return f
        abl = {}
        for T in range(L + 1, nL):
            for h in range(nH):
                o = fwd(ids, extra=[(layers[T].self_attn.o_proj, "pre", ohook(T, zero_h=h))])
                abl[(T, h)] = Lc - lo_stats(o.logits[0, -1], tid)[0]
        cap_c, cap_i = {}, {}
        fwd(ids, extra=[(layers[T].self_attn.o_proj, "pre", ohook(T, cap_c)) for T in range(L + 1, nL)])
        srec = {}
        oS = fwd(ids, [(L, proj_hook(Ppat, F_["patient"], L, srec))],
                 extra=[(layers[T].self_attn.o_proj, "pre", ohook(T, cap_i)) for T in range(L + 1, nL)])
        Li = lo_stats(oS.logits[0, -1], tid)[0]
        head_shift = {(T, h): float((cap_i[T][h * hd:(h + 1) * hd] - cap_c[T][h * hd:(h + 1) * hd]).norm()) for T in range(L + 1, nL) for h in range(nH)}
        cands = [k for k in abl if attn_to[k] >= 0.05]
        pick = max(cands or list(abl), key=lambda k: abs(abl[k]))
        T, h = pick; gq = h // (nH // nKV); galt = (gq + 1) % nKV
        xin = {}
        def cap_in(m, args, kwargs): xin["x"] = (kwargs.get("hidden_states") if "hidden_states" in kwargs else args[0]).detach().clone()
        fwd(ids, extra=[(layers[T].self_attn, "pre", cap_in)])
        def kv_restore(group, positions):
            def f(m, i, o):
                clean = F.linear(xin["x"], m.weight, m.bias); o = o.clone()
                for g_ in group: o[0, positions, g_ * hd:(g_ + 1) * hd] = clean[0, positions, g_ * hd:(g_ + 1) * hd]
                return o
            return f
        def attn_in_cap(store):
            def f(m, args, kwargs): store["x"] = (kwargs.get("hidden_states") if "hidden_states" in kwargs else args[0]).detach().clone()
            return f
        def restored(group):
            cap_r, ain = {}, {}
            o = fwd(ids, [(L, proj_hook(Ppat, F_["patient"], L, {}))],
                    extra=[(layers[T].self_attn.k_proj, "fwd", kv_restore(group, Ppat)),
                           (layers[T].self_attn.v_proj, "fwd", kv_restore(group, Ppat)),
                           (layers[T].self_attn.o_proj, "pre", ohook(T, cap_r)), (layers[T].self_attn, "pre", attn_in_cap(ain))])
            Lr = lo_stats(o.logits[0, -1], tid)[0]
            oc, oi_, orr = (c[T][h * hd:(h + 1) * hd] for c in (cap_c, cap_i, cap_r))
            den = float((oi_ - oc).norm())
            return dict(group=list(group), L=Lr, logit_recovery=(Lr - Li) / (Lc - Li) if abs(Lc - Li) > 1e-6 else None,
                        head_out_recovery=1 - float((orr - oc).norm()) / den if den > 0 else None, ain=ain["x"])
        ain_int = {}
        fwd(ids, [(L, proj_hook(Ppat, F_["patient"], L, {}))], extra=[(layers[T].self_attn, "pre", attn_in_cap(ain_int))])
        r_own, r_alt, r_all = restored([gq]), restored([galt]), restored(list(range(nKV)))
        iso = all(torch.equal(r["ain"], ain_int["x"]) for r in (r_own, r_alt, r_all))
        for r in (r_own, r_alt, r_all): r.pop("ain")
        top = sorted(abl, key=lambda k: -abs(abl[k]))[:8]
        R["p3"].append(dict(family=fam, prompt=p, layer_substance=L, L_clean=Lc, L_substance_intervened=Li, substance_rec=srec,
                            top_heads_by_ablation=[dict(layer=t, head=hh, dL=abl[(t, hh)], attn_to_patient=attn_to[(t, hh)],
                                                        head_out_shift_under_S=head_shift[(t, hh)]) for t, hh in top],
                            chosen=dict(layer=T, head=h, kv_group=gq, rule="max |ablation dL| among heads with attn(last->patient) >= 0.05",
                                        dL=abl[pick], attn_to_patient=attn_to[pick], head_out_shift_under_S=head_shift[pick],
                                        n_candidates=len(cands)),
                            restore_own_kv_group=r_own, restore_other_kv_group=r_alt, restore_all_kv_layer=r_all,
                            isolation_layer_input_unchanged=iso,
                            gqa_note=f"{nH // nKV} query heads share KV group {gq}; restoration is per KV group, not per query head"))
        if time_left() < 300: R["errors"]["p3"] = "stopped early: time budget"; break
    stage("p3_feasibility")

    # ---- optional: Gemma Scope SAE route (feasibility of an SAE feature basis for G5)
    if sae_loader is not None and time_left() > 300:
        try:
            W = sae_loader(Lstar)
            Wenc, benc, thr, Wdec = (torch.tensor(W[k], dtype=torch.float64) for k in ("W_enc", "b_enc", "threshold", "W_dec"))
            def acts(x): pre = x @ Wenc + benc; return pre * (pre > thr)
            A_c = torch.stack([acts(r[Lstar]) for r in reps["vase"]["train"]]).mean(0)
            A_p = torch.stack([acts(r[Lstar]) for r in pool["train"]]).mean(0)
            topv = torch.topk(A_c - A_p, 5).indices
            G_c = torch.stack([acts(r[Lstar]) for r in reps["guitar"]["train"]]).mean(0)
            topg = torch.topk(G_c - A_p, 5).indices
            te_c = np.array([float(acts(r[Lstar])[topv].sum()) for r in reps["vase"]["test"]])
            te_p = np.array([float(acts(r[Lstar])[topv].sum()) for r in pool["test"]])
            pr_c = np.array([float(acts(r[Lstar])[topv].sum()) for r in reps["vase"]["pron"]])
            pr_p = np.array([float(acts(r[Lstar])[topv].sum()) for r in pool["pron"]])
            Wd = Wdec.to(device=device, dtype=torch.float32); We = Wenc.to(device, torch.float32); be = benc.to(device, torch.float32); th = thr.to(device, torch.float32)
            def sae_hook(P, lat, rec, target_norm=None):
                def f(m, i, o):
                    hh = out_t(o); x = hh[0, P, :]; pre = x @ We[:, lat] + be[lat]; a = pre * (pre > th[lat])
                    delta = -(a @ Wd[lat]); raw = float(delta.norm()); sc = 1.0
                    if target_norm is not None and raw > 0: sc = target_norm / raw; delta = delta * sc
                    hh[0, P, :] = x + delta
                    rec.update(layer=Lstar, operator="SAE latent ablation (subtract decoder contribution)", dim=len(lat), positions=P,
                               l2_raw=raw, scale=sc, l2_total=float(delta.norm()), active=int((a > 0).sum()))
                    return o
                return f
            rows = []
            for k, p in enumerate(FAMILIES["impact"]["prompts"]):
                b = base[("impact", k)]
                if "L" not in b: continue
                ids, offs, _ = enc(p); P = word_positions(p, offs, "vase"); tid = b["target"]["id"]
                rs, rc, rm = {}, {}, {}
                Ls = measure(ids, tid, Lstar, [(Lstar, sae_hook(P, topv.tolist(), rs))])
                Lu = measure(ids, tid, Lstar, [(Lstar, sae_hook(P, topg.tolist(), rc))])
                Lm = measure(ids, tid, Lstar, [(Lstar, sae_hook(P, topg.tolist(), rm, target_norm=rs["l2_total"]))]) if rs["l2_total"] > 0 else None
                rows.append(dict(idx=k, D_sae_vase=b["L"] - Ls, D_sae_guitar_raw=b["L"] - Lu, D_sae_guitar_matched=(b["L"] - Lm) if Lm is not None else None,
                                 rec_vase=rs, rec_guitar=rc, rec_guitar_matched=rm))
            R["sae"] = dict(release="google/gemma-scope-2b-pt-res", file=W.get("_path"), layer=Lstar, vase_latents=topv.tolist(),
                            guitar_latents=topg.tolist(), heldout_auc=auc(te_c, te_p), pronoun_auc=auc(pr_c, pr_p), impact_rows=rows)
        except Exception as e:  # optional block must never kill the probe
            R["errors"]["sae"] = repr(e)[:500]
        stage("sae_optional")

    # ---- 7/8. noise, variability, sample size (exploratory; NOT an effect-size claim)
    def ss(sd_, delta): return math.ceil(((1.96 + 0.8416) * sd_ / delta) ** 2) if sd_ and delta else None
    summ = {}
    for L in layers_test:
        rows = [r for r in R["interventions"] if r["layer"] == L]
        def col(name): return [r["recs"][name]["D"] for r in rows if name in r["recs"]]
        def diff(a, b_): return [r["recs"][a]["D"] - r["recs"][b_]["D"] for r in rows if a in r["recs"] and b_ in r["recs"]]
        d1 = diff("S_patient", "C1_unrelated_matched"); d1r = diff("S_patient", "C1_random_matched"); d2 = diff("S_patient_c2", "C2_bystander_matched")
        def desc(x):
            if len(x) < 2: return dict(n=len(x), values=x)
            a = np.array(x); rng = np.random.default_rng(SEED); bs = [np.std(rng.choice(a, len(a)), ddof=1) for _ in range(2000)]
            s = float(a.std(ddof=1))
            return dict(n=len(a), mean=float(a.mean()), sd=s, sd_boot95=[float(np.quantile(bs, .025)), float(np.quantile(bs, .975))],
                        n_required_80pct_power={str(dl): ss(s, dl) for dl in (0.1, 0.25, 0.5, 1.0)})
        summ[str(L)] = dict(D_S_patient=desc(col("S_patient")), D_C1_unrelated_matched=desc(col("C1_unrelated_matched")),
                            D_C1_random_matched=desc(col("C1_random_matched")), D_C2_matched=desc(col("C2_bystander_matched")),
                            S_minus_C1_unrelated=desc(d1), S_minus_C1_random=desc(d1r), S_minus_C2=desc(d2),
                            D_interchange_diag=desc(col("DIAG_interchange_patient")))
    R["summary_by_layer"] = summ
    R["baseline_L"] = dict(values=[b.get("L") for b in base.values()],
                           sd_across_prompts=float(np.std([b["L"] for b in base.values() if "L" in b], ddof=1)))
    stage("summary")
    return R


# ----------------------------------------------------------------------------------------------- Modal wrapper
try:
    import modal
    app = modal.App("alignedlab-exp3-g4-probe")
    image = (modal.Image.debian_slim(python_version="3.12").pip_install(*PINS)
             .env({"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "HF_HUB_ENABLE_HF_TRANSFER": "0"}))
    vol = modal.Volume.from_name("kant-hf-cache", create_if_missing=True)

    @app.function(image=image, gpu=GPU, timeout=TIMEOUT_S, volumes={"/root/.cache/huggingface": vol},
                  secrets=[modal.Secret.from_name("huggingface-secret")])
    def probe():
        import torch, subprocess, numpy as np
        t0 = time.time(); logs = []
        def log(s): print(s, flush=True); logs.append(s)
        env = dict(hf_token_injected=bool(os.environ.get("HF_TOKEN")), gpu_requested=GPU,
                   device=torch.cuda.get_device_name(0), torch=torch.__version__, cuda=torch.version.cuda,
                   pip_freeze=subprocess.run(["pip", "freeze"], capture_output=True, text=True).stdout.split("\n"))
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        from huggingface_hub import snapshot_download, hf_hub_download, list_repo_files
        path = snapshot_download(MODEL_ID, revision=REV); vol.commit(); log(f"snapshot ready {time.time()-t0:.0f}s")
        manifest = {f: hashlib.sha256(open(os.path.join(path, f), "rb").read()).hexdigest() for f in sorted(os.listdir(path))
                    if f.endswith((".json", ".safetensors", ".model"))}
        log(f"manifest {time.time()-t0:.0f}s")
        from transformers import AutoTokenizer, AutoModelForCausalLM
        tok = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32, attn_implementation="eager").to("cuda").eval()
        ident = dict(model_id=MODEL_ID, revision=REV, commit_hash=getattr(model.config, "_commit_hash", None),
                     file_sha256=manifest, dtype="float32", tokenizer_class=type(tok).__name__, bos_token=tok.bos_token,
                     add_bos_default=True, config=model.config.to_dict())
        log(f"model loaded {time.time()-t0:.0f}s")
        def sae_loader(L):
            files = [f for f in list_repo_files("google/gemma-scope-2b-pt-res") if f.startswith(f"layer_{L}/width_16k/") and f.endswith("params.npz")]
            f = min(files, key=lambda s: abs(int(re.search(r"average_l0_(\d+)", s).group(1)) - 70))
            d = dict(np.load(hf_hub_download("google/gemma-scope-2b-pt-res", f))); d["_path"] = f; vol.commit(); return d
        R = run_probe(model, tok, "cuda", log, sae_loader=sae_loader, time_left=lambda: TIMEOUT_S - 120 - (time.time() - t0))
        rt = time.time() - t0
        out = dict(status="EXPLORATORY G4 PROBE — not evidence for P1/P2/P3", identity=ident, env=env, results=R,
                   probe_only=probe_only_manifest(), runtime_s=round(rt, 1),
                   gpu_cost_usd_estimate=round(rt / 3600 * GPU_USD_PER_H, 3), logs=logs)
        return json.dumps(out, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)).encode()

    @app.local_entrypoint()
    def main():
        t = time.time(); b = probe.remote()
        os.makedirs("modal_out/g4", exist_ok=True); open("modal_out/g4/g4_results.json", "wb").write(b)
        d = json.loads(b)
        print(f"DONE in {time.time()-t:.0f}s wall; remote runtime {d['runtime_s']}s; est GPU cost ${d['gpu_cost_usd_estimate']}")
        print("errors:", d["results"].get("errors")); print("wrote modal_out/g4/g4_results.json")
except ImportError:
    pass
