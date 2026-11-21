"""Kantian Positional Logic Test — G5 FEASIBILITY VERIFICATION (exploratory; PI-authorized 2026-11-21, cap +$0.50).
Uses ONLY G4 exploratory materials (imported from g4_probe.py) plus their Decision-2 pronoun-format conversion.
No final held-out evaluation items exist or are used. Nothing here is evidence for P1/P2/P3.

Objectives: (1) SAE feature selection runs and yields candidates; (2) C1/C2 matching under Decision 2 limits;
(3) layer-selection feasibility (representation criteria only, never causal effects); (4) exclusion rates.

Run from the Mac, in the folder that also contains g4_probe.py:
    MODAL_IMAGE_BUILDER_VERSION=2025.06 python3 -m modal run g5v_probe.py
Writes ./modal_out/g5v/g5v_results.json"""
import json, os, re, time, math, hashlib
import g4_probe as G4

MODEL_ID, REV = G4.MODEL_ID, G4.REV
SAE_REPO = "google/gemma-scope-2b-pt-res"
GPU, GPU_USD_PER_H = "A10G", 1.10
TIMEOUT_S = 20 * 60                       # GPU hard cap ≈ $0.37 at list price
SEED = 20261121
# Decision 1-4 settings (PI-approved 2026-11-21)
A_MAX, RHO_MIN, TOL, ACTIVE_RATIO = 2.0, 0.5, 0.10, (0.5, 2.0)
K_MAX, GAIN_MIN, PREFILTER = 8, 0.01, 64
P_MIN, RANK_MAX, BYSTANDER_SHIFT_MAX = 0.10, 5, 1.0
THR = dict(recog=0.85, perm_frac=0.80, worst_group=0.75, property_max=0.65)
PATIENT_KINDS = ["vase", "ice", "rope"]
BYSTANDER = "clock"
FRAGILE, ROBUST = {"glass vase", "ceramic vase"}, {"steel vase"}       # only property structure available in G4 nouns


def convert(prompt, patient):
    return prompt.replace("causing the %s to" % patient, "causing it to")


def derived_manifest():
    items = []
    for f in G4.FAMILIES.values():
        for p in f["prompts"]:
            q = convert(p, f["patient"]); items += [q, G4.C2_PREFIX + q[0].lower() + q[1:]]
    return dict(rule="G5v derived items (pronoun-format conversions of G4 probe prompts): probe-only, excluded from any held-out set",
                n=len(items), sha256=[hashlib.sha256(s.encode()).hexdigest()[:16] for s in items], items=items)


def run_verify(model, tok, device, log, sae_loader, time_left=lambda: 1e9):
    import torch, numpy as np
    torch.manual_seed(SEED)
    T0 = time.time(); R = dict(stages={}, errors={}, settings=dict(A_MAX=A_MAX, RHO_MIN=RHO_MIN, TOL=TOL, ACTIVE_RATIO=ACTIVE_RATIO,
                                K_MAX=K_MAX, GAIN_MIN=GAIN_MIN, PREFILTER=PREFILTER, P_MIN=P_MIN, RANK_MAX=RANK_MAX,
                                BYSTANDER_SHIFT_MAX=BYSTANDER_SHIFT_MAX, THR=THR))
    def stage(n): R["stages"][n] = round(time.time() - T0, 1); log(f"[{R['stages'][n]:7.1f}s] {n}")
    layers_mod = model.model.layers; nL = model.config.num_hidden_layers; Vv = len(tok)
    LAYERS = list(range(2, min(18, nL - 3) + 1))
    R["layers_scanned"] = LAYERS

    def enc(t):
        e = tok(t, return_offsets_mapping=True, add_special_tokens=True)
        return torch.tensor([e["input_ids"]], device=device), e["offset_mapping"], e["input_ids"]
    def word_pos(text, offs, w):
        out = []
        for m in re.finditer(r"\b%s\b" % re.escape(w), text):
            a, b = m.span(); out += [i for i, (s, e) in enumerate(offs) if e > s and s < b and e > a]
        return sorted(set(out))
    def out_t(o): return o[0] if isinstance(o, tuple) else o
    def fwd(ids, hooks=()):
        hs = [layers_mod[L].register_forward_hook(f) for L, f in hooks]
        try:
            with torch.no_grad(): return model(ids, attention_mask=torch.ones_like(ids), use_cache=False)
        finally:
            for h in hs: h.remove()
    def capture(ids, pos):
        st = {}
        def mk(L):
            def f(m, i, o): st[L] = out_t(o)[0, pos, :].detach().float().clone()
            return f
        o = fwd(ids, [(L, mk(L)) for L in LAYERS]); return st, o
    def lo(logits, tid):
        lp = torch.log_softmax(logits[:Vv].double(), -1); l = lp[tid]
        return float(l - torch.log1p(-torch.exp(l))), float(torch.exp(l)), int((lp > l).sum()) + 1
    def tgt(prompt, w):
        _, _, a = enc(prompt); _, _, b = enc(prompt + " " + w)
        return b[-1] if (len(b) == len(a) + 1 and b[:len(a)] == a) else None

    # ---------------- recognition data (G4 templates/nouns; D/V split by template)
    recs = []   # dict(kind, phrase, split, ptype, st{L: [d]})
    def add(kind, phrase, templates, split, ptype):
        for t in templates:
            text = t.format(phrase); ids, offs, _ = enc(text)
            if ptype == "pron": pos = ids.shape[1] - 1
            else:
                a = t.index("{}"); b = a + len(phrase); ha = a + phrase.rfind(" ") + 1 if " " in phrase else a
                pos = max(i for i, (s, e) in enumerate(offs) if e > s and s < b and e > ha)
            st, _ = capture(ids, [pos]); recs.append(dict(kind=kind, phrase=phrase, split=split, ptype=ptype, st={L: v[0] for L, v in st.items()}))
    PT = G4.PRONOUN_TEMPLATES
    for kind, phrases in list(G4.CONCEPTS.items()) + [("pool", G4.POOL)]:
        for ph in phrases:
            add(kind, ph, G4.TEMPLATES_TRAIN, "D", "noun"); add(kind, ph, G4.TEMPLATES_TEST, "V", "noun")
            add(kind, ph, PT[:2], "D", "pron"); add(kind, ph, PT[2:], "V", "pron")
    R["recognition_counts"] = {k: sum(r["kind"] == k for r in recs) for k in list(G4.CONCEPTS) + ["pool"]}
    stage("recognition_capture")

    # ---------------- causal items in Decision-2 format (base + C2 variant), baseline eligibility
    items = []
    for fam, F in G4.FAMILIES.items():
        for k, p0 in enumerate(F["prompts"]):
            p = convert(p0, F["patient"]); p2 = G4.C2_PREFIX + p[0].lower() + p[1:]
            ids, offs, _ = enc(p); ids2, offs2, _ = enc(p2)
            it = dict(family=fam, idx=k, kind=F["patient"], prompt=p, prompt_c2=p2,
                      P=word_pos(p, offs, F["patient"]), P2=word_pos(p2, offs2, F["patient"]), Pc=word_pos(p2, offs2, BYSTANDER))
            t1, t2 = tgt(p, F["target"]), tgt(p2, F["target"]); it["codes"] = []
            if t1 is None or t2 is None: it["codes"].append("TOKENIZATION")
            st1, o1 = capture(ids, it["P"]); st2, o2 = capture(ids2, it["P2"] + it["Pc"])
            it["_st"], it["_st2"], it["_ids"], it["_ids2"], it["tid"] = st1, st2, ids, ids2, t1
            if t1 is not None and t2 is not None:
                it["L"], it["p"], it["rank"] = lo(o1.logits[0, -1], t1); it["L_c2"], it["p_c2"], it["rank_c2"] = lo(o2.logits[0, -1], t2)
                if not (it["p"] >= P_MIN and it["rank"] <= RANK_MAX and it["p_c2"] >= P_MIN and it["rank_c2"] <= RANK_MAX): it["codes"].append("BASELINE_INELIGIBLE")
                if abs(it["L"] - it["L_c2"]) > BYSTANDER_SHIFT_MAX: it["codes"].append("BYSTANDER_DESTABILIZES")
            if len(it["P"]) != 1 or len(it["Pc"]) != 1 or len(it["P2"]) != 1: it["codes"].append("C2_SPAN")
            items.append(it)
    stage("causal_capture")

    # ---------------- per-layer SAE analysis
    def auc_cols(P, N):            # P [m,F], N [n,F] -> [F]
        out = torch.zeros(P.shape[1], device=P.device)
        for s in range(0, P.shape[1], 4096):
            a, b = P[:, None, s:s + 4096], N[None, :, s:s + 4096]
            out[s:s + 4096] = (a > b).float().mean((0, 1)) + 0.5 * (a == b).float().mean((0, 1))
        return out
    def auc1(p, n): return float(auc_cols(p[:, None], n[:, None])[0])
    def youden(p, n):
        cand = torch.cat([p, n]).unique(); best, tb = -1, 0.0
        for t in cand:
            j = float((p >= t).float().mean() - (n >= t).float().mean())
            if j > best: best, tb = j, float(t)
        return tb

    per_layer, feats = {}, {}
    for L in LAYERS:
        if time_left() < 240: R["errors"]["layers"] = f"stopped at layer {L}: time budget"; break
        W = sae_loader(L)
        We, be, th, Wd = (torch.tensor(W[k], device=device, dtype=torch.float32) for k in ("W_enc", "b_enc", "threshold", "W_dec"))
        dn = Wd.norm(dim=-1)
        def acts(x): pre = x @ We + be; return pre * (pre > th)
        X = torch.stack([r["st"][L] for r in recs]).to(device); A = acts(X)
        def sel(kind=None, split=None, ptype=None, phrases=None):
            return torch.tensor([(kind is None or r["kind"] == kind) and (split is None or r["split"] == split) and (ptype is None or r["ptype"] == ptype)
                                 and (phrases is None or r["phrase"] in phrases) for r in recs], device=device)
        row = {}
        for K in PATIENT_KINDS + [BYSTANDER]:
            mD_n, nD_n, mD_p, nD_p = sel(K, "D", "noun"), sel("pool", "D", "noun"), sel(K, "D", "pron"), sel("pool", "D", "pron")
            a_n, a_p = auc_cols(A[mD_n], A[nD_n]), auc_cols(A[mD_p], A[nD_p])
            score = torch.minimum(a_n, a_p)
            if K == "vase":   # drop property latents (fragile vs robust within members, discovery noun)
                pa = auc_cols(A[sel(K, "D", "noun", FRAGILE)], A[sel(K, "D", "noun", ROBUST)])
                score = torch.where((pa - 0.5).abs() >= 0.25, torch.full_like(score, -1.0), score)
            cand = torch.topk(score, PREFILTER).indices.tolist(); S, best = [], 0.5
            while len(S) < K_MAX:
                trial = []
                for c in cand:
                    if c in S: continue
                    sc = A[:, S + [c]].sum(1)
                    trial.append((min(auc1(sc[mD_n], sc[nD_n]), auc1(sc[mD_p], sc[nD_p])), float(A[mD_n][:, c].mean()), c))
                if not trial: break
                v, _, c = max(trial)
                if S and v - best < GAIN_MIN: break
                S.append(c); best = v
            sc = A[:, S].sum(1)
            tau = youden(torch.cat([sc[mD_n], sc[mD_p]]), torch.cat([sc[nD_n], sc[nD_p]]))
            mV_n, nV_n, mV_p, nV_p = sel(K, "V", "noun"), sel("pool", "V", "noun"), sel(K, "V", "pron"), sel("pool", "V", "pron")
            m = dict(k=len(S), latents=S, disc_min_auc=best, tau=tau,
                     recog_noun=auc1(sc[mV_n], sc[nV_n]), recog_pron=auc1(sc[mV_p], sc[nV_p]),
                     perm_frac=float((torch.cat([sc[mV_n], sc[mV_p]]) >= tau).float().mean()),
                     worst_group=min(auc1(sc[sel(K, "V", "noun", {ph})], sc[nV_n]) for ph in G4.CONCEPTS[K]))
            if K == "vase":
                pa = auc1(sc[sel(K, "V", "noun", FRAGILE)], sc[sel(K, "V", "noun", ROBUST)]); m["property_auc"] = max(pa, 1 - pa)
            crit = [(min(m["recog_noun"], m["recog_pron"]) - THR["recog"]) / (1 - THR["recog"]),
                    (m["perm_frac"] - THR["perm_frac"]) / (1 - THR["perm_frac"]), (m["worst_group"] - THR["worst_group"]) / (1 - THR["worst_group"])]
            if "property_auc" in m: crit.append((THR["property_max"] - m["property_auc"]) / THR["property_max"])
            m["margin"] = min(crit); m["passes_available_criteria"] = m["margin"] >= 0
            m["criteria_not_testable_with_G4_data"] = ["relationality (roles/causal contexts)", "property-matched non-members", "noun-free descriptions"] + ([] if K == "vase" else ["property-only invariance"])
            # C1 candidates: active at member noun positions (D), identity-unselective, not in S, not property
            if K != BYSTANDER:
                actfrac = (A[mD_n] > 0).float().mean(0)
                ok = (actfrac >= 0.5) & ((a_n - 0.5).abs() < 0.05)
                if K == "vase": ok &= (pa_l := auc_cols(A[sel(K, "D", "noun", FRAGILE)], A[sel(K, "D", "noun", ROBUST)]) - 0.5).abs() < 0.15
                ok[S] = False
                contrib = (A[mD_n] * dn).mean(0)
                cS = float(contrib[S].mean()) if S else 0.0
                idx = torch.nonzero(ok).flatten().tolist()
                idx = sorted(idx, key=lambda j: (abs(float(contrib[j]) - cS), j))[:len(S)]
                m["c1_latents"], m["c1_pool_size"], m["c1_feasible_selection"] = idx, int(ok.sum()), len(idx) == len(S) and len(S) > 0
            else:
                m["ranked"] = torch.topk(score, K_MAX).indices.tolist()
            row[K] = m
        # matching feasibility on causal items (natural norms, no extra forward)
        for it in items:
            K = it["kind"]; S = row[K]["latents"]; C1 = row[K].get("c1_latents", []); C2 = row[BYSTANDER]["ranked"][:len(S)]
            def contrib(st, rows, lat):
                if not lat: return 0.0, 0
                x = st[L][rows].to(device); a = acts(x)[:, lat]
                return float((a @ Wd[lat]).norm()), int((a > 0).sum())
            nS, aS = contrib(it["_st"], list(range(len(it["P"]))), S); nC1, aC1 = contrib(it["_st"], list(range(len(it["P"]))), C1)
            nP2 = len(it["P2"]); nS2, aS2 = contrib(it["_st2"], list(range(nP2)), S); nC2, aC2 = contrib(it["_st2"], list(range(nP2, nP2 + len(it["Pc"]))), C2)
            codes = []
            Nst = min(nS, nS2, A_MAX * min(nC1, nC2)) if min(nC1, nC2) > 0 else 0.0
            if Nst <= 0 or Nst < RHO_MIN * max(nS, nS2): codes.append("MATCH_INFEASIBLE")
            def ratio(a, b): return (a / b) if b else float("inf")
            if not (ACTIVE_RATIO[0] <= ratio(aC1, aS) <= ACTIVE_RATIO[1]) or not (ACTIVE_RATIO[0] <= ratio(aC2, aS2) <= ACTIVE_RATIO[1]): codes.append("ACTIVE_UNITS")
            it.setdefault("match", {})[L] = dict(n_S=nS, n_C1=nC1, n_S2=nS2, n_C2=nC2, act_S=aS, act_C1=aC1, act_S2=aS2, act_C2=aC2, N_star=Nst, codes=codes)
        per_layer[L] = row
        feats[L] = dict(We=None)  # placeholder; SAE kept only for L* below
        del We, be, th, Wd, A, X; torch.cuda.empty_cache() if device == "cuda" else None
        stage(f"layer_{L}")
    R["per_layer"] = per_layer

    # ---------------- layer selection (representation criteria only)
    done = list(per_layer)
    glob = {L: min(per_layer[L][K]["margin"] for K in PATIENT_KINDS) for L in done}
    passing = [L for L in done if glob[L] >= 0]
    Lstar = max(passing or done, key=lambda L: (glob[L], -L))
    R["layer_selection"] = dict(global_margin=glob, passing_layers=passing, L_star=Lstar, L_star_passes=Lstar in passing,
                                per_kind_best={K: max(done, key=lambda L: (per_layer[L][K]["margin"], -L)) for K in PATIENT_KINDS},
                                rule="argmax over scanned layers of min over patient kinds of normalized margin on available V criteria; ties -> shallower")
    stage("layer_selection")

    # ---------------- exclusion rates
    def summarize(L):
        out = {}
        for fam in G4.FAMILIES:
            its = [i for i in items if i["family"] == fam]
            codes = [sorted(set(i["codes"] + i["match"][L]["codes"])) for i in its]
            out[fam] = dict(n=len(its), eligible=sum(not c for c in codes),
                            code_counts={c: sum(c in cc for cc in codes) for c in ["TOKENIZATION", "BASELINE_INELIGIBLE", "BYSTANDER_DESTABILIZES", "C2_SPAN", "MATCH_INFEASIBLE", "ACTIVE_UNITS"]},
                            matching_fail_rate=sum(any(x in cc for x in ("MATCH_INFEASIBLE", "ACTIVE_UNITS")) for cc in codes) / len(its))
        return out
    R["exclusions_by_layer"] = {L: summarize(L) for L in done}
    stage("exclusions")

    # ---------------- realized interventions at L* (norm-tolerance verification; D recorded, NOT used for any selection)
    L = Lstar; W = sae_loader(L)
    We, be, th, Wd = (torch.tensor(W[k], device=device, dtype=torch.float32) for k in ("W_enc", "b_enc", "threshold", "W_dec"))
    def hook(P, lat, alpha, rec):
        def f(m, i, o):
            h = out_t(o); x = h[0, P, :]; pre = x @ We[:, lat] + be[lat]; a = pre * (pre > th[lat])
            delta = -alpha * (a @ Wd[lat]); h[0, P, :] = x + delta; rec["realized"] = float(delta.norm()); return o
        return f
    realized = []
    for it in items:
        mt = it["match"][L]
        if "MATCH_INFEASIBLE" in mt["codes"] or not per_layer[L][it["kind"]]["latents"] or it["tid"] is None: continue
        K = it["kind"]; S = per_layer[L][K]["latents"]; C1 = per_layer[L][K]["c1_latents"]; C2 = per_layer[L][BYSTANDER]["ranked"][:len(S)]
        N = mt["N_star"]; row = dict(family=it["family"], idx=it["idx"], N_star=N)
        for name, ids, P, lat, n, Lref in [("S", it["_ids"], it["P"], S, mt["n_S"], it["L"]), ("C1", it["_ids"], it["P"], C1, mt["n_C1"], it["L"]),
                                           ("S2", it["_ids2"], it["P2"], S, mt["n_S2"], it["L_c2"]), ("C2", it["_ids2"], it["Pc"], C2, mt["n_C2"], it["L_c2"])]:
            rec = {}; alpha = N / n
            tid = it["tid"] if name in ("S", "C1") else tgt(it["prompt_c2"], G4.FAMILIES[it["family"]]["target"])
            o = fwd(ids, [(L, hook(P, lat, alpha, rec))]); Lv = lo(o.logits[0, -1], tid)[0]
            row[name] = dict(alpha=alpha, realized=rec["realized"], tol_ok=abs(rec["realized"] - N) / N <= TOL, D=Lref - Lv)
        row["Delta1_exploratory"] = row["S"]["D"] - row["C1"]["D"]; row["Delta2_exploratory"] = row["S2"]["D"] - row["C2"]["D"]
        realized.append(row)
    R["realized_at_L_star"] = realized
    R["tolerance_all_ok"] = all(r[n]["tol_ok"] for r in realized for n in ("S", "C1", "S2", "C2")) if realized else None
    R["items"] = [{k: v for k, v in it.items() if not k.startswith("_")} for it in items]
    stage("realized_interventions")
    return R


try:
    import modal
    app = modal.App("alignedlab-exp3-g5-verification")
    image = (modal.Image.debian_slim(python_version="3.12").pip_install(*G4.PINS)
             .env({"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "HF_HUB_ENABLE_HF_TRANSFER": "0"})
             .add_local_python_source("g4_probe"))
    vol = modal.Volume.from_name("kant-hf-cache", create_if_missing=True)

    @app.function(image=image, gpu=GPU, timeout=TIMEOUT_S, volumes={"/root/.cache/huggingface": vol},
                  secrets=[modal.Secret.from_name("huggingface-secret")])
    def verify():
        import torch, subprocess, numpy as np
        t0 = time.time(); logs = []
        def log(s): print(s, flush=True); logs.append(s)
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        from huggingface_hub import snapshot_download, hf_hub_download, list_repo_files
        path = snapshot_download(MODEL_ID, revision=REV)
        from transformers import AutoTokenizer, AutoModelForCausalLM
        tok = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32, attn_implementation="eager").to("cuda").eval()
        log(f"model ready {time.time()-t0:.0f}s")
        files = list_repo_files(SAE_REPO); chosen = {}
        def sae_loader(L):
            if L not in chosen:
                fs = [f for f in files if f.startswith(f"layer_{L}/width_16k/") and f.endswith("params.npz")]
                chosen[L] = min(fs, key=lambda s: (abs(int(re.search(r"average_l0_(\d+)", s).group(1)) - 70), s))
            p = hf_hub_download(SAE_REPO, chosen[L]); return dict(np.load(p))
        R = run_verify(model, tok, "cuda", log, sae_loader, time_left=lambda: TIMEOUT_S - 90 - (time.time() - t0))
        vol.commit()
        rt = time.time() - t0
        out = dict(status="EXPLORATORY G5 VERIFICATION — G4 materials only; not evidence for P1/P2/P3",
                   identity=dict(model_id=MODEL_ID, revision=REV, commit_hash=getattr(model.config, "_commit_hash", None),
                                 sae_repo=SAE_REPO, sae_files=chosen, sae_repo_files_listed=len(files)),
                   env=dict(device=torch.cuda.get_device_name(0), torch=torch.__version__, hf_token_injected=bool(os.environ.get("HF_TOKEN")),
                            pip_freeze=subprocess.run(["pip", "freeze"], capture_output=True, text=True).stdout.split("\n")),
                   results=R, derived_probe_only=derived_manifest(), g4_probe_only=G4.probe_only_manifest(),
                   runtime_s=round(rt, 1), gpu_cost_usd_estimate=round(rt / 3600 * GPU_USD_PER_H, 3), logs=logs)
        return json.dumps(out, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)).encode()

    @app.local_entrypoint()
    def main():
        t = time.time(); b = verify.remote()
        os.makedirs("modal_out/g5v", exist_ok=True); open("modal_out/g5v/g5v_results.json", "wb").write(b)
        d = json.loads(b)
        print(f"DONE in {time.time()-t:.0f}s wall; remote runtime {d['runtime_s']}s; est GPU cost ${d['gpu_cost_usd_estimate']}")
        print("errors:", d["results"].get("errors")); print("wrote modal_out/g5v/g5v_results.json")
except ImportError:
    pass
