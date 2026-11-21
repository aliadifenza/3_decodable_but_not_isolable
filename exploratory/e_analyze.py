"""Analysis of EXPLORATORY E1-E3 results (no new model runs). Usage: python3 e_analyze.py results/e_results_*.json
Writes results/e_summary.json, results/e_summary.md, results/fig_e1e3.png. Medians are over all 12 kinds (9 patients + 3 bystanders)."""
import json, sys, os, statistics as st

LABEL = "EXPLORATORY E1-E3 — not confirmatory; discovery/validation only; evaluation sealed"
paths = sys.argv[1:]; out = os.path.dirname(paths[0]) or "."
layers, cost, rt = {}, 0.0, 0.0
for p in paths:
    d = json.load(open(p)); cost += d["gpu_cost_usd_estimate"]; rt += d["runtime_s"]
    assert d["status"] == LABEL and not d["results"]["errors"], (p, d["results"]["errors"])
    for L, LO in d["results"]["layers"].items(): layers[int(L)] = LO
Ls = sorted(layers); conds = list(next(iter(layers[Ls[0]]["kinds"].values()))["conditions"])
fin = lambda v: [x for x in v if x == x]
def vals(c, key, L=None): return fin([i["conditions"][c][key] for LL in ([L] if L else Ls) for i in layers[LL]["kinds"].values()])
def med(v): return round(st.median(v), 3) if v else None
def iqr(v): q = st.quantiles(v, n=4) if len(v) > 1 else [float("nan")] * 3; return [round(q[0], 3), round(q[2], 3)]

table = {c: dict(n=len(vals(c, "r_down")), met_R1_bar=sum(i["conditions"][c]["R1_criteria_met"] for L in Ls for i in layers[L]["kinds"].values()),
                 r_down_median=med(vals(c, "r_down")), r_down_iqr=iqr(vals(c, "r_down")), r_down_max=round(max(vals(c, "r_down")), 3),
                 amnesic_median=med(vals(c, "amnesic_auc")), role_drop_median=med(vals(c, "role_auc_drop")),
                 rel_norm_median=med(vals(c, "rel_norm_mean")), kl_median=med(vals(c, "kl_mean"))) for c in conds}
per_layer = {c: {L: med(vals(c, "r_down", L)) for L in Ls} for c in conds}
met = [dict(layer=L, kind=K, cond=c, r_down=round(m["r_down"], 3), ci95=[round(x, 2) for x in m["r_down_ci95"]], role_drop=round(m["role_auc_drop"], 3))
       for L in Ls for K, i in layers[L]["kinds"].items() for c, m in i["conditions"].items() if m["R1_criteria_met"]]
D = [dict(layer=L, kind=K, cond=c, **{k: y["eligible"].get(k) for k in ("n_items", "n_events", "delta1_mean", "delta1_ci95", "delta2_mean", "delta2_ci95")})
     for L in Ls for K, x in layers[L]["D_exploratory"].items() for c, y in x.items()]
S = dict(label=LABEL, layers=Ls, cost_usd_est=round(cost, 3), runtime_s=round(rt, 1), conditions=table, r_down_median_by_layer=per_layer,
         conditions_meeting_R1_bar=met, D_exploratory=D, down_clean_auc_median_by_layer={L: med(vals("E1_k8_last", "down_clean_auc", L)) for L in Ls})
json.dump(S, open(os.path.join(out, "e_summary.json"), "w"), indent=1)

md = [f"# {LABEL}", "", f"Layers {Ls[0]}–{Ls[-1]}, 12 kinds, validation items. Cost ≈ ${cost:.2f} (runtime {rt / 60:.0f} GPU-min).", "",
      "| condition | met R1 bar (of 156) | median r (IQR) | max r | median amnesic AUC | median role drop | median ‖Δ‖/‖h‖ | median KL |", "|---|---|---|---|---|---|---|---|"]
for c, t in table.items():
    md.append(f"| {c} | {t['met_R1_bar']} | {t['r_down_median']} ({t['r_down_iqr'][0]}, {t['r_down_iqr'][1]}) | {t['r_down_max']} | {t['amnesic_median']} | {t['role_drop_median']} | {t['rel_norm_median']} | {t['kl_median']} |")
md += ["", "## Conditions meeting the R1 bar (descriptive)", ""] + [f"- L{m['layer']} {m['kind']} {m['cond']}: r = {m['r_down']} (95% CI {m['ci95']}), role drop {m['role_drop']}" for m in met]
md += ["", "## Exploratory D comparison (discovery causal items, eligible only)", ""] + [
    f"- L{x['layer']} {x['kind']} {x['cond']}: n = {x['n_items']} items / {x['n_events']} events; Δ1 = {x['delta1_mean']}, Δ2 = {x['delta2_mean']}" for x in D]
open(os.path.join(out, "e_summary.md"), "w").write("\n".join(md) + "\n")

# ---- figure: (A) dose (k) at the best E1 placement vs random; (B) r by layer for whole-direction erasure vs controls
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
C1, C2, C3, C4, INK, INK2, SURF = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "figure.facecolor": SURF, "axes.facecolor": SURF})
fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
ks = [8, 16, 32, 64, 128]
for c, col, mk, lab in ((lambda k: f"E1_k{k}_last", C1, "o", "E1 identity latents, last token"), (lambda k: f"E1_k{k}_all", C2, "s", "E1 identity latents, all span tokens"),
                        (lambda k: f"RAND_k{k}_all", C4, "^", "random active latents, all tokens")):
    y = [table[c(k)]["r_down_median"] for k in ks]; a.plot(ks, y, color=col, marker=mk, lw=2, ms=8, label=lab)
a.set_xscale("log", base=2); a.set_xticks(ks); a.set_xticklabels(ks); a.set_xlabel("number of SAE latents ablated (k)"); a.set_ylabel("identity removal downstream, r (median)")
a.set_title("E1 dose × E2 place: SAE ablation removes ~nothing", loc="left", color=INK, fontsize=10)
for c, col, mk, lab in (("E3_meandiff_all", C1, "o", "E3 mean-difference erasure"), ("E3_leace_all", C3, "D", "E3 LEACE erasure"),
                        ("E1_k128_all", C2, "s", "E1 k=128 identity latents"), ("E3_randdir_all", C4, "^", "random direction (control)")):
    y = [per_layer[c][L] for L in Ls]; b.plot(Ls, y, color=col, marker=mk, lw=2, ms=8, label=lab)
b.set_xlabel("intervention layer L (all reference-span tokens; probe at L+4, final token)"); b.set_xticks(Ls)
b.set_title("E3 whole-direction erasure: partial removal (peak L6)", loc="left", color=INK, fontsize=10)
for ax in (a, b):
    ax.axhline(0.67, color=INK2, lw=1, ls="--"); ax.axhline(0, color=INK2, lw=0.8)
    ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", color="#e4e3df", lw=0.8)
a.text(8, 0.69, "frozen R1 bar r ≥ 0.67", color=INK2, fontsize=8); a.set_xlim(7, 150); b.set_xlim(1.5, 14.5); a.set_ylim(-0.1, 0.8)
a.legend(loc="center left", bbox_to_anchor=(0, 0.55), frameon=False, fontsize=8, labelcolor=INK); b.legend(loc="upper right", frameon=False, fontsize=8, labelcolor=INK)
fig.suptitle("Exploratory E1–E3 (Gemma-2-2B, validation items; not confirmatory)", x=0.01, ha="left", color=INK, fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(os.path.join(out, "fig_e1e3.png"), dpi=160)
print(json.dumps(dict(cost=round(cost, 3), met=len(met), D=len(D)), indent=0))
