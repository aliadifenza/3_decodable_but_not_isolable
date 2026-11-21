"""Analysis of EXPLORATORY E4 ("already travelled?") with the matching E3 conditions for comparison. No model runs.
Usage: python3 e4_analyze.py results/e4_results.json results/e_results_layers_*.json
Writes results/e4_summary.json, results/e4_summary.md, results/fig_e4.png. Medians over 12 kinds."""
import json, sys, os, statistics as st

e4 = json.load(open(sys.argv[1])); out = os.path.dirname(sys.argv[1]) or "."
assert e4["status"].startswith("EXPLORATORY E4") and not e4["results"]["errors"]
E3 = {}
for p in sys.argv[2:]:
    for L, LO in json.load(open(p))["results"]["layers"].items(): E3[int(L)] = LO
R = {int(L): LO for L, LO in e4["results"]["layers"].items()}; Ls = sorted(R)
fin = lambda v: [x for x in v if x == x]
def med(v): v = fin(v); return round(st.median(v), 3) if v else None
conds = list(next(iter(R[Ls[0]]["kinds"].values())))
rows = {}
for c in conds:
    rows[c] = {L: dict(clean_auc_at_L_final=med([k[c]["clean_auc_now"] for k in R[L]["kinds"].values()]),
                       r_now=med([k[c]["r_now"] for k in R[L]["kinds"].values()]),
                       r_down=med([k[c]["r_down"] for k in R[L]["kinds"].values()]),
                       clean_auc_down=med([k[c]["clean_auc_down"] for k in R[L]["kinds"].values()]),
                       kl=med([k[c]["kl_mean"] for k in R[L]["kinds"].values()])) for L in Ls}
e3 = {c: {L: med([i["conditions"][c]["r_down"] for i in E3[L]["kinds"].values()]) for L in Ls} for c in ("E3_meandiff_all", "E3_leace_all", "E3_randdir_all")}
S = dict(label=e4["status"], cost_usd_est=e4["gpu_cost_usd_estimate"], runtime_s=e4["runtime_s"], layers=Ls, e4=rows, e3_same_layers=e3)
json.dump(S, open(os.path.join(out, "e4_summary.json"), "w"), indent=1)
md = [f"# {e4['status']}", "", f"Layers {Ls}; 12 kinds; validation items. Cost ≈ ${e4['gpu_cost_usd_estimate']:.2f}.", "",
      "Median over kinds. `clean AUC @L final` = identity readable at the final token at the intervention layer, BEFORE any edit. "
      "`r_now` = normalized probe-AUC reduction at layer L, final token; `r_down` = same at L+4.", "",
      "| condition | " + " | ".join(f"L{L}" for L in Ls) + " |", "|---|" + "---|" * len(Ls)]
md.append("| clean AUC @L final | " + " | ".join(str(rows[conds[0]][L]["clean_auc_at_L_final"]) for L in Ls) + " |")
for c in conds:
    md.append(f"| {c} r_now | " + " | ".join(str(rows[c][L]["r_now"]) for L in Ls) + " |")
    md.append(f"| {c} r_down | " + " | ".join(str(rows[c][L]["r_down"]) for L in Ls) + " |")
for c, v in e3.items(): md.append(f"| (E3) {c} r_down | " + " | ".join(str(v[L]) for L in Ls) + " |")
open(os.path.join(out, "e4_summary.md"), "w").write("\n".join(md) + "\n")

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
C1, C2, C3, C4, INK, INK2, SURF = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "figure.facecolor": SURF, "axes.facecolor": SURF})
fig, ax = plt.subplots(figsize=(7.5, 4.3))
for y, col, mk, ls, lab in ((e3["E3_meandiff_all"], C1, "o", "--", "E3: erase at description only (mean-diff.)"),
                            ({L: rows["E4a_spanfinal_meandiff"][L]["r_down"] for L in Ls}, C1, "o", "-", "E4a: erase at description + final token (mean-diff.)"),
                            ({L: rows["E4a_spanfinal_leace"][L]["r_down"] for L in Ls}, C3, "D", "-", "E4a: same, LEACE"),
                            ({L: rows["E4a_spanfinal_randdir"][L]["r_down"] for L in Ls}, C4, "^", "-", "E4a: random direction (control)")):
    ax.plot(Ls, [y[L] for L in Ls], color=col, marker=mk, ls=ls, lw=2, ms=8, label=lab)
ax.axhline(1.0, color=INK2, lw=1, ls=":"); ax.text(2, 1.02, "complete removal (r = 1)", color=INK2, fontsize=8)
ax.axhline(0, color=INK2, lw=0.8); ax.set_ylim(-0.1, 1.12); ax.set_xticks(Ls)
ax.set_xlabel("intervention layer L (probe reads final token at L+4)"); ax.set_ylabel("normalized probe-AUC reduction, r (median)")
ax.set_title("E4: erasing at the reading position too removes more, but not all", loc="left", color=INK, fontsize=10)
ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", color="#e4e3df", lw=0.8)
ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.92), frameon=False, fontsize=8, labelcolor=INK)
fig.tight_layout(); fig.savefig(os.path.join(out, "fig_e4.png"), dpi=160)
print(json.dumps({c: {L: rows[c][L]["r_down"] for L in Ls} for c in conds}))
