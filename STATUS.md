# STATUS — Kantian Positional Logic (AlignedLab Exp. 3)

**Single source of truth.** Read this before writing any brief or starting any work. Claude updates it after every step; anything not listed here does not exist yet.

**Last updated:** 2026-11-22 12:40 CET · Claude · see git log

## Current state — SUBMITTED (2026-11-22 ~12:34 CET, 1 day 26 min before the deadline)

**G5 SPEC FREEZE: FROZEN** (PI 2026-11-21 18:32 CET; Amendment 1 18:47 CET). **Scientific authority: Protocol V1.1 + `protocol/G5_SPEC_FREEZE.md`** (V1.2 is a consolidated reissue; the freeze document governs on conflict).

**G5r verification: DONE → NO VALID LAYER.** Under the frozen §7/§14 rules, **P2 and P3 are INCONCLUSIVE**. The confirmatory pipeline (G6 ENGINEER → G10 FULL RUN) is **closed, not executed** — documented NO-GO by the frozen rule. **The evaluation set (v1.1) remains sealed and unused.**

**Submitted:** "Decodable but Not Isolable" is entered in Mechanistic Interpretability Hackathon 5: Autointerp (itch.io jam, host victorlf4), submitted by Narintea; downloads: report PDF + slides PDF; cover image set. Rating opens 2026-11-23 13:00. Verified by PI screenshot of the jam submission page.

**Done:** frozen G5r (no valid layer; P2/P3 inconclusive; evaluation sealed) · exploratory E1–E4 · G12 reviews (Gemini + DeepSeek independent, Qualam internal audit) and consolidation · final PI-revised report in `report/` · README led by the paper title · repo description · itch.io cover · 13-slide presentation (private Claude Slides artifact, PI-owned; includes roles and philosophy-to-measurement slides) · compute ≈ $2.21 of $30 for reported model runs.

**Open, optional:** §3.2 provenance wording for the discovery set (lab notes suggest it was assembled from several AI generator streams; PI to confirm).

### Gate report — G5r (universal format)

- **DONE:** G5r run on discovery v1.2 + validation v1.1 under the frozen rules, layers 2–14, 12 kinds (9 patients + 3 bystanders). Two runs merged (`merge_g5r.py`; per-layer results independent).
- **EVIDENCE:** `g5_verification/results/g5r_results_layers_2-10.json` (Modal ap-HjkhUsCPA7dYzyRHoiX4Qw, 937.7 s, ≈ $0.287; stopped at the 20-min budget), `g5r_results_layers_11-12-13-14.json` (ap-wVzoKopHVYB4wQwhExVCx7, 500.6 s, ≈ $0.153), merged `g5r_merged.json` (commit ab13994). Input `g5_verification/g5r_input.jsonl` (sha256 0bb8d11c…).
- **RESULT [HIGH]:**
  - **Identity is strongly represented:** linear readout at the ablation site separates each kind from property-matched look-alikes at AUC ≈ 0.93–1.0 at every layer.
  - **R1 efficacy failed everywhere, 0/156 layer×kind:** ablating up to k = 8 selected SAE latents at the referent position removes ≈ 0 identity information downstream (removal r ≈ 0.00 vs required ≥ 0.67; amnesic AUC ≈ clean AUC; ‖Δ‖/‖h‖ ≈ 3–7%; next-token KL ≈ 0).
  - **R3 controls** failed almost everywhere (few discovery causal items pass baseline eligibility: beverage can 0/8, egg 2/8, windshield 5/8).
  - **P1 (R2, held-out validation):** passed for some kind×layer pairs only — egg 6, 8, 14; can 7, 12; windshield 3, 5, 6, 8, 9, 12 — never all three impact kinds at one layer.
- **DECISION:** frozen rule → **no valid layer → P2/P3 inconclusive**; confirmatory run not executed (rule-determined, not a choice).
- **NEXT:** PI decision on exploratory E1–E3; then G11–G13.
- **BLOCKER:** none beyond that decision.

### Interpretation guard (agreed in conversation, 19:26–19:31)

The G5r result is **consistent with** the PI's working definition of identity ("organized togetherness; no single part essential") but is **not evidence for it**: a failed intervention is a null result of the tool; mundane explanations remain (identity re-readable from other tokens of the description; SAE reconstruction error; only one position ablated); and "identity is distributed" was not the pre-registered hypothesis. Honest claim now: *the model clearly represents identity; sparse local SAE ablation cannot remove it, so the Substance → Causality dependency could not be tested.*

## What exists (verified, in this repo)

| Item | Path | Commit |
|---|---|---|
| Protocol V1.1 (PI-approved) + fidelity record | `protocol/` | a7de87b |
| **G5 spec — FROZEN** (+ Amendment 1) | `protocol/G5_SPEC_FREEZE.md` | 99f093c, 71670bc |
| Protocol V1.2 (consolidated reissue; V1.1 kept as cited authority) | `protocol/experimental_protocol_v1.2_G5_frozen.docx` (+ `.md`) | d5a3729 |
| G4 feasibility probe | `g4_probe/` | a7de87b |
| G5v verification (interventions too weak) + C2 bug fix | `g5_verification/2026-11-21_g5_verification.md`, `..._c2_bugfix.md` | 088b6d3, e3978f3 |
| G5r design, adapter, tests, merge | `g5_verification/` (`g5r_probe.py`, `g5r_from_dataset.py`, `merge_g5r.py`, `test_g5r_*`) | 2e90e26, 88b9c87, de20455 |
| **G5r results (no valid layer)** | `g5_verification/results/g5r_*.json` | ab13994 |
| Dataset v1 (superseded versions kept) | `datasets/kantian_v1/` | b0e61bb |
| **Dataset current: discovery v1.2 (492) · validation v1.1 (492) · evaluation v1.1 (1,152, sealed)** | `datasets/kantian_v1/` (`DATASET_CARD.md` §8) | 2602ac3 |
| Tokenizer check results | `datasets/kantian_v1/audit/tokcheck_results.json` | 934fb43 |

Discovery = provided set (v1.1: 231 logged fixes; v1.2: fixed bystander + 60 bystander records). Validation/evaluation generated by Claude from templates. Causal items are event-clustered (eval impact 24 events; after tokenization 90 items / 22 events).

**Do not regenerate the dataset.** The Modal/Qwen cutting generator is superseded.

## Open decisions (PI)

1. ~~Exploratory follow-up E1–E3~~ → **APPROVED 19:38–19:39** (see below). No open PI decisions until results.

### Exploratory E1–E3 (labelled EXPLORATORY; never feeds P2/P3)

- **Code:** `exploratory/e_probe.py`, tests `exploratory/test_e_local.py` (7 tests PASS on tiny CPU model + real G5r input: evaluation guards, exact LEACE property, determinism, **G5r replication — k ≤ 8 latent order and k = 8 R1 metrics identical to g5r_probe**, nested sets, controls disjoint, D path).
- **Scope (PI-approved $3 version, cap $10):** layers 2–14, 12 kinds, discovery + validation; evaluation refused by loader.
- **Conditions (26 per layer × kind):** E1 k ∈ {8,16,32,64,128} identity latents (first 8 = G5r greedy; continued greedily from top-512 pool — exploratory extension); E2 last token vs all tokens of reference span (crossed with E1 and E3); E3 mean-difference projection and LEACE (10% shrinkage) on the full residual incl. SAE error; controls: same-k random active latents, covariance-matched random direction.
- **Measures:** = R1 (downstream r, amnesic AUC, role-AUC drop, ‖Δ‖/‖h‖, KL) + item-bootstrap 95% CIs. Frozen R1 thresholds used only as a descriptive bar.
- **Optional D comparison:** discovery causal items only, for conditions meeting the R1 bar; C1 = random control, C2 = same method on the bystander; **no magnitude matching (deviation from Decision 2, stated)**; event-cluster means + bootstrap CI.
- **Cost guard:** 2 runs (layers 2–8, 9–14), 45-min hard timeout each → worst case ≈ $1.65 total.
- **RUN DONE (2026-11-21 ~20:30):** Modal ap-xz03ZHllg5j0TTRSkrwGdC (layers 2–8, 2167 s, ≈ $0.66) + ap-W1YapGqj2QUqgYZME8cuZr (9–14, 2504 s, ≈ $0.77); all 13 layers, no errors. Raw `exploratory/results/e_results_layers_*.json` (commit 1f25778). Analysis `exploratory/e_analyze.py` → `e_summary.md/json`, `fig_e1e3.png`.
- **RESULT (exploratory):** E1/E2 — up to 128 identity latents, last or all reference tokens: downstream removal r ≈ 0 (median ≤ 0.005); random active latents with ~5× larger edits also ≈ 0 [HIGH]. E3 — whole-direction erasure on the full residual: median r 0.26 (mean-difference, all tokens) / 0.18 (LEACE), peak L6 0.55; random direction ≈ 0.01; role drop ≈ 0 [MED]. 9/312 E3 cases meet the R1 bar (wide CIs). D comparison: only windshield (5 items) and acrylic (3) usable, effects ≤ 0.1 log-odds → no conclusion [HIGH].
- **Caveats found in analysis (for reviewers):** amnesic AUC = 0.50 under E3 is by construction (equal class means ⇒ logistic optimum w = 0); identity may already be copied to the final token before layer L (interventions edit only the span) — untested; r has small denominators at L2, L5–L8.

### E4 "already traveled?" (exploratory) and G12 reviews

- **E4 run:** Modal ap-GK63sOtfiHMoseAls3RcmP, 820 s, ≈ $0.25, layers 2,4,…,14, no errors (commit 217d657). Summary `exploratory/results/e4_summary.md`, figure `fig_e4.png`.
- **Result:** identity already readable at the final token at layer L before any edit (AUC 0.71–0.92) [HIGH]. Erasing at span + final token: r at L 0.76–1.15; r at L+4 0.22–0.65 (mean-diff.), 0.19–0.44 (LEACE); random ≈ 0 [MED]. E4b (one pooled direction at all tokens) failed as a test (does not remove identity at the final token) [HIGH].
- **G12 reviews:** Gemini (independent), DeepSeek (independent; substituted for Grok, unavailable), Qualam (internal audit, not independent). All agree claims 2, 6; all judge 3–4 overstated; claim 5 moved to Discussion as a V2 hypothesis (PI 21:05). Consolidation `review/G12_consolidation_2026-11-21.md`; v2 package with E4 addendum `review/G12_review_package_v2_2026-11-22.md`.
- **Report draft:** `report/AlignedLab_Exp3_Kantian_Positional_Logic_report.docx` (Apart template structure, Baskerville, US English; build script `report/build_report.js`). Open items for PI: confirm discovery-set provenance wording (§3.2), confirm hackathon name in footnote, revise text in own voice (template asks for primarily team-written text).
- **README** rewritten in the house style of 2_epistemic_vacuum.

### 2026-11-22 morning

- README: hackathon name corrected to "Mechanistic Interpretability Hackathon 5: Autointerp", hosted by victorlf4 (Victor Levoso) on itch.io — verified from the jam page (PI-supplied PDF, 10:29). Not an Apart Research event; "(Apart Research)" removed. **Submission window: 2026-11-20 17:00 → 2026-11-23 13:00** (itch.io display time) — the deadline is Monday, not Sunday. Official theme "Autointerp Evals" is a suggestion, not a rule. Judging: default itch.io voting. G2 reviewers (Gemini, Grok) and the dataset-generation history added. No engineering changes; no runs.
- **Final report committed** (PI-revised, 2026-11-22): `report/AlignedLab_Exp3_Kantian_Positional_Logic_report.docx`, replacing the Claude draft at the same path. Claude's edits on PI instruction ("do all the updates you see fit"): footnote hackathon name/host; repo URL; appendix cross-references after the PI's reordering (A philosophy, B workflow, C what went wrong, D additional results, E context); "overtime" → "over time"; removed the line "We operationalized identity as it follows:" (the definition is philosophical, the operationalization is A.2); "creative overhead" → "creative dimension"; G2 design review by Gemini + Grok in Table A1/B roles; new item C.7 on the dataset-generation history (Adaption Labs export, Grok capacity, undelivered Gemini ZIP, Qwen2.5-7B Modal generator); compute note that a separate dataset-generator test's cost was not recorded. itch.io cover image `report/cover.png`. Presentation deck: private Claude Slides artifact (owner: PI).
- Repository renamed to `aliadifenza/3_kantian_positional_logic_test` (PI, 2026-11-22 ~10:33); README notes the rename; report link updated in `report/build_report.js` (the PI's edited report must use the new URL too).
- Lab notes (private AlignedLab repo, `docs/lab_notes/3_kantian_positions/`) read for the report: they add the G2 review (Gemini + Grok, CONDITIONAL GO) and the dataset-generation failures (Adaption Labs export rejected; Grok capacity exhausted; Gemini-reported ZIP never delivered; separate Modal Qwen2.5-7B cutting generator, cost unverified).

## PI decisions today (2026-11-21, CET)

18:01 one fixed C2 bystander per family · 18:08 freeze layer/k/strength as rules · 18:32 **G5 SPEC FREEZE** · 18:33 G5r run authorized ($0.37) · 18:47 **Amendment 1** (inclusion by target token only) · 19:10 G5r cap raised to $0.55 · 19:15 Protocol V1.2 pushed with four non-scientific edits · 19:23 direction B (exploratory follow-up) accepted in principle · 19:34 **close this phase; continue E1–E3 discussion in a new session.** · 19:38 **E1–E3 approved, $3 version (all layers, random controls, CIs, E1×E2, optional D); hard spend cap $10** · 19:39 go. · 20:47 standing rule: spending within €30 total needs no separate permission · 20:52–21:00 DeepSeek substitutes for Grok as independent reviewer; Qualam reported as internal audit · 21:05 claim 5 (organized togetherness) moved from claims to Discussion as V2 hypothesis · 21:23 report + README + figures requested.

## Spending

Cap $30 (Modal). Spent (estimates from run logs): G4 ≈ $0.056 · G5v ≈ $0.029 · tokenizer < $0.01 · G5r ≈ $0.44 (0.287 + 0.153) · E1–E3 exploratory ≈ $1.43 (0.662 + 0.765) · E4 ≈ $0.25. **Total ≈ $2.21.** Pilot + full run ceiling ($5, frozen §15) unused — confirmatory run not executed.

## Rules

- PI decides science and spending; Qualam writes briefs **from this file**; Claude executes and updates this file.
- No regenerating, re-running or re-deciding anything listed as done without a PI decision. Exploratory work is labelled as such and never feeds the confirmatory result.
