# G5 SPEC FREEZE — Kantian Positional Logic Test (DRAFT for PI approval)

**Date:** 2026-11-21 · **Drafted by:** Claude · **Owner / approver:** Ali · **Authority:** Protocol V1.1 (`protocol/experimental_protocol_v1.1.md`), fidelity record, PI Decisions 1–4, G5v report, G5r repair spec, dataset v1.

**Status legend:** **FROZEN** = already PI-approved, restated here. **RULE** = a fixed procedure is frozen; its numerical outcome is produced by G5r on discovery/validation data only, then recorded without change. **PROPOSED** = needs Ali's yes/no in this document.

**Freeze principle (PI decision 2026-11-21):** intervention layer, feature-set size and strength are frozen as *selection rules*, not as numbers. They are chosen only from discovery and validation data, never from evaluation data or causal-target log-odds. If the rules find no valid configuration, the experiment reports P2/P3 as **inconclusive**; nothing is re-tuned.

---

## 1. Model, tokenizer, environment — FROZEN

`google/gemma-2-2b` @ `c5ebcd40d208330abc697524c919956e692655cf` (model and tokenizer). SAEs: `google/gemma-scope-2b-pt-res`, width 16k, per layer the release with mean L0 closest to 70 (exact files logged per run). Modal, NVIDIA A10G, fp32, eager attention; torch 2.5.1, transformers 4.46.3, numpy 1.26.4, huggingface-hub 0.26.5. Weights stay on Modal. Seed 20261121 unless stated.

## 2. Prompt families and inclusion rules — FROZEN / RULE

Families: physical_impact (raw hen's egg, aluminium beverage can, car windshield), heating (corn kernel, raw egg white, beeswax block), cutting (parchment, balsa strip, acrylic sheet); 6 property-matched strata per family. **Confirmatory family: physical_impact (Decision 4). Heating and cutting: secondary, reported separately.**

Causal item inclusion (Decision 4 + B6), applied on the intact model before any intervention:
target and every alternative are single Gemma tokens appended to the prompt; target p ≥ 0.10; target rank ≤ 5; inserting/removing the bystander shifts target log-odds by ≤ 1.0; patient and bystander each named exactly once. Excluded items are logged with reason codes, never replaced.

## 3. Feature-discovery and validation datasets — FROZEN

`datasets/kantian_v1/` (SHA256SUMS): **discovery v1.2** (492, sha256 `65c614bd47f7bfeb…`), **validation v1.1** (492, `67dd259bd750d2c3…`). Templates disjoint across splits; no G4 overlap; no kind-noun leaks; audit clean except 38 reviewed parser artifacts.
C2 bystander (PI decision 2026-11-21): one fixed kind per family in every causal prompt — impact **umbrella**, heating **wristwatch**, cutting **teapot** — with 10 members + 10 property-matched non-members per kind in discovery and in validation.

## 4. P1 identity contrasts and thresholds — RULE (thresholds PROPOSED in G5r)

Contrasts: members vs property-matched non-members sharing head nouns; noun-free descriptions; ≥2 grammatical roles; leave-one-role-out. Pronoun positions do **not** count as non-lexical evidence.
Validated Substance at a layer requires **all** of: noun AUC ≥ 0.85; worst property-matched stratum AUC ≥ 0.75; description AUC ≥ 0.75; leave-one-role-out AUC ≥ 0.75 for every role; lexical gap (noun − description AUC) ≤ 0.20; ≥10 items per stratum. Missing any with adequate counts → identity not established at that layer; inadequate counts → inconclusive. **[PROPOSED — Ali: freeze these numbers?]**

## 5. Held-out evaluation data — FROZEN

`evaluation/v1.1` (1,152 records, sha256 `e0c02030eaae8007…`; impact 96 causal items in **24 distinct events**). Never loaded by any selection or verification code (loader refuses `evaluation`); first opened at G10. Template, bundle and near-duplicate disjointness verified.

## 6. Feature identification and downstream localization — RULE

Per layer 2–14 and per kind: greedy SAE latent order (≤8), each step maximizing the worst-stratum AUC, from the top 64 latents (discovery data). Candidate sets k ∈ {1, 2, 4, 8}.
Downstream (P3 feasibility, discovery causal items only, no target log-odds): for layers T ≥ L+2, per KV head group, attention final-token→patient ≥ 0.10 and information-use ratio (Substance vs C1 output change) ≥ 1.5.

## 7. Intervention layer, site, strength, matching — RULE

Operator: SAE latent ablation at the patient-noun token span, residual stream at layer L. Feature-set size: smallest k that passes R1 efficacy (downstream removal r ≥ 0.67, amnesic AUC ≤ 0.70, role-AUC drop ≤ 0.05). Layer: among layers 2–14 passing R1, R2 (=§4) , R3 and P3 feasibility for **every** impact patient kind, the one with the highest worst-case R2 margin; ties → shallower. None → "no valid layer" → P2/P3 inconclusive.
Strength/matching (Decision 2, FROZEN): one common change norm N* per item; control boost ≤ ×2; Substance at ≥ 0.5 of natural strength; realized ‖Δ‖ within ±10%; equal k; same layer, hook, operator. Active-feature ratio measured and logged, never an exclusion. ‖Δ‖ and ‖Δ‖/‖h‖ logged per position.

## 8. C1 and C2 implementations — FROZEN (C2 bystander per §3)

C1: k latents active at the site on discovery data, identity-unselective (|AUC − 0.5| < 0.05), disjoint from Substance, closest contribution size; applied at the patient span.
C2: the bystander kind's own greedy set truncated to k (bug-fixed rule), applied at the bystander noun span. C2 is **not** position-matched and is never described as such; position, role and salience differences are logged.
Control validity (R3): valid-pair retention ≥ 75%; C1 removes ≤ 0.10 target identity; C2 removes ≥ 0.67 bystander identity; role drop ≤ 0.05. A comparison failing a required check is flagged non-confirmatory.

## 9. Matching tolerance, diagnostics, failure handling — FROZEN

±10% norm tolerance; per item log: layer, hook, k, latent IDs, ‖Δ‖, ‖Δ‖/‖h‖, token positions, boost factor, ρ, active-feature ratio, exclusion code. Items failing matching are excluded from the confirmatory test and counted; exclusion rates reported per condition.

## 10. P3 pathway evidence — RULE + PROPOSED

Necessity claim requires (i) validated Substance at L and an identified downstream head group at T (§6); (ii) Substance ablation changes that group's output more than C1 (ratio ≥ 1.5); (iii) **pathway-specific restoration**: patching the clean layer-L patient-span activation into that head group's input only, under the Substance ablation, recovers ≥ 50% of the group's output change and of D. **[PROPOSED — 50% recovery threshold]** If (iii) cannot be implemented or fails isolation checks → P3 reported **inconclusive/unestablished**, never inferred from P2.

## 11. Measurements — FROZEN

Primary (B6): D = L_intact − L_intervened, L = log(p/(1−p)) of the target token. Comparisons Δ1 = D_Substance − D_C1, Δ2 = D_Substance − D_C2, paired per item. Secondary (B7): feature selectivity/generalization, downstream activation change, individual vs combined entity interventions, restoration fraction, alternative-outcome shifts, property/variation sensitivity. Secondary measures cannot replace the primary.

## 12. Sample size — PROPOSED

Fixed by the dataset, no optional stopping: all eligible impact evaluation items (max 96). **Causal items are clustered by event** (96 items = 24 distinct events × wording variants; `template_id` = event), so the unit of inference is the event. **Minimum for a conclusive primary result: ≥ 30 eligible items covering ≥ 12 distinct events**; otherwise inconclusive. A planning estimate from G5r variability (discovery items only) is reported before G10 but cannot change n. G5v gave no usable variability estimate (|D| ≤ 0.05).

## 13. Effect size, uncertainty, multiplicity — PROPOSED

Per comparison: average the paired difference within each event, then take the mean over events (each event weighted equally); 95% CI by bootstrap over events (10,000 resamples, logged seed); one-sided sign-flip test on event means (exact ≤ 24 events, else Monte Carlo 100,000). Minimum effect of interest: 0.5 log-odds. Both Δ1 and Δ2 must pass (intersection–union), so no correction on the primary. Secondary families: same analysis, Holm across the four secondary tests.

## 14. Outcome criteria — PROPOSED (mapping B8)

| Outcome | Rule |
|---|---|
| **P2 supported** | Δ1 and Δ2 both: p < 0.05 and mean ≥ 0.5 |
| **P2 not supported** | Either Δ1 or Δ2 has 95% CI upper bound < 0.5 (with the §12 minimum met) |
| **Inconclusive** | Anything else; or no valid layer; or §12 minimum not met; or Substance not validated (§4) |
| **P1** | Reported separately per §4 |
| **P3** | Reported separately per §10; P2 alone never establishes P3 |

## 15. Expenditure and engineering contract — PROPOSED

Cap remains $30 (Modal). Spent ≈ $0.09. G5r verification: hard cap $0.37 (20-min timeout). Proposed ceiling for G8 pilot + G10 full run: **$5**, any excess escalated. Claude implements (G6), Gemini + Grok review code (G7) against Protocol V1.1 and this document.

## 16. Known deviations and risks (disclosed)

1. Layer/k/strength frozen as rules, not numbers (§ principle) — PI decision.
2. No separate "P3-discovery" split exists; P3 feasibility uses discovery causal items.
3. Validation/evaluation stimuli were authored by Claude from templates (style shift from discovery).
4. g5r_probe.py must be adapted from the vase/ice/rope/clock schema to dataset v1 — engineering only, no rule change.
5. Gemma tokenization check pending (on the new dataset versions).
6. Causal items are event-clustered (impact: 24 events); inference is by event (§12–13). Generator limitation, disclosed in the dataset card.

---

## Sign-off

☐ Ali approves this specification as the G5 freeze (with PROPOSED items accepted or amended as noted).
Date / time: ______ · Commit: ______
