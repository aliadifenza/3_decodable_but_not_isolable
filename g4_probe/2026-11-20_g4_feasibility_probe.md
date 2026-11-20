# G4 — FEASIBILITY PROBE

**Project:** AlignedLab Experiment 3 — Kantian Positional Logic Test (Research Agenda Ch. 2, Exp. 2) · **Date:** 2026-11-20
**Authority:** Protocol V1.1 (unchanged). **Executed by:** Claude (code, analysis); Ali (Modal launch from Mac). **Review:** Qualam.
**Status: EXPLORATORY.** Nothing below is evidence for P1, P2 or P3. Every probe prompt, template and noun is excluded from all G5+ held-out sets.

**Artifacts:**
- Code: `experiments/kantian_positional/g4_probe/g4_probe.py`. It includes the probe-only data lists.
- Local code-path test: `experiments/kantian_positional/g4_probe/test_g4_local.py`.
- Raw results: `experiments/kantian_positional/g4_probe/results/g4_results.json`, sha256 `0568e6b07c737baddeec89f48db3c3c2944e585447586c4e69da384c91c41ca7`. It contains the identity, environment/pip freeze, every measurement and the probe-only manifest with hashes.

## Execution record

- **Command (Mac):** `MODAL_IMAGE_BUILDER_VERSION=2025.06 python3 -m modal run g4_probe.py`
  - The builder-version override is the same workaround used in the Epistemic Vacuum experiment.
  - Without it, the first attempt failed at a client/builder version check before any build or GPU use.
- **Modal app:** `ap-5NyKBjUSyqADi7tZgb1t33`. GPU requested A10G; device reported "NVIDIA A10".
- **Model:** `google/gemma-2-2b`.
  - Revision `c5ebcd40d208330abc697524c919956e692655cf`, confirmed equal to `config._commit_hash`.
  - float32, eager attention.
  - SHA256 of all 10 checkpoint, config and tokenizer files are stored in the JSON.
- **Architecture** [VERIFIED]: 26 layers, 8 query heads / 4 KV heads (2 per group), head_dim 256, d_model 2304, vocabulary 256,000. Final logit softcap 30, attention softcap 50.
- **Pinned stack** [VERIFIED by pip freeze]: torch 2.5.1+cu124, transformers 4.46.3, tokenizers 0.20.3, huggingface-hub 0.26.5, safetensors 0.4.5, numpy 1.26.4.
- **Settings:** TF32 off, deterministic algorithms, seed 20261120.
- **HF secret** injected inside the container [VERIFIED].
- **Timing:**

  | Phase | Time |
  |---|---|
  | Snapshot download | 46 s |
  | Checksums | 11 s |
  | Model load | 9 s |
  | All probe checks | 119 s |
  | **Total remote** | **184 s** |

  This is well inside the 90-min cap.

### Spending ledger (cap: $30 cumulative)

| Item | Estimate before | Recorded |
|---|---|---|
| G3 | — | $0.00 |
| G4 attempt 1 (version error, no build) | $0 | $0.00 |
| G4 run (A10G, 184 s) | ≤ $0.40 (hard cap $0.85) | **$0.056** GPU estimate at $1.10/h; billed figure on Modal dashboard [UNVERIFIED]. Container CPU/memory/image build are additional and small. |
| **Cumulative** | | **≈ $0.06 of $30** |

## Checklist

| # | Check | Result | Evidence and limitations |
|---|---|---|---|
| 1 | Model loads; forward-pass activations | **PASS** | Residual activations [14 × 2304] at every layer; logits finite. Repeat forward is **bitwise identical**. |
| 2 | Measurable correct-target prediction | **PASS** | All three targets are single tokens (`▁shatter`, `▁melt`, `▁snap`); all alternatives too. Baselines (15 prompts): **shatter** p 0.29–0.41, rank 1–2; **melt** p 0.43–0.77, rank 1; **snap** p 0.09–0.18, rank 1–2 (weakest). |
| 3 | Activations recorded and modified | **PASS** | Hooks on decoder-layer outputs at chosen token positions. Layer input at the restoration site was unchanged by downstream hooks (isolation check true in 3/3). |
| 4 | One candidate entity representation | **PARTIAL** | Difference-of-means (DoM) directions per concept and layer; the layer chosen for interventions (L\* by pronoun-position AUC) was **5**. Held-out AUC = 1.00 at every layer: **saturated and largely lexical** (same noun token). Pronoun-position AUC rests on only 3 positive items and is ceilinged; Gemma Scope SAE latents (layer 5) scored 0.39 there. **Nothing here addresses P1**, which needs property-matched, non-lexical contrasts. |
| 5 | One targeted intervention changes a measurable signal | **PASS** | Rank-1 projection of the patient direction at layer 5. D = drop in correct-target log-odds: heating +1.70 to +2.03, cutting +2.37 to +3.25, impact −0.16 to +0.54. |
| 6 | C1 and C2 constructed with magnitude diagnostics | **PASS** (with design caveats) | Same layer, hook, operator (rank-1 projection to non-entity mean) and dim = 1. Actual total L2 matched **exactly** by rescaling. Per-intervention logs: positions, raw and matched L2, scale, per-position L2. C1 = "guitar" direction and a seeded random direction at the patient positions. C2 = "clock" direction at the bystander position. Caveats below (§ Matching). |
| 7 | Noise, variability, expense | **PASS** | Baseline log-odds SD across 15 prompts 0.97, driven mostly by differences between families. Within-family SDs of paired differences: 0.05–0.62 (n = 3–8, very uncertain). Expense: 119 s of GPU for the whole probe. |
| 8 | Failure modes; sample size | **PASS** | See the two sections below. |
| 9 | All probe data exploratory and excluded | **PASS** | `probe_only` manifest in the JSON: 42 prompt/template items with SHA-prefix hashes, plus 40 probe nouns. Rule: exclude from any G5+ held-out set. |
| 10 | GO / CONDITIONAL GO / NO-GO | **CONDITIONAL GO** | See Recommendation. |

## B6 measurements and control diagnostics — exploratory

The table shows mean D (drop in correct-target log-odds) at layer 5 for each family. Matched controls have the same total L2 as the Substance intervention in the same prompt.

| Family (n) | S_patient | C1 unrelated (matched) | C1 random (matched) | S_patient (C2 prompt) | C2 bystander (matched) |
|---|---|---|---|---|---|
| Impact / vase→shatter (8) | +0.14 | +0.14 | −0.12 | +0.31 | −0.02 |
| Heating / ice→melt (4) | +1.88 | +0.05 | +0.12 | +2.03 | −0.04 |
| Cutting / rope→snap (3) | +2.72 | +0.82 | +0.33 | +3.68 | −0.11 |

Other layers tested (6, 13, 19) show the same pattern with effects weakening with depth; by layer 19 all D < 0.4. The agent-entity intervention has small effects (mean +0.05 at layer 5). A diagnostic that swaps the patient's full residual for "pillow" gives D +0.8 to +4.7, so information at the entity position clearly matters for the output.

### Matching (key G5 input)
- Matching total L2 required scaling the unrelated directions far beyond their natural range: C1-guitar ×4.4–5.9; C1-random ×23–822.
- C2 matches total L2, but the bystander is 1 token against 2 patient tokens. Its per-position change is therefore **1.24–1.37×** larger, at a different position and grammatical role (subordinate-clause subject).
- These follow V1.1's rule literally. Whether matched-L2 controls pushed this far off-distribution are fair controls is a scientific choice to be frozen at G5.

## P3 pathway feasibility — technically feasible, scientifically inconclusive

Method, per family (first prompt; Substance at layer 5):
1. Localize downstream heads by zeroing each head's output at the final position.
2. Pick the head with the largest effect among those attending ≥ 0.05 to the patient.
3. Under the Substance intervention, restore clean K/V **only for that head's KV group, only at the patient positions**.
4. Compare with restoring another group and with restoring the whole layer.

| Family | Chosen head (attention to patient) | Own-group restoration: logit / head output recovered | Other group: logit / head output | Whole layer: logit |
|---|---|---|---|---|
| Heating | L18 H2 (0.81) | 44% / 75% | 29% / 0% | 62% |
| Cutting | L18 H2 (0.89) | 55% / 87% | 9% / 0% | 61% |
| Impact | L25 H4 (0.21) | not interpretable: the Substance intervention barely changed the output (D = −0.06), so there was nothing to restore | | |

**Observation:** the same head, L18 H2 (KV group 1, shared with H3), was selected independently in two families and reads heavily from the patient.

**Limitations:**
- Heads were selected on the same prompts (circular).
- One prompt per family.
- Restoration granularity is per KV group (2 query heads), not per query head.
- The "other group" head-output recovery of 0% is a built-in consistency check, not a finding.
- Under V1.1 this establishes **technical feasibility of pathway-specific restoration**, not P3.

## Failure modes and risks observed

1. **Primary vase/shatter family shows no Substance > C1 specificity with this operator.**
   - Layer 5: S − C1-unrelated = +0.005 (SD 0.21). Layer 13: −0.19.
   - Diagnostic: on the steel vase, *shatter* is still preferred over *dent* (log-prob −2.0 vs −4.5).
   - Possible readings, untested: *shatter* may be driven mostly by "glass/fragile" (material) rather than vase identity, or the rank-1 lexical direction misses identity.
   - **This is exactly the Substance-vs-property distinction V1.1 asks P1 to resolve; it is not a result.**
2. **Lexical association confound.** Large ice→melt and rope→snap effects may reflect the noun's own word association with the outcome rather than identity structure. P1 property-matched contrasts are essential before any P2 reading.
3. **Feature-basis choice is open.**
   - DoM directions are saturated and lexical.
   - The Gemma Scope SAE route works technically (`layer_5/width_16k/average_l0_68`), but:
     - the top vase latents give small effects (D +0.02 to +0.10);
     - latent 1059 appears in both the vase and guitar lists (a generic noun latent);
     - active-latent counts differ (10 vs 2), which complicates "dimensionality" matching.
4. **Off-distribution matched controls** (see Matching).
5. **Weak snap baseline** (p ≈ 0.09–0.18): a risk under B8's "already weak" exclusion.
6. **Layer choice:** effects shrink with depth; the L\* rule picked the earliest saturated layer. It must be pre-registered at G5.

## Sample-size estimate [ESTIMATE]

This uses within-family SDs of paired differences (S − control), *not* probe effect sizes.
- Observed SDs: 0.05–0.62 across families, layers and controls, from tiny n (3–8). True SDs may be ~1.5× larger.
- Formula: two-sided α = 0.05, 80% power, per family and comparison, n ≈ ((1.96 + 0.84)·SD / δ)², before multiplicity adjustment.

| Planning SD | δ = 0.5 log-odds | δ = 0.25 log-odds |
|---|---|---|
| 0.62 (observed max) | ≈ 12 items | ≈ 48 items |
| 0.9 (1.5× inflation) | ≈ 26 items | ≈ 102 items |

**Compute is not the constraint:** ~100 items × ~15 conditions is a few GPU-minutes (<$0.50). The binding constraints are building valid property-matched stimuli and choosing the matching rule. δ, multiplicity and family structure are G5 decisions.

## Recommendation — CONDITIONAL GO to G5

The pipeline is technically sound on the real model: loading, pinning, determinism, targets, interventions, both controls with exact L2 matching, and isolated KV-group restoration. Cost is negligible ($0.06 of $30). **No model switch is justified**: Gemma 2 2B measured all targets, so the Gemma 3 4B fallback is not triggered.

Conditions are scientific decisions for Qualam/Ali at G5, not engineering blockers:
- **(a)** Feature basis and operator for Substance (DoM vs SAE latents vs other), with property-matched, non-lexical P1 contrasts (material, fragility, grammatical role).
- **(b)** C1/C2 matching rule, given the off-distribution scaling observed.
- **(c)** Layer-selection rule and stimulus families: how to treat the vase family's lack of specificity, and whether *snap* stays given its weak baseline.
- **(d)** P3 head selection on data separate from evaluation, and acceptance of KV-group granularity.
- **(e)** δ and sample size.

**Stopping here.** G4 is complete pending Qualam's review and Ali's G5 decision. No protocol change has been made or implemented.
