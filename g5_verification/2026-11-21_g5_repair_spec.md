# G5 Repair Specification — Kantian Positional Logic Test

**Date:** 2026-11-21 · **Status:** design plus local tests only. No paid run, held-out data untouched, G5 not frozen, G6 not started.

**Code:**
- `g5_verification/g5r_probe.py` (sha256 prefix `525fd3c0`)
- `test_g5r_local.py` (`26b26e46`)
- `g5v_probe.py` with the bug fixed (`bd585d25`)

**Rule throughout:** no step selects anything using causal-target log-odds. Every threshold below is **PROPOSED**.

## 0. C2 bug — fixed and documented

- **Bug:** C2 used the bystander's top-k latents ranked by score, instead of the bystander's own selected set. At k = 1 this gave latent 14182 instead of the selected 13005.
- **Fix:**
  - The greedy feature-selection order now always continues to 8 latents.
  - The selected set is a prefix of that order.
  - C2 is the bystander's greedy order truncated to `k`, so it equals the bystander's selected set whenever `k` ≤ that set's size, and otherwise extends it with the bystander's next greedy picks.
- **Verified:** the G5v local test passes, and a regression check confirms C2 is a prefix of the bystander's greedy order at every layer.
- **Effect on G5v conclusions:** none. The repaired script supersedes it.

## Input data requirement (Adaption → JSONL; schema in `g5r_probe.SCHEMA`)

- **Recognition records** for each patient kind (vase, ice, rope) and each bystander kind (clock). Each has:
  - members (identity variants) and **property-matched non-members**, by stratum: material, fragility, shape, function;
  - **noun-free descriptions** for both members and non-members;
  - **at least 2 grammatical roles**;
  - template IDs, with **discovery and validation templates disjoint**;
  - **at least 10 per stratum** in validation.
- **Causal records** (discovery and P3-discovery splits only) in Decision-2 format: patient and bystander each named once.
- **Guards in the loader:** it refuses any `evaluation` record and any G4 probe-only item.
- **Recommended extra field:** `mentioned_attribute` on descriptions, so the semantic-shortcut check (R2) can stratify by it.

## R1 — Intervention efficacy

**Algorithm (per layer in 2–14 and per kind):**
1. On discovery data, build a greedy order of up to 8 latents. Each step maximizes the **worst stratum AUC** over property-matched strata and descriptions, starting from the top 64 latents by that score.
2. The candidate sets are the first 1, 2, 4 and 8 latents of that order.
3. Measure each set on validation data:
   - **Identity removal, downstream:** train a probe on clean discovery activations at the final token, 4 layers after the ablation layer. Compare its AUC on clean vs ablated validation data, giving a removal score `r = 1 − (AUC_abl − 0.5)/(AUC_clean − 0.5)`.
   - **Identity removal, same site:** retrain a probe on ablated discovery data and test it on ablated validation data ("amnesic" AUC).
   - **Preservation:** the role probe's AUC drop at the ablation site.
   - **Magnitude:** ‖Δ‖ and ‖Δ‖/‖h‖ (the G5v logging gap, now filled).
   - **Next-token KL:** reported only.
4. Choose the **smallest k** that passes.

| | Criterion |
|---|---|
| **Success** | `r` ≥ 0.67, amnesic AUC ≤ 0.70, role-AUC drop ≤ 0.05 |
| **Failure** | No k ≤ 8 passes, meaning this SAE basis cannot remove identity information at that layer. Recorded as "efficacy FAIL". |

**Cost:** part of the single run below.

## R2 — Non-lexical identity

**Algorithm:**
- Selection uses the worst-stratum objective from R1, so a latent must work in **every** property-matched stratum **and** on noun-free descriptions.
- Validation uses **unseen templates**.
- **Leave-one-role-out:** for each role, reselect latents on discovery data with that role removed, then test on validation items of the held-out role.
- **Shortcut guards:**
  - Lexical: flag any latent with noun AUC ≥ 0.95 but description AUC ≤ 0.60, and require a **lexical gap** (noun AUC − description AUC) ≤ 0.20.
  - **Pronoun positions no longer count as non-lexical evidence** (G5v finding: at "it" the model can copy the noun's word identity).
  - Semantic: property- and function-matched non-members (e.g. a glass jar, a flower pot); optionally, stratify descriptions by `mentioned_attribute`.

| | Criterion |
|---|---|
| **Success** | Noun AUC ≥ 0.85; worst property-matched stratum ≥ 0.75; description AUC ≥ 0.75; leave-one-role-out AUC ≥ 0.75 for every role; lexical gap ≤ 0.20; ≥ 10 items per stratum |
| **Failure** | Any criterion misses with adequate counts → identity not established at that layer. Inadequate counts → inconclusive. |

## R3 — Controls

**Algorithm:**
- All Decision 2 rules stay: one common `N*`, boost at most ×2, Substance at least half strength, ±10%, equal `k`, same layer, hook and operator, one mention each.
- The **active-feature ratio is measured and logged, not used to exclude** (as you instructed).
- C1 is drawn from latents that are active at the site on discovery data, are identity-unselective (\|AUC − 0.5\| < 0.05), are disjoint from the Substance set, and have the closest contribution size to it.
- C2 is the bystander's own greedy set (bug fixed).
- Measured:
  - **valid-pair retention** on discovery and P3-discovery causal items;
  - **collateral damage**: whether C1 removes target-kind identity (it should not), whether C2 removes bystander identity (it should), and the role-probe drop.

| | Criterion |
|---|---|
| **Success** | Retention ≥ 75%; C1 identity removal ≤ 0.10; C2 removal of bystander identity ≥ 0.67; role drop ≤ 0.05 |
| **Failure** | Retention < 75% → matching-compromised. C1 removes identity → not an unrelated control. C2 removes nothing → not a valid noun-representation control. |

## Layers 2–14 and P3 feasibility

**P3 feasibility algorithm** (P3-discovery prompts only; no target log-odds):
- For downstream layers `T ≥ L + 2`, for each key/value (KV) head group:
  - attention from the final token to the patient;
  - an **information-use ratio**: how much the group's output changes under the Substance ablation, relative to the change under the matched C1.
- A layer counts as feasible if at least one group has attention ≥ 0.10 and ratio ≥ 1.5.

**Layer selection:** among layers that pass R1, R2, R3 and P3 feasibility for **every** patient kind, pick the one with the highest worst-case R2 margin, preferring the shallower layer on ties. If no layer qualifies, report "no valid layer".

## Local deterministic tests (CPU, tiny random Gemma 2, synthetic fixture) — ALL PASS

| Test | What it checks |
|---|---|
| T1 | Loader refuses evaluation-split records and G4 probe-only items |
| T2 | Two runs give identical results |
| T3 | Nested sets; leave-one-role-out reselection; C2 = bystander greedy prefix; C1 disjoint from Substance; active ratio measured but never an exclusion code; `N*` bounds |
| T4 | P3 path runs and KL > 0 |

On the random toy model every layer verdict is FAIL, as expected. That validates the plumbing only.

## Minimum GPU verification (needs authorization)

- **One Modal A10G job** on validated Adaption discovery, validation and P3-discovery data, covering:
  - 13 layers (SAEs already cached from G5v) × 4 feature-set sizes × 4 kinds;
  - batched forward passes, with probes fitted on CPU.
- **Estimate:** 6–15 minutes, about $0.11–0.28 of GPU.
- **Hard cap:** 20-minute timeout, about $0.37.
- **Cumulative afterwards:** ≤ about $0.46 of $30.
- **Blocked until the Adaption data is validated and converted** to the schema.
