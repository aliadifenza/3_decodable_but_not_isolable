# G5 Feasibility Verification — Kantian Positional Logic Test

**Date:** 2026-11-21 · **Authorized by:** Ali (Decisions 1–4 approved; +$0.50 cap) · **Executed by:** Claude (code, analysis); Ali (Modal launch)
**Status: EXPLORATORY.** G4 materials only, plus their conversion to the Decision-2 pronoun format.
- No held-out evaluation items exist or were used.
- Nothing here is evidence for P1, P2 or P3.
- Protocol V1.1 is unchanged; G5 is not frozen.

## Record

- **Code:** `g5_verification/g5v_probe.py` (sha256 prefix `f85347d7`). It imports `g4_probe/g4_probe.py` for the G4 data lists (`04f7fca4`).
- **Local plumbing test:** `test_g5v_local.py` — PASS.
- **Raw results:** `results/g5v_results.json`, sha256 `9ad8bfef436ba1e37402d7ff23544e46ff40c9712118aa7279d4e39dad531b2f`.
- **Command (Mac):** `MODAL_IMAGE_BUILDER_VERSION=2025.06 python3 -m modal run g5v_probe.py`
- **Model:** `google/gemma-2-2b` @ `c5ebcd40d208330abc697524c919956e692655cf`, equal to the loaded commit hash.
- **Environment:** fp32, eager attention, NVIDIA A10. torch 2.5.1, transformers 4.46.3, numpy 1.26.4, huggingface-hub 0.26.5. HF token injected.
- **SAEs:** `google/gemma-scope-2b-pt-res`, width 16k, layers 2–18. For each layer, the release whose average L0 is closest to 70 (L0 from 53 to 83); exact files are listed in the JSON.
- **Settings (Decisions 1–4):**

  | Setting | Value |
  |---|---|
  | Maximum control boost `A_max` | 2 |
  | Minimum Substance strength `ρ_min` | 0.5 |
  | Change-size tolerance | ±10% |
  | Active-feature ratio (control vs Substance) | 0.5–2 |
  | Feature-set size `k` | ≤ 8 |
  | Eligibility | target p ≥ 0.10, rank ≤ 5, bystander shift ≤ 1.0 |
  | Recognition threshold | 0.85 |
  | Stability (permanence) threshold | 0.80 |
  | Worst-group threshold | 0.75 |
  | Property-only maximum | 0.65 |

- **Runtime:** 95.5 s remote.
- **Spending ledger:**
  - GPU estimate this run: **$0.029** at $1.10/h.
  - Cumulative estimate: **≈ $0.085** of the $30 ceiling; this run used $0.029 of its $0.50 allowance.
  - Billed amount: **[PENDING: Modal dashboard]**.
- **Derived probe-only items:** 30 (15 converted prompts + 15 bystander variants), hashes in the JSON, excluded from any held-out set.

## Results per objective

| # | Objective | Result | Evidence |
|---|---|---|---|
| 1 | SAE feature selection | **PASS (technical) / scientific adequacy unknown** | Ran for every kind (vase, ice, rope, clock) at all 17 layers; selected 1–6 latents. Most late-layer sets are a **single latent** with recognition AUC 1.00 at both noun and pronoun. That is consistent with a word-identity feature (see F2). |
| 2 | C1/C2 matching | **PARTIAL** | All realized interventions stayed within ±10% and every boost was ≤ ×2 [VERIFIED, 40/40]. But feasibility depends heavily on layer: eligible items range from 0/15 (layers 2–3) to 15/15 (layer 14). At the selected layer 18, only **3/15** are eligible: impact 1/8 (ACTIVE_UNITS ×7), heating 0/4 (MATCH_INFEASIBLE: the C1 change size is 0–8 against Substance's 45–57), cutting 2/3. |
| 3 | Layer selection (representation criteria only) | **PARTIAL** | The rule ran. Only **layer 18** passed, and it is the deepest layer scanned. Margins swing sharply between neighbouring layers (vase pronoun AUC: 0.03 at layer 9, 1.00 at layer 10). Validation sets are tiny (pronoun set: 4 vase vs 15 pool), so the selection is **not reliable on G4-sized data**. |
| 4 | Exclusion rates | **PASS (measured)** | In the pronoun format: tokenization, baseline eligibility, bystander stability and span all **0/15 excluded**. Every item has a single patient mention and a single bystander mention. Cutting's baseline improved (p 0.18–0.27 vs 0.09–0.18 in G4). Matching is the only significant source of exclusions (layer 18: 12/15). |

Eligible items by layer, out of 15:

| Layer | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Eligible | 0 | 0 | 4 | 10 | 6 | 14 | 10 | 7 | 3 | 2 | 7 | 8 | 15 | 12 | 11 | 11 | 3 |

## Findings that block a G5 freeze

1. **The interventions are too weak to test P2.** At layer 18, matched SAE ablations changed the target log-odds by |D| ≤ 0.05 in every eligible and ineligible row recorded: impact ≈ 0.00, cutting S ≈ +0.02 to +0.05. The planned minimum effect of interest is 0.5. A 1–3-latent ablation that leaves the SAE reconstruction error untouched likely removes too little identity information to matter. An experiment run this way is predictably uninformative for P2: an expected null result caused by intervention weakness, not by the hypothesis. The change relative to the residual size (‖Δ‖/‖h‖) was not logged; that is a gap to fix.
2. **The pronoun test is not truly non-lexical.** At the pronoun "it", the model can copy the noun's word identity through attention. Single latents reach AUC 1.00 there, so passing the pronoun test does not rule out noun recognition. D1's "not noun recognition" requirement still depends on items the G4 data lacks: property-matched non-members (e.g. a glass jar against a glass vase) and noun-free descriptions.
3. **The layer-selection rule can pick a layer with no room downstream.** The rule picked layer 18, which leaves only layers 19–25 for downstream components. In G4, the P3 candidate head was itself in layer 18. An upper bound on the layer range is needed (e.g. layer ≤ 14). That is a scientific setting.
4. **The active-feature ratio rule is too coarse for small feature sets.** With k = 1–3, one active latent against three is a ratio of 0.33, which fails. At layer 18 this rule alone excluded 7/8 impact items.
5. **Validation data is too small.** Reliable layer selection and the P1 criteria (≥ 20 per property group) need the Adaption dataset.

## Engineering issues (Claude; no approval needed)

- **C2 used the wrong clock features.** C2 took the top-k clock latents by score rather than the clock's greedily selected set. At k = 1 this used latent 14182 instead of the selected 13005. This does not change the conclusions above, but must be fixed.
- **Missing logging:** add ‖Δ‖/‖h‖ for each position.

## Decisions needed before G5 can freeze (Ali, with Qualam)

**(a) Intervention strength.** Pick one route to make the Substance intervention capable of a 0.5-log-odds effect.
- Options:
  - Larger feature sets (e.g. a fixed k = 8).
  - Ablate the SAE reconstruction error together with the selected features, with C1 and C2 treated the same way.
  - Combine SAE selection with a supervised subspace.
- Without one of these, P2 is expected to come out insensitive.

**(b) Layer range.** Set an upper bound that leaves room for P3 (proposed: layer ≤ 14).

**(c) Active-feature ratio.** Either drop it in favour of exact equality of the selected feature count `k`, which is already enforced, or apply it only when k ≥ 4.

**(d) Dataset requirement.** Confirm that the Adaption bundles include property-matched non-members and noun-free descriptions. These are the only non-lexical P1 tests left.

**Recommendation:** do not freeze G5 yet. After (a)–(d), repeat this verification once on the Adaption discovery and validation splits. Estimated cost is about $0.05–0.10 of GPU.
