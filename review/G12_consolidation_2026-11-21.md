# G12 review consolidation — decisions (2026-11-21, ~21:05 CET)

**Reviews received (archived verbatim in this folder):**
- **Gemini** — independent (`G12_review_gemini_2026-11-21.md`).
- **DeepSeek** — independent, free web chat, reply in Chinese (`G12_review_deepseek_2026-11-21_zh.md`). Substituted for Grok, which was unavailable (usage limit). PI decision 20:52–21:00.
- **Qualam (GPT-6)** — internal audit by the Research Director, **not independent**, self-declared as such (`G12_review_qualam_internal_2026-11-21.md`). A fresh GPT-6 session was considered and rejected because it would see project context.

All three reviewed the same package (`G12_review_package_2026-11-22.md`, pre-E4).

## Verdicts

| Claim | Gemini | DeepSeek | Qualam (internal) | Consolidated |
|---|---|---|---|---|
| 1 identity represented | supported (caveats) | overstated (available ≠ used) | overstated (decodable ≠ validated invariant Form) | **narrow**: "linearly decodable; availability, not causal use; strict P1 criteria passed only sporadically" |
| 2 P2/P3 inconclusive | supported | supported | supported | **keep** |
| 3 not dose/placement | overstated | overstated (silent latents; span-only) | overstated | **narrow**: "for the selected SAE latents within the reference span" |
| 4 local erasure, downstream return | overstated | overstated | overstated | **narrow**: drop "unrecoverable" and "reconstitutes"; report as a modest normalized probe-AUC reduction with raw AUCs; mechanism unresolved |
| 5 organized togetherness | risky framing | unsupported (not testable as stated) | compatibility only | **removed from claims**; moved to Discussion as a V2 hypothesis with falsifiable predictions (PI decision 21:05) |
| 6 no causal conclusion | supported | supported | supported | **keep**; D numbers to footnote |

## Accepted corrections (engineering / reporting)
1. r is described as **normalized downstream probe-AUC reduction**, never as "% of identity removed" (Qualam). Raw clean/ablated AUCs reported next to every r (all three).
2. Amnesic AUC = 0.50 under E3 is by construction; it is reported as "freshly fitted linear probe at chance under this procedure" (all three).
3. **Reporting bug (Claude):** for "all"-token conditions `rel_norm_mean` is measured at the last reference position only, not the aggregate of all edited positions (Qualam). Disclosed; it does not change r.
4. Role probe uses only the first two sorted roles; a NaN role drop passes the descriptive bar (Qualam). Disclosed; the 9 qualifying cases all have numerical role drops.
5. Bootstrap CIs condition on fitted probes/latents/erasers; they do not include fitting or selection uncertainty (Qualam). Disclosed.
6. L6 peak and the 9 bar-crossings are not headlines (all three).
7. Template/style risk from Claude-authored validation data is stated as a limitation (all three).

## Headline (consolidated)
> **Substance remains decodable but not causally isolable at the tested intervention site; P2/P3 inconclusive.** (DeepSeek short form; compatible with Gemini's and Qualam's longer forms, which are used in the abstract.)

## Discussion paragraph for claim 5 (PI-approved placement)
The PI's working conception of identity — "organized togetherness: no single part essential" — echoes V1.1's Form as "a multidimensional, distributed structure". The present results neither support nor contradict it. To give it empirical content, V2 tests three predictions (from DeepSeek's review): (a) several distinct directions each carry part of identity; (b) erasing some leaves downstream decoding intact; (c) erasing all of them at the read position makes it collapse. E4 (exploratory) bears on (c).

## Pending
- E4 ("already travelled?") results — decides the wording of the mechanism sentence in claim 4 and the Discussion.
- Optional: a final-version check by Grok (if usage resets) or a temporary-chat model on 2026-11-22 morning.
