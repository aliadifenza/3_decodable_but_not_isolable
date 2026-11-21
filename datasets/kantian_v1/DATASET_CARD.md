# Kantian Positional Logic — Dataset v1 (Workstream 1)

**Experiment:** AlignedLab Exp. 3 (Research Agenda Ch.2 Exp.2) · **Authority:** Protocol V1.1 + fidelity record · **Model:** Gemma-2-2B @ `c5ebcd40d208330abc697524c919956e692655cf`
**Built:** 2026-11-21 by Claude (lead engineer) · **Status:** BLOCKED (see §7). Not a G5 freeze.

## 1. Layout

```
discovery/v1.0/    original 432-record set as delivered (preserved, unmodified)
discovery/v1.1/    corrected set + changelog + per-family files (IDs and allocation unchanged)
validation/v1.0/   432 records, consolidated + per-family JSONL/CSV
evaluation/v1.0/   1,152 records, consolidated + per-family JSONL/CSV
scripts/           kinds.py, banks.py, repair_discovery.py, generate_splits.py, audit_dataset.py, tokcheck_modal.py
audit/             audit_all_splits.json, audit_discovery_v1.0_original.json, reviewed_parser_artifacts.json
SHA256SUMS         checksums of every data/script/audit file
```

Schema: 17-field JSONL (unchanged from discovery); 6-column mapping CSV (`unique_id, bundle_id, split, focal_kind, property_group, template_id`).

## 2. Counts (verified by `audit_dataset.py`)

| split | per family: identity_preserving / matched_nonmember / causal_prediction | per family | total |
|---|---|---|---|
| discovery v1.1 | 60 / 60 / 24 | 144 | 432 |
| validation v1.0 | 60 / 60 / 24 (24 bundles = 96 + 48 supplements) | 144 | 432 |
| evaluation v1.0 | 192 / 96 / 96 (96 bundles, no supplements) | 384 | 1,152 |

Six strata per family in every split; 4 validation and 16 evaluation bundles per stratum. Families: physical_impact (raw hen's egg, aluminium beverage can, car windshield), heating (corn kernel, raw egg white, beeswax block), cutting (parchment, balsa strip, acrylic sheet).

## 3. Phase A — discovery audit and repairs

Audit found no ID, CSV, bundle, label, duplicate, G4-overlap or kind-noun-leak errors. 231 logged corrections (`discovery/v1.1/discovery_v1.1_changelog.csv`): 133 role-label spelling harmonisations (e.g. `direct object` → `direct_object`); 84 role corrections where the label contradicted the sentence (57 → passive_subject, 22 → direct_object, 2 → subject, 3 → prepositional_object), checked against a transformer dependency parse plus manual review; 14 cutting causal prompts rewritten so the bystander clause is inert (e.g. "while the glass weight held the far edge" → "while the glass weight sat on a nearby shelf"). Targets, labels, IDs and allocation untouched.

## 4. Phase B — generation method

`generate_splits.py` composes sentences from `banks.py`: role-specific observational frames, agents, noun-free entity cores (10 member + 10 non-member per stratum, sharing head nouns), inert everyday bystanders and per-kind causal events. Template and core banks are partitioned so validation and evaluation share no templates, cores, frames, agents or bystanders. Seeds 20261121 (validation), 20261122 (evaluation). Each B record reuses its A1's frame, agent and adjective (minimal pair). Candidates are rejected if near-duplicate (token Jaccard ≥ 0.8) to any discovery sentence, any sentence of the other split, or any earlier sentence of the same split. Rerun: `python3 scripts/generate_splits.py discovery/v1.1/discovery_v1.1.jsonl <outdir>` reproduces the archived files byte-for-byte (checksums in SHA256SUMS).

## 5. Phase C — integrity results

All checks in `audit/audit_all_splits.json`: schema and CSV consistency, ID uniqueness, label consistency, target ∉ alternatives, role vocabulary, role vs parse, kind-noun leaks in descriptions, causal patient/bystander mentions, exact and near duplicates within and across splits (0 / 0), template and bundle disjointness across splits (0 shared), G4 exact/near/noun overlap (0). Only residual flags: 37 role-vs-parse disagreements (13 discovery, 6 validation, 18 evaluation), each manually reviewed as parser error and listed in `audit/reviewed_parser_artifacts.json`. Head-noun balance: validation/evaluation shared-head fraction ≥ 0.88 (members) and ≥ 0.94 (non-members) in every stratum.

**Tokenization (Gemma tokenizer): PENDING** — `scripts/tokcheck_modal.py` (CPU-only Modal job, tokenizer only) checks that every causal target and alternative is a single token after the prompt and records the recognition span's last token.

## 6. Fidelity audit and limitations

Unchanged: families, kinds, strata, record types, bundle structure, label semantics, C1/C2 requirements, evaluation isolation. Known limitations: (a) 129 discovery `entity_description` fields paraphrase rather than quote the sentence — measurement spans must be located in `sentence`; (b) discovery heating bystanders are matched non-member kinds (C2 overlap risk), while validation/evaluation use inert everyday bystanders — a distribution shift; (c) five discovery strata have non-member shared-head fraction < 0.5 (K03_CTX 0.2, PG1_sealed_moisture_hull 0.3, PG1_coagulable_albumin_fluid 0.3, PG2_molded_apiary_slab 0.3, balsa_porous_float 0.1) — a residual lexical cue, left as is because it is not a demonstrable error; (d) generated text is compositional and framed, a style shift from discovery; (e) validation and evaluation were authored by Claude, not by an independent source; (f) baseline eligibility of causal items is unknown until the model check.

## 7. Blockers

1. **Tokenization check not yet run** (requires PI's Modal account; < $0.01 CPU).
2. **C2 data gap (PI decision):** the allocation contains no recognition records for bystander kinds, so C2 cannot be drawn from this dataset as specified. Options: (a) add a bystander-recognition supplement (changes allocation); (b) resolve by method change in the specification workstream.
