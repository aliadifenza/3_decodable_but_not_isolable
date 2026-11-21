# AlignedLab Experiment 3 — The Kantian Positional Logic Test (Research Agenda Ch. 2, Experiment 2)

**Project: AlignedLab / Mechanistic Interpretability Hackathon \#5  
Principal Investigator: Dr. Ali Antonio Di Fenza  
Research Director: Qualam (GPT-6)  
Engineering Lead: Claude  
Independent Reviewers: Gemini and Grok  
Version: 1.2 — G5-frozen protocol reissue (21 November 2026)  
Status: G5 SPEC FREEZE FROZEN; PI approval 21 November 2026, 18:32 CET.
Next gate: G6 ENGINEER. No real causal experiment has been run.  
Scientific authority: Protocol V1.1 and the approved G5 specification,
including Amendment 1 (21 November 2026, 18:47 CET).  
V1.2 is a consolidated reissue for readability. If V1.2 and
protocol/G5_SPEC_FREEZE.md (with Amendment 1) ever disagree,
G5_SPEC_FREEZE.md governs.**

## A. Scientific foundation

### Philosophical question

Can Plato’s Forms and Kant’s conditions of possibility explain and
predict the organization of neural-network representations and their
downstream circuits?

### Core hypothesis

A neural network learns organizing structures that permit recognition
across different manifestations of a phenomenon. These structures,
interpreted as *Forms*, function as conditions of possibility for
downstream recognition and processing.

A Form is not necessarily a perfect exemplar or a single feature. It may
be a multidimensional, distributed structure defining which variations
can still be recognized as manifestations of the same phenomenon.

### Specific experimental hypothesis

**Substance → Causality**

The internal representation establishing the identity of an entity is a
necessary upstream component of a downstream circuit processing its
causal interactions.

This is the original Kantian Positional Logic hypothesis.

### Three predictions

**P1 — Form / invariance**

The model contains identifiable representational structure that remains
sufficiently stable across different manifestations of an entity while
distinguishing it from other entities.

**P2 — Causal dependency**

Disrupting that structure damages the correct causal-outcome prediction
more than either mandatory control.

**P3 — Mandatory pathway**

The downstream causal computation relies on the identified upstream
structure. Disrupting the relevant pathway impairs the computation, and
restoring its information should restore some or all of the lost
functionality.

P1, P2, and P3 have different evidential requirements. P1 does not
establish P2; P2 does not automatically establish P3.

### Scope

- Primary system: one lightweight pretrained language model.
- Primary phenomenon: entity representations and causal physical
  interactions expressed in language.
- Primary method: mechanistic interpretability with internal activation
  measurements and interventions.
- Excluded claims: universal Kantian necessity, consciousness, visual
  perception, or a complete account of all Forms.
- Broader philosophical hypothesis remains unchanged regardless of the
  outcome of this particular experiment.

**A methodological definition of Identity:**

**Recognizable Permanence Within Change**

*Identity is the recognizable permanence of a being through change: an
organized togetherness of qualities that can tolerate the individual
transformation of some of them while retaining its recognizable Form,
for itself and/or others, depending on the kind of being.*

*A philosophical reflection by Ali Antonio Di Fenza*

Identity is that which remains during change. Not that which remains the
same literally, but that which we can recognize as such even though
change has occurred.

So, Form, both as *idea* and *morphē*. Not something that exists
independently from the thing, but the organization of the thing that we
can distinguish through change.

For now, let us pretend that names and things are the same stuff, so
that, methodologically, the definition of identity we reach marks,
circumscribes, identifies (pun intended) what we are talking about.

**The Square: Essence and Accident**

Take a square. A square has four equal sides and four 90-degree angles.
That's its essence. Everything else is accident.

A square remains a square even if you change the length of its sides,
the thickness of its lines, its color, or its material, if it exists in
the material world. None of these changes matters for its definition.

But if you change the length of one side without changing the others,
you destroy the square.

On the one hand, yes, you transform it. But on the other, you destroy
it. There is no square anymore if one of its four sides or angles
changes independently in a way that breaks its defining organization.

Identity, then, is not the absence of change. It is the persistence of a
certain organization through change.

**The Human Body: Recognizable Transformation**

Now, let's move to humans. To me, for instance.

Take my body to begin with.

I'm not the same kid I was. And I'm not the same man I was, say, a year
ago. I think every six months all my atoms are renewed, or something
like that.

As a child, I grew and morphed, and became me today. I grew muscles and
lost them, then grew them again. I had no wrinkles and black hair; now I
have many more wrinkles and gray hair. Salt and pepper.

And yet, you look at my photograph, and some basic shapes of me have
remained the same. My eyes, my lips, my ears, my hands, my nose.

A set of patterns of my body has morphed inessentially—or
accidentally—meaning that these changes did not affect the capacity in
me and others to recognize permanence within change.

The changes pertained to accident, not essence.

No part of my body *per se* is essential for you to say it's me. I could
have lost my hands, cut my nose, or lost my ears.

I'm thinking of Wittgenstein's *family resemblance*, when he tries to
define a game.

But the overall impression you get from all my parts—hands, eyes, mouth,
and so forth—that's my identity.

No single quality is necessarily fundamental. Yet the qualities, in
their organized togetherness, allow the being to remain recognizable.

**Identity as Trace, Echo, and Givenness**

So, the identity of a being is the overall givenness of such a being,
its *being so and so*, through which both the being and other beings, if
conscious, can say:

**This is X!**

It's a trace, an echo that spans across time.

A trace, an echo, and an overall givenness—Form, *morphē*, *idea*—that
emerges for us to contemplate and experience, but that does **not**
exist independently, somewhere up there, abstracted from the places from
which it emerges.

These hands. This nose. These teeth.

The Form is not detached from its manifestations. It is recognizable
through them.

**The Relationality of Identity**

Identity is relational, even when this relation is between me and me.

There is an ontologically social dimension of identity. It begs getting
out of yourself. It *ex-ists*, etymologically.

Heidegger, *Being and Time*, rules. Always did and always will.

Recognition is essential, whether I do it or others do it. Who
recognizes depends on the kind of being we're talking about: conscious
or non-conscious.

A square does not recognize itself. A human being can.

But identity entails recognizability and relation.

**Provisional Definition**

**Identity is the recognizable permanence of a being through change: an
organized togetherness of qualities that can tolerate the transformation
of some individual qualities and yet retain its recognizable Form, for
itself and/or others, depending on the kind of being.**

Identity is thus permanence within change, organized togetherness, and
recognizability. And recognition is relational—even when the relation is
between a being and itself.

Identity is not simply a list of unchanging attributes. It is the
recognizable persistence of an organized being across its
transformations.

It is a trace. An echo. The continuity of a *being so and so* across
time.

*Research memo — AlignedLab, Kantian Positional Logic Test, G5
philosophical clarification.*

## B. Proposed experimental design

google/gemma-2-2b @ c5ebcd40d208330abc697524c919956e692655cf (model and
tokenizer). SAEs: google/gemma-scope-2b-pt-res, width 16k, per layer the
release with mean L0 closest to 70 (exact files logged per run). Modal,
NVIDIA A10G, fp32, eager attention; torch 2.5.1, transformers 4.46.3,
numpy 1.26.4, huggingface-hub 0.26.5. Weights stay on Modal. Seed
20261121 unless stated.

### B1. Stimuli

The experiment uses controlled causal sentences.

| Family  | Example                                                              | Target outcome |
|---------|----------------------------------------------------------------------|----------------|
| Impact  | The bowling ball struck the fragile glass vase, causing the vase to… | shatter        |
| Heating | The flame heated the ice cube, causing the ice to…                   | melt           |
| Cutting | The knife sliced through the rope, causing the rope to…              | snap           |

The examples above are illustrative candidates, not the frozen dataset.
Their baseline predictions and tokenization must be checked. The frozen
families and inclusion rules are specified below.

Families: physical_impact (raw hen's egg, aluminium beverage can, car
windshield), heating (corn kernel, raw egg white, beeswax block),
cutting (parchment, balsa strip, acrylic sheet); 6 property-matched
strata per family. Confirmatory family: physical_impact (Decision 4).
Heating and cutting: secondary, reported separately.  
  
Causal item inclusion (Decision 4 + B6), applied on the intact model
before any intervention:  
target is a single Gemma token appended to the prompt (Amendment 1:
multi-token alternatives are dropped from the alternative list, not used
to exclude the item); target p ≥ 0.10; target rank ≤ 5;
inserting/removing the bystander shifts target log-odds by ≤ 1.0;
patient and bystander each named exactly once. Excluded items are logged
with reason codes, never replaced.

For example, a steel vase may remain recognizable as a vase while making
*shatter* an inappropriate outcome. This distinction helps separate the
structural representation of vasehood from material properties relevant
to the causal prediction.

Frozen datasets: datasets/kantian_v1/ (SHA256SUMS): discovery v1.2 (492,
sha256 65c614bd47f7bfeb…), validation v1.1 (492, 67dd259bd750d2c3…).
Templates disjoint across splits; no G4 overlap; no kind-noun leaks;
audit clean except 38 reviewed parser artifacts.  
C2 bystander (PI decision 2026-11-21): one fixed kind per family in
every causal prompt — impact umbrella, heating wristwatch, cutting
teapot — with 10 members + 10 property-matched non-members per kind in
discovery and in validation.

Held-out evaluation: evaluation/v1.1 (1,152 records, sha256
e0c02030eaae8007…; impact 96 causal items in 24 distinct events). Never
loaded by any selection or verification code (loader refuses
evaluation); first opened at G10. Template, bundle and near-duplicate
disjointness verified.

Current counts per family are 70 identity-preserving, 70 matched
non-member and 24 causal-prediction records in discovery and validation;
evaluation has 192 identity-preserving, 96 matched non-member and 96
causal-prediction records per family. Evaluation causal items cover 24
impact events, 15 heating events and 15 cutting events. The frozen data
define property-matched non-members; no separate ambiguous-object
category is defined.

### B2. Feature identification

**Objective:** Find internal representations associated with the entity,
not simply the occurrence of its name.

Frozen procedure:

1.  Present multiple manifestations of the entity in varied sentence
    contexts.
2.  Record internal activations across candidate layers and token
    positions.
3.  Identify candidate features, directions, or distributed feature sets
    associated with that entity.
4.  Test whether these candidates generalize to held-out contexts and
    distinguish the entity from relevant non-entities.
5.  Check whether their selectivity persists across property changes
    rather than merely tracking one material, descriptor, token, or
    grammatical position.
6.  **Required P1 identity validation (PI-approved G2 refinement):** In
    held-out, property-matched contrasts, test whether the candidate
    representation still discriminates the intended entity from
    non-entities when material, fragility, affordances, and descriptive
    context are matched or varied independently. Also vary grammatical
    role and position so selectivity cannot be explained solely by noun
    placement or object/patient tracking. For the same entity, verify
    stability across such changes; for matched non-entities, verify
    discrimination.
7.  The candidate-selection procedure, property-matched contrast
    construction, held-out split, quantitative selectivity/stability
    criteria, and failure/inconclusive thresholds are frozen at G5
    before confirmatory tests. A candidate driven by property or
    grammatical role without residual entity identity does not qualify
    as validated Substance for P1 or subsequent Substance-specific
    claims.

Contrasts: members vs property-matched non-members sharing head nouns;
noun-free descriptions; ≥2 grammatical roles; leave-one-role-out.
Pronoun positions do not count as non-lexical evidence.  
Validated Substance at a layer requires all of: noun AUC ≥ 0.85; worst
property-matched stratum AUC ≥ 0.75; description AUC ≥ 0.75;
leave-one-role-out AUC ≥ 0.75 for every role; lexical gap (noun −
description AUC) ≤ 0.20; ≥10 items per stratum. Missing any with
adequate counts → identity not established at that layer; inadequate
counts → inconclusive.

Per layer 2–14 and per kind: greedy SAE latent order (≤8), each step
maximizing the worst-stratum AUC, from the top 64 latents (discovery
data). Candidate sets k ∈ {1, 2, 4, 8}.

**Operational definition of a candidate Form:** An identified internal
representation with demonstrated selectivity and stability across the
tested variations.

This is a measurable proxy for the philosophical Form, not a definition
of Form itself.

Important limitation: a text model tested on sentences establishes
linguistic or conceptual invariance only. It does not establish visual
recognition across actual images.

Dataset limitations: 129 discovery entity_description fields paraphrase
rather than quote the sentence; measurement spans must be located in
sentence. Five discovery strata retain non-member shared-head fractions
below 0.5. Validation and evaluation were authored by Claude from
templates, with a style shift from discovery. Baseline eligibility
remains unknown until the model check.

### B3. Downstream circuit identification

Identify candidate downstream attention heads or connected components
whose activity contributes to the causal-outcome prediction.

Test whether these components:

- respond to relevant causal relationships;
- contribute causally to the target prediction;
- receive or use information from the identified entity representations.

The original diagram’s Layer N → Layer N+1 is conceptual. Actual
components need not occupy adjacent layers.

Downstream (P3 feasibility, discovery causal items only, no target
log-odds): for layers T ≥ L+2, per KV head group, attention
final-token→patient ≥ 0.10 and information-use ratio (Substance vs C1
output change) ≥ 1.5. No separate P3-discovery split exists; P3
feasibility uses discovery causal items.

### B4. Intervention

Leave the input sentence unchanged.

Operator: SAE latent ablation at the patient-noun token span, residual
stream at layer L. Feature-set size: smallest k that passes R1 efficacy
(downstream removal r ≥ 0.67, amnesic AUC ≤ 0.70, role-AUC drop ≤ 0.05).
Layer: among layers 2–14 passing R1, R2 (=G5 §4) , R3 and P3 feasibility
for every impact patient kind, the one with the highest worst-case R2
margin; ties → shallower. None → "no valid layer" → P2/P3
inconclusive.  
Strength/matching (Decision 2, FROZEN): one common change norm N\* per
item; control boost ≤ ×2; Substance at ≥ 0.5 of natural strength;
realized ‖Δ‖ within ±10%; equal k; same layer, hook, operator.
Active-feature ratio measured and logged, never an exclusion. ‖Δ‖ and
‖Δ‖/‖h‖ logged per position.

Intervention layer, feature-set size and strength are frozen as
selection rules, not as numbers. They are chosen only from discovery and
validation data, never from evaluation data or causal-target log-odds.
If the rules find no valid configuration, the experiment reports P2/P3
as inconclusive; nothing is re-tuned.

Individual versus combined entity interventions are secondary
measurements. Run separately:

- Intervention on the first causal entity.
- Intervention on the second causal entity.
- Combined entity intervention.
- Activation restoration/patching where technically feasible.

Use the same prompt and model weights throughout each comparison.

**P3 pathway-specific evidence (PI-approved G2 refinement):** A claim of
mandatory Substance → Causality dependency requires (i) a localized
upstream Substance representation and identified downstream interaction
component(s), (ii) targeted disruption with measurements showing the
downstream component uses information from that upstream site, and (iii)
pathway-specific restoration/patching that recovers downstream
processing and, where applicable, the output log-odds signal. Global
recovery or reintroduction of correlated information through an
unspecified alternate channel does not by itself establish the claimed
pathway. If pathway isolation or specific restoration is technically
infeasible, report P3 as inconclusive/unestablished rather than
inferring necessity from P2. These are pathway-evidence requirements,
not a replacement or co-primary metric for B6.

Necessity claim requires (i) validated Substance at L and an identified
downstream head group at T (G5 §6); (ii) Substance ablation changes that
group's output more than C1 (ratio ≥ 1.5); (iii) pathway-specific
restoration: patching the clean layer-L patient-span activation into
that head group's input only, under the Substance ablation, recovers ≥
50% of the group's output change and of D. If (iii) cannot be
implemented or fails isolation checks → P3 reported
inconclusive/unestablished, never inferred from P2.

### B5. Mandatory controls

**Control C1 — Unrelated features**

Suppress unrelated internal features using interventions matched in
magnitude and, where possible, location and other relevant properties.

Purpose: distinguish entity-specific dependency from generic damage
caused by altering model activations.

C1: k latents active at the site on discovery data, identity-unselective
(\|AUC − 0.5\| \< 0.05), disjoint from Substance, closest contribution
size; applied at the patient span.

**Control C2 — Irrelevant bystander noun**

Introduce a grammatically present but causally irrelevant noun.

Example:

*As a clock ticked on the shelf, the bowling ball struck the fragile
glass vase, causing the vase to…*

Suppress the representation associated with *clock*, using a comparable
intervention.

Purpose: distinguish the hypothesized dependency from generic noun
tracking or attention routing.

**C1/C2 intervention-matching rules (PI-approved G2 refinement; exact
numerical settings frozen at G5, PI-approved 21 November 2026):** For
each confirmatory comparison, use the same model checkpoint, target
continuation, activation-ablation operator, and intervention layer/hook
class. Match and record the actual L2 norm of the intervention-induced
activation change at the affected site, together with the
number/dimensionality of intervened features; do not infer equality
merely from applying the same scalar coefficient. For C1, apply
unrelated-feature ablation at the same candidate activation site and
target token position(s), where feature selectivity permits. For C2,
apply the corresponding noun-representation intervention at the
bystander noun token position(s); match layer, hook, operator,
intervention dimensionality, and actual change magnitude, while
explicitly recording unavoidable differences in token position,
grammatical role, or salience. Never describe C2 as position-matched
when it is not. The G5-frozen feature/position matching algorithm,
numerical magnitude tolerance, token-span handling, paired-stimulus
rules, exclusion/failure criteria, and logging fields below apply
identically without consulting confirmatory outcomes. Comparisons that
fail required matching checks are flagged as non-confirmatory, not
interpreted as positive evidence. Both original controls remain
mandatory and distinct.

C2: the bystander kind's own greedy set truncated to k (bug-fixed rule),
applied at the bystander noun span. C2 is not position-matched and is
never described as such; position, role and salience differences are
logged.  
Control validity (R3): valid-pair retention ≥ 75%; C1 removes ≤ 0.10
target identity; C2 removes ≥ 0.67 bystander identity; role drop ≤ 0.05.
A comparison failing a required check is flagged non-confirmatory.

±10% norm tolerance; per item log: layer, hook, k, latent IDs, ‖Δ‖,
‖Δ‖/‖h‖, token positions, boost factor, ρ, active-feature ratio,
exclusion code. Items failing matching are excluded from the
confirmatory test and counted; exclusion rates reported per condition.

**Exploratory diagnostic — Material property**

Separately intervene on a candidate material or fragility
representation, if one can be identified. This helps distinguish
Substance from outcome-relevant properties. It does not replace C1 or C2
and is not a third mandatory control.

### B6. Primary measurement

Measure the change in **log-odds of the correct causal-outcome token**
between the intact and intervened model.

For a target token with probability (p):

L = log(p/(1−p))

For each intervention:

D = L_intact − L_intervened

Positive (D) means the intervention reduced the target prediction.

The primary comparisons are:

Δ1 = D_Substance − D_C1

Δ2 = D_Substance − D_C2

Comparisons are paired per item. Both Δ1 and Δ2 must have p \< 0.05 and
mean ≥ 0.5 log-odds for P2 to be supported under the frozen analysis
plan.

Only targets with valid, documented tokenization will enter the primary
next-token analysis. Multi-token outcomes require a separately approved
measurement rule.

Fixed by the dataset, no optional stopping: all eligible impact
evaluation items (max 96). Causal items are clustered by event (96 items
= 24 distinct events × wording variants; template_id = event), so the
unit of inference is the event. Minimum for a conclusive primary result:
≥ 30 eligible items covering ≥ 12 distinct events; otherwise
inconclusive. A planning estimate from G5r variability (discovery items
only) is reported before G10 but cannot change n. G5v gave no usable
variability estimate (\|D\| ≤ 0.05).

Per comparison: average the paired difference within each event, then
take the mean over events (each event weighted equally); 95% CI by
bootstrap over events (10,000 resamples, logged seed); one-sided
sign-flip test on event means (exact ≤ 24 events, else Monte Carlo
100,000). Minimum effect of interest: 0.5 log-odds. Both Δ1 and Δ2 must
pass (intersection–union), so no correction on the primary. Secondary
families: same analysis, Holm across the four secondary tests.

### B7. Secondary measurements

- Selectivity and generalization of candidate entity features.
- Changes in identified downstream circuit activations.
- Effects of individual versus combined entity interventions.
- Degree of recovery following activation restoration.
- Changes in alternative outcome predictions.
- Sensitivity to property and sentence variations.

Secondary measurements cannot retroactively replace the primary metric.
For P3, downstream-component intervention and pathway-specific
restoration evidence is required **before making a necessity claim**;
these measurements remain secondary to the unchanged primary B6 log-odds
measurement.

### B8. Falsification and interpretation

| Observation                                                               | Interpretation                                                                                |
|---------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------|
| Δ1 and Δ2 both: p \< 0.05 and mean ≥ 0.5 log-odds                         | P2 supported; P3 requires separate pathway evidence                                           |
| Substance intervention is no more damaging than unrelated-feature control | Does not support the predicted specificity; apply the frozen P2 outcome criteria              |
| Bystander intervention produces comparable damage                         | Does not support the proposed special entity dependency; apply the frozen P2 outcome criteria |
| Entity features cannot be identified or validated                         | Inconclusive                                                                                  |
| Target prediction is already weak or unstable                             | Inconclusive for that stimulus                                                                |
| Downstream prediction survives the purportedly complete intervention      | Counts against necessity of that targeted pathway                                             |
| Selective disruption followed by restoration recovers performance         | Stronger causal evidence for the targeted pathway                                             |

A mandatory pathway requires stronger evidence than a statistically
significant output change. Even successful interventions establish
necessity only relative to the tested representations, model, tasks, and
intervention coverage. **Interpretation rule:** P1 selectivity, P2
log-odds specificity, and P3 pathway necessity must be reported
separately. P2 superiority over both controls supports causal
contribution to the output; it cannot alone establish a specific
downstream Causality circuit or a mandatory Substance → Causality route.
P3 necessity is supported only by adequately selective, pathway-specific
disruption and restoration evidence; otherwise mark P3 inconclusive or
contradicted as indicated by the observations.

Frozen outcome criteria: P2 supported — Δ1 and Δ2 both: p \< 0.05 and
mean ≥ 0.5. P2 not supported — either Δ1 or Δ2 has 95% CI upper bound \<
0.5, with the G5 §12 minimum met. Inconclusive — anything else; or no
valid layer; or the G5 §12 minimum not met; or Substance not validated
under G5 §4. P1 is reported separately under G5 §4. P3 is reported
separately under G5 §10; P2 alone never establishes P3.

## C. Gate execution and verification

| Gate | Name                       | Required outcome                                                                                   | Owner / Review             |
|------|----------------------------|----------------------------------------------------------------------------------------------------|----------------------------|
| G0   | DESIGN                     | Agree workflow, roles, deadlines, resources, authority boundaries, independent reviews             | Ali + Qualam               |
| G1   | ALIGN                      | Confirm philosophical question, hypothesis, objectives, scope, predictions, falsification, success | Ali + Qualam               |
| G2   | SCIENTIFIC DRY RUN         | Verify question → scientific problem → variables → measurements → controls → falsification         | Qualam; Gemini + Grok; Ali |
| G3   | RECON                      | Verify repositories, model, Modal, prior infrastructure, feasibility, time, cost and GO status     | Qualam + Claude            |
| G4   | FEASIBILITY PROBE          | Real-model exploratory test of intervention sensitivity, controls and variability                  | Claude; Qualam; Ali        |
| G5   | SPEC FREEZE                | Freeze complete approved experimental and analysis specification                                   | Ali                        |
| G6   | ENGINEER                   | Implement in small verified chunks                                                                 | Claude                     |
| G7   | CODE DRY RUN               | Test implementation, intervention isolation, metrics, reproducibility and fidelity                 | Claude; Gemini + Grok      |
| G8   | PILOT                      | Small real-model test of the completed implementation                                              | Claude + Qualam            |
| G9   | EXPERIMENT FREEZE          | Lock tested code, model, data, seeds, configuration and commit                                     | Claude + Qualam            |
| G10  | FULL RUN                   | Execute approved experiment and preserve evidence                                                  | Claude                     |
| G11  | ANALYSIS                   | Calculate predefined results, effects, uncertainty and limitations                                 | Qualam + Claude            |
| G12  | INDEPENDENT RESULTS REVIEW | Independent examination of evidence, methods and claims                                            | Gemini + Grok; Ali         |
| G13  | SUBMISSION                 | Final public artifacts and verified hackathon submission                                           | Team                       |

### G0 — DESIGN

**Required outcome:** Approved operational framework.

**Owner:** Ali + Qualam.

- [ ] Revised G0–G13 workflow accepted without renumbering or skipped
  gates.
- [ ] Ali’s scientific authority and Claude’s engineering delegation
  recorded.
- [ ] Gemini and Grok’s independent-review responsibilities recorded.
- [ ] Deadline and Saturday work constraint recognized.
- [ ] Resource budget, spending authorization boundary and escalation
  rules agreed.
- [ ] Required evidence and gate-signoff format established.

**Evidence:** Approved workflow and roles record.

**Exit:** Ali explicitly approves the operating framework.

### G1 — ALIGN

**Required outcome:** Locked scientific intent.

**Owner:** Ali + Qualam.

- ~~☐ Original philosophical question reproduced faithfully.~~
- ~~☐ Relationship between Form, Substance, recognition and Causality
  explained.~~
- ~~☐ Specific Substance → Causality hypothesis confirmed.~~
- ~~☐ P1, P2 and P3 explicitly accepted, modified, or rejected.~~
- ~~☐ Scope limited to claims the experiment can investigate.~~
- ~~☐ Primary measurement verified against the original agenda.~~
- ~~☐ Both original mandatory controls retained.~~
- ~~☐ Falsification, inconclusive outcomes and success distinguished.~~
- ~~☐ New cross-manifestation validation and restoration proposals
  explicitly decided.~~
- ~~☐ Scientific success and hackathon completion defined separately.~~

**Evidence:** G1 alignment record with PI decisions.

**Exit:** Ali can explain what is predicted, what is manipulated, what
is measured and what would count against the hypothesis.

### G2 — SCIENTIFIC DRY RUN

**Required outcome:** Demonstrate that the proposed experiment genuinely
tests the intended hypothesis.

**Owner:** Qualam. Independent reviews by Gemini + Grok. Ali approves
substantive decisions.

- ~~☐ Trace philosophical Form → observable representation.~~
- ~~☐ Trace Substance → upstream neural feature or feature set.~~
- ~~☐ Trace Causality → downstream processing component and output.~~
- ~~☐ Define the precise intervention site and causal comparison.~~
- ~~☐ Check that entity identity is distinguishable from lexical,
  grammatical, material and other property information.~~
- ~~☐ Verify that held-out property-matched entity/non-entity contrasts
  and grammatical-role variations can test residual identity
  selectivity.~~
- ~~☐ Explain how feature validation avoids circular selection.~~
- ~~☐ Verify intervention magnitude matching for C1 and C2.~~
- ~~☐ Verify comparability and logging of intervention layer, hook,
  affected feature dimensionality, actual activation-change norm, and
  token position for each mandatory control.~~
- ~~☐ Define evidence for contribution versus mandatory dependency.~~
- ~~☐ Separate output-level P2 from downstream pathway-specific P3
  disruption and restoration evidence.~~
- ~~☐ Check that measurements and proposed comparisons could falsify the
  hypothesis.~~
- ~~☐ Verify fidelity to every relevant original experimental
  specification.~~
- ~~☐ Gemini and Grok review independently with the original agenda.~~
- ~~☐ Resolve criticisms without unapproved scientific substitutions.~~

**Evidence:** Scientific traceability matrix, independent reviews, and
decision ledger.

**Exit:** Scientifically coherent protocol or explicit NO-GO with
reasons.

### ~~G3 — RECON~~

~~**Required outcome:** Verified technical feasibility and resource
estimates.~~

~~**Owner:** Qualam + Claude.~~

- ~~☐ Confirm access to the canonical AlignedLab repository.~~
- ~~☐ Inspect existing Epistemic Vacuum infrastructure for reusable
  components.~~
- ~~☐ Confirm model files, architecture, revision, tokenizer and
  licensing.~~
- ~~☐ Compare the original Gemma/Llama suggestions with any proposed
  substitute, including Qwen.~~
- ~~☐ Confirm activation-hook and patching support.~~
- ~~☐ Confirm local hardware feasibility and Modal availability.~~
- ~~☐ Verify GPU quotas, available credits, prices and permitted
  spending.~~
- ~~☐ Estimate implementation, GPU, review, debugging and submission
  effort.~~
- ~~☐ Identify the most likely technical blockers.~~
- ~~☐ Provide GO / CONDITIONAL GO / NO-GO recommendation.~~

**Evidence:** Claude reconnaissance report with commands, observed
system facts, resource estimates and blockers.

**Exit:** Ali approves the chosen infrastructure and authorized
expenditure, including any scientific model substitution.

### G4 — FEASIBILITY PROBE

**Required outcome:** Establish whether the experiment can work
technically and be measured meaningfully.

**Owner:** Claude executes; Qualam evaluates; Ali approves changes.

This is a time-boxed exploratory real-model test, not a miniature full
experiment.

- ~~☐ Confirm the model loads and produces forward-pass activations.~~
- ~~☐ Confirm a selected causal prompt has a measurable target
  prediction.~~
- ~~☐ Confirm activations can be recorded and modified.~~
- ~~☐ Try identifying one candidate entity representation.~~
- ~~☐ Verify that one targeted intervention changes a measurable
  signal.~~
- ~~☐ Construct both mandatory controls.~~
- ~~☐ Estimate noise, prompt variability and computational expense.~~
- ~~☐ Identify failure modes and practical sample-size requirements.~~
- ~~☐ Mark every probe result as exploratory and exclude probe examples
  from final held-out evaluation.~~
- CONDITIONAL GO.

**Evidence:** Probe configuration, raw measurements, logs, actual
runtime and cost.

**Exit:** Evidence that the intended experiment is technically
measurable, or a documented feasibility failure.

### G5 — SPEC FREEZE

**Required outcome:** One immutable experimental specification.

**Owner:** Ali.

FROZEN — PI approval 21 November 2026, 18:32 CET. Checklist completion
means the rules are approved, not that the pending G5r run or
independent review has succeeded. The tokenizer check is recorded as
complete in STATUS.md. Numerical layer/k outcomes are produced by the
frozen rules; this does not reopen the specification.

- ☑ Select model, exact revision, tokenizer and execution environment.
- ☑ Freeze experimental prompt families and inclusion rules.
- ☑ Freeze feature-discovery and validation datasets.
- ☑ Freeze property-matched P1 identity/non-identity contrasts,
  grammatical-role variation, and selectivity/stability thresholds.
- ☑ Freeze held-out testing data with no selection leakage.
- ☑ Freeze feature identification and downstream circuit localization
  procedures.
- ☑ Freeze intervention layer, site, strength, and matching rules.
- ☑ Freeze C1 and C2 implementations.
- ☑ Freeze the actual-activation-change norm matching tolerance,
  hook/layer/dimensionality and token-position matching rules,
  diagnostic records, and failure/exclusion handling for C1 and C2.
- ☑ Freeze pathway-specific disruption, downstream-component measures,
  and restoration criteria for any P3 necessity claim, including what to
  report if infeasible.
- ☑ Freeze primary and secondary measurements.
- ☑ Freeze sample size and the predeclared discovery-only
  probe-variability planning procedure within available resources.
- ☑ Freeze effect-size criteria, uncertainty method and multiplicity
  handling.
- ☑ Freeze falsification, inconclusive and success criteria.
- ☑ Freeze maximum expenditure and engineering contract.
- ☑ Ali approves the complete specification.

Cap remains \$30 (Modal). Spent ≈ \$0.38 (G4 ≈ \$0.056, G5v ≈ \$0.029,
tokenizer check \< \$0.01, G5r layers 2–10 ≈ \$0.287). G5r verification:
hard cap \$0.37 (20-min timeout) as frozen, raised by the PI to \$0.55
on 21 November 2026, 19:10 CET, to complete layers 11–14. Ceiling for G8
pilot + G10 full run: \$5, any excess escalated. Claude implements (G6),
Gemini + Grok review code (G7) against Protocol V1.1 and this document.

**Evidence:** Versioned, PI-approved frozen protocol.

**Exit:** No scientific design changes without explicit change control.

### G6 — ENGINEER

**Required outcome:** Implementation of the frozen specification.

**Owner:** Claude.

- [ ] Implement model loading and deterministic configuration.
- [ ] Implement datasets, prompt formatting and token verification.
- [ ] Implement activation recording and feature identification.
- [ ] Implement independent feature-validation procedure.
- [ ] Implement downstream component localization.
- [ ] Implement Substance intervention and activation restoration.
- [ ] Implement both mandatory controls.
- [ ] Implement primary and secondary measurements.
- [ ] Implement raw-result logging, failure capture and cost tracking.
- [ ] Produce reproducible execution commands.
- [ ] Report DONE / TESTED / NEXT / BLOCKER after each engineering
  chunk.
- [ ] Document all deviations, with scientific changes blocked pending
  Ali’s approval.

**Evidence:** Source files, tests, incremental commits and Claude
reports.

**Exit:** Complete implementation ready for deterministic verification.

### G7 — CODE DRY RUN

**Required outcome:** Correct implementation, not scientific
conclusions.

**Owner:** Claude. Independent reviews by Gemini + Grok.

- [ ] Check model inputs, tensor shapes and token indices.
- [ ] Verify that ablations modify only intended activations.
- [ ] Verify intervention magnitudes and matched controls.
- [ ] Test and log site/position/dimensionality matching, actual
  activation-change norms, tolerance failures, and the distinction
  between C1 and C2.
- [ ] Verify numerical calculation of target-token log-odds.
- [ ] Verify activation restoration behaves correctly on deterministic
  fixtures.
- [ ] Verify that P3 pathway-specific restoration affects only the
  intended path and that non-specific recovery cannot be labeled
  necessity evidence.
- [ ] Check dataset separation and prevent test-data leakage.
- [ ] Check random seeds and configuration logging.
- [ ] Run deterministic repeatability tests.
- [ ] Compare every implemented property against the frozen
  original-aligned specification.
- [ ] Gemini and Grok independently inspect code and tests.
- [ ] Resolve critical findings before proceeding.

**Evidence:** Test logs, failing and passing cases, independent code
reviews.

**Exit:** Correctness established to the degree required for a
real-model pilot.

### G8 — PILOT

**Required outcome:** Demonstrate that the completed pipeline operates
reliably on a real model.

**Owner:** Claude + Qualam.

- [ ] Execute the full pipeline on a small pilot dataset.
- [ ] Validate end-to-end feature selection and intervention.
- [ ] Confirm both controls execute correctly.
- [ ] Check metric stability and numerical health.
- [ ] Examine whether the feature-validation procedure is informative.
- [ ] Confirm downstream circuit measurements are available.
- [ ] Measure actual runtime, memory use and costs.
- [ ] Identify failures or unusable cases.
- [ ] Separate exploratory pilot observations from confirmatory
  evidence.
- [ ] Report readiness or request explicit scientific change control.

**Evidence:** Pilot raw data, summary diagnostics and readiness
decision.

**Exit:** Stable full pipeline or explicit return to the appropriate
gate.

### G9 — EXPERIMENT FREEZE

**Required outcome:** Lock the exact experiment to be executed.

**Owner:** Claude + Qualam.

- [ ] Record exact tested source-code commit.
- [ ] Record model identity, revision and checksum where available.
- [ ] Hash datasets, including held-out data.
- [ ] Record seeds, configurations and dependencies.
- [ ] Freeze intervention parameters and analysis scripts.
- [ ] Confirm no scientific design drift after G5.
- [ ] Confirm pilot observations have not been used to optimize held-out
  results.
- [ ] Verify repository status and reproducibility instructions.
- [ ] Record final execution authorization.

**Evidence:** Frozen manifest with paths, identifiers, hashes and
verified commit.

**Exit:** Exact execution state is reproducible and authorized.

### G10 — FULL RUN

**Required outcome:** Complete empirical dataset.

**Owner:** Claude.

- [ ] Execute the frozen experiment without unapproved modifications.
- [ ] Preserve intact-model predictions.
- [ ] Preserve targeted Substance intervention results.
- [ ] Preserve unrelated-feature control results.
- [ ] Preserve bystander-noun control results.
- [ ] Preserve downstream circuit and restoration measurements.
- [ ] Preserve each prompt’s metadata and inclusion status.
- [ ] Log errors, failed examples, retries and interruptions.
- [ ] Record actual compute time and costs.
- [ ] Verify completeness and raw-data integrity.

**Evidence:** Raw results, execution logs, cost records and integrity
manifest.

**Exit:** Complete, auditable dataset or documented partial run.

### G11 — ANALYSIS

**Required outcome:** Results and appropriately bounded interpretations.

**Owner:** Qualam + Claude.

- [ ] Calculate primary Substance-versus-C1 comparison.
- [ ] Calculate primary Substance-versus-C2 comparison.
- [ ] Calculate predefined effect sizes and uncertainty intervals.
- [ ] Analyze results across stimulus families and variations.
- [ ] Assess candidate Form selectivity and invariance.
- [ ] Report property-matched identity discrimination and sensitivity to
  grammatical role separately from material/affordance tracking.
- [ ] Analyze downstream circuit dependency and restoration.
- [ ] Treat any P3 necessity conclusion as conditional on
  pathway-specific intervention and restoration evidence, independently
  of P2 log-odds comparisons.
- [ ] Produce intervention-comparison figures.
- [ ] Produce a circuit diagram showing observed versus hypothetical
  pathways.
- [ ] Distinguish statistical evidence from causal interpretation.
- [ ] Assess confounds, null results and alternative explanations.
- [ ] State whether P1, P2 and P3 are independently supported,
  contradicted or inconclusive.
- [ ] Report limitations and untested claims.

**Evidence:** Reproducible analysis scripts, figures, result tables and
report draft.

**Exit:** All central claims traceable to actual data and predefined
comparisons.

### G12 — INDEPENDENT RESULTS REVIEW

**Required outcome:** Independent scientific challenge before
publication.

**Owner:** Gemini + Grok. Qualam integrates; Ali resolves.

- [ ] Give each reviewer the original agenda and frozen protocol.
- [ ] Supply raw data, code, tests, configurations and analysis.
- [ ] Have reviewers independently inspect methodological fidelity.
- [ ] Have reviewers independently examine statistical claims and
  uncertainty.
- [ ] Challenge feature identity and intervention specificity.
- [ ] Challenge any assertion of mandatory dependency.
- [ ] Attempt independent reproduction or targeted verification.
- [ ] Record disagreements and critical failures.
- [ ] Reconcile findings without concealing negative evidence.
- [ ] Ali approves final scientific interpretation.

**Evidence:** Two separate reviews, reconciliation record, approved
conclusions.

**Exit:** All material concerns resolved or transparently disclosed.

### G13 — SUBMISSION

**Required outcome:** Verified hackathon submission.

**Owner:** Team; Ali authorizes publication.

- [ ] Prepare clear project title and short research abstract.
- [ ] Explain the philosophical question and falsifiable prediction.
- [ ] Create judge-readable README with a minimal reproduction
  procedure.
- [ ] Include source code, dependencies, configurations and permissible
  data.
- [ ] Include experimental results with both mandatory controls.
- [ ] Include figures and a clear observed-versus-hypothesized circuit
  diagram.
- [ ] Include negative or inconclusive results and limitations.
- [ ] Disclose AI assistance and research contributions.
- [ ] Check repository accessibility and scientific claims.
- [ ] Prepare visual demonstration or concise project presentation.
- [ ] Submit to the official hackathon page before the deadline.
- [ ] Verify the public submission and preserve its URL and final
  commit.

**Evidence:** Accessible GitHub repository, final report, submission URL
and verification record.

**Exit:** Submission independently verifiable.

## D. Universal gate reporting rules

Every completed gate must produce:

**DONE:** What was completed.  
**EVIDENCE:** The exact output, file, commit, measurement or review
proving completion.  
**DECISION:** GO, CONDITIONAL GO or NO-GO, with the decision owner
identified.  
**NEXT:** The next authorized gate or corrective action.  
**BLOCKER:** Only an issue requiring Ali’s scientific decision or
expenditure authorization.

No gate is marked complete merely because an assistant says it is
complete.

### Scientific change control

For any substantive deviation:

**Original → Proposed change → Scientific impact → Recommendation →
Ali’s approval**

Scientific substitutions are forbidden without explicit approval.
Engineering details that do not alter scientific meaning remain
delegated to Claude.

## E. Success definitions

**Scientific success:** A valid, reproducible test with appropriately
controlled evidence, whether positive, negative or inconclusive.

**Hypothesis support:** Candidate entity representations generalize
sufficiently, targeted interventions have stronger effects than both
original controls, and pathway evidence supports the proposed
dependency.

**Strong necessity evidence:** A sufficiently characterized pathway is
shown to be required under specified conditions, with appropriate
intervention coverage and pathway-specific downstream
disruption/restoration evidence. A positive P2 output-log-odds effect
alone is insufficient. Universal necessity is outside scope.

**Hackathon completion:** A coherent empirical or clearly labeled
exploratory project is publicly submitted with documented methods,
evidence, limitations and accessible artifacts.

**Competitive ambition:** Maximize novelty, clarity, interpretability
contribution, reproducibility and judge accessibility without
sacrificing scientific validity.

## F. Current status and immediate next decision

**G0: DESIGN approved by the PI on 20 November 2026.  
G1: ALIGN approved by the PI on 20 November 2026.  
G2: SCIENTIFIC DRY RUN — independent Gemini and Grok reports received;
both recommended CONDITIONAL GO. Qualam consolidated the substantive
findings, and Ali explicitly approved the three refinements incorporated
in V1.1 on 20 November 2026. G2 scientific design decision: CONDITIONAL
GO. Quantitative implementation rules are now frozen at G5; this does
not imply empirical validation.  
G4: Exploratory feasibility probe exists in the canonical repository.  
G5: SPEC FREEZE FROZEN, PI-approved 21 November 2026, 18:32 CET. G5v
exploratory verification, the C2 selection bug fix, G5r repair design
and local tests, and the current versioned datasets exist.  
G6: ENGINEER is the next gate. No real causal experiment has been run.**

Verified repository state: aliadifenza/3_kantian_positions, commit
71670bcce82f91fca0b9e15ce4294d5a7d67fb3c. Scientific authority remains
Protocol V1.1 + protocol/G5_SPEC_FREEZE.md. G5r verification was
authorized; layers 2–10 ran on 21 November 2026 (Modal app
ap-HjkhUsCPA7dYzyRHoiX4Qw, ≈ \$0.287) and stopped at the 20-minute
budget; layers 11–14 are being run under the PI-raised \$0.55 cap and
merged under the frozen §7 rule (merge_g5r.py). Tokenizer verification
on the current dataset versions is recorded as DONE on 21 November 2026
(Modal app ap-gpbV9nmiJa8xUMBCZzE6RR); the full JSON is archived in
datasets/kantian_v1/audit/. G5r completion and independent review remain
pending. Spending recorded in STATUS.md: total ≈ \$0.38 of the \$30 cap.
Existing G4/G8 exploratory-versus-held-out safeguards remain mandatory
and unchanged.

**Tokenizer result under PI-approved Amendment 1: multi-token targets
excluded (impact evaluation 6, heating evaluation 10); alternatives no
longer exclude items. After tokenization: impact 90 items / 22 events /
3 kinds; heating 86 items; cutting 96 items. These are
tokenization-eligible counts, before the remaining inclusion and
matching checks. g5r_probe.py has been adapted to the dataset schema
(commits 88b9c87 and de20455), with local deterministic tests and no
frozen-rule changes. Next actions: complete G5r (layers 11–14), merge,
and record the selected layer/k or “no valid layer” (P2/P3
inconclusive). Proceed to G6 ENGINEER, then G7 CODE DRY RUN with
Gemini + Grok review against Protocol V1.1 + G5_SPEC_FREEZE.md. Failures
are reported without changing the scientific design. Any further
substantive scientific change requires explicit PI approval.**
