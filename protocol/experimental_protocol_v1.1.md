# Experiment 2 — The Kantian Positional Logic Test

**Project:** AlignedLab / Mechanistic Interpretability Hackathon #5  
**Principal Investigator:** Dr. Ali Antonio Di Fenza  
**Research Director:** Qualam (GPT-6)  
**Engineering Lead:** Claude  
**Independent Reviewers:** Gemini and Grok  
**Version:** 1.1 — Final PI-approved G2 revision (20 November 2026)  
**Status:** G0 DESIGN and G1 ALIGN approved; G2 SCIENTIFIC DRY RUN CONDITIONAL GO, with PI-approved refinements; G3 RECON authorized (not yet executed)

---

## A. Scientific foundation

### Philosophical question

Can Plato's Forms and Kant's conditions of possibility explain and predict the organization of neural-network representations and their downstream circuits?

### Core hypothesis

A neural network learns organizing structures that permit recognition across different manifestations of a phenomenon. These structures, interpreted as *Forms*, function as conditions of possibility for downstream recognition and processing.

A Form is not necessarily a perfect exemplar or a single feature. It may be a multidimensional, distributed structure defining which variations can still be recognized as manifestations of the same phenomenon.

### Specific experimental hypothesis

**Substance → Causality**

The internal representation establishing the identity of an entity is a necessary upstream component of a downstream circuit processing its causal interactions.

This is the original Kantian Positional Logic hypothesis.

### Three predictions

**P1 — Form / invariance**

The model contains identifiable representational structure that remains sufficiently stable across different manifestations of an entity while distinguishing it from other entities.

**P2 — Causal dependency**

Disrupting that structure damages the correct causal-outcome prediction more than either mandatory control.

**P3 — Mandatory pathway**

The downstream causal computation relies on the identified upstream structure. Disrupting the relevant pathway impairs the computation, and restoring its information should restore some or all of the lost functionality.

P1, P2, and P3 have different evidential requirements. P1 does not establish P2; P2 does not automatically establish P3.

### Scope

- Primary system: one lightweight pretrained language model.
- Primary phenomenon: entity representations and causal physical interactions expressed in language.
- Primary method: mechanistic interpretability with internal activation measurements and interventions.
- Excluded claims: universal Kantian necessity, consciousness, visual perception, or a complete account of all Forms.
- Broader philosophical hypothesis remains unchanged regardless of the outcome of this particular experiment.

---

## B. Proposed experimental design

### B1. Stimuli

The experiment uses controlled causal sentences.

| Family | Example | Target outcome |
|---|---|---|
| Impact | The bowling ball struck the fragile glass vase, causing the vase to… | shatter |
| Heating | The flame heated the ice cube, causing the ice to… | melt |
| Cutting | The knife sliced through the rope, causing the rope to… | snap |

Examples are candidates, not validated stimuli. Their baseline predictions and tokenization must be checked.

For the primary vase family, vary properties such as material, shape, and context to investigate representational invariance. Include non-vases and ambiguous objects to test category boundaries.

For example, a steel vase may remain recognizable as a vase while making *shatter* an inappropriate outcome. This distinction helps separate the structural representation of vasehood from material properties relevant to the causal prediction.

### B2. Feature identification

**Objective:** Find internal representations associated with the entity, not simply the occurrence of its name.

Proposed procedure:

1. Present multiple manifestations of the entity in varied sentence contexts.
2. Record internal activations across candidate layers and token positions.
3. Identify candidate features, directions, or distributed feature sets associated with that entity.
4. Test whether these candidates generalize to held-out contexts and distinguish the entity from relevant non-entities.
5. Check whether their selectivity persists across property changes rather than merely tracking one material, descriptor, token, or grammatical position.
6. **Required P1 identity validation (PI-approved G2 refinement):** In held-out, property-matched contrasts, test whether the candidate representation still discriminates the intended entity from non-entities when material, fragility, affordances, and descriptive context are matched or varied independently. Also vary grammatical role and position so selectivity cannot be explained solely by noun placement or object/patient tracking. For the same entity, verify stability across such changes; for matched non-entities, verify discrimination.
7. At G5, preregister the candidate-selection procedure, property-matched contrast construction, held-out split, quantitative selectivity/stability criteria, and failure/inconclusive thresholds before confirmatory tests. A candidate driven by property or grammatical role without residual entity identity does not qualify as validated Substance for P1 or subsequent Substance-specific claims.

**Operational definition of a candidate Form:** An identified internal representation with demonstrated selectivity and stability across the tested variations.

This is a measurable proxy for the philosophical Form, not a definition of Form itself.

Important limitation: a text model tested on sentences establishes linguistic or conceptual invariance only. It does not establish visual recognition across actual images.

### B3. Downstream circuit identification

Identify candidate downstream attention heads or connected components whose activity contributes to the causal-outcome prediction.

Test whether these components:

- respond to relevant causal relationships;
- contribute causally to the target prediction;
- receive or use information from the identified entity representations.

The original diagram's Layer N → Layer N+1 is conceptual. Actual components need not occupy adjacent layers.

### B4. Intervention

Leave the input sentence unchanged.

At a validated upstream activation site, suppress or replace the component corresponding to the candidate entity representation. Preserve other activation components as far as the chosen method permits.

Run separately:

- Intervention on the first causal entity.
- Intervention on the second causal entity.
- Combined entity intervention.
- Activation restoration/patching where technically feasible.

Use the same prompt and model weights throughout each comparison.

**P3 pathway-specific evidence (PI-approved G2 refinement):** A claim of mandatory Substance → Causality dependency requires (i) a localized upstream Substance representation and identified downstream interaction component(s), (ii) targeted disruption with measurements showing the downstream component uses information from that upstream site, and (iii) pathway-specific restoration/patching that recovers downstream processing and, where applicable, the output log-odds signal. Global recovery or reintroduction of correlated information through an unspecified alternate channel does not by itself establish the claimed pathway. If pathway isolation or specific restoration is technically infeasible, report P3 as inconclusive/unestablished rather than inferring necessity from P2. These are pathway-evidence requirements, not a replacement or co-primary metric for B6.

### B5. Mandatory controls

**Control C1 — Unrelated features**

Suppress unrelated internal features using interventions matched in magnitude and, where possible, location and other relevant properties.

Purpose: distinguish entity-specific dependency from generic damage caused by altering model activations.

**Control C2 — Irrelevant bystander noun**

Introduce a grammatically present but causally irrelevant noun.

Example:

*As a clock ticked on the shelf, the bowling ball struck the fragile glass vase, causing the vase to…*

Suppress the representation associated with *clock*, using a comparable intervention.

Purpose: distinguish the hypothesized dependency from generic noun tracking or attention routing.

**C1/C2 intervention-matching rules (PI-approved G2 refinement; exact numerical settings frozen at G5):** For each confirmatory comparison, use the same model checkpoint, target continuation, activation-ablation operator, and intervention layer/hook class. Match and record the actual L2 norm of the intervention-induced activation change at the affected site, together with the number/dimensionality of intervened features; do not infer equality merely from applying the same scalar coefficient. For C1, apply unrelated-feature ablation at the same candidate activation site and target token position(s), where feature selectivity permits. For C2, apply the corresponding noun-representation intervention at the bystander noun token position(s); match layer, hook, operator, intervention dimensionality, and actual change magnitude, while explicitly recording unavoidable differences in token position, grammatical role, or salience. Never describe C2 as position-matched when it is not. At G5 freeze the feature/position matching algorithm, numerical magnitude tolerance, token-span handling, paired-stimulus rules, exclusion/failure criteria, and logging fields; apply them identically without consulting confirmatory outcomes. Comparisons that fail required matching checks are flagged as non-confirmatory, not interpreted as positive evidence. Both original controls remain mandatory and distinct.

**Exploratory diagnostic — Material property**

Separately intervene on a candidate material or fragility representation, if one can be identified. This helps distinguish Substance from outcome-relevant properties. It does not replace C1 or C2 and is not a third mandatory control.

### B6. Primary measurement

Measure the change in **log-odds of the correct causal-outcome token** between the intact and intervened model.

For a target token with probability \(p\):

\[
L=\log\left(\frac{p}{1-p}\right)
\]

For each intervention:

\[
D=L_{\mathrm{intact}}-L_{\mathrm{intervened}}
\]

Positive \(D\) means the intervention reduced the target prediction.

The primary comparisons are:

\[
D_{\mathrm{Substance}}-D_{\mathrm{Unrelated}}
\]

\[
D_{\mathrm{Substance}}-D_{\mathrm{Bystander}}
\]

Both should be positive, with sufficient statistical evidence under the analysis plan frozen at G5.

Only targets with valid, documented tokenization will enter the primary next-token analysis. Multi-token outcomes require a separately approved measurement rule.

### B7. Secondary measurements

- Selectivity and generalization of candidate entity features.
- Changes in identified downstream circuit activations.
- Effects of individual versus combined entity interventions.
- Degree of recovery following activation restoration.
- Changes in alternative outcome predictions.
- Sensitivity to property and sentence variations.

Secondary measurements cannot retroactively replace the primary metric. For P3, downstream-component intervention and pathway-specific restoration evidence is required **before making a necessity claim**; these measurements remain secondary to the unchanged primary B6 log-odds measurement.

### B8. Falsification and interpretation

| Observation | Interpretation |
|---|---|
| Substance intervention produces greater damage than both controls | Supports a specific dependency |
| Substance intervention is no more damaging than unrelated-feature control | Does not support the predicted specificity |
| Bystander intervention produces comparable damage | Does not support the proposed special entity dependency |
| Entity features cannot be identified or validated | Inconclusive |
| Target prediction is already weak or unstable | Inconclusive for that stimulus |
| Downstream prediction survives the purportedly complete intervention | Counts against necessity of that targeted pathway |
| Selective disruption followed by restoration recovers performance | Stronger causal evidence for the targeted pathway |

A mandatory pathway requires stronger evidence than a statistically significant output change. Even successful interventions establish necessity only relative to the tested representations, model, tasks, and intervention coverage. **Interpretation rule:** P1 selectivity, P2 log-odds specificity, and P3 pathway necessity must be reported separately. P2 superiority over both controls supports causal contribution to the output; it cannot alone establish a specific downstream Causality circuit or a mandatory Substance → Causality route. P3 necessity is supported only by adequately selective, pathway-specific disruption and restoration evidence; otherwise mark P3 inconclusive or contradicted as indicated by the observations.

---

## C. Gate execution and verification

| Gate | Name | Required outcome | Owner / Review |
|---|---|---|---|
| G0 | DESIGN | Agree workflow, roles, deadlines, resources, authority boundaries, independent reviews | Ali + Qualam |
| G1 | ALIGN | Confirm philosophical question, hypothesis, objectives, scope, predictions, falsification, success | Ali + Qualam |
| G2 | SCIENTIFIC DRY RUN | Verify question → scientific problem → variables → measurements → controls → falsification | Qualam; Gemini + Grok; Ali |
| G3 | RECON | Verify repositories, model, Modal, prior infrastructure, feasibility, time, cost and GO status | Qualam + Claude |
| G4 | FEASIBILITY PROBE | Real-model exploratory test of intervention sensitivity, controls and variability | Claude; Qualam; Ali |
| G5 | SPEC FREEZE | Freeze complete approved experimental and analysis specification | Ali |
| G6 | ENGINEER | Implement in small verified chunks | Claude |
| G7 | CODE DRY RUN | Test implementation, intervention isolation, metrics, reproducibility and fidelity | Claude; Gemini + Grok |
| G8 | PILOT | Small real-model test of the completed implementation | Claude + Qualam |
| G9 | EXPERIMENT FREEZE | Lock tested code, model, data, seeds, configuration and commit | Claude + Qualam |
| G10 | FULL RUN | Execute approved experiment and preserve evidence | Claude |
| G11 | ANALYSIS | Calculate predefined results, effects, uncertainty and limitations | Qualam + Claude |
| G12 | INDEPENDENT RESULTS REVIEW | Independent examination of evidence, methods and claims | Gemini + Grok; Ali |
| G13 | SUBMISSION | Final public artifacts and verified hackathon submission | Team |

### G0 — DESIGN

**Required outcome:** Approved operational framework.

**Owner:** Ali + Qualam.

- [ ] Revised G0–G13 workflow accepted without renumbering or skipped gates.
- [ ] Ali's scientific authority and Claude's engineering delegation recorded.
- [ ] Gemini and Grok's independent-review responsibilities recorded.
- [ ] Deadline and Saturday work constraint recognized.
- [ ] Resource budget, spending authorization boundary and escalation rules agreed.
- [ ] Required evidence and gate-signoff format established.

**Evidence:** Approved workflow and roles record.

**Exit:** Ali explicitly approves the operating framework.

### G1 — ALIGN

**Required outcome:** Locked scientific intent.

**Owner:** Ali + Qualam.

- [ ] Original philosophical question reproduced faithfully.
- [ ] Relationship between Form, Substance, recognition and Causality explained.
- [ ] Specific Substance → Causality hypothesis confirmed.
- [ ] P1, P2 and P3 explicitly accepted, modified, or rejected.
- [ ] Scope limited to claims the experiment can investigate.
- [ ] Primary measurement verified against the original agenda.
- [ ] Both original mandatory controls retained.
- [ ] Falsification, inconclusive outcomes and success distinguished.
- [ ] New cross-manifestation validation and restoration proposals explicitly decided.
- [ ] Scientific success and hackathon completion defined separately.

**Evidence:** G1 alignment record with PI decisions.

**Exit:** Ali can explain what is predicted, what is manipulated, what is measured and what would count against the hypothesis.

### G2 — SCIENTIFIC DRY RUN

**Required outcome:** Demonstrate that the proposed experiment genuinely tests the intended hypothesis.

**Owner:** Qualam. Independent reviews by Gemini + Grok. Ali approves substantive decisions.

- [ ] Trace philosophical Form → observable representation.
- [ ] Trace Substance → upstream neural feature or feature set.
- [ ] Trace Causality → downstream processing component and output.
- [ ] Define the precise intervention site and causal comparison.
- [ ] Check that entity identity is distinguishable from lexical, grammatical, material and other property information.
- [ ] Verify that held-out property-matched entity/non-entity contrasts and grammatical-role variations can test residual identity selectivity.
- [ ] Explain how feature validation avoids circular selection.
- [ ] Verify intervention magnitude matching for C1 and C2.
- [ ] Verify comparability and logging of intervention layer, hook, affected feature dimensionality, actual activation-change norm, and token position for each mandatory control.
- [ ] Define evidence for contribution versus mandatory dependency.
- [ ] Separate output-level P2 from downstream pathway-specific P3 disruption and restoration evidence.
- [ ] Check that measurements and proposed comparisons could falsify the hypothesis.
- [ ] Verify fidelity to every relevant original experimental specification.
- [ ] Gemini and Grok review independently with the original agenda.
- [ ] Resolve criticisms without unapproved scientific substitutions.

**Evidence:** Scientific traceability matrix, independent reviews, and decision ledger.

**Exit:** Scientifically coherent protocol or explicit NO-GO with reasons.

### G3 — RECON

**Required outcome:** Verified technical feasibility and resource estimates.

**Owner:** Qualam + Claude.

- [ ] Confirm access to the canonical AlignedLab repository.
- [ ] Inspect existing Epistemic Vacuum infrastructure for reusable components.
- [ ] Confirm model files, architecture, revision, tokenizer and licensing.
- [ ] Compare the original Gemma/Llama suggestions with any proposed substitute, including Qwen.
- [ ] Confirm activation-hook and patching support.
- [ ] Confirm local hardware feasibility and Modal availability.
- [ ] Verify GPU quotas, available credits, prices and permitted spending.
- [ ] Estimate implementation, GPU, review, debugging and submission effort.
- [ ] Identify the most likely technical blockers.
- [ ] Provide GO / CONDITIONAL GO / NO-GO recommendation.

**Evidence:** Claude reconnaissance report with commands, observed system facts, resource estimates and blockers.

**Exit:** Ali approves the chosen infrastructure and authorized expenditure, including any scientific model substitution.

### G4 — FEASIBILITY PROBE

**Required outcome:** Establish whether the experiment can work technically and be measured meaningfully.

**Owner:** Claude executes; Qualam evaluates; Ali approves changes.

This is a time-boxed exploratory real-model test, not a miniature full experiment.

- [ ] Confirm the model loads and produces forward-pass activations.
- [ ] Confirm a selected causal prompt has a measurable target prediction.
- [ ] Confirm activations can be recorded and modified.
- [ ] Try identifying one candidate entity representation.
- [ ] Verify that one targeted intervention changes a measurable signal.
- [ ] Construct both mandatory controls.
- [ ] Estimate noise, prompt variability and computational expense.
- [ ] Identify failure modes and practical sample-size requirements.
- [ ] Mark every probe result as exploratory and exclude probe examples from final held-out evaluation.
- [ ] Report GO / CONDITIONAL GO / NO-GO.

**Evidence:** Probe configuration, raw measurements, logs, actual runtime and cost.

**Exit:** Evidence that the intended experiment is technically measurable, or a documented feasibility failure.

### G5 — SPEC FREEZE

**Required outcome:** One immutable experimental specification.

**Owner:** Ali.

- [ ] Select model, exact revision, tokenizer and execution environment.
- [ ] Freeze experimental prompt families and inclusion rules.
- [ ] Freeze feature-discovery and validation datasets.
- [ ] Freeze property-matched P1 identity/non-identity contrasts, grammatical-role variation, and selectivity/stability thresholds.
- [ ] Freeze held-out testing data with no selection leakage.
- [ ] Freeze feature identification and downstream circuit localization procedures.
- [ ] Freeze intervention layer, site, strength, and matching rules.
- [ ] Freeze C1 and C2 implementations.
- [ ] Freeze the actual-activation-change norm matching tolerance, hook/layer/dimensionality and token-position matching rules, diagnostic records, and failure/exclusion handling for C1 and C2.
- [ ] Freeze pathway-specific disruption, downstream-component measures, and restoration criteria for any P3 necessity claim, including what to report if infeasible.
- [ ] Freeze primary and secondary measurements.
- [ ] Freeze sample size using probe variability and available resources.
- [ ] Freeze effect-size criteria, uncertainty method and multiplicity handling.
- [ ] Freeze falsification, inconclusive and success criteria.
- [ ] Freeze maximum expenditure and engineering contract.
- [ ] Ali approves the complete specification.

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
- [ ] Report DONE / TESTED / NEXT / BLOCKER after each engineering chunk.
- [ ] Document all deviations, with scientific changes blocked pending Ali's approval.

**Evidence:** Source files, tests, incremental commits and Claude reports.

**Exit:** Complete implementation ready for deterministic verification.

### G7 — CODE DRY RUN

**Required outcome:** Correct implementation, not scientific conclusions.

**Owner:** Claude. Independent reviews by Gemini + Grok.

- [ ] Check model inputs, tensor shapes and token indices.
- [ ] Verify that ablations modify only intended activations.
- [ ] Verify intervention magnitudes and matched controls.
- [ ] Test and log site/position/dimensionality matching, actual activation-change norms, tolerance failures, and the distinction between C1 and C2.
- [ ] Verify numerical calculation of target-token log-odds.
- [ ] Verify activation restoration behaves correctly on deterministic fixtures.
- [ ] Verify that P3 pathway-specific restoration affects only the intended path and that non-specific recovery cannot be labeled necessity evidence.
- [ ] Check dataset separation and prevent test-data leakage.
- [ ] Check random seeds and configuration logging.
- [ ] Run deterministic repeatability tests.
- [ ] Compare every implemented property against the frozen original-aligned specification.
- [ ] Gemini and Grok independently inspect code and tests.
- [ ] Resolve critical findings before proceeding.

**Evidence:** Test logs, failing and passing cases, independent code reviews.

**Exit:** Correctness established to the degree required for a real-model pilot.

### G8 — PILOT

**Required outcome:** Demonstrate that the completed pipeline operates reliably on a real model.

**Owner:** Claude + Qualam.

- [ ] Execute the full pipeline on a small pilot dataset.
- [ ] Validate end-to-end feature selection and intervention.
- [ ] Confirm both controls execute correctly.
- [ ] Check metric stability and numerical health.
- [ ] Examine whether the feature-validation procedure is informative.
- [ ] Confirm downstream circuit measurements are available.
- [ ] Measure actual runtime, memory use and costs.
- [ ] Identify failures or unusable cases.
- [ ] Separate exploratory pilot observations from confirmatory evidence.
- [ ] Report readiness or request explicit scientific change control.

**Evidence:** Pilot raw data, summary diagnostics and readiness decision.

**Exit:** Stable full pipeline or explicit return to the appropriate gate.

### G9 — EXPERIMENT FREEZE

**Required outcome:** Lock the exact experiment to be executed.

**Owner:** Claude + Qualam.

- [ ] Record exact tested source-code commit.
- [ ] Record model identity, revision and checksum where available.
- [ ] Hash datasets, including held-out data.
- [ ] Record seeds, configurations and dependencies.
- [ ] Freeze intervention parameters and analysis scripts.
- [ ] Confirm no scientific design drift after G5.
- [ ] Confirm pilot observations have not been used to optimize held-out results.
- [ ] Verify repository status and reproducibility instructions.
- [ ] Record final execution authorization.

**Evidence:** Frozen manifest with paths, identifiers, hashes and verified commit.

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
- [ ] Preserve each prompt's metadata and inclusion status.
- [ ] Log errors, failed examples, retries and interruptions.
- [ ] Record actual compute time and costs.
- [ ] Verify completeness and raw-data integrity.

**Evidence:** Raw results, execution logs, cost records and integrity manifest.

**Exit:** Complete, auditable dataset or documented partial run.

### G11 — ANALYSIS

**Required outcome:** Results and appropriately bounded interpretations.

**Owner:** Qualam + Claude.

- [ ] Calculate primary Substance-versus-C1 comparison.
- [ ] Calculate primary Substance-versus-C2 comparison.
- [ ] Calculate predefined effect sizes and uncertainty intervals.
- [ ] Analyze results across stimulus families and variations.
- [ ] Assess candidate Form selectivity and invariance.
- [ ] Report property-matched identity discrimination and sensitivity to grammatical role separately from material/affordance tracking.
- [ ] Analyze downstream circuit dependency and restoration.
- [ ] Treat any P3 necessity conclusion as conditional on pathway-specific intervention and restoration evidence, independently of P2 log-odds comparisons.
- [ ] Produce intervention-comparison figures.
- [ ] Produce a circuit diagram showing observed versus hypothetical pathways.
- [ ] Distinguish statistical evidence from causal interpretation.
- [ ] Assess confounds, null results and alternative explanations.
- [ ] State whether P1, P2 and P3 are independently supported, contradicted or inconclusive.
- [ ] Report limitations and untested claims.

**Evidence:** Reproducible analysis scripts, figures, result tables and report draft.

**Exit:** All central claims traceable to actual data and predefined comparisons.

### G12 — INDEPENDENT RESULTS REVIEW

**Required outcome:** Independent scientific challenge before publication.

**Owner:** Gemini + Grok. Qualam integrates; Ali resolves.

- [ ] Give each reviewer the original agenda and frozen protocol.
- [ ] Supply raw data, code, tests, configurations and analysis.
- [ ] Have reviewers independently inspect methodological fidelity.
- [ ] Have reviewers independently examine statistical claims and uncertainty.
- [ ] Challenge feature identity and intervention specificity.
- [ ] Challenge any assertion of mandatory dependency.
- [ ] Attempt independent reproduction or targeted verification.
- [ ] Record disagreements and critical failures.
- [ ] Reconcile findings without concealing negative evidence.
- [ ] Ali approves final scientific interpretation.

**Evidence:** Two separate reviews, reconciliation record, approved conclusions.

**Exit:** All material concerns resolved or transparently disclosed.

### G13 — SUBMISSION

**Required outcome:** Verified hackathon submission.

**Owner:** Team; Ali authorizes publication.

- [ ] Prepare clear project title and short research abstract.
- [ ] Explain the philosophical question and falsifiable prediction.
- [ ] Create judge-readable README with a minimal reproduction procedure.
- [ ] Include source code, dependencies, configurations and permissible data.
- [ ] Include experimental results with both mandatory controls.
- [ ] Include figures and a clear observed-versus-hypothesized circuit diagram.
- [ ] Include negative or inconclusive results and limitations.
- [ ] Disclose AI assistance and research contributions.
- [ ] Check repository accessibility and scientific claims.
- [ ] Prepare visual demonstration or concise project presentation.
- [ ] Submit to the official hackathon page before the deadline.
- [ ] Verify the public submission and preserve its URL and final commit.

**Evidence:** Accessible GitHub repository, final report, submission URL and verification record.

**Exit:** Submission independently verifiable.

---

## D. Universal gate reporting rules

Every completed gate must produce:

**DONE:** What was completed.  
**EVIDENCE:** The exact output, file, commit, measurement or review proving completion.  
**DECISION:** GO, CONDITIONAL GO or NO-GO, with the decision owner identified.  
**NEXT:** The next authorized gate or corrective action.  
**BLOCKER:** Only an issue requiring Ali's scientific decision or expenditure authorization.

No gate is marked complete merely because an assistant says it is complete.

### Scientific change control

For any substantive deviation:

**Original → Proposed change → Scientific impact → Recommendation → Ali's approval**

Scientific substitutions are forbidden without explicit approval. Engineering details that do not alter scientific meaning remain delegated to Claude.

---

## E. Success definitions

**Scientific success:** A valid, reproducible test with appropriately controlled evidence, whether positive, negative or inconclusive.

**Hypothesis support:** Candidate entity representations generalize sufficiently, targeted interventions have stronger effects than both original controls, and pathway evidence supports the proposed dependency.

**Strong necessity evidence:** A sufficiently characterized pathway is shown to be required under specified conditions, with appropriate intervention coverage and pathway-specific downstream disruption/restoration evidence. A positive P2 output-log-odds effect alone is insufficient. Universal necessity is outside scope.

**Hackathon completion:** A coherent empirical or clearly labeled exploratory project is publicly submitted with documented methods, evidence, limitations and accessible artifacts.

**Competitive ambition:** Maximize novelty, clarity, interpretability contribution, reproducibility and judge accessibility without sacrificing scientific validity.

---

## F. Current status and immediate next decision

**G0:** DESIGN approved by the PI on 20 November 2026.  
**G1:** ALIGN approved by the PI on 20 November 2026.  
**G2:** SCIENTIFIC DRY RUN — independent Gemini and Grok reports received; both recommended CONDITIONAL GO. Qualam consolidated the substantive findings, and Ali explicitly approved the three refinements incorporated in V1.1 on 20 November 2026. G2 scientific design decision: CONDITIONAL GO. Remaining quantitative implementation details are to be fixed at G5; this does not imply empirical validation.  
**G3:** RECON authorized as the next gate; not executed in this protocol revision.  
**G4–G13:** Not executed.

The two independent G2 design reviews have been received. No model, feature, circuit, empirical result, compute expenditure, or GitHub commit/push has been verified by this protocol revision. Existing G4/G8 exploratory-versus-held-out safeguards remain mandatory and unchanged.

**Next action:** Proceed to G3 RECON with Claude. Preserve the original agenda, this V1.1 protocol, and the independent G2 reviews. Resolve model choice, resources, and practical feasibility at the proper gates; freeze all quantitative rules and criteria at G5 before confirmatory execution. Any further substantive scientific change requires explicit PI approval.