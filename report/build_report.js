// Builds report/AlignedLab_Exp3_Kantian_Positional_Logic_report.docx (Apart template structure, Baskerville, US Letter).
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, WidthType, ShadingType,
  AlignmentType, BorderStyle, LevelFormat, FootnoteReferenceRun, PageBreak, HeadingLevel, ExternalHyperlink,
} = require("docx");

const FONT = "Baskerville";
const R = path.join(__dirname, "..");
const INK = "1A1A1A", MUTED = "555555", RULE = "1A1A1A";

// ---------- inline markup: **bold**, *italic*
function runs(text, base = {}) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*)/g;
  let last = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), font: FONT, ...base }));
    const t = m[0];
    if (t.startsWith("**")) out.push(new TextRun({ text: t.slice(2, -2), bold: true, font: FONT, ...base }));
    else out.push(new TextRun({ text: t.slice(1, -1), italics: true, font: FONT, ...base }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), font: FONT, ...base }));
  return out;
}
const P = (t, o = {}) => new Paragraph({ children: runs(t, o.run || {}), spacing: { after: 120, line: 276 }, alignment: o.align || AlignmentType.JUSTIFIED, ...(o.p || {}) });
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun({ text: t, font: FONT })] });
const H3 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun({ text: t, font: FONT })] });
const B = (t, lvl = 0) => new Paragraph({ numbering: { reference: "bul", level: lvl }, children: runs(t), spacing: { after: 60, line: 264 } });
const N = (t, ref = "num") => new Paragraph({ numbering: { reference: ref, level: 0 }, children: runs(t), spacing: { after: 60, line: 264 } });
const cap = (t) => new Paragraph({ children: runs(t, { size: 19, color: MUTED }), spacing: { before: 60, after: 200 }, alignment: AlignmentType.LEFT });
const quote = (t) => new Paragraph({ children: runs(t, { italics: true }), indent: { left: 567, right: 567 }, spacing: { after: 140 } });

function table(header, rows, widths, opts = {}) {
  const total = widths.reduce((a, b) => a + b, 0);
  const border = { style: BorderStyle.SINGLE, size: 4, color: "BBBBBB" };
  const borders = { top: border, bottom: border, left: border, right: border };
  const cell = (t, i, head) => new TableCell({
    borders, width: { size: widths[i], type: WidthType.DXA },
    shading: head ? { fill: "EFEDE8", type: ShadingType.CLEAR, color: "auto" } : undefined,
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    children: [new Paragraph({ children: runs(String(t), { size: opts.size || 18, bold: head }), spacing: { after: 0, line: 240 } })],
  });
  return new Table({
    width: { size: total, type: WidthType.DXA }, columnWidths: widths,
    rows: [new TableRow({ tableHeader: true, children: header.map((h, i) => cell(h, i, true)) }),
           ...rows.map((r) => new TableRow({ cantSplit: true, children: r.map((c, i) => cell(c, i, false)) }))],
  });
}
function figure(file, wIn, hIn) {
  return new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 40 },
    children: [new ImageRun({ type: "png", data: fs.readFileSync(path.join(R, file)), transformation: { width: wIn * 96, height: hIn * 96 },
      altText: { title: path.basename(file), description: path.basename(file), name: path.basename(file) } })] });
}
const space = () => new Paragraph({ children: [], spacing: { after: 60 } });

// =====================================================================================
const C = [];

// ---------- title block (mirrors the Apart template)
C.push(new Paragraph({ border: { top: { style: BorderStyle.SINGLE, size: 12, color: RULE, space: 12 } }, spacing: { before: 600, after: 0 }, children: [] }));
C.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 120 }, children: [
  new TextRun({ text: "Decodable but Not Isolable: Testing Whether an Object’s Identity Is a Condition of Possibility for Causal Prediction in Gemma-2-2B", font: FONT, size: 34, bold: true }),
  new FootnoteReferenceRun(1)] }));
C.push(new Paragraph({ border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: RULE, space: 12 } }, spacing: { after: 240 }, children: [] }));
C.push(new Paragraph({ alignment: AlignmentType.CENTER, children: runs("Ali Antonio Di Fenza", { size: 22 }), spacing: { after: 0 } }));
C.push(new Paragraph({ alignment: AlignmentType.CENTER, children: runs("AlignedLab, Berlin", { size: 20, italics: true }), spacing: { after: 160 } }));
C.push(new Paragraph({ alignment: AlignmentType.CENTER, children: runs("**With**", { size: 20 }), spacing: { after: 0 } }));
C.push(new Paragraph({ alignment: AlignmentType.CENTER, children: runs("Apart Research", { size: 20 }), spacing: { after: 280 } }));
C.push(new Paragraph({ alignment: AlignmentType.CENTER, children: runs("**Abstract**", { size: 26 }), spacing: { after: 120 } }));
C.push(new Paragraph({ alignment: AlignmentType.JUSTIFIED, indent: { left: 720, right: 720 }, spacing: { after: 200, line: 264 }, children: runs(
  "Can ideas from philosophy (Plato’s Forms and Kant’s conditions of possibility) be turned into testable claims about how a language model is organized? We test one such claim: that a model’s representation of *what a thing is* (Kant’s Substance) is a necessary upstream condition for its prediction of *what happens to it* (Causality). In Gemma-2-2B we froze a gated design in advance: find a representation that tells nine kinds of objects apart from property-matched look-alikes, switch it off, and check whether the correct causal outcome (“the egg fell … and *cracked*”) suffers more than under two controls. Identity was easy to read: simple linear probes separated the kinds almost perfectly (AUC 0.93–1.0) at every layer tested. But it could not be switched off: removing up to eight sparse-autoencoder (SAE) features removed none of it downstream in all 156 layer-by-kind tests. Our frozen rules therefore stopped the experiment before the causal test, and the held-out evaluation set was never opened. Exploratory follow-ups showed that even 128 SAE features remove nothing; erasing a whole linear identity direction removes a modest, specific part; and erasing it where the model later reads it removes more, yet four layers on much of the signal is readable again. The Kantian dependency remains untested. We contribute a reproducible operationalization of a philosophical hypothesis, evidence that a highly decodable concept resists sparse SAE ablation, which matters for SAE-based unlearning and steering, and a transparent record of a human–AI research workflow, failures included.",
  { italics: true, size: 21 }) }));
C.push(new Paragraph({ children: [new PageBreak()] }));

// ---------- 1. Introduction
C.push(H2("1. Introduction"));
C.push(H3("1.1 The goal behind this work"));
C.push(P("This study is the third experiment of AlignedLab, an independent research lab in Berlin led by Ali Antonio Di Fenza, a philosopher (PhD, Cornell) working on AI alignment. The lab’s goal is to **find new research paths by applying philosophical concepts to machine learning**: to translate a philosophical idea into something that can be operationalized and tested, and to gather initial evidence toward new, more radical ways of understanding and aligning AI systems. The experiments are deliberately small and fast. Each one asks whether an idea can be made testable at all and what the first evidence looks like, not whether the idea is settled. Negative and inconclusive results count as results."));
C.push(H3("1.2 From the research agenda to this experiment"));
C.push(P("The lab works from a research agenda by the principal investigator (PI), *Aligning Latent Spaces: From Possibility to Power — A Research Agenda for a Modal Theory of Native Alignment*. Its starting intuition is that a model’s learned representations can be studied as a structured space of possibilities, and that philosophical categories may describe how that space is organized. Chapter 1 of the agenda led to AlignedLab Experiment 2, the “I Don’t Know” Vacuum Test (inconclusive primary test, reported separately). This report covers **Chapter 2, Experiment 2 of the agenda, which is AlignedLab Experiment 3**: the Kantian Positional Logic Test."));
C.push(P("The agenda’s original version of the experiment was purely Kantian. It proposed that “a relational circuit (Causality) cannot function without drawing from an upstream entity circuit (Substance).” The proposed test was to knock out the features that track an entity (the vase in “The heavy bowling ball rolled into the fragile glass vase, causing the vase to *shatter*”) and measure whether the model’s confidence in the correct outcome drops more than when an equal amount of unrelated features is knocked out, or when a grammatically present but causally irrelevant noun (a *bystander*) is knocked out. The approved protocol (V1.1, 20 November 2026) kept this test unchanged and widened its philosophical frame: *can Plato’s Forms and Kant’s conditions of possibility explain and predict the organization of neural-network representations and their downstream circuits?* It also added a stricter requirement, that “Substance” must mean the entity’s identity and not merely the occurrence of its name."));
C.push(H3("1.3 Key concepts, as we use them"));
C.push(P("The project depends on four ideas that are easy to blur. Table 1 states each one in plain terms and says how it was measured. These are working definitions for this experiment, not claims about Plato or Kant scholarship."));
C.push(table(["Concept", "Meaning in this project", "How it was made measurable"], [
  ["**Form** (Plato)", "The organizing structure that lets many different manifestations count as *the same kind of thing*. Following Protocol V1.1, a Form “is not necessarily a perfect exemplar or a single feature. It may be a multidimensional, distributed structure.”", "A candidate Form is an internal representation that stays stable across very different descriptions of a kind and still separates it from look-alikes. This is a measurable proxy, not a definition of Form."],
  ["**Condition of possibility** (Kant)", "Something that must already be in place for a kind of experience or judgment to be possible at all. In Kant’s Analogies of Experience, the permanence of substance is a condition for perceiving change as causal succession.", "Necessity is tested by removal: if the Substance representation is a condition for causal prediction, removing it should damage the correct causal outcome more than matched control removals do."],
  ["**Substance → Causality**", "The specific hypothesis: the representation that establishes an entity’s identity is a necessary upstream component of the circuit that predicts what happens to that entity.", "P1: the identity representation exists. P2: removing it hurts the correct-outcome log-odds more than both controls. P3: the dependency runs through a specific pathway that can be disrupted and restored."],
  ["**Identity** (PI’s working definition)", "“Organized togetherness; no single part essential.” Identity is the way a thing’s aspects hang together, not any one of them.", "Not pre-registered and not directly tested. It enters this report as an interpretation to be tested in a future version (Section 5)."],
], [1700, 4000, 3660]));
C.push(cap("**Table 1.** The four concepts the experiment rests on, in plain terms, and their operational translations."));
C.push(H3("1.4 Why this matters for AI safety"));
C.push(P("Mechanistic interpretability promises that we can find the concepts a model uses and act on them: remove dangerous knowledge, steer behavior, or monitor what a model is representing. These promises usually rest on two steps. First a concept is shown to be *decodable* (a probe or a sparse-autoencoder feature picks it up). Then the decoded features are ablated or edited as if they *were* the concept. If decodability and removability come apart, interventions built on decoded features can give false confidence that a concept is gone. Separately, if some representations really are conditions of possibility for others, they would be unusually high-leverage points for monitoring and alignment. Testing whether such necessity exists is a first step toward knowing where to look."));
C.push(P("Our main contributions are:"));
C.push(N("**A pre-registered, gated operationalization of a philosophical hypothesis.** We turned Kant’s Substance → Causality claim and Plato’s Form into frozen, falsifiable selection rules, and the rules worked as designed: they stopped the experiment honestly when the tool could not do its job, before any confirmatory data were touched."));
C.push(N("**Evidence that a highly decodable concept is not removable by sparse SAE ablation.** Entity identity is read almost perfectly by linear probes, yet ablating up to 128 SAE features removes none of it downstream; erasing a whole linear direction removes a modest part, and the signal partly persists even when erased where the model reads it."));
C.push(N("**A transparent record of a human–AI research workflow**, including the design decisions, data bottlenecks and errors that shaped the outcome (Appendices A and B), with all code, data and raw results public."));

// ---------- 2. Related work
C.push(H2("2. Related Work"));
C.push(P("**Probing and its limits.** Linear probes show what information can be read from a representation, but not whether the model uses it (Belinkov, 2022; Hewitt and Liang, 2019). Our results are a clear instance of that gap: near-perfect decodability with no measurable effect of removal on the model’s outputs."));
C.push(P("**Concept erasure.** Iterative nullspace projection (Ravfogel et al., 2020), amnesic probing (Elazar et al., 2021) and LEACE (Belrose et al., 2023) remove linearly decodable information and test what the model does without it. We use mean-difference projection and LEACE as exploratory tools. Belrose et al. prove that LEACE leaves no linear signal on the data it is fit to; we report this explicitly, because it makes “chance-level probe after erasure” a guarantee rather than a finding."));
C.push(P("**Sparse autoencoders.** Gemma Scope provides open SAEs for every layer of Gemma 2 (Lieberum et al., 2024; Gemma Team, 2024). Known limits include the part of the activation the SAE does not reconstruct (Engels et al., 2024) and concepts split or absorbed across features (Chanin et al., 2024). Our frozen design ablated selected SAE features, and its failure is consistent with these limits."));
C.push(P("**Information flow.** Activation-patching studies show that attributes of an entity are moved to later token positions by attention before the model uses them (Geva et al., 2023; see Heimersheim and Nanda, 2024, on interpreting patching). This predicts our E4 finding that identity is already present at the final token before the layer where we intervened."));
C.push(P("**How this work differs.** As far as we know, no prior work has operationalized Kant’s Analogies of Experience as a pre-registered necessity test between an entity representation and a causal-prediction circuit, with controls for generic damage and for ordinary noun tracking. Our contribution is less a new method than a careful translation of a philosophical claim into a test, and an honest account of where the available tools stopped."));

// ---------- 3. Methods
C.push(H2("3. Methods"));
C.push(H3("3.1 Model and tools"));
C.push(P("We used **Gemma-2-2B** (base, revision c5ebcd40), run in 32-bit precision with eager attention on a single NVIDIA A10G on Modal, with deterministic settings (seed 20261121, torch 2.5.1, transformers 4.46.3). Features came from **Gemma Scope** residual-stream SAEs (width 16k, per layer the release with average L0 closest to 70). Interventions were plain PyTorch hooks on the output of a decoder layer. The model was chosen because the agenda suggested it, it is small enough to run cheaply, and it has open per-layer SAEs."));
C.push(H3("3.2 Data"));
C.push(P("Nine object kinds in three families: **impact** (raw hen’s egg, aluminum beverage can, car windshield; the confirmatory family), **heating** (corn kernel, raw egg white, beeswax block) and **cutting** (parchment, balsa strip, acrylic sheet), plus one fixed bystander per family (umbrella, wristwatch, teapot). Each kind has *members* and *property-matched non-members* that share shape, material or setting. Half the sentences never name the object. For example:"));
C.push(B("*Member, description:* “The brown oval a hen had left in the straw that dawn, its fragile casing hiding a raw, runny centre, was still faintly warm.”"));
C.push(B("*Look-alike, same setting:* “The ceramic nest decoy sat in the straw to coax young hens into laying there.”"));
C.push(B("*Causal item:* “The egg fell from the nesting shelf onto the stone floor and” → target “cracked”; the bystander version begins “With a folded umbrella lying nearby, …”."));
C.push(P("The data were split three ways: **discovery** (used to choose features), **validation** (used to check them) and **evaluation** (held out for the confirmatory test; never opened). The discovery set was delivered to engineering as a ready-made 432-record set and corrected in 231 logged places [PI: confirm the provenance wording]. Validation and evaluation were generated by Claude from templates on 21 November, because the source set had no held-out splits. The analysis input had 1,680 recognition sentences (840 per split) and 72 discovery causal items. Appendix A describes why the data were the main bottleneck."));
C.push(H3("3.3 The frozen design"));
C.push(P("Before any confirmatory data were used, the design was frozen as **selection rules** rather than numbers (G5 spec freeze, PI approval 21 November 18:32). For every layer 2–14 and every kind, the rules chose up to eight SAE features whose summed activation best separated members from their hardest group of look-alikes (a greedy search over the top 64 features). A layer counted as valid only if four checks passed for every impact kind:"));
C.push(B("**R1, efficacy:** switching off the chosen features at the object’s last word must remove at least 67% of the identity signal read four layers later at the end of the sentence, while leaving grammatical-role information intact."));
C.push(B("**R2, identity:** the features must separate the kind from look-alikes, in noun-free descriptions and across grammatical roles, above fixed thresholds."));
C.push(B("**R3, controls:** the two controls (unrelated active features at the same word; the bystander’s own features) must be valid on enough eligible causal items."));
C.push(B("**P3 feasibility:** some downstream attention head must attend to the object and use information from it."));
C.push(P("Only on a valid layer would the confirmatory test run: on held-out evaluation items, compare the drop in the correct outcome’s log-odds (D = log-odds intact − log-odds after removal) under Substance removal with the drops under both controls, by event-level bootstrap, with a minimum effect of interest of 0.5 log-odds. If no layer was valid, P2 and P3 would be reported as **inconclusive**, with nothing re-tuned."));
C.push(H3("3.4 How we measure “identity removed”"));
C.push(P("A **probe** is a simple logistic classifier trained on the model’s internal state to answer “is this the egg or a look-alike?” Its **AUC** measures separation (0.5 is chance, 1.0 is perfect). To ask whether an intervention removed identity downstream, we train a probe on clean discovery sentences at the final token, four layers after the intervention, and compare its AUC on clean and on edited validation sentences. The **normalized probe-AUC reduction** is r = 1 − (AUC edited − 0.5) / (AUC clean − 0.5): 0 means nothing changed, 1 means the probe is at chance. r measures one probe’s loss of accuracy, not the share of information removed from the network. We also report the size of the edit relative to the activation (‖Δ‖/‖h‖) and the change in the model’s next-word predictions (KL divergence)."));
C.push(H3("3.5 Exploratory follow-ups (E1–E4)"));
C.push(P("After the frozen rules stopped the experiment, the PI approved exploratory follow-ups to diagnose *why*. They use discovery and validation data only, are labelled exploratory everywhere, and cannot change the confirmatory result. Table 2 lists them."));
C.push(table(["Run", "Question", "Manipulation", "Control"], [
  ["E1 Dose", "Were eight features too few?", "Switch off 8, 16, 32, 64, 128 identity features (first 8 = the frozen choice)", "Same number of random active features"],
  ["E2 Place", "Was one word the wrong place?", "Switch features off at the last word vs at every word of the description", "As E1"],
  ["E3 Whole", "Does the SAE miss part of identity?", "Erase one linear identity direction from the full activation, including what the SAE does not reconstruct (mean-difference projection; LEACE)", "A random direction of matched variance"],
  ["E4 Travel", "Had identity already moved to where it is read?", "E3 erasure at the description **and** at the final token (E4a); one pooled direction at every word (E4b)", "Random direction at the same positions"],
], [1050, 2200, 4150, 1960]));
C.push(cap("**Table 2.** Exploratory follow-ups. Each targets one ordinary explanation of the frozen result. E1–E3 covered layers 2–14 and 12 kinds; E4 covered layers 2, 4, …, 14. All CIs are 1,000-sample item bootstraps."));

// ---------- 4. Results
C.push(H2("4. Results"));
C.push(H3("4.1 The frozen test: identity is readable but could not be removed"));
C.push(P("Identity is strongly present. At the object’s last word, a probe separates each kind from its look-alikes with AUC 0.93–1.0 at every layer from 2 to 14, including for descriptions that never name the object. At the end of the sentence four layers later, the clean AUC is 0.70–0.94 depending on the layer."));
C.push(P("The intervention failed. Table 3 summarizes the verdicts. Switching off the selected SAE features removed essentially no identity downstream (r ≈ 0.00) in **all 156 layer-by-kind tests**, although the edits changed the activation by a median of about 5%. With R1 failing everywhere, no layer was valid. By the frozen rule, **P2 and P3 are inconclusive**: not supported and not refuted. The confirmatory pipeline was closed without running, and the evaluation set was never opened."));
C.push(table(["Check", "Passed (of 117 patient-kind × layer cases)", "What it means"], [
  ["R1 efficacy", "0", "The chosen features never removed identity downstream (0 of 156 including bystanders)"],
  ["R2 identity", "14", "Scattered kind–layer pairs; never all three impact kinds at one layer"],
  ["R3 controls", "0", "Too few causal items passed baseline eligibility (beverage can 0/8, egg 2/8, windshield 5/8)"],
  ["P3 feasibility", "24", "Downstream heads that attend to the object exist at some layers"],
  ["Valid layer (all checks)", "0", "Experiment stopped by rule; P2/P3 inconclusive"],
], [2000, 2600, 4760]));
C.push(cap("**Table 3.** Frozen verification (G5r) on discovery and validation data, layers 2–14. Source: g5_verification/results/g5r_merged.json."));
C.push(H3("4.2 Why it failed: dose, place and the whole direction (E1–E3)"));
C.push(P("Figure 1 shows the answer. **More features do not help** (left): switching off 128 identity features gives a median r of 0.005, and switching them off at every word of the description instead of one changes nothing. A control of 128 random active features, which changes the activation about five times more (30% vs 6%), also removes nothing. The problem is not the count or the position of the selected SAE features. One caveat from review: the selected features fire weakly, so this is not a test of a large dose in general."));
C.push(figure("exploratory/results/fig_e1e3.png", 6.5, 2.48));
C.push(cap("**Figure 1.** Normalized downstream probe-AUC reduction r (median over 12 kinds). Left: E1 dose × E2 placement, SAE ablation; every condition stays near 0. Right: E3 by layer, all description tokens; erasing a whole linear direction removes a modest, specific part, and a matched random direction removes nothing. Dashed line: the frozen R1 bar (r ≥ 0.67)."));
C.push(P("**Erasing the whole direction helps somewhat** (right). Removing one linear identity direction from the full activation, including what the SAE fails to reconstruct, gives a median r of 0.26 (mean-difference) and 0.18 (LEACE), against 0.01 for a random direction. It peaks at layer 6 (0.55), but that layer has a weak clean signal (AUC 0.74), so the ratio is unstable there; 9 of 624 erasure tests cross the frozen bar, all with mean-difference and all with very wide intervals. The model’s next-word predictions barely move (median KL ≤ 0.004). A fresh linear probe at the erased word is at exactly chance, but this is guaranteed by the erasure method (Section 2) and is not evidence that identity is gone there."));
C.push(H3("4.3 Had identity already traveled? (E4)"));
C.push(P("All three reviewers raised the same objection: by the layer where we intervene, attention may already have copied identity to the end of the sentence, where the downstream probe reads it. E4 shows this is **true**. Before any edit, identity is already readable at the final token at the intervention layer (AUC 0.71–0.92). Erasing it there as well as at the description removes it at that layer (r = 0.76–1.15, partly by construction). Four layers later, the reduction rises above E3 at every layer, most clearly at layers 6–8 (0.62–0.65 with mean-difference), but it stays partial, and at layers 12–14 it is back to about 0.2 (Figure 2). Prediction “erasing at the reading position makes identity collapse” is therefore not observed. The pooled-direction variant (E4b) failed as a test: one direction fit across all words does not remove identity at the final token, so its downstream numbers cannot be interpreted."));
C.push(figure("exploratory/results/fig_e4.png", 5.6, 3.21));
C.push(cap("**Figure 2.** E4: normalized probe-AUC reduction four layers after the intervention (median over 12 kinds). Dashed: E3, erase at the description only. Solid: E4a, erase at the description and at the final token. Removal increases but remains partial; the random direction stays near 0. Source: exploratory/results/e4_summary.md."));
C.push(H3("4.4 The causal test itself"));
C.push(P("An optional exploratory version of the causal comparison ran only for conditions that crossed the R1 bar. It had 3–5 usable items per case and effects of at most 0.1 log-odds, against a pre-registered minimum of 30 items in 12 events and an effect of interest of 0.5. **No conclusion about Substance → Causality is possible from it.**"));
C.push(H3("4.5 What the evidence supports"));
C.push(table(["Level", "Status", "Confidence"], [
  ["Entity identity is linearly decodable in Gemma-2-2B (layers 2–14)", "Demonstrated (availability, not use)", "High"],
  ["A validated invariant identity representation (strict P1 criteria)", "Not established (passed only sporadically)", "High"],
  ["Sparse SAE ablation removes identity downstream", "Not achieved, up to 128 features, one or all description tokens", "High (exploratory beyond k = 8)"],
  ["Linear erasure removes identity downstream", "Partially: modest at the description, more when also erased at the reading position", "Medium (exploratory)"],
  ["Substance → Causality dependency (P2) and pathway (P3)", "Untested: inconclusive by the frozen rules", "High"],
], [3900, 3700, 1760]));
C.push(cap("**Table 4.** Three levels of conclusion, as recommended by the reviewers: what is shown, what is not established, and what remains untested."));

// ---------- 5. Discussion
C.push(H2("5. Discussion and Limitations"));
C.push(P("**Decodable is not removable.** The clearest finding is a gap that matters beyond philosophy. A concept that a probe reads almost perfectly could not be removed by ablating the SAE features that best tracked it, even 128 of them, and only partly by erasing a whole linear direction, even where the model later reads it. Methods that infer “the concept is gone” from ablating decoded features (SAE-based unlearning, steering or monitoring) should verify removal downstream, at the positions and layers where the information is actually used. Otherwise they risk false confidence."));
C.push(P("**Information spreads ahead of the intervention.** Identity was already present at the end of the sentence before the layer where we intervened. Any test of necessity in a transformer has to follow the information to where it is used; editing the word where it is first computed is not enough. Our frozen design, which edited one word at one layer, could not have succeeded against this."));
C.push(P("**A hypothesis for the next version.** The PI’s working conception of identity, “organized togetherness; no single part essential,” echoes the protocol’s description of a Form as “a multidimensional, distributed structure.” The results neither support nor contradict it. Sparse ablation failing and linear erasure partly succeeding are equally compatible with ordinary explanations: copies at other words of the sentence, or identity carried in several directions or in non-linear structure that a single linear edit does not touch. As one reviewer put it, an idea that can explain every null result explains none. To give it empirical content, a future version should test three predictions: (a) several distinct directions each carry part of identity; (b) erasing some of them leaves downstream decoding intact; (c) erasing all of them at the reading position makes it collapse. E4 bears on (c) with a single direction and does not show collapse."));
C.push(P("**What happened to the philosophy.** The frozen design narrowed a possibly distributed Form into at most eight sparse features at one word. That narrowing was a design choice made at the freeze (drafted by Claude, approved by the PI), not a change of hypothesis, but it is the main reason the test could not reach the Kantian question. Its failure tells us about the tool, not about Kant."));
C.push(H3("Limitations"));
C.push(B("**One model, one tool family, text only.** Gemma-2-2B with Gemma Scope SAEs. Text tests linguistic and conceptual identity, not perception."));
C.push(B("**The data were the binding constraint.** Validation and evaluation sentences were written by Claude from templates, so probes may partly read template style. Causal items cluster into few events (24 in the evaluation impact family), and few discovery causal items met the baseline criteria (beverage can 0/8)."));
C.push(B("**Measures with known weaknesses.** r is a ratio that is unstable when the clean signal is weak; chance-level probes after erasure are guaranteed by construction; the role-preservation check used only two roles; bootstrap intervals ignore uncertainty in feature selection and erasure fitting."));
C.push(B("**Exploratory follow-ups were run after seeing the frozen result** and use the same validation data repeatedly. They generate hypotheses; they do not confirm them."));
C.push(B("**The Kantian claim itself was never tested.** Nothing here supports or refutes it."));
C.push(H3("Future Work"));
C.push(B("Test the distributed-identity predictions (a)–(c) with low-rank subspaces rather than single directions or sparse features, erasing at the reading positions and at every position that carries a copy."));
C.push(B("Return to the causal test with an intervention verified to remove identity downstream, and with causal items written so that most pass baseline eligibility, authored independently of the engineering team."));
C.push(B("Revisit the first glimpse from the feasibility probe (Appendix A): a dense direction at layer 5 changed causal predictions strongly for heating and cutting while controls did not, on 3–4 items. It was lexical and tiny, but it is the only causal signal the project saw."));

// ---------- 6. Conclusion
C.push(H2("6. Conclusion"));
C.push(P("We tried to turn a Kantian claim, that knowing what a thing is comes before predicting what happens to it, into a falsifiable test inside a language model, with Plato’s Form as the name for the representation that makes many descriptions count as one thing. Gemma-2-2B clearly represents object identity, but the frozen tool, sparse SAE features at one word, could not remove it, and the rules we set in advance stopped the experiment before the causal question. That outcome is a legitimate result for the tool and says nothing yet about Kant."));
C.push(P("Exploratory work explains why: identity is not held in a few sparse features, it has already traveled to where it is used, and even erasing it there leaves much of it readable later. For AI safety, the lesson is practical: decodability is not removability, and interventions must be verified where the information is used. For the lab’s philosophical program, the next step is clear and testable: is identity carried by several directions that together, and only together, make it what it is?"));

// ---------- Code and data
C.push(H2("Code and Data"));
C.push(P("**Code repository:** https://github.com/aliadifenza/3_kantian_positional_logic_test (public). Frozen protocol and spec in protocol/; frozen verification in g5_verification/; exploratory E1–E4 in exploratory/; reviews in review/; live record in STATUS.md.", { align: AlignmentType.LEFT }));
C.push(P("**Data/Datasets:** datasets/kantian_v1/ (discovery v1.2, validation v1.1, evaluation v1.1 sealed; checksums in SHA256SUMS; DATASET_CARD.md).", { align: AlignmentType.LEFT }));
C.push(P("**Other artifacts:** raw results for every run (JSON), Modal run identifiers, local test suites, figures and analysis scripts. Total compute: about $2.21 of a $30 cap.", { align: AlignmentType.LEFT }));

// ---------- Author contributions
C.push(H2("Author Contributions"));
C.push(P("A.A.D.F. (PI) conceived the research agenda and the experiment, made every scientific and spending decision, launched all model runs, and is responsible for the conclusions. Qualam (GPT-6), acting as research director, translated the philosophical question into the protocol, coordinated methodology and reviews, and audited the results. Claude (Anthropic), acting as lead engineer, implemented and tested all code, generated the validation and evaluation data, ran the analyses and drafted this report. Gemini (Google) and DeepSeek served as independent reviewers of the results; Grok (xAI) was planned as the second reviewer but was unavailable at that stage. Appendix A gives the full division of labor."));

// ---------- References
C.push(H2("References"));
[
  "Belinkov, Y. (2022). Probing classifiers: Promises, shortcomings, and advances. *Computational Linguistics*, 48(1), 207–219.",
  "Belrose, N., et al. (2023). LEACE: Perfect linear concept erasure in closed form. *NeurIPS 2023*. arXiv:2306.03819.",
  "Chanin, D., et al. (2024). A is for absorption: Studying feature splitting and absorption in sparse autoencoders. arXiv:2409.14507.",
  "Elazar, Y., Ravfogel, S., Jacovi, A., and Goldberg, Y. (2021). Amnesic probing: Behavioral explanation with amnesic counterfactuals. *Transactions of the ACL*, 9, 160–175. arXiv:2006.00995.",
  "Engels, J., Riggs, L., and Tegmark, M. (2024). Decomposing the dark matter of sparse autoencoders. arXiv:2410.14670.",
  "Gemma Team (2024). Gemma 2: Improving open language models at a practical size. arXiv:2408.00118.",
  "Geva, M., Bastings, J., Filippova, K., and Globerson, A. (2023). Dissecting recall of factual associations in auto-regressive language models. *EMNLP 2023*. arXiv:2304.14767.",
  "Heimersheim, S., and Nanda, N. (2024). How to use and interpret activation patching. arXiv:2404.15255.",
  "Hewitt, J., and Liang, P. (2019). Designing and interpreting probes with control tasks. *EMNLP 2019*. arXiv:1909.03368.",
  "Kant, I. (1998). *Critique of Pure Reason* (P. Guyer and A. W. Wood, Trans.). Cambridge University Press. (Original work published 1781/1787.) Analogies of Experience, A176–218/B218–265.",
  "Lieberum, T., et al. (2024). Gemma Scope: Open sparse autoencoders everywhere all at once on Gemma 2. arXiv:2408.05147.",
  "Plato. *Phaedo* and *Republic* (Books VI–VII), on the Forms. In J. M. Cooper (Ed.), *Plato: Complete Works* (1997). Hackett.",
  "Ravfogel, S., Elazar, Y., Gonen, H., Twiton, M., and Goldberg, Y. (2020). Null it out: Guarding protected attributes by iterative nullspace projection. *ACL 2020*. arXiv:2004.07667.",
].forEach((r) => C.push(new Paragraph({ children: runs(r, { size: 20 }), indent: { left: 360, hanging: 360 }, spacing: { after: 80 } })));

// ---------- Appendix
C.push(new Paragraph({ children: [new PageBreak()] }));
C.push(H2("Appendix"));
C.push(H3("A. How we worked: roles, workflow, bottlenecks, and how we stopped"));
C.push(P("**Roles.** The lab runs as a human-directed, AI-assisted team. The division below is the agreed one and describes what each party actually did in this experiment."));
C.push(table(["Who", "Role", "What they did here"], [
  ["Ali A. Di Fenza (human)", "Principal investigator", "Research agenda and question; approval of every scientific change, spending limit and gate; launched every model run from his own computer; final conclusions."],
  ["Qualam (GPT-6)", "Research director", "Translated philosophy into protocol V1.1 with the PI; wrote briefs for engineering; coordinated reviews; internal audit of the results (not independent, and declared as such)."],
  ["Claude (Anthropic)", "Lead engineer", "Feasibility, code, tests, data repair and generation, analysis, figures, review packages, this report draft. Could not access the PI’s computer or Modal account; every run was a command the PI pasted."],
  ["Gemini (Google)", "Independent reviewer", "Reviewed the results package against the original plan (G12)."],
  ["DeepSeek", "Independent reviewer", "Substituted for Grok at G12 (PI decision). Reviewed the same package independently."],
  ["Grok (xAI)", "Planned reviewer", "Unavailable at G12 (usage limit)."],
], [2000, 1700, 5660], { size: 17 }));
C.push(cap("**Table A1.** Roles and actual contributions."));
C.push(P("**Agreed workflow.** The lab uses a gated sequence (G0–G13): design and alignment of the question with the PI, a scientific dry run, technical reconnaissance, a feasibility probe on the real model before anything is frozen, a spec freeze, engineering, a code dry run, a pilot, an experiment freeze, the full run, analysis, independent review, and submission. Scientific changes always go back to the PI in the form *original → proposed change → impact → recommendation*; routine engineering is delegated. The rule “scientific fidelity before engineering progress” governed every decision below."));
C.push(P("**What actually happened.** Table A2 compresses roughly two days of work."));
C.push(table(["When (2026)", "Step", "Outcome"], [
  ["Before Nov 20", "Research agenda; seed version of the experiment", "Purely Kantian test with log-odds and two controls"],
  ["Nov 20", "Protocol V1.1 approved (G0–G2)", "Forms framing added; identity requirement (P1) and pathway evidence (P3) added"],
  ["Nov 20", "Technical reconnaissance (G3)", "Gemma-2-2B chosen; $30 cap; Modal on the PI’s account"],
  ["Nov 20", "Feasibility probe (G4), exploratory", "Dense direction at layer 5 changed causal predictions (heating +1.88 vs control +0.05; cutting +2.72 vs +0.82; 3–4 items) but was lexical and saturated"],
  ["Nov 21", "G5 verification with SAE features", "Interventions too weak (|D| ≤ 0.05); pronoun test not truly non-lexical; layer rule picked an edge layer; C2 selection bug found and fixed"],
  ["Nov 21", "Dataset workstream", "Most of the day: repair, held-out generation, bystander redesign, tokenizer check (Appendix B)"],
  ["Nov 21, 18:32", "G5 spec freeze (+ Amendment 1 at 18:47)", "Rules frozen; R1 efficacy gate added"],
  ["Nov 21, ~19:00", "G5r verification, layers 2–14", "No valid layer; P2/P3 inconclusive; confirmatory phase closed (19:34)"],
  ["Nov 21, 19:38–21:12", "Exploratory E1–E4; reviews", "Results in Section 4; three reviews consolidated"],
], [1500, 3000, 4860], { size: 17 }));
C.push(cap("**Table A2.** Timeline. Times are CET. The G4 numbers are exploratory, on probe-only items, and are not evidence for P2."));
C.push(P("**The bottleneck was the dataset.** The experiment needed sentences that separate identity from everything correlated with it: look-alikes sharing material, shape or setting; descriptions that never name the object; several grammatical roles; and causal items where the model already predicts the right outcome. Building and repairing these took most of the second day and set the limits of every result (Appendix B, items 5–9)."));
C.push(P("**How and why we stopped.** The freeze made efficacy a gate: if the chosen features could not remove identity, the causal test would be uninformative, so it would not be run. R1 failed at every layer, so the pipeline stopped by rule, not by choice. The PI then closed the confirmatory phase, kept the evaluation set sealed, and approved exploratory diagnostics under a separate spending rule (within €30 total, no further permission needed; science changes still to the PI). Total compute: G4 $0.06, G5 verification $0.03, tokenizer check < $0.01, G5r $0.44, E1–E3 $1.43, E4 $0.25, about **$2.21**."));
C.push(H3("B. What went wrong"));
C.push(P("We list problems in the order they arose, including our own mistakes. None was hidden at the time; all are documented in the repository."));
[
  "**The design narrowed the Form.** Protocol V1.1 allowed a Form to be distributed. The freeze tested it as at most eight sparse SAE features at one word. Claude drafted that rule; the PI approved it. Dense directions had been dropped because they were lexical and could not be magnitude-matched for the control, and SAE features made matching possible. The G5 verification had already warned that sparse ablation leaving the SAE’s unreconstructed part untouched “likely removes too little identity information to matter.” The R1 gate caught the problem before any confirmatory spending, but the risk had been foreseen.",
  "**C2 control bug (G5 verification).** The bystander control used the bystander’s top-ranked features instead of its selected set. Fixed and documented; it did not affect the conclusions of that stage.",
  "**The pronoun test was not non-lexical.** At “it”, the model can copy the noun’s identity through attention, so passing there proved nothing. Replaced by noun-free descriptions and property-matched look-alikes.",
  "**The layer rule first picked an edge layer** (18), leaving little room downstream. The range was capped at 14.",
  "**The discovery set needed 231 corrections**: 133 role-label spellings, 84 wrong role labels, and 14 bystander clauses that were not actually inert.",
  "**No held-out splits existed.** Claude generated validation and evaluation from templates, which introduces a style difference from discovery and the risk that probes read templates.",
  "**Bystanders had no recognition data** (about 95 different bystanders). The PI decided on one fixed bystander per family.",
  "**Causal items cluster by event** (24 events behind 96 evaluation impact items), so the real sample is the event, not the item. Eight target words were more than one token, which led to Amendment 1.",
  "**Few causal items were eligible**: the model rarely gave the right outcome enough probability at baseline (beverage can 0/8, egg 2/8, windshield 5/8), which made the controls unusable.",
  "**E4b failed as a test**: one direction pooled over all words did not remove identity where it is read.",
  "**Reporting bug (Claude):** for “all words” conditions, the edit size was measured at one word only. r values are unaffected.",
  "**Weak checks disclosed by review**: role preservation used two roles, and a missing value counted as a pass.",
  "**Over-reading, caught by review.** Claude’s first summary said identity was “unrecoverable at the site” and “largely returns downstream,” and offered the PI’s idea of identity as “consistent with” the data. All three reviewers judged this overstated; the claims were narrowed and the idea moved to the Discussion.",
  "**Process friction.** Claude could not reach the PI’s computer or Modal, so every run was a copy-paste command; one launch failed because files had not been downloaded, and long runs made the terminal look frozen. Claude also cannot work between messages, and at one point promised work “by morning” that it could not do unattended; this was corrected the same evening. Claude could not access the private lab-notes repository while drafting this report, which relies on the public repository and the conversation record.",
  "**Reviewer availability.** Grok hit a usage limit. A GPT-6 session would have seen the project context, so Qualam’s review is reported as an internal audit, and DeepSeek served as the second independent reviewer.",
  "**Fatigue and time.** The PI worked about ten hours on the final day. The deadline shaped scope: no new confirmatory design was attempted.",
].forEach((t) => C.push(N(t, "num2")));
C.push(H3("C. Additional results"));
C.push(table(["Condition (median over 12 kinds × 13 layers)", "r (IQR)", "Probe at edited word", "‖Δ‖/‖h‖", "KL"], [
  ["SAE, k = 8, last word (frozen choice)", "0.000 (−0.005, 0.006)", "0.99", "0.05", "0.000"],
  ["SAE, k = 128, all description words", "0.005 (−0.006, 0.014)", "0.99", "0.06*", "0.000"],
  ["Random active features, k = 128, all words", "0.015 (−0.019, 0.069)", "0.99", "0.30*", "0.004"],
  ["Erase direction, mean-difference, all words", "0.261 (0.166, 0.346)", "0.50†", "0.20*", "0.002"],
  ["Erase direction, LEACE, all words", "0.180 (0.116, 0.271)", "0.50†", "0.14*", "0.001"],
  ["Random direction, all words", "0.013 (0.000, 0.053)", "0.99", "0.13*", "0.001"],
], [3700, 1900, 1400, 1100, 1260], { size: 17 }));
C.push(cap("**Table C1.** E1–E3 summary. *Edit size measured at the last description word only. †Chance by construction. Full tables: exploratory/results/e_summary.md and e4_summary.md."));

// ---------- LLM usage
C.push(H2("LLM Usage Statement"));
C.push(P("Large language models were central to this project, and their roles are given in Appendix A. Claude wrote all code and ran all analyses; the PI launched every model run. The first draft of this report was written by Claude from the repository record and the project conversation, at the PI’s request and following his instructions on structure and content; the PI is responsible for the final text. Every number in this report was checked against the raw result files in the repository. The results were reviewed by Gemini and DeepSeek independently and audited by Qualam; their reviews are archived verbatim in review/. No AI review is presented as human peer review."));

// =====================================================================================
const doc = new Document({
  creator: "AlignedLab", title: "Decodable but Not Isolable", language: "en-US",
  styles: {
    default: { document: { run: { font: FONT, size: 22, color: INK, language: { value: "en-US" } } } },
    paragraphStyles: [
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 30, bold: true, color: INK }, paragraph: { spacing: { before: 360, after: 140 }, keepNext: true, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 24, bold: true, italics: true, color: INK }, paragraph: { spacing: { before: 240, after: 100 }, keepNext: true, outlineLevel: 2 } },
    ],
  },
  numbering: { config: [
    { reference: "bul", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 270 } } } }] },
    { reference: "num", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 360 } } } }] },
    { reference: "num2", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 360 } } } }] },
  ] },
  footnotes: { 1: { children: [new Paragraph({ children: runs("Research conducted at the Mechanistic Interpretability Hackathon #5 (Apart Research), November 2026. AlignedLab Experiment 3; Research Agenda Chapter 2, Experiment 2.", { size: 18 }) })] } },
  sections: [{ properties: { page: { size: { width: 12240, height: 15840 }, margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } } }, children: C }],
});
Packer.toBuffer(doc).then((b) => { const o = path.join(__dirname, "AlignedLab_Exp3_Kantian_Positional_Logic_report.docx"); fs.writeFileSync(o, b); console.log("wrote", o); });
