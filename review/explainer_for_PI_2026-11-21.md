# What we did, what we found, what it means — a plain-English explainer

*For Ali, 2026-11-21 evening. No jargon without a translation. Confidence labels [HIGH/MED/LOW] on the claims that matter.*

## 1. The question, in one breath

Kant says some categories are conditions of possibility for others: you cannot think a causal relation without first having things that persist through it — Substance before Causality. The experiment asks whether a language model is built that way: is there an internal representation of *what a thing is* that the model's causal reasoning must draw on? V1.1 widened the frame to Plato. The thing's identity is the measurable face of a Form, meaning whatever lets the model recognize "the same egg" across very different descriptions while telling it apart from look-alikes. V1.1 also warned that a Form "may be a multidimensional, distributed structure", not a single feature.

## 2. The tools, translated

**The model.** Gemma-2-2B, a small open model. Inside it, every word is a long list of 2,304 numbers per layer; the list changes as it passes through 26 layers. This list is the "residual stream", the model's working memory for that word.

**A probe.** A simple detector we train on those numbers to answer a yes/no question, e.g. "is this about a raw hen's egg, or about a look-alike that shares its shape or setting (a ceramic nest decoy in the straw)?" If a simple detector can answer, the information is there and readable.

**AUC.** How well the probe separates the two groups. 0.5 is coin-flipping; 1.0 is perfect.

**SAE latents.** A sparse autoencoder (Google's Gemma Scope) is a dictionary that rewrites the 2,304 numbers as a handful of active "concepts" chosen from 16,000. Each is called a latent. Interpretability work hopes these are the model's real units of meaning.

**Ablation.** Switching off chosen latents and letting the model run on, to see what breaks.

**Erasure (E3).** Instead of switching off dictionary entries, we take the single direction in the 2,304-number space along which "egg vs not-egg" differs most, and flatten it. This touches everything, including the part the dictionary fails to capture (its "reconstruction error").

**Downstream.** We do the intervention at the word itself at some layer L, then ask a probe four layers later, at the end of the sentence, whether identity is still there. The removal score **r** says how much of the identity signal disappeared there: 0 means nothing, 1 means all of it.

## 3. What the confirmatory phase found (G5r)

**Identity is there, clearly [HIGH].** At every layer from 2 to 14, a probe separates each kind from its property-matched look-alikes almost perfectly (AUC 0.93–1.0). That holds for descriptions that never name the object ("the brown oval a hen had left in the straw…") and across grammatical roles. In the protocol's terms, a candidate Form exists.

**But our scalpel couldn't cut it [HIGH].** The frozen plan said: find the ≤ 8 dictionary latents that best carry identity, switch them off at the last word of the reference, and check that identity is gone downstream. It never was (0 of 156 layer × kind cases). The frozen rule therefore found no valid layer, so the actual Kantian test — does removing Substance damage the causal prediction ("the egg… *cracked*") more than the controls do? — was never run. **P2 and P3 are inconclusive, not refuted.** The sealed evaluation set stayed sealed. That is the rule working as designed.

## 4. Tonight's exploratory follow-up (E1–E3)

Three boring explanations could account for the failure, so we tested each:

**E1 — "not enough latents."** We switched off 8, 16, 32, 64, then 128. Still nothing downstream [HIGH]. Switching off 128 *random* active latents, which changes the representation about five times more, also does nothing. So the problem is not quantity.

**E2 — "wrong place."** We switched them off at every word of the description, not just the last. Still nothing [HIGH]. So the problem is not position, at least within the reference itself.

**E3 — "the dictionary misses it."** We erased the whole identity direction from the full working memory. This is the only intervention that does something: at the word, identity becomes unreadable to a fresh probe, and downstream about a fifth to a third of it is gone (median r 0.18–0.26). The peak is at layer 6 (0.55). Erasing a random direction does nothing, so the effect is specific to identity [MED]. But most of the identity signal is back by the end of the sentence.

## 5. What this means, and what it doesn't

**The defensible reading [MED].** Identity is real and specific in the model. It is not stored in a few sparse dictionary entries, and it does not sit in one place you can switch off. Remove it locally and the model mostly has it again a few layers later. That is why the frozen plan's tool was the wrong shape for V1.1's hypothesis. The plan tested a Form as a few parts in one spot; the evidence points to something more spread out or more redundant.

**The tempting reading, held at arm's length [LOW].** This sounds like your "organized togetherness: no single part essential". It is *consistent* with that. It is not *evidence* for it, for three concrete reasons:

1. **The information may simply have travelled ahead [MED].** Before layer L, the model's attention has already copied identity from the egg words into other words, including the end of the sentence where our probe reads. We erased it only at the egg words. A copy that left earlier explains the "return" without any organized whole. This is the strongest objection, and the test that answers it (erasing at the reading position too) has not been run.
2. **"Unreadable at the word" is partly automatic [HIGH].** The erasure method guarantees, by its mathematics, that the simplest detector trained on the same kind of data finds nothing. A cleverer, non-linear detector might.
3. **The best numbers are the shakiest [MED].** The layer-6 peak and the few cases that cleared the bar sit where the downstream signal was weak to begin with, so their uncertainty is large.

**What remains untested [HIGH].** The Kantian claim itself — Substance as a necessary condition for Causality. Our small exploratory attempt had 3–5 usable items and effects ten times smaller than the pre-registered minimum. It says nothing either way.

## 6. The story in one paragraph (draft you can edit)

We asked whether a language model's representation of what a thing *is* functions as a condition of possibility for its reasoning about what *happens* to that thing. The model clearly represents identity: it tells a raw egg from its look-alikes across descriptions and sentence roles. The pre-registered intervention — switching off a few sparse features — could not remove that identity, so the frozen rules declared the Kantian test inconclusive and we did not run it. Exploratory follow-ups showed why. No number or placement of sparse features removes identity, while erasing its whole direction removes it locally, yet most of it returns downstream. Identity in this model behaves less like a part than like a pattern the network keeps reconstituting. Whether that is an organized whole or simply information that has already spread is the next experiment's question.

## 7. Questions worth your thinking (not mine to answer)

- If identity is something the model *reconstitutes*, is that closer to a Form (a principle that regenerates its instances) or to Kantian synthesis (an act the understanding keeps performing)?
- The protocol treated Substance as a *representation* to be removed. If it is instead a *process*, what would the right intervention be — and what would count as removing a condition of possibility?
- Is "no single part essential" testable at all, or only falsifiable when some part turns out to be essential?
