# **G12 — Critical scientific review**

AlignedLab Experiment 3 · Kantian Positional Logic · 21 November 2026

**Overall verdict [HIGH]:** The frozen experiment was stopped for a valid methodological reason. The exploratory work contains useful observations, but two central interpretations are too strong: that identity is *reconstituted* downstream and that increasing the SAE latent count rules out a dose problem.

I reviewed the package against the repository's protocol, `STATUS.md`, exploratory analysis and intervention code. This is a critical internal audit, not an independent experimental replication; I am named as Research Director in the protocol.

## **1. Assessment of the six claims**

| Claim                                           | Verdict                                   | Assessment                                                                                                                                                                                                                                                      |
| ----------------------------------------------- | ----------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **1. Strong identity representation**           | **Overstated**                            | Same-site linear decodability is supported (AUC ≈ 0.93–1.00). But the stricter P1/R2 invariance criteria pass only for scattered kind–layer combinations. Strong decodability is not equivalent to a validated invariant Form.                                  |
| **2. No valid layer; P2/P3 inconclusive**       | **Supported [HIGH]**                      | Correct. R1 failed in all 156 configurations, and the confirmatory pipeline was closed according to the freeze. The narrow intervention failed; the philosophical hypothesis was neither confirmed nor falsified.                                               |
| **3. Not a dose or placement problem**          | **Overstated**                            | Increasing k to 128 and editing the reference span did not materially change downstream AUC. But identity latents produced weak edits, and other context positions were untouched. Neither dose nor position has been decisively excluded.                      |
| **4. Local erasure, downstream reconstitution** | **Overstated**                            | Downstream probe-score changes are real reported measurements. However, local AUC ≈ 0.50 is largely guaranteed by construction, and downstream identity may have been available at untouched positions before intervention. Reconstitution is not demonstrated. |
| **5. Compatible with distributed Form**         | **Supported only as compatibility [LOW]** | The observations do not contradict the distributed-Form hypothesis. They also do not distinguish it from mundane explanations. Keep this as philosophical interpretation, not experimental evidence.                                                            |
| **6. No causal-dependency conclusion**          | **Supported [HIGH]**                      | Only 3–5 eligible causal items per applicable condition, no matched-magnitude controls, and effects below the 0.5 log-odds threshold. No admissible P2 conclusion.                                                                                              |

## **2. Important problems beyond the package's existing caveats**

**[HIGH] AUC removal is not information removal.** The quantity

\\[ r=1-\frac{\mathrm{AUC}\_{\mathrm{ablated}}-0.5} {\mathrm{AUC}\_{\mathrm{clean}}-0.5} \\]

measures relative loss of one trained probe's discriminative performance above chance. It does not measure the percentage of identity information removed from the network. The report should consistently describe \\(r\\) as *normalized downstream probe-AUC reduction*.

**[HIGH] The E3 amnesic result is not an independent success.** In `fit_erasers`, the direction is fitted to the discovery class means. The same fitting data are subsequently used to train the fresh logistic probe. Equalized class means force the optimal linear discriminator toward a constant, making AUC ≈ 0.50 expected. It does not establish that identity is absent from the representation, especially under nonlinear decoding.

**[HIGH] All-token intervention magnitude is incompletely measured.** In [`e_probe.py`](https://github.com/aliadifenza/3_kantian_positions/blob/main/exploratory/e_probe.py?utm_source=chatgpt.com), `rel_norm_mean` is computed on the selected last-reference-position representation. The `"all"` intervention subsequently edits multiple positions in the actual forward pass. Thus the reported norm is not the aggregate magnitude of the full-span intervention. This matters when comparing intervention strength and making dose claims.

**[MED] Role preservation has a weaker implementation than the experimental ambition.** The exploratory role probe uses only the first two sorted grammatical roles. Furthermore, the descriptive R1 gate allows a missing role-drop measurement (`NaN`) to pass. The nine reported qualifying cases have numerical role-drop values, so this does not appear to explain their qualification, but the implementation should be disclosed or repaired in future work.

**[MED] The bootstrap does not fully reflect experimental uncertainty.** Item-level resampling conditions on fitted probes, selected latents and eraser directions. It does not account for fitting or selection uncertainty, or dependence among templated examples. The extremely wide intervals around some \\(r\\) estimates make threshold crossings unstable.

**[MED] The SAE intervention and the full-residual eraser do not represent equally strong tests of the same construct.** The former removes selected decoder contributions; the latter changes a full residual-space direction. Their difference is compatible with SAE limitations, but does not prove that SAE representations intrinsically miss a coherent entity representation.

## **3. Answers to §7**

**a. Local unrecoverability:** Yes, overstated. Replace it with: *the erasure eliminates the fitted linear class-mean contrast, yielding chance-level performance for a freshly fitted probe under this evaluation procedure.* This is much narrower than information being unrecoverable.

**b. Information already transferred before layer L:** This is arguably the most serious alternative explanation. Editing reference positions at L cannot erase representations already incorporated into the final token by earlier attention operations. The observed downstream decodability could persist without any reconstruction mechanism. This substantially undermines the causal interpretation of claim 4 and removes any distinctive evidential force for claim 5.

**c. Small denominator:** A clean AUC of 0.70–0.76 means that \\(r=0.55\\) corresponds to an absolute AUC reduction of approximately 0.11–0.14. Report absolute AUC changes alongside \\(r\\), with uncertainty. The L6 peak is exploratory, not a confirmed layer-specific effect.

**d. LEACE versus mean-difference:** Both methods remove the same binary class-mean direction, but apply different readout functionals and therefore different edits to individual activations. Covariance weighting, residual anisotropy, nonlinear downstream responses and differences in realized edit magnitude can explain their unequal effects. The 10% shrinkage may contribute, but cannot be singled out without a sensitivity analysis.

**e. Rare identity latents:** This substantially weakens the dose conclusion. If the selected latents barely activate, increasing k does not necessarily increase the relevant effective intervention. Compare activation frequency, actual removed contribution, and matched-norm edits before declaring a dose-independent null.

**f. Synthetic data:** This chiefly threatens construct and external validity. Claude-generated templates and imperfect lexical balancing may support classification through stylistic or lexical regularities rather than the proposed invariant identity representation. Poor baseline causal eligibility separately prevents a meaningful downstream dependency test.

**g. Exploratory selection:** The exploratory label is appropriate and sufficient for presenting these observations, provided all conditions remain visible, the nine threshold crossings are not promoted into confirmatory evidence, and the validation split is not treated as a fresh untouched test after repeated analysis. No results need be deleted; the interpretations need narrowing.

**h. Headline:** The proposed headline remains slightly too strong because it uses *identity* and *returns* as if the underlying mechanisms were established.

My preferred wording:

> **Entity-kind labels are linearly decodable in Gemma-2-2B, but the frozen sparse-SAE intervention fails to reduce their downstream decodability. Exploratory full-residual erasure changes downstream probe performance, without demonstrating reconstruction or a mandatory Substance → Causality pathway.**

## **4. Fidelity to the original hypothesis**

**[HIGH]** I agree with the package's distinction, with one important qualification.

G5 did **not** change the philosophical hypothesis. Substance → Causality remained the central dependency claim.

However, restricting a potentially multidimensional, distributed Form to at most eight SAE latents at one reference position introduces a substantial auxiliary assumption: that the relevant representation is accessible and sufficiently removable through that narrow intervention.

This is a scientifically consequential operational narrowing, not merely a technical implementation detail.

Its failure limits what the experiment can conclude. In particular, it cannot be used to reject the wider V1.1 hypothesis, which explicitly permits directions and distributed structures.

## **5. What I would recommend for submission**

Preserve the frozen result without alteration. Present the exploratory findings separately, and distinguish three levels of conclusion: **linear decodability demonstrated, sparse intervention efficacy not established, and causal dependency untested.**

For reproducibility, retain the exact datasets, model and SAE identifiers, code revision, seeds, exclusion logs, probe definitions, uncertainty procedures and raw results. The repository already contains much of this material, though the public `README.md` currently provides virtually no guidance to reproduce or interpret the results.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

aliadifenza/3_kantian_positions · GitHub



For a future experiment—not a retrospective change to this one—the most informative next test would compare erasure at the reference span, final-token read position, and all context positions, with matched realized intervention strengths and an independently authored held-out set.

**Final assessment [HIGH]:** The experiment has produced a legitimate negative result for a specific frozen operationalization and a useful exploratory diagnosis. It has not yet tested the Kantian necessity claim successfully. That boundary should be made explicit, rather than obscuring either the scientific achievement or its limitations.

**Next move:** Give Claude this review for G12 consolidation, keeping the confirmatory and exploratory conclusions strictly separate.