## 结论先行

**§6 各条评述：**

| Claim | 评级 | 核心问题 |
|---|---|---|
| 1 | **overstated** | 线性可解码 ≠ 模型"使用"该表征；跨角色/描述的可解码性不能排除浅层启发式 |
| 2 | **supported** | 冻结规则下无有效层，P2/P3 确实 inconclusive |
| 3 | **overstated** | E1 的剂量测试不公平（identity latents 几乎不激活）；"≈0 下游移除"的结论受 §7b 混淆 |
| 4 | **overstated** | "unrecoverable at the site"由构造保证（§7a）；"reconstitution"可能只是信息在 L 之前已复制（§7b） |
| 5 | **unsupported** | 将 null 模式解释为"distributed whole"是事后叙事，无可检验预测 |
| 6 | **supported** | 效应量远低于冻结最小兴趣效应，样本量不足，无法推断 |

**§7 关键答案：**
- **(a) 是，overstated。** Amnesic AUC=0.50 是 LEACE/mean-difference 的数学必然，不是"不可恢复"的证据。
- **(b) 严重削弱 claims 4–5。** 未在读取位置或所有位置同步擦除，无法排除信息在干预层之前已被 attention 复制到下游。
- **(c) 是，r 的比值在分母小时不可靠。** L6 峰值恰在 clean AUC 最低处，CI 上界 >1 表明该指标在此设置下不稳定。
- **(d) 是，LEACE 与 mean-difference 的差异来自 shrinkage 和 read-out，不是不同机制。**
- **(e) E1 不是公平剂量测试。** 选择标准偏向稀有 latents，‖Δ‖/‖h‖ 仅 ~6% vs 随机 latents ~30%。
- **(f) 数据风格偏移威胁有限但不可忽略。** 核心 null 在 discovery+validation 上稳定，但确认性结论不应依赖 Claude 生成的数据。
- **(g) 探索性标签足够，但 claims 3–5 不应以现在的措辞进入任何摘要。**
- **(h) 建议标题见末段。**

---

## §6 逐条评述

### Claim 1 [HIGH]：**overstated**

**支持的部分：** 同位置 probe AUC ≈ 0.93–1.0 确实表明身份信息在参考位置线性可解码。这是坚实的结果。

**问题在于"represents"的歧义。** 线性可解码性只说明信息在表征中**可获得**（available），不说明模型在因果计算中**使用**它。LEACE 原文明确指出，"concept erasure"的必要条件是隐藏状态与概念有显著互信息，但互信息存在不等于下游电路依赖它。你的 G5r 结果恰恰是反例：身份高度可解码，但稀疏 SAE 消融后下游几乎不受影响——这说明 probe 检测到的线性方向可能是一个"旁观者"表征，而非因果瓶颈。

此外，P1 的"invariance across manifestations"测试（角色、描述、noun-free）只证明解码器能跨这些变换泛化，**不证明模型的因果电路以 kind-invariant 方式使用该信息**。一个在每种角色/描述上都重新计算身份的浅层启发式，同样会产生可解码的 kind 信息，但不会在参考位置形成一个统一的 Substance 表征。

**建议降级为：** "Identity is linearly decodable at the reference position with high AUC across descriptions and roles. This establishes availability, not causal use."

---

### Claim 2 [HIGH]：**supported**

冻结规则（G5 spec）明确：无有效层 ⇒ P2/P3 inconclusive，nothing re-tuned。G5r 报告 0/156 layer×kind 通过 R1，R3 因因果项基线资格不足而大面积失败，R2 仅散点通过。评估集未打开。**这与协议一致，报告诚实。** 唯一需要补充的是：R1 的失败本身可能部分归因于 SAE 特征选择的性质（§7e），而非纯"Substance 不可移除"——但这不改变 P2/P3 inconclusive 的结论。

---

### Claim 3 [HIGH, exploratory]：**overstated**

**核心问题：E1 不是公平的剂量测试。** 你报告 identity latents k=128 的 ‖Δ‖/‖h‖ ≈ 6%，而随机 active latents k=128 的 ‖Δ‖/‖h‖ ≈ 30%。这意味着所谓"剂量加到 128"实际上是在消融一组**几乎不激活的 latents**。选择标准（worst-stratum AUC）偏向的是那些在区分 worst stratum 上有判别力、但整体激活率极低的特征。用一组"沉默的" latents 做剂量测试，然后说"加到 128 也没用"，是不公平的。

**第二个问题：placement 测试不完整。** E2 比较了"last token"与"all reference tokens"，但**两者都在层 L 的参考跨度内**。如果 attention 在层 <L 已经把身份信息从参考跨度复制到了最终读取位置或其他上下文 token，那么在参考跨度上的任何消融都不会影响下游 probe 读取到的信息。这不是"SAE 漏掉了分布表征"，而是**信息已经不在你编辑的位置了**。

**建议改为：** "Increasing the number of selected SAE latents and editing all reference tokens did not produce downstream removal in this setup. However, the selected latents had very low activation magnitude, and interventions were confined to the reference span at layer L. This does not rule out that identity information was already routed to downstream positions before the intervention layer."

---

### Claim 4 [MED, exploratory]：**overstated**

这条 claim 有三个独立的问题，每一个都足以削弱"reconstitution"的解释。

**问题一：Amnesic AUC=0.50 是构造性的。** 你自己在 §7a 已经指出这一点。LEACE 的数学保证是：擦除后各类均值在**所有线性方向上**相等，因此任何线性 probe 的最优解是常数。这意味着 amnesic AUC=0.50 **不是**"身份不可恢复"的证据，而是"线性均值差异被消除了"。说"unrecoverable at the site"是 overstatement——准确的说法是"linear mean-difference information was removed from the fitting data"。

**问题二：下游 r ≈ 0.15–0.35 可能完全由信息迁移解释，而非"分布表征"。** Transformer 的 residual stream 是累积的：每层的 attention 可以从参考位置读取信息并写入最终 token 或其他上下文位置。你在层 L 的参考跨度上擦除线性方向时，层 <L 的 attention 可能已经将身份信息以某种形式复制到了最终读取位置。下游 probe 读的是**最终 token 的表征**，而你在**参考 token**上做了擦除。下游 r 不为零，最简约的解释是：信息已经在 L 之前被路由走了，而非"SAE 漏掉的分布表征在层 L 被下游电路重建"。

**决定性测试未做：** 在读取位置（最终 token）也做擦除，或在所有位置、所有层 <L 同步擦除。没有这个控制，"reconstitution" claim 不成立。

**问题三：LEACE 和 mean-difference 的对比不提供"分布 vs 线性"的证据。** 你自己指出两者删除的方向 cosine=1.00。LEACE 在二分类下沿 Σ_xz ∝ mean difference 方向编辑，差异仅在于 read-out vector 和 shrinkage。mean-difference 移除更多（median r 0.261 vs 0.180）更可能反映的是 10% covariance shrinkage 对 LEACE 的约束，而非"mean-difference 更接近真正的分布结构"。

**建议改为：** "Linear erasure from the full residual at the reference span reduces downstream linear decodability by a median of ~15–35%, while a matched random direction produces ≈0. The mechanism (distributed representation vs. information routing before the intervention layer) is unresolved because erasure was not applied at the read position or across layers."

---

### Claim 5 [LOW, exploratory]：**unsupported**

这条 claim 的问题不是"证据弱"，而是**它不是一个可检验的命题**。

"organized togetherness; no single part essential"——这个描述可以解释任何 null 结果。SAE 消融无效？那是因为"organized whole"不是稀疏可分的。线性擦除部分有效？那是因为你只碰了线性可及的部分。下游恢复？那是因为"whole"在别处重建。**一个能解释一切的模式不能解释任何东西。**

V1.1 的"multidimensional, distributed Form"如果要有经验内容，需要给出**可证伪的预测**：例如，分布表征应当表现为 (a) 多个不重叠的线性方向各自携带部分身份信息，(b) 随机擦除其中一部分后剩余部分仍可支持下游解码，(c) 在读取位置擦除全部方向后下游崩溃。E3 只做了 (a) 的必要条件（一个方向），没有检验 (b) 或 (c)。

**建议：要么删除这条 claim，要么将其明确标注为"post-hoc narrative, not a testable hypothesis in the current design."**

---

### Claim 6 [HIGH]：**supported**

效应量 ≤ 0.1 log-odds，冻结最小兴趣效应 0.5 log-odds，样本 ≤5 items / 5 events，冻结最小样本 30 items / 12 events。**明确不可推断。** 这条报告准确。

---

## §7 具体回答

### a. Amnesic AUC = 0.50

**是，overstated。** 如上述，这是 LEACE 的数学定义决定的：擦除后类均值相等，线性 probe 最优解为常数。它说明的是"linear mean-difference was removed from the fitting data"，不说明"identity is unrecoverable"。一个在验证集上训练的 probe，或一个非线性 probe，或一个在擦除后数据上重新拟合的 probe，都可能恢复信息。Claim 4 的"unrecoverable at the site"应改为"linearly guarded with respect to the fitting data's mean structure"。

---

### b. 信息可能已经在 L 之前迁移

**严重削弱 claims 4–5。** 这是整个探索性分析中最大的未控制混淆。Transformer 的 attention 在每一层都可以将信息从参考位置复制到最终 token。你在层 L 擦除参考跨度的线性方向时，最终 token 可能已经携带了身份信息——来自层 <L 的复制。下游 probe 在最终 token 上读取，自然不受你在参考跨度上擦除的影响。

**这不是"分布表征"的证据，而是"你擦错了位置"的证据。** 决定性控制：在层 L 的**读取位置**（最终 token）也擦除同一方向，或在所有层 <L 的所有位置同步擦除。没有这个控制，claim 4 的"reconstitution"和 claim 5 的"organized whole"都缺乏最小可信度。

---

### c. r 的比值问题

**是，L6 峰值和达标案例的可靠性被高估了。** r = 1 − (AUC_ablated − 0.5)/(AUC_clean − 0.5)。当 AUC_clean 在 L2/L5–L8 为 0.70–0.76 时，分母仅 0.20–0.26。一个 0.05 的 AUC 波动就会导致 r 波动 ~0.20–0.25。你报告 CI 上界 >1 是直接的证据：这个指标在分母小时不稳定。**L6 的 r=0.55 不应该被解读为"层 6 特别重要"**，更可能是该层 clean AUC 恰好较低、分母小导致的比值放大。建议报告**绝对 AUC 变化**而非比值，或至少并列报告。

---

### d. LEACE vs mean-difference

**10% shrinkage 很可能是原因，但不排除 read-out 差异的贡献。** LEACE 在二分类下的编辑方向确实与 mean-difference 平行（cosine=1.00），但 LEACE 的变换同时考虑协方差结构：它最小化擦除对非目标方向的扰动，而 mean-difference 直接投影。10% covariance shrinkage 会**约束** LEACE 的编辑幅度，使其更保守。mean-difference 没有这个约束，可能移除更多，但也造成更多附带损伤（虽然你的 role-probe drop ≈0 表明附带损伤在角色信息上不明显）。

**但这不是"mean-difference 更接近真实分布结构"的证据。** 两者都是线性方法，都只移除一个方向。差异在幅度，不在机制。如果要用这个对比支持"分布表征"，需要证明 mean-difference 移除的是**多个不重叠方向中的一个**，而其余方向仍携带身份信息——E3 没有做这个分析。

---

### e. E1 剂量测试的公平性

**不公平。** 选择标准是 worst-stratum AUC，这确实会偏向那些在困难负例上有判别力、但整体激活率极低的 latents。‖Δ‖/‖h‖ 仅 ~6% vs 随机 active latents ~30% 是直接证据：你消融的是"沉默的判别器"，不是"活跃的身份编码器"。

**正确的剂量测试**应该：(1) 选择标准改为整体激活率或方差解释量，而非 worst-stratum AUC；(2) 或者报告"在给定 ‖Δ‖/‖h‖ 匹配下的移除效果"，而非"在给定 k 下的移除效果"。当前 E1 只能说"消融这组特定选择的 low-activation latents 无效果"，不能说"SAE latents 剂量加到 128 也无效果"。

---

### f. 数据问题

**对 claim 1–2 威胁有限，对 claim 3–4 威胁不可忽略。**

Claim 1–2 的核心结果（线性可解码性强，G5r null）在 discovery+validation 上都稳定，且 G5r 在 discovery 上独立运行。Claude 生成的 validation 风格偏移可能影响可解码性的**绝对值**，但不太可能逆转"强可解码"的定性结论。

Claim 3–4 的威胁在于：validation 和 evaluation 都是 Claude 从模板生成的，如果模板引入了系统性的语言模式（例如固定的句式、词汇分布），下游 probe 可能在学习**模板特征**而非真正的身份泛化。E1–E3 只在 discovery+validation 上运行，没有在 evaluation 上验证。**探索性结论不应进入任何确认性表述。**

---

### g. 选择效应与探索性标签

**探索性标签足够，但 claims 3–5 的措辞需要大幅收紧。** 探索性分析在 G5r 之后运行是合理的（测试 null 的替代解释），但问题在于 claim 的措辞超出了数据能支持的范围。具体：

- Claim 3 的"not a dose or placement problem"应改为"the tested doses and placements did not produce removal under the current selection criteria"。
- Claim 4 的"unrecoverable at the site"必须删除或重述（§7a）。
- Claim 5 应删除或明确标注为"narrative, not tested"。

不需要移除任何分析，但摘要/标题不应包含 claim 3–5 的当前措辞。

---

### h. 诚实标题

**你提议的标题基本诚实，但需要两处修正：**

> "Identity is present but not removable by sparse SAE ablation; linear erasure removes it locally but it largely returns downstream; the Kantian dependency remains untested."

**问题一：** "removes it locally"暗示擦除在局部是有效的，但 amnesic AUC=0.50 是构造性的（§7a）。更准确的说法是"linear mean-difference erasure reduces downstream decodability modestly"。

**问题二：** "largely returns downstream"暗示有一个"返回"的过程，但更简约的解释是"信息在擦除之前已经到达下游"（§7b）。"returns"是一个因果叙事，"was already present at the read position"是描述性的。

**建议标题：**

> **"Identity is linearly decodable at the reference position, but neither ≤8 sparse SAE latents nor single-direction linear erasure at that span produced substantial downstream removal. The null is consistent with information routing before the intervention layer and with low-activation SAE feature selection; the Substance → Causality hypothesis remains untested."**

这个标题不宣称任何正面理论（"distributed Form"），不宣称"reconstitution"，只报告可观察到的和不可推断的。如果 PI 要求一个更简洁的版本：

> **"Substance remains decodable but not causally isolable at the tested intervention site; P2/P3 inconclusive."**