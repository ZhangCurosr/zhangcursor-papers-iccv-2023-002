---
title: "Self-supervised-Cross-view-Representation-Reconstruction-for"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Tu_Self-supervised_Cross-view_Representation_Reconstruction_for_Change_Captioning_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 21:18:55"
field: "变化描述与跨视图表示学习"
keywords: ["change captioning", "cross-view representation", "self-supervised learning", "contrastive alignment", "pseudo change", "vision-language"]
innovations: ["通过自监督跨视图对比对齐学习视角不变表示以抵抗伪变化", "多头token级匹配实现细粒度跨视图特征子空间交互", "跨模态反向推理强制caption携带充分差异信息"]
benchmarks: ["CLEVR-Change", "CLEVR-DC", "Image Editing Request", "Spot-the-Diff"]
---

# 论文速读：Self-supervised-Cross-view-Representation-Reconstruction-for-Change-Captioning

## 一句话总结
本文提出 **SCORER**（Self-supervised CrOss-view REpresentation Reconstruction）网络，通过自监督跨视图对比对齐学习视角不变的图像表示，并据此重建不变对象表示以获取稳定的差异表示；进一步引入 **CBR**（Cross-modal Backward Reasoning）模块，从跨模态反向推理角度强制caption携带差异信息，最终在四个变化描述数据集上均取得SOTA性能。

## 研究问题与动机
- **核心问题**：Change captioning 需要在存在**伪变化**（pseudo changes，主要由视角变化引起）的条件下准确描述两幅相似图像之间的真实语义变化。
- **现有方法不足**：基于匹配的方法（如 MCCFormers-D、NCT 等）直接在两幅图像的特征间建模跨视图关系，但视角变化会导致对应对象特征发生偏移（feature shift），局部特征变化容易被全局特征偏移淹没，难以学习稳定的差异表示。
- **关键洞察**：（1）不同图像对之间的差异对比能更突出特征变化，从而抵抗特征偏移；（2）伪变化本质是对象的不同形变，不改变图像对的相似性，因此可通过最大化跨视图对比对齐学习视角不变的表示。
- **动机**：先学习视角不变的表示 → 重建不变对象表示 → 隐式推断差异 → 生成高质量 caption。

## 核心贡献（创新点）
1. **SCORER 框架**：通过自监督跨视图对比对齐学习视角不变的表示，再重建不变对象表示以获取稳定差异表示；与直接匹配两幅图像的方法（如 MCCFormers-D）本质不同，后者在剧烈视角变化下易受特征偏移干扰。
2. **Multi-head Token-wise Matching (MTM)**：在多个特征子空间上分别执行 token 级最大相似度交互，实现更细粒度的跨视图特征匹配；与仅做 token 级匹配或全局池化相比，能更好地捕捉局部弱特征的变化。
3. **Cross-modal Backward Reasoning (CBR)**：用生成的 caption 和"before"表示反向生成"幻觉"表示，并通过跨视图对比对齐将其推向"after"表示；与已有工作仅约束变化对象一致性不同，CBR 利用完整表示进行反向推理，使 caption 更丰富。
4. **端到端统一训练**：整体网络通过联合损失函数（caption loss + 跨视图对比 loss + 跨模态反向推理 loss）进行端到端训练，训练效率高。

## 方法详解
### 整体架构
由四部分组成：预训练 CNN 编码器 → SCORER（MTM + 视角不变表示学习 + 跨视图表示重建）→ Transformer Decoder → CBR 模块。

### 1) 跨视图图像对编码
- 输入："before" 图像 $I_{bef}$ 和 "after" 图像 $I_{aft}$。
- 用预训练 ResNet-101 提取网格特征 $X \in \mathbb{R}^{C \times H \times W}$，经 2D 卷积投影至维度 $D$，并加可学习位置嵌入：
  $$\tilde{X}_o = \text{conv}_2(X_o) + \text{pos}(X_o), \quad o \in (bef, aft)$$

### 2) 多头 Token-wise Matching (MTM)
- **单头 TM**：对 query $Q$ 和 key $K$，先对每个 query token 取与所有 key token 的最大相似度，再对所有 query token 求平均；同时反向计算，取两者均值：
  $$\text{TM}(Q, K) = \left[\frac{1}{N}\sum_i \max_j e_{i,j} + \frac{1}{N}\sum_j \max_i e_{i,j}\right]/2, \quad e_{i,j}=q_i^\top k_j$$
- **多头 MTM**：在不同子空间上分别计算 TM，再拼接：
  $$\text{MTM}(Q,K)=\text{Concat}_{i'=1..h}(\text{TM}(QW_{i'}^Q, KW_{i'}^K))$$

### 3) 视角不变表示学习
- 在 batch 中，同对的 bef/after 为正样本，其他为负样本。
- 用 MTM 计算 bef→aft 和 aft→bef 的相似度矩阵，通过 InfoNCE loss 最大化正对的跨视图对比对齐：
  $$\mathcal{L}_{b2a} = -\frac{1}{B}\sum_k \log \frac{\exp(\text{MTM}(\tilde{X}_k^b, \tilde{X}_k^a)/\tau)}{\sum_r \exp(\text{MTM}(\tilde{X}_k^b, \tilde{X}_r^a)/\tau)}$$
  $$\mathcal{L}_{cv} = \frac{1}{2}(\mathcal{L}_{b2a} + \mathcal{L}_{a2b})$$
- 目的：使 $\tilde{X}_{bef}$ 和 $\tilde{X}_{aft}$ 的表示对伪变化不变。

### 4) 跨视图表示重建
- 用多头交叉注意力（MHCA）从另一幅图像中挖掘共同特征，重建每幅图像的不变对象表示：
  $$\tilde{X}_{bef}^u = \text{MHCA}(\tilde{X}_{bef}, \tilde{X}_{aft}, \tilde{X}_{aft})$$
  $$\tilde{X}_{aft}^u = \text{MHCA}(\tilde{X}_{aft}, \tilde{X}_{bef}, \tilde{X}_{bef})$$
- 将重建表示与原表示融合（LayerNorm）以突出不变对象、隐式推断差异：
  $$\tilde{X}_o^c = \text{LN}(\tilde{X}_o + \tilde{X}_o^u)$$
- 拼接 bef 和 aft 的差异表示：
  $$\tilde{X}_c = \text{ReLU}([\tilde{X}_{bef}^c; \tilde{X}_{aft}^c]W_h + b_h)$$

### 5) Caption 生成
- Transformer Decoder 以 $\tilde{X}_c$ 为条件，通过多头自注意力（词间关系）和多头交叉注意力（词-图对齐）生成词序列，负对数似然损失 $\mathcal{L}_{cap}$ 优化。

### 6) 跨模态反向推理 (CBR)
- 将 decoder 输出的隐藏表示 $\tilde{H}$ 经 mean-pooling 得句子特征 $\tilde{T}$，与 $\tilde{X}_{bef}$ 拼接后经 2D 卷积生成"幻觉"表示 $\hat{X}_{hal}$。
- 对 $\hat{X}_{hal}$ 做 MHSA 得到 $\tilde{X}_{hal}$。
- 在 batch 内，$\tilde{X}_{hal}$ 与同对 $\tilde{X}_{aft}$ 为正，其他为负，用 InfoNCE loss 最大化跨视图对齐 $\mathcal{L}_{cm}$。
- 目的：强制 caption 充分描述差异信息。

### 7) 联合训练
$$\mathcal{L} = \mathcal{L}_{cap} + \lambda_v \mathcal{L}_{cv} + \lambda_m \mathcal{L}_{cm}$$

## 实验与结果
### 数据集
- **CLEVR-Change**：79,606 对，中等视角变化，5 种变化类型（Color/Texture/Add/Drop/Move），67,660/3,976/7,970 划分。
- **CLEVR-DC**：48,000 对，极端视角变化，85%/5%/10% 划分。
- **Image Editing Request (IER)**：3,939 对对齐图像，3,061/383/495 划分。
- **Spot-the-Diff**：13,192 对监控图像，单变化设置，8:1:1 划分。

### 评估指标
BLEU-4 (B)、METEOR (M)、ROUGE-L (R)、CIDEr (C)、SPICE (S)。

### 主要结果
- **CLEVR-Change**：SCORER+CBR 取得 Total 性能最优：B=41.2, M=56.3, R=74.5, C=126.8, S=33.3；较 NCT 相对提升约 2.5%。在最难类别"Move"上较 R³Net+SSP 相对提升 4.7%。
- **CLEVR-DC**：SCORER+CBR CIDEr 较 VACC 提升 16.7%。
- **IER**：SCORER+CBR BLEU-4 较 NCT 相对提升 23.5%（B=10.0 vs NCT 的 8.1）。
- **Spot-the-Diff**：SCORER 取得最优，但 SCORER+CBR 在 M 和 S 上略降（因该数据集多含多个变化，与单变化假设不完全匹配）。

### 消融结论
- Subtraction（直接相减）< RR（基础重建）< SCORER（加对比学习）< SCORER+CBR，证明各模块有效性。
- MTM 优于 max/mean pooling 和单头 TM；head number=8 最优；layer number 因数据集而异（2/1/3/2），过深易过拟合。

## 相关工作脉络
1. **DUDA / DUDA+**：早期基于直接图像减法的方法，在不对齐图像上引入噪声；SCORER 用重建替代减法，避免信息损失。
2. **M-VAM / M-VAM+RAF**：基于强化学习的匹配方法；SCORER 用自监督对比学习替代 RL，训练更高效。
3. **VACC**：引入 cycle consistency 约束；SCORER 的 CBR 扩展为利用完整表示的反向推理，而非仅约束变化对象。
4. **MCCFormers-D**：经典跨视图直接匹配方法；SCORER 通过先学习视角不变表示再重建，从根本上抵抗特征偏移。
5. **NCT**：neighborhood contrastive transformer；SCORER 在对比学习粒度上更细（token 级子空间交互）。
6. **PCL w/ pre-training**：预训练+对比学习提升跨模态对齐；SCORER 无需额外预训练阶段，端到端联合优化。

## 局限性与未来方向
- **多变化场景适配有限**：在 Spot-the-Diff 上 CBR 效果不明显，因该数据集常含多个变化，与单变化 caption 的假设不完全匹配。
- **层数敏感性**：SCORER 层数过深易过拟合，需按数据集调优（2~3 层）。
- **未来方向**：（1）扩展至多变化场景；（2）探索更鲁棒的反向推理机制；（3）结合更强的预训练视觉 backbone。

## 研究启发与可借鉴点
1. **跨视图对比对齐学习视角不变表示**的思路可有效抵抗伪变化，可迁移到其他需要消除视角/域偏移的跨视图任务（如变化检测、视频对齐）。
2. **MTM 的细粒度子空间匹配策略**：在 token 级最大相似度基础上叠加多头子空间交互，比单纯全局池化或单头匹配更能捕捉局部弱特征变化，值得在其他细粒度匹配任务中借鉴。
3. **CBR 的反向推理机制**：用生成结果反推视觉表示并与目标对齐，是一种自监督的质量校准思路，可推广至其他视觉描述/grounding 任务的辅助训练。
4. **重建代替减法的差异表示学习**：避免了减法导致的 referent 信息损失，同时保留上下文，对变化感知任务有参考价值。
5. **端到端联合优化 vs 两阶段预训练**：SCORER 无需额外预训练阶段即可达到 SOTA，对计算资源有限的研究场景有借鉴意义。

## 关键术语表
**Change Captioning**：描述两幅相似图像之间差异的自然语言生成任务。
**Pseudo Change**：由视角、光照等非语义因素引起的对象外观变化，非真实语义差异。
**View-invariant Representation**：对伪变化（如视角变换）具有不变性的图像表示。
**Multi-head Token-wise Matching (MTM)**：在多个特征子空间上分别进行 token 级最大相似度匹配的跨视图交互模块。
**Cross-modal Backward Reasoning (CBR)**：利用生成的 caption 和"before"表示反向生成"幻觉"表示，并通过对比对齐强制其接近"after"表示的自监督模块。
**InfoNCE Loss**：对比学习中的噪声对比估计损失，用于拉近正样本对、推远负样本对。
**Cross-view Contrastive Alignment**：最大化跨视图正样本对的相似度、最小化负样本对相似度的自监督对齐目标。
**Representation Reconstruction**：从另一幅图像中提取共同特征来重建当前图像中不变对象表示的过程。

## 可复现要素
- **数据集**：CLEVR-Change、CLEVR-DC、IER、Spot-the-Diff 均官方公开。
- **代码**：已开源，https://github.com/tuyunbin/SCORER。
- **权重**：论文未提及预训练权重发布。
- **关键超参**：CNN 用 ResNet-101，embedding 维度 512，word embedding 维度 300，SCORER head number=8，layer number 依数据集设为 2/1/3/2，temperature $\tau$ 见补充材料，优化器 Adam。
