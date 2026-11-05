---
title: "Self-supervised-Cross-view-Representation-Reconstruction-for"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Tu_Self-supervised_Cross-view_Representation_Reconstruction_for_Change_Captioning_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 00:55:10"
field: "视觉语言理解"
keywords: ["change captioning", "self-supervised learning", "cross-view representation", "contrastive alignment", "view-invariant features"]
innovations: ["通过自监督跨视角对比对齐学习视角不变表示", "设计多头标记级匹配实现细粒度跨视角特征交互", "引入跨模态反向推理自监督强化描述信息量"]
benchmarks: ["CLEVR-Change", "CLEVR-DC", "Image Editing Request", "Spot-the-Diff"]
---

# 论文速读：Self-supervised-Cross-view-Representation-Reconstruction-for

## 一句话总结
本文提出 SCORER（Self-supervised Cross-view Representation Reconstruction）网络，通过自监督跨视角对比对齐学习视角不变图像表示，并重建未变对象表示以获取稳定差异特征，从而在存在视角变化（伪变化）的情况下实现高质量图像变化描述（change captioning）。

## 研究问题与动机
- 变化描述任务需描述相似图像对之间的差异，但真实场景中常伴随视角变化，导致物体在尺度、位置上发生偏移（伪变化），使现有基于直接特征匹配或相减的方法难以学习稳定的差异表示。
- 现有方法直接对两幅图像的特征进行交叉注意力匹配，当视角变化剧烈时，特征偏移会淹没局部真实变化，导致描述质量下降。
- 作者观察到：（1）不同图像对之间的特征差异更难被混淆，对比学习能帮助模型聚焦特征变化并抵抗特征偏移；（2）伪变化本质上是对物体的不同畸变，仅构成跨视角比较，不改变图像对本身的相似性。
- 因此，本文旨在通过自监督方式最大化相似图像对的跨视角对比对齐，学习视角不变的表示，再重建未变对象表示，最终获得稳定的差异表示用于描述生成。

## 核心贡献（创新点）
- 提出 SCORER 框架，首次通过最大化跨视角对比对齐自监督学习视角不变图像表示，并据此重建未变对象表示以获取稳定差异特征，与直接匹配/相减的方法本质不同。
- 设计多头标记级匹配（MTM）模块，通过交互不同特征子空间实现细粒度跨视角匹配，解决了伪变化下 token 级匹配不足的问题。
- 引入跨模态反向推理（CBR）模块，利用生成句子和“之前”图像反向构建“幻觉”表示，并通过对比对齐推动其接近“之后”图像表示，从而自监督地强制句子包含丰富的差异信息。
- 在四个公开数据集（CLEVR-Change、CLEVR-DC、Image Editing Request、Spot-the-Diff）上达到 SOTA，并在不同变化类型（尤其是 Move）下显著提升。

## 方法详解
- **输入编码**：使用预训练 ResNet-101 提取前后图像的网格特征 $X_{bef}, X_{aft} \in \mathbb{R}^{C \times H \times W}$，经 2D 卷积和可学习位置嵌入投影到维度 D：$\tilde{X}_o = \text{conv}_2(X_o) + \text{pos}(X_o)$。
- **多头标记级匹配（MTM）**：单头 TM 计算查询 Q 与键 K 之间所有 token 对的相似度矩阵 $e_{i,j}=q_i^\top k_j$，取每行和每列的最大值后平均得到相似性 $\text{TM}(Q,K)=\frac{1}{N}\sum_i\max_j e_{i,j} + \frac{1}{N}\sum_j\max_i e_{i,j}$ 后除以 2。多头版本将 Q、K 分别投影到 h 个子空间后计算 TM 并拼接：$\text{MTM}(Q,K)=\text{Concat}_{i'=1..h}(\text{TM}(QW_{i'}^Q, KW_{i'}^K))$。
- **视角不变表示学习**：在 batch 中，第 k 个 before 图像与其对应的 after 图像为正样本，其余 after 图像为负样本。使用 MTM 计算相似度后，通过 InfoNCE 损失 $\mathcal{L}_{b2a}$ 和 $\mathcal{L}_{a2b}$ 最大化跨视角对比对齐，总损失 $\mathcal{L}_{cv}=\frac{1}{2}(\mathcal{L}_{b2a}+\mathcal{L}_{a2b})$，得到伪变化不变的表示 $\tilde{X}_{bef}, \tilde{X}_{aft}$。
- **跨视角表示重建**：利用多头交叉注意力（MHCA）从两个不变表示中挖掘共同特征，重建未变对象表示：$\tilde{X}_{bef}^u = \text{MHCA}(\tilde{X}_{bef}, \tilde{X}_{aft}, \tilde{X}_{aft})$，$\tilde{X}_{aft}^u = \text{MHCA}(\tilde{X}_{aft}, \tilde{X}_{bef}, \tilde{X}_{bef})$。然后将重建表示与原表示相加并归一化：$\tilde{X}_o^c = \text{LN}(\tilde{X}_o + \tilde{X}_o^u)$，最后拼接两个 $\tilde{X}^c$ 经全连接层得到差异表示 $\tilde{X}_c$。
- **描述生成**：使用 Transformer decoder，以词嵌入 $E[W]$ 为输入，经自注意力得到 $\hat{E}[W]$，再通过交叉注意力从 $\tilde{X}_c$ 查询相关特征 $\hat{H}$，经前馈网络得到 $\tilde{H}$，最后 Softmax 输出词分布。
- **跨模态反向推理（CBR）**：对生成表示 $\tilde{H}$ 做均值池化得到句子特征 $\tilde{T}$，广播后与 $\tilde{X}_{bef}$ 拼接并卷积得到“幻觉”表示 $\hat{X}_{hal}$，再用多头自注意力（MHSA）建模内部关系得到 $\tilde{X}_{hal}$。同样采用 InfoNCE 损失 $\mathcal{L}_{cm}$ 推动 $\tilde{X}_{hal}$ 与 $\tilde{X}_{aft}$ 对齐，使句子充分描述差异。
- **联合训练**：总损失 $\mathcal{L}=\mathcal{L}_{cap}+\lambda_v\mathcal{L}_{cv}+\lambda_m\mathcal{L}_{cm}$，端到端优化。

## 实验与结果
- **数据集**：CLEVR‑Change（中等视角变化）、CLEVR‑DC（极端视角变化）、Image Editing Request（对齐图像对）、Spot‑the‑Diff（监控场景对齐图像对）。
- **评估指标**：BLEU‑4 (B)、METEOR (M)、ROUGE‑L (R)、CIDEr (C)、SPICE (S)。
- **主要结果**：
  - **CLEVR‑Change**：SCORER+CBR 在总性能上 B=41.2、M=56.3、R=74.5、C=126.8、S=33.3；在语义变化上 B=54.4、M=37.6、R=71.7、C=122.4、S=31.6，均优于所有端到端基线（如 NCT、VARD‑Trans、MCCFormers‑D）。相比 SCORER，CBR 进一步带来约 1–2% 的 CIDEr 提升。
  - **CLEVR‑Change 各变化类型**（Table 2）：在最具挑战的 Move 类别上，SCORER+CBR 达 CIDEr 92.2，相对 R³Net+SSP（88.1）提升 4.7%。
  - **CLEVR‑DC**：SCORER+CBR 在 B、M、C、S 上均达最高（B=49.5、M=33.4、C=83.7、S=16.2），较 VACC 在 CIDEr 上提升 16.7%。
  - **IER**：SCORER+CBR 在 BLEU‑4 上达 10.0，较最新方法 NCT（9.6）提升约 4.2%，相对提升 23.5%（原文表述）。
  - **Spot‑the‑Diff**：SCORER 在 B（10.2）、C（38.9）、S（19.3）上领先；SCORER+CBR 在 M（12.2）和 S（18.4）上略有下降，作者解释因数据集常含多变化而单变化设定下 CBR 受限。
- **消融**：Subtraction < RR < SCORER，证明视角不变学习的关键作用；CBR 对 RR 和 SCORER 均有增益；MTM 优于 max/mean‑pooling 及单头 TM；head 数最优为 8，layer 数依数据集设为 1–3。

## 相关工作脉络
- **直接相减法**（DUDA、DUDA+）：对两图像进行像素或特征相减，易受伪变化噪声干扰，本文通过对比学习和表示重建避免直接相减。
- **跨视角特征匹配法**（MCCFormers‑D、NCT、VARD‑Trans）：直接利用交叉注意力匹配两图特征，在剧烈视角变化下特征偏移严重；本文引入多头子空间匹配和自监督对齐以提升鲁棒性。
- **预训练增强方法**（PCL w/ pre‑training）：通过三个自监督任务改进跨模态对齐，但未专门处理视角不变性；本文的 MTM 和 CBR 均从特征层面强化跨视角对齐。
- **循环一致性约束**（VACC、SRDRL+AVS）：通过重建一致性验证描述质量，但未显式学习视角不变表示；本文 CBR 以反向推理方式自监督强化句子信息量。
- **Token‑wise 匹配在检索中的应用**：多用于图文检索，但仅在 token 级交互；本文扩展为多子空间交互，以应对伪变化下的局部弱特征。

## 局限性与未来方向
- 论文指出在 Spot‑the‑Diff 等多变化场景中，CBR 因单变化假设而与真实 after 表示不完全匹配，导致 METEOR 和 SPICE 略降。
- 网络层数增加易过拟合（尤其在背景复杂的监控场景），当前调参较为依赖实验启发。
- 方法依赖 ResNet‑101 提取网格特征，未探索更高效的视觉 backbone 或端到端学习。
- 未来可探索：多变化场景下的 CBR 自适应扩展、更深层稳定架构、跨数据集泛化及实时应用场景。

## 研究启发与可借鉴点
- **视角不变表示学习**：通过跨视角对比对齐自监督学习不变表示的思路，可迁移至其他视角敏感任务（如视频目标跟踪、跨视角目标检测）。
- **表示重建而非相减**：用交叉注意力重建未变部分并与原表示融合，比直接相减更能保留参照物信息，可应用于图像编辑、视频差分解码等。
- **跨模态反向推理**：利用生成结果反向构建视觉表示并施加对比约束，可推广至视频描述、图文生成等需要反馈优化的任务。
- **细粒度子空间匹配**：MTM 在多子空间交互上的设计，可为多模态匹配任务提供轻量且有效的替代方案。
- **实验设计**：对同一数据集按变化类型细分评测（如 CLEVR‑Change 的五类变化），以及 IoU 量化的视角难度分析，值得借鉴。

## 关键术语表
- **Change Captioning**：变化描述，要求模型理解两幅相似图像并生成描述其差异的自然语言句子。
- **Pseudo Change**：伪变化，由视角、光照、尺度等引起的物体外观变化，非真实内容改变，会干扰差异学习。
- **Multi‑head Token‑wise Matching (MTM)**：多头标记级匹配，通过多个子空间交互计算跨视角图像对相似度，支持细粒度特征对齐。
- **View‑invariant Representation**：视角不变表示，经自监督对比学习后对伪变化鲁棒的图像特征。
- **Cross‑view Representation Reconstruction**：跨视角表示重建，利用交叉注意力从对方图像中提取共同特征以恢复未变对象的表示。
- **Cross‑modal Backward Reasoning (CBR)**：跨模态反向推理，根据生成的句子和“之前”图像反向构建“幻觉”表示，并通过对比对齐迫使句子包含丰富差异信息。
- **InfoNCE Loss**：对比学习常用的信息噪声对比估计损失，用于最大化正样本对相似度并最小化负样本对相似度。
- **CLEVR‑Change / CLEVR‑DC**：两个合成变化描述数据集，分别模拟中等和极端视角变化。

## 可复现要素
- **数据集**：CLEVR‑Change、CLEVR‑DC、Image Editing Request、Spot‑the‑Diff（均为公开数据集，可用官方划分）。
- **代码/权重**：代码已开源（https://github.com/tuyunbin/SCORER），模型权重未明确提及是否公开。
- **关键超参**：ResNet‑101 提取 1024 维网格特征，投影至 512 维；隐藏层及词嵌入维度均为 512；Transformer decoder 8 头 2 层；损失权重 $\lambda_v、\lambda_m$ 见补充材料；Adam 优化器；训练/推理均在 RTX 3090 上实现。
