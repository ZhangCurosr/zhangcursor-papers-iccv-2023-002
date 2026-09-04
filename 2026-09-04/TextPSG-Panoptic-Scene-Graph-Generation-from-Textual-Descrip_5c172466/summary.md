---
title: "TextPSG-Panoptic-Scene-Graph-Generation-from-Textual-Descrip"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Zhao_TextPSG_Panoptic_Scene_Graph_Generation_from_Textual_Descriptions_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 21:20:05"
field: "弱监督视觉-语言场景理解"
keywords: ["Panoptic Scene Graph", "弱监督场景理解", "图文对齐", "自回归生成", "开放词汇", "文本监督语义分割"]
innovations: ["首次定义纯文本描述生成全景场景图（Caption-to-PSG）问题，摆脱位置先验和预定义概念集限制", "设计 four-module 协作框架 TextPSG，通过 fine-grained contrastive grounding 产出 pseudo labels 驱动 segment merger 和 label generator", "将标签预测重构为自回归生成问题，结合 PET 技术利用预训练 VLM 共同知识实现开放词汇标签预测"]
benchmarks: ["COCO Caption", "Panoptic Scene Graph Dataset [49]", "PhrDet", "SGDet", "TSSS (COCO mIoU)"]
---

# 论文速读：TextPSG-Panoptic-Scene-Graph-Generation-from-Textual-Descrip

## 一句话总结
本文首次提出**纯文本描述生成全景场景图（Caption-to-PSG）**问题，旨在仅利用图像-文本对弱监督信号、无需位置先验和预定义概念集的情况下，生成基于像素级分割掩码的结构化全景场景图。为此设计了 TextPSG 框架，通过区域分组器、实体定位器、片段合并器和标签生成器协同工作，在 COCO Caption 上训练、PSG 数据集上评估，显著超越基线并展现强 OOD 鲁棒性。

## 研究问题与动机
1. **现有 PSG 方法过度依赖密集标注数据**：全监督学习需要像素级分割掩码与关系标注，人工标注成本极高，难以扩展至更复杂的场景和概念。
2. **弱监督场景图生成仍受限于强前提条件**：现有方法依赖预训练的 Region Proposal Network（RPN）和预定义的概念集（物体语义与关系谓词），限制了对未见物体的泛化能力。
3. **文本描述蕴含丰富的结构化场景知识但未被充分利用**：Web 上存在大量免费图像-文本对，若能从中学习全景场景理解，可突破标注瓶颈，实现更全面的结构化场景理解。
4. ** bbox-based 场景图粒度不足**：边界框无法精确覆盖不规则形状物体，且重叠区域会引入噪声；全景场景图（PSG）采用像素级分割掩码，能提供更细粒度的结构化表示。

## 核心贡献（创新点）
1. **首次定义 Caption-to-PSG 问题**：在三个严格约束（无位置先验、无显式区域-实体链接、无预定义概念集）下，探索纯文本弱监督下的全景场景图生成，与先前依赖 RPN 和固定词表的弱监督方法形成本质区别。
2. **提出模块化框架 TextPSG**：由区域分组器（Region Grouper）、实体定位器（Entity Grounder）、片段合并器（Segment Merger）和标签生成器（Label Generator）四个模块协同，分别解决图像分割、图文对齐、片段合并与标签预测问题，相比 prior 工作将标签预测重新定义为自回归生成而非分类问题。
3. **设计基于 fine-grained contrastive learning 的图文对齐策略**：借鉴 FILIP，在每个分组阶段计算 token-wise 相似度，引入过滤阈值 $\theta$ 处理"图像中有区域无描述"或"描述中有实体无对应区域"的情况，实现隐式图文对齐并产出 pseudo labels。
4. **提出 PET（Prompt-embedding-based Technique）技术**：利用预训练 VLM（BLIP）的共同知识，通过设计 prompt（如 "a photo of [ENT]"、"a photo of [SUB] and [OBJ], what is their relation [REL]"）引导标签自回归生成，避免预定义词表限制，相比使用 WordNet 匹配的分类方法更准确鲁棒。
5. **证明模块的迁移价值**：Entity grounder 和 segment merger 可提升 text-supervised semantic segmentation（TSSS）性能，在 COCO Caption 上使 GroupViT 的 mIoU 提升 2.15%。

## 方法详解
**整体框架**：TextPSG 由四个协作模块组成，训练时输入图像-文本对批次，推理时输入单张图像。

1. **Text Graph Preprocessing（文本图预处理）**：
   - 使用规则语言解析器（基于 [37]）和 Stanford CoreNLP 的 OpenIE 系统提取 caption 的语言结构。
   - 构建 text graph：节点表示实体（entity），有向边表示实体对之间的关系（relation）。

2. **Region Grouper（区域分组器）**：
   - 采用分层设计（GroupViT [48]），输入图像被切分为 $N$ 个不重叠 patch 作为初始 segment $\{\mathbf{s}_i^0\}_{i=1}^N$。
   - 经过 $K$ 个 grouping layer，每层 $k$ 有 $H_k$ 个 learnable grouping centers $\{\mathbf{c}_i^k\}$，通过 attention 机制将 $H_{k-1}$ 个输入 segment 合并为 $H_k$ 个更大 segment：
     $$\{\mathbf{s}_i^k\}_{i=1}^{H_k} = \mathbf{Grp}_k(\{\mathbf{c}_i^k\}_{i=1}^{H_k}, \{\mathbf{s}_i^{k-1}\}_{i=1}^{H_{k-1}})$$
   - 输出多个分组阶段的 segment 组 $\{\mathbf{s}_i^k\}$。

3. **Entity Grounder（实体定位器）**：
   - 图像侧：每个阶段 $k$ 的 segment $\{\mathbf{s}_i^k\}$ 通过 MLP $\text{Proj}^I$ 映射到共享特征空间 $\mathcal{F}$，得到 segment embeddings $\{\mathbf{x}_i^k\}$。
   - 文本侧：caption tokenized 为 $M$ 个 token，经 Transformer $\text{Tfm}^T$ 交互后，RNN $\text{Rnn}$ 将同一实体的 token 合并为 entity embedding，再经 MLP $\text{Proj}^T$ 映射到 $\mathcal{F}$，得到 $\{\mathbf{y}_j\}_{j=1}^E$。
   - 计算 token-wise cosine similarity，引入阈值 $\theta$ 过滤低相似度对，计算 fine-grained image-to-text 相似度 $p^k$ 和 text-to-image 相似度 $q^k$。
   - 对比损失：
     $$\mathcal{L}_{fine}^{k, I \to T} = -\frac{1}{B}\sum_{i=1}^B \frac{\exp(p^{k,i\to i}/\tau)}{\sum_{j=1}^B \exp(p^{k,i\to j}/\tau)}, \quad \mathcal{L}_{fine}^{k, T \to I} = -\frac{1}{B}\sum_{i=1}^B \frac{\exp(q^{k,i\to i}/\tau)}{\sum_{j=1}^B \exp(q^{k,i\to j}/\tau)}$$
     $$\mathcal{L}_{fine}^k = \frac{1}{2}(\mathcal{L}_{fine}^{k, I \to T} + \mathcal{L}_{fine}^{k, T \to I})$$
   - 最终 grounding 结果：segment $\mathbf{s}_i^k$ 对齐到的 entity 为 $l_i^k = \arg\max_j \cos[\mathbf{x}_i^k, \mathbf{y}_j]$。

4. **Segment Merger（片段合并器）**：
   - 计算 segment 间余弦相似度并线性缩放至 $[0,1]$：
     $$\mathbf{Sim}_k[i,j] = \frac{1}{2}(\cos[\mathbf{x}_i^k, \mathbf{x}_j^k] + 1)$$
   - 利用 grounding pseudo labels 构建目标矩阵：
     $$\mathbf{Sim}_k^{target}[i,j] = \begin{cases} 1, & \text{if } l_i^k = l_j^k \land \cos[\mathbf{x}_i^k, \mathbf{y}_{l_i^k}] > \theta \land \cos[\mathbf{x}_j^k, \mathbf{y}_{l_j^k}] > \theta \\ 0, & \text{otherwise} \end{cases}$$
   - 相似度损失：
     $$\mathcal{L}_{sim}^k = \frac{1}{H_k^2}\|\mathbf{Sim}_k - \mathbf{Sim}_k^{target}\|_F^2$$
   - 训练时学习 similarity matrix，推理时用 graph cut 进行聚类合并。

5. **Label Generator（标签生成器）**：
   - **重构为自回归生成问题**：不再使用固定词表的分类，而是利用预训练 VLM（BLIP decoder）的共同知识生成标签。
   - **PET 技术**：
     - 物体预测 prompt："a photo of [ENT]"，[ENT] 期望为 pseudo label $b_i^k$。
     - 关系预测 prompt："a photo of [SUB] and [OBJ], what is their relation [REL]"，[SUB] 和 [OBJ] 由 pseudo labels 嵌入，[REL] 期望为关系谓词。
     - 引入三个可学习位置嵌入 $\mathbf{f}_{sub}, \mathbf{f}_{obj}, \mathbf{f}_{region}$ 指示不同区域。
   - 两个 cross-entropy 损失 $\mathcal{L}_{ent}^k, \mathcal{L}_{rel}^k$ 分别监督 [ENT] 和 [REL] token 的生成。

6. **Inference**：
   - 输入图像 $I$，经 region grouper 得到 candidate segments，再通过 segment merger（spectral clustering + graph cut + matrix recovery）合并为最终 segment。
   - Label generator 对每个 segment/pair 生成物体语义和关系谓词的概率分布，从已知概念集 $\mathcal{C}_o, \mathcal{C}_r$ 中选择概率最高的标签。
   - 语义分割转实例分割：简单地将每个连通分量视为一个 instance。

## 实验与结果
**数据集**：
- 训练：COCO Caption [4]，123,287 张图像，每张 5 个人工标注 caption，取 118,287 张用于训练。
- 评估：Panoptic Scene Graph dataset [49]，合并歧义类别后得到 127 个物体语义和 56 个关系谓词。

**评估协议与指标**：
- 任务：Visual Phrase Detection (PhrDet) 和 Scene Graph Detection (SGDet)。
- 指标：No-Graph-Constraint-X Recall@K (NXR@K, %)——允许每个 subject-object 对最多预测 X 个谓词标签。

**主要结果（Table 1，bbox mode）**：
- TextPSG（Ours）在 PhrDet N5R100 达到 **14.37%**，SGDet N5R100 达到 **5.48%**，显著超越所有基线。
- 最强 baseline SGGNLS-c（有位置先验+有概念集）PhrDet N5R100 为 12.22%，SGDet N5R100 为 8.65%；Ours mask mode 略低但 bbox mode 超过 SGGNLS-o（6.79 vs 14.37 PhrDet N5R100）。
- Ours 在完全无位置先验和无概念集约束下优于使用预训练检测器的 SGGNLS-o。

**OOD 鲁棒性（Table 2）**：
- 在 OOD 集上，SGGNLS-c/o 性能骤降至接近 0（N5R100 PhrDet 为 0），而 Ours 保持 9.76%（mask）/11.69%（bbox），展现强 OOD 泛化能力。

**消融实验**：
- Segment Merger：在 stage 1（64 segments）应用 graph cut 效果最佳（PhrDet N5R100: 14.37%）。
- Label Generator：Gen w/ PET（BLIP）最优（PhrDet N5R100: 14.28%），远优于分类+WordNet（9.36%）和无 PET 的生成（2.58%）。

**TSSS 应用（Table 5）**：
- GroupViT + Ours 模块：mIoU 26.87%，相比 GroupViT 微调（24.72%）提升 2.15%。

## 相关工作脉络
1. **Bbox-based Scene Graph Generation**：多数工作 [50, 47, 41, 12, 24] 采用全监督学习，依赖密集标注数据集 [18, 13]；弱监督方法 [32, 55, 57, 40] 试图减少标注需求，但多数仍需 region proposal。本文相比彻底摆脱 RPN 依赖。
2. **Weakly-supervised SGG from image-caption pairs**：[52, 58, 22] 探索从图像-文本对学习场景图，但仍依赖 pre-trained RPN 和固定概念集；本文设定更严格的三个约束，更具挑战性但泛化性更强。
3. **Panoptic Scene Graph (PSG)**：[49, 45] 提出 PSG 概念并使用全监督学习；本文首次探索 PSG 的纯文本弱监督生成。
4. **Text-supervised Semantic Segmentation (TSSS)**：[48, 20, 26, 9, 27, 59] 从图像-文本对学习像素级语义标注；本文在 TSSS 基础上进一步学习关系，实现高阶结构化理解。
5. **Visual Grounding**：早期方法 [54, 53, 7] 先检测物体 proposal 再匹配文本；弱监督方法 [15, 36, 5] 通过 multiple instance learning 或 reconstruction 减轻标注需求；本文在无 proposal 网络且无固定词表的条件下进行细粒度图文对齐。
6. **Fine-grained Image-Text Alignment**：FILIP [51] 采用 token-wise contrastive learning；本文借鉴并引入阈值过滤机制处理不对等情况，同时产出 pseudo labels 供下游模块使用。

## 局限性与未来方向
1. **语义分割转实例分割策略简单**：当前仅将连通分量视为独立 instance，在物体重叠或遮挡情况下会导致 instance 数量估计偏差（高估或低估）。
2. **小物体定位困难**：受限于图像分辨率和分组策略，对小物体的定位效果不佳。
3. **关系预测对图像条件依赖不足**：label generator 有时过度依赖物体语义而忽视实际图像内容，需要更合适的 image-conditioned reasoning 机制。
4. **训练数据粒度不足**：现有图像-文本对 caption 粒度较粗（常将同类物体合并为复数形式描述），限制了从学习中实现 panoptic segmentation 的能力，需要更细粒度的数据集。
5. **未来方向**：a) 更精细的分割转换策略；b) 提高输入分辨率；c) 改进关系预测的图像条件推理机制；d) 构建更细粒度标注的图像-文本-对数据集。

## 研究启发与可借鉴点
1. **无先验条件下的图文对齐策略**：引入阈值过滤的 fine-grained contrastive learning 有效处理"多对一"或"一对多"不对等情况，可迁移至其他弱监督多模态对齐任务。
2. **自回归生成替代分类的标签预测范式**：将标签预测重构为生成问题并借助预训练 VLM 共同知识，避免了预定义词表的限制，适合开放词汇/零样本场景。
3. **Pseudo label 驱动的多模块协同**：grounding 结果作为 pseudo label 同时服务于 segment merger 和 label generator，实现模块间信息流动，这种"软监督传递"设计值得借鉴。
4. **PET（Prompt-embedding-based Technique）的设计思路**：通过结构化 prompt 引导 VLM 生成特定类型标签，结合可学习位置嵌入区分不同角色，可在其他条件生成任务中复用。
5. **跨任务迁移验证**：证明所提模块可提升 TSSS 性能，说明该方法具有通用性，为多任务联合学习提供了新思路。

## 关键术语表
**Panoptic Scene Graph (PSG)**：一种综合场景表示，每个物体由 panoptic segmentation mask 定位并附带语义标签，边表示物体间的关系谓词。
**Caption-to-PSG**：本文提出的新问题，指仅利用图像-文本对（纯文本描述）弱监督信号生成全景场景图。
**Region Grouper**：将输入图像分层合并为多个 segment 的模块，借鉴 GroupViT 的分层注意力机制。
**Entity Grounder**：基于 fine-grained contrastive learning 将文本实体定位到图像 segment 的模块，产出 pseudo labels。
**Segment Merger**：学习 segment 间相似度矩阵并通过 graph cut 合并相似 segment 的模块，利用 grounding pseudo labels 作为显式监督。
**Label Generator**：利用预训练 VLM（BLIP decoder）和 PET 技术自回归生成物体语义和关系谓词标签的模块。
**PET (Prompt-embedding-based Technique)**：通过设计结构化 prompt（含 [ENT], [SUB], [OBJ], [REL] 等特殊 token）引导 VLM 生成标签的技术。
**No-Graph-Constraint-X Recall@K (NXR@K)**：允许每个 subject-object 对最多预测 X 个谓词标签的 Recall@K 指标，适用于非互斥谓词场景。

## 可复现要素
- **数据集**：COCO Caption（训练，公开）、Panoptic Scene Graph dataset（评估，公开）；论文未提及自定义数据集。
- **代码/权重**：论文声明代码、数据和结果在项目页面开源：https://vis-www.cs.umass.edu/TextPSG。
- **关键超参**：分组层数 $K=2$，第一阶段 segment 数 $H_1=64$，第二阶段 $H_2=8$；推理阶段 $k_{inf}=1$；相似度过滤阈值 $\theta$（具体数值论文未在主文中明确，见 supplementary）；温度参数 $\tau$（learnable）；batch size $B$。
- **预训练模型**：GroupViT（region grouper）、BLIP [21]（label generator decoder）、Transformer $\text{Tfm}^T$（entity grounder）；训练时 $\text{Tfm}^T$ 和 label generator 冻结。
