---
title: "Partition-and-Debias-Agnostic-Biases-Mitigation-via-A-Mixtur"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Li_Partition-And-Debias_Agnostic_Biases_Mitigation_via_a_Mixture_of_Biases-Specific_Experts_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 19:15:09"
field: "计算机视觉 - 公平性与偏差缓解"
keywords: ["bias mitigation", "agnostic biases", "mixture of experts", "debiased classification", "counterfactual learning", "image classification"]
innovations: ["提出 agnostic biases 新场景，同时未知偏差类型与数量", "设计 PnD 框架，按网络深度划分偏差子空间并用多 expert 并行去偏", "引入两阶段训练（初始训练+反事实对比训练）与 KL 多样性正则"]
benchmarks: ["Biased MNIST", "BAR", "Modified IMDB", "MIMIC-CXR + NIH", "CelebA"]
---

# 论文速读：Partition-and-Debias-Agnostic-Biases-Mitigation-via-A-Mixtur

## 一句话总结
提出 Partition-and-Debias (PnD) 方法，通过在不同网络深度插入多个偏差特定专家（biases-specific experts）并结合门控模块，解决图像分类中"不可知偏差"（agnostic biases，即偏差类型与数量均未知）场景下的去偏问题。

## 研究问题与动机
- **真实场景中存在多重未知偏差**：以 CelebA 年龄分类为例，young/old 类别中同时混杂 gender、attractiveness、lipstick 等多种偏差属性，43.75% 的 young 样本同时含三种偏差，60%+ 含两种及以上，现实偏差具有普遍共存性。
- **既有方法假设过强**：已有去偏工作通常假设图像中仅含单类已知/未知偏差，且依赖强先验（如偏差易于学习、目标特征更简单、偏差可编辑等），无法应对多重未知偏差共存的场景。
- **单层去偏不充分**：仅去除某一偏差并不能消除所有偏差的影响，不同偏差特征散落在网络不同深度，需要在多层次并行处理。
- **填补研究空白**：引入"agnostic biases"新场景，同时放宽对偏差类型和数量的先验假设，推动去偏方法向真实应用靠拢。

## 核心贡献（创新点）
1. **定义 agnostic biases 新场景**：同时未知偏差类型与数量，超越现有工作仅考虑单一或已知偏差的局限。
2. **Partition-and-Debias (PnD) 框架**：提出分治策略，将偏差空间按网络深度划分为多个子空间，各子空间由独立的 biases-specific expert 处理。
3. **两阶段训练机制**：初始训练（initial training）通过 GCE 损失与权重 CE 损失联合预热编码器；反事实训练（counterfactual training）引入对比损失显式分离目标与偏差特征。
4. **KL 散度多样性正则**：通过约束相邻专家偏差检测结果之间的 KL 散度，迫使不同 expert 关注不同层次特征，最大化偏差捕获多样性。
5. **自适应门控聚合**：门控模块对各 expert 的去偏预测结果进行 softmax 加权融合，输出最终一致预测。

## 方法详解
- **双编码器结构**：去偏编码器 $\mathcal{D} = \{D^{(i)}\}_{i=1}^{M}$ 与偏差编码器 $\mathcal{B} = \{B^{(i)}\}_{i=1}^{M}$ 分别提取目标特征 $\mathbf{z}_d^{(i)}$ 与偏差特征 $\mathbf{z}_b^{(i)}$，其中 $M=4$，对应 ResNet-18 的四个残差块。
- **偏差特定专家 $E^{(i)}$**：每个专家含去偏分类器 $C_d^{(i)}$ 与偏差分类器 $C_b^{(i)}$，输入为拼接特征 $\mathbf{z}^{(i)} = [\mathbf{z}_d^{(i)}; \mathbf{z}_b^{(i)}]$。
- **初始训练损失**：
  - 偏差检测损失：$\mathcal{L}_{\text{bias}} = \sum_{i=1}^{M} \text{GCE}(\hat{\mathbf{y}}_b^{(i)}, \mathbf{y})$，GCE 损失利于捕捉易学特征。
  - 去偏分类损失：$\mathcal{L}_{\text{debias}} = \sum_{i=1}^{M} w^{(i)} \times \text{CE}(\hat{\mathbf{y}}_d^{(i)}, \mathbf{y})$，其中 $w^{(i)} = \frac{\text{CE}(\hat{\mathbf{y}}_b^{(i)}, \mathbf{y})}{\text{CE}(\hat{\mathbf{y}}_d^{(i)}, \mathbf{y}) + \text{CE}(\hat{\mathbf{y}}_b^{(i)}, \mathbf{y})}$，赋予难分类样本更高权重。
  - 多样性损失：$\mathcal{L}_{\text{div}} = \sum_{i=2}^{M} \exp(-\text{KL}(\hat{\mathbf{y}}_b^{(i)}, \hat{\mathbf{y}}_b^{(i-1)}))$，鼓励不同 expert 捕获不同偏差。
  - 总分类损失：$\mathcal{L}_{\text{cls}} = \alpha \times \mathcal{L}_{\text{debias}} + \mathcal{L}_{\text{bias}}$。
- **反事实训练**：在 mini-batch 内随机配对目标特征与偏差特征，构造正样本（相同目标、不同偏差）与负样本（相同偏差、不同目标），通过对比损失 $\mathcal{L}_{\text{con}}$ 强制目标特征独立于偏差特征。
- **门控模块**：$\hat{\mathbf{y}}_d = \sum_{i=1}^{M} p^{(i)} \times \hat{\mathbf{y}}_d^{(i)}$，其中 $p^{(i)}$ 为门控 softmax 输出，最终损失 $\mathcal{L} = \mathcal{L}_{\text{cls}} + \mathcal{L}_{\text{gate}} + \mathcal{L}_{\text{div}} + \beta \times \mathcal{L}_{\text{con}}$。

## 实验与结果
- **数据集**：Biased MNIST（7 类偏差）、BAR（1 类偏差，训练集完全有偏）、Modified IMDB（2 类偏差，自建）、MIMIC-CXR + NIH（1 类数据源偏差，自建）、CelebA（真实人脸数据集，4 类偏差属性）。
- **对比基线**：ResNet-18、LfF、DFA、OccamNet、DebiAN、UBNet。
- **主要结果**：
  - **Biased MNIST（7 偏差，ratio=0.95）**：PnD 达 **70.43%**，超越 OccamNet（66.85%）第二，提升 **+3.58pp**。
  - **MIMIC-CXR + NIH（1 偏差）**：PnD 达 **60.73%**，超越 DebiAN（60.00%）。
  - **Modified IMDB（2 偏差）**：PnD 达 **67.87%**（ratio=0.99），为最好。
  - **BAR（1 偏差，全有偏）**：PnD 69.83%，仅次于 DebiAN 的 69.88%（论文解释因 PnD 要求无偏训练数据）。
  - **CelebA 最坏组**：PnD 平均准确率 **60.00%**，优于 PAD（60.00% 持平）与 DebiAN（58.16%）。
- **鲁棒性**：偏差数量从 1 增至 7，PnD 始终最优，且在新增偏差后性能下降最小，趋于稳定。
- **消融**：各损失项均有贡献；两阶段训练优于单阶段；多 expert 在多重偏差场景下显著优于仅最后一块插入 MoE 的变体（Biased MNIST 上 70.43% vs 52.06%）。

## 相关工作脉络
- **LfF / DFA**：基于偏差分类器置信度加权去偏的无监督方法，但假设单类偏差且依赖偏差易学习的先验；本文放松该假设并支持多重未知偏差。
- **OccamNet**：通过偏好简单假设缓解偏差，但同样对偏差类型有隐式假设；本文完全无需偏差类型先验。
- **DebiAN**：利用 Equal Opportunity Violation loss 发现最显著偏差并重采样，理论上仅能消除一个主导偏差；本文通过多专家并行处理多重偏差。
- **UBNet**：从浅层特征提取无偏目标特征，忽略数据中可能存在多个未知偏差；本文显式建模多层偏差并分层去偏。
- **Mixture of Experts（MoE）**：传统 MoE 用于领域划分或加速推理；本文将其创新性地应用于偏差去偏，按网络深度划分偏差子空间。

## 局限性与未来方向
- **参数量增加**：多 expert 结构带来额外参数，未来可通过专家稀疏化（如 Switch Transformers 思路）缓解。
- **需要无偏训练数据**：在完全有偏数据集（如 BAR）上表现略逊于专用方法，这是方法设计的固有trade-off。
- **无归纳偏置难以消除所有多重未知偏差**：论文引用 Whac-a-Mole 困境，指出无额外约束时同时处理多个未知偏差存在理论困难。
- **当前专家数量为固定值 4**：未探索最优 expert 数量自适应确定策略。

## 研究启发与可借鉴点
- **分治策略用于复杂偏差场景**：将难以全局解决的"多重未知偏差"问题分解为多个子空间的独立去偏，是解决高维混杂问题的有效思路，可迁移至 NLP 或时序数据去偏。
- **反事实对比训练解耦特征**：通过 minibatch 内特征重配对构造反事实样本并施加对比损失，是一种无需标签的解耦手段，值得在公平学习、鲁棒表示学习中借鉴。
- **多样性正则的 KL 散度设计**：利用相邻 expert 输出间的 KL 散度强制多样性，比传统 dropout 正则更直接地约束表征空间，可推广至其他 MoE 结构。
- **门控加权聚合优于简单平均**：自适应门控显著提升多 expert 融合效果，且消融显示比无权重平均高出约 2.6pp，提示在多任务/多视角学习中需谨慎设计融合策略。

## 关键术语表
**Agnostic Biases**：指偏差类型与数量均未知的场景，比"unknown bias"更进一步，强调现实数据的复杂混杂性。
**Biases-Specific Expert**：针对特定层次/类型偏差设计的独立处理模块，内含去偏分类器与偏差分类器，嵌入网络不同深度。
**Gating Module**：基于 softmax 的门控网络，根据各 expert 输出自适应计算融合权重，生成最终一致预测。
**Counterfactual Training**：通过特征重配对构造反事实样本，利用对比损失强制目标特征与偏差特征解耦的训练阶段。
**Diversity Penalty**：以 KL 散度约束相邻 expert 偏差检测结果的一致性，确保各 expert 捕获不同层次的偏差特征。
**GCE Loss (Generalized Cross Entropy)**：对噪声标签更具鲁棒性的交叉熵变体，本文用于偏差检测损失以更好地捕捉易学偏差特征。
**Debiased Encoder / Bias Encoder**：分别从输入图像中提取目标相关特征与偏差相关特征的双分支编码器结构。

## 可复现要素
- **代码**：已开源，GitHub 地址 https://github.com/Jiaxuan-Li/PnD
- **数据集**：Biased MNIST（公开）、BAR（公开）、CelebA（公开）；Modified IMDB 与 MIMIC-CXR + NIH 为作者自建数据集，论文未提供公开链接
- **模型架构**：ResNet-18 backbone，每个 block 后接偏差特定专家（两个线性分类器），门控模块为单线性层
- **输入尺寸**：除 BAR 外统一 Resize 至 160×160；BAR 随机裁剪至 224×224 并水平翻转
- **超参数**：$\alpha$（$\mathcal{L}_{\text{debias}}$ 与 $\mathcal{L}_{\text{bias}}$ 平衡系数）、$\beta$（对比损失权重）、bias ratio 默认 0.95；论文未完整列出具体数值
- **硬件**：NVIDIA RTX A4000 GPU
- **重复次数**：所有实验报告三次运行的均值±标准差
