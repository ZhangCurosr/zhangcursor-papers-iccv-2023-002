---
title: "ShiftNAS-Improving-One-shot-NAS-via-Probability-Shift"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Zhang_ShiftNAS_Improving_One-shot_NAS_via_Probability_Shift_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 00:55:41"
field: "神经架构搜索"
keywords: ["Neural Architecture Search", "One-shot NAS", "Weight Entanglement", "Probability Shift", "Architecture Generator", "Vision Transformer", "Convolutional Neural Network"]
innovations: ["提出基于训练充分性的概率偏移策略动态调整采样分布", "设计可微LSTM架构生成器直接输出满足任意计算约束的子网", "在CNN与ViT搜索空间上实现无需重训的SOTA性能"]
benchmarks: ["ImageNet Classification"]
---

# 论文速读：ShiftNAS:Improving One-shot NAS via Probability Shift

## 一句话总结
本文针对单样本神经架构搜索（One-shot NAS）中均匀采样导致的子网训练不充分、性能差距问题，提出一种可学习的概率偏移（Probability Shift）策略与基于LSTM的架构生成器（AG），能够根据子网计算复杂度动态调整采样概率，并直接生成满足任意指定资源约束的优化子网，无需额外搜索或重训成本。

## 研究问题与动机
1. **子网性能继承劣势**：单样本NAS中，通过权重共享获得的子网性能往往不如重新训练，存在明显的性能差距（performance gap）。
2. **均匀采样导致训练资源分配偏差**：现有方法假设所有候选架构同等重要，采用均匀采样，但实际子网所需训练资源不同（例如1.0 GFLOPs子网可能30000步收敛，2.0 GFLOPs需50000步），均匀采样使训练资源集中在中间复杂度子网，导致边缘复杂度子网欠训练或过训练。
3. **采样效率低下**：均匀采样下子网计算复杂度近似正态分布，小/大复杂度子网采样概率极低，难以高效获取满足特定约束的子网。
4. **缺乏端到端的子网生成机制**：如何在给定资源约束下快速、准确地生成对应子网，并保持权重纠缠（weight entanglement）下的可微训练，是现有方法未充分解决的问题。

## 核心贡献（创新点）
1. **可学习的概率偏移（Probability Shift）策略**：提出通过评估子网训练充分性（梯度范数）动态调整计算资源采样概率，将训练资源重新分配给欠训练的子网；与AttentiveNAS/GreedyNAS等仅关注Pareto前沿或筛除弱子网的方法不同，本方法直接从训练充分性角度优化资源分布，适用于任意复杂度区域。
2. **基于LSTM的架构生成器（AG）**：设计可微训练的LSTM架构生成器，结合Gumbel Softmax与矩阵映射技巧，实现任意目标计算约束下的子网生成，并与Supernet联合端到端训练；与FocusFormer基于资源分布采样、FairNAS基于严格公平性更新不同，本方法直接生成满足精确约束的子网架构，避免大量无效采样。
3. **模型无关（Model‑Agnostic）的One‑shot NAS框架**：在CNN和ViT搜索空间上均验证了方法的有效性，ShiftFormer系列在多种FLOPs约束下取得SOTA或具有竞争力的结果，且无需重训/微调即可直接部署；与AutoFormer、BigNAS等特定于某类模型的工作相比，本方法展现出更强的通用性。
4. **权重纠缠搜索空间下的可微优化**：针对权重纠缠（weight entanglement）特性，引入矩阵映射技术将one‑hot策略转换为可微掩码，使AG与Supernet可联合反向传播；与DARTS等连续化搜索方法不同，本方法严格保持权重共享与低索引优先的纠缠约束，更适合实际部署场景。

## 方法详解
**整体框架**：ShiftNAS包含两个核心组件——**概率偏移（Probability Shift）**与**架构生成器（AG）**，两者可端到端联合训练。

1. **概率偏移（Probability Shift）**
   - **计算资源离散化**：将FLOPs范围划分为K个区间（如每0.1 GFLOPs一档），初始化为均匀分布的概率向量 $B \in \mathbb{R}^{1 \times K}$。
   - **训练充分性度量**：利用验证集损失梯度范数 $\nabla_w = \nabla_w L_{val}(w, \alpha \mid b)$ 判断子网是否收敛；梯度小（趋于零）表示训练充分，梯度大表示欠训练。
   - **贪婪采样优化**：每次更新时，通过Gumbel Softmax从分布 $B$ 中采样一个计算资源 $b$，并最小化该资源下子网的梯度范数期望：
     $$\arg\min_B E_{b \sim B}\left[\nabla_w L_{val}(w, \alpha \mid b)\right]$$
   - **梯度近似**：为避免二阶导数计算，采用有限差分近似（每两步训练）估算 $B$ 的梯度：
     $$\nabla_{B_t} = \left(\nabla_{B_t} L_{val}(w_t, \alpha \mid G(B_t)) - \nabla_{B_t} L_{val}(w_{t-1}, \alpha \mid G(B_t))\right) \nabla_{B_t} G(B_t)$$
   - **更新频率**：每100次迭代更新一次 $B$，训练初期（前50 epoch）仅优化Supernet权重与AG。

2. **架构生成器（AG）与矩阵映射**
   - **AG结构**：LSTM网络（隐藏维度64），逐序列生成每个操作（operation）的one‑hot策略向量 $p \in \{0,1\}^n$（$n$ 为候选操作数）。
   - **矩阵映射技巧**：为适配权重纠缠搜索空间，定义掩码矩阵 $M = [m_0^T, m_1^T, \dots, m_n^T]$，其中 $m_j^T$ 为第 $j$ 个候选操作的掩码向量（如ViT中head数量掩码为 $[1,0,0], [1,1,0], [1,1,1]$）。映射后策略 $p' = M \cdot p^T$ 可与Supernet前向传播兼容，梯度可通过链式法则回传到AG。
   - **联合损失函数**：
     $$L = L_{task} + \lambda L_{RC}$$
     $$L_{RC} = \left(\sum_{i=1}^D \sum_{j=1}^n b_j^i p_j'^{i} - C\right)^2$$
     其中 $b_j^i$ 为第 $i$ 个操作中第 $j$ 个候选的操作计算量，$C$ 为从分布 $B$ 采样的目标资源约束，$\lambda$ 为平衡系数。

3. **训练流程**
   - **阶段一（0–50 epoch）**：固定 $B$，仅用均匀采样训练Supernet与AG，使AG初步具备生成能力。
   - **阶段二（50 epoch后）**：每100次迭代更新 $B$，同时用AG生成满足采样资源 $C$ 的子网进行Supernet训练，AG与Supernet联合优化。
   - **搜索阶段**：训练完成后，AG直接生成概率最高（即预期性能最优）的子网，权重从Supernet继承，无需微调或重训练。

## 实验与结果
- **数据集**：ImageNet（训练集1.2M，验证集50K）。
- **评估基线**：
  - ViT模型：DeiT, AutoFormer, FocusFormer, ViT-S/B, BoTNet等。
  - CNN模型：MobileNetV2/V3, EfficientNet, BigNAS, Once‑For‑All, GreedyNAS等。
- **关键结果**（Table 1）：
  - **ShiftFormer‑T**：1.3 GFLOPs, 5.8M参数，Top‑1 Acc 76.0%（比AutoFormer‑T高1.3%，比FocusFormer‑T高0.9%）。
  - **ShiftFormer‑S**：5.0 GFLOPs, 23.6M参数，Top‑1 Acc 82.2%（超过AutoFormer‑S的81.4%与FocusFormer‑S的81.6%）。
  - **ShiftFormer‑B**：11.0 GFLOPs, 52.8M参数，Top‑1 Acc 82.8%（超过AutoFormer‑B的81.4%与FocusFormer‑B的81.9%）。
  - **ShiftCNN‑S**：0.24 GFLOPs, 4.5M参数，Top‑1 Acc 77.2%（比BigNAS‑S的76.5%高0.7%）。
  - **ShiftCNN‑B**：0.42 GFLOPs, 5.6M参数，Top‑1 Acc 79.6%（比BigNAS‑M的78.9%高0.7%）。
- **无重训优势**（Table 2）：继承权重性能与微调/重训几乎一致（差异在±0.2%内），证明无需额外计算开销即可部署。
- **消融实验**（Figure 4）：加入概率偏移后，FLOPs‑Accuracy曲线覆盖范围显著扩大，各复杂度区域性能均提升。
- **采样概率分布**（Figure 5）：训练后期分布向高FLOPs偏移（如ViT‑tiny中1.6–1.9 GFLOPs概率更高），符合大子网需要更多训练资源的直觉。
- **排名相关性**（Figure 6）：继承性能与微调后性能在高性能子网（Top‑1 > 65%）中高度相关，验证Supernet可作为可靠性能估计器。

## 相关工作脉络
1. **Uniform Sampling（One‑Shot NAS基础）**：如SP‑NAS [7]、FBNet [30] 等假设所有架构同等重要，采用均匀采样。ShiftNAS指出该方法导致训练资源向中间复杂度集中，提出概率偏移作为根本性改进。
2. **Pareto‑Aware / Greedy Sampling**：AttentiveNAS [29] 通过筛选极端性能子网引导采样，GreedyNAS [31] 仅从潜在优秀子网中采样。本文方法同样关注优质子网，但不同之处在于直接从训练充分性（梯度范数）角度动态调整分布，而非依赖单次迭代的性能筛选。
3. **Resource‑Focused Sampling**：FocusFormer [19] 提出基于资源分布的架构采样器，FairNAS [5] 强调参数更新次数公平性。ShiftNAS与二者定位不同：FocusFormer仍需在训练后搜索子网，而本文通过AG直接生成满足精确约束的子网；FairNAS关注更新公平，而本文关注训练充分性。
4. **Weight Entanglement NAS**：AutoFormer [4]、BigNAS [33] 等采用权重纠缠搜索空间，避免三明治规则带来的额外开销。ShiftNAS明确针对此类空间设计矩阵映射技巧，实现AG与Supernet的端到端联合训练，而传统DARTS类方法不适用于权重纠缠设置。
5. **Differentiable NAS with Architecture Generator**：DSNAS [11] 等使用预测器直接生成架构。本文的AG基于LSTM且与概率偏移联合优化，不仅生成架构还同时学习最优采样分布，形成闭环。

## 局限性与未来方向
1. **仅适用于权重纠缠搜索空间**：作者明确指出ShiftNAS无法用于DARTS类连续化搜索空间（如每个操作独立权重），限制了在其他NAS范式中的直接应用。
2. **计算资源离散化粒度影响精度**：将FLOPs划分为固定区间（每0.1 GFLOPs一档）可能导致边界附近的子网生成不够精确，未来可探索连续资源建模。
3. **梯度近似误差**：使用有限差分近似二阶梯度会引入一定噪声，可能影响概率偏移的稳定性；未来可研究更精确的梯度估计方法。
4. **未探索跨领域泛化**：目前仅在图像分类任务上验证，未测试于目标检测、语义分割等其他视觉任务，也未涉及NLP或音频领域。
5. **超参数敏感性**：如更新频率（每100步）、LSTM隐藏维度、$\lambda$ 等超参数需手动调优，缺乏自适应机制。

## 研究启发与可借鉴点
1. **训练充分性作为采样依据**：用梯度范数衡量子网训练状态，进而动态调整采样概率，这一思路可迁移至其他多任务学习、课程学习场景，实现训练资源的智能分配。
2. **矩阵映射技巧适配权重纠缠**：将one‑hot策略转换为可微掩码以保持权重共享，该技巧可直接用于其他需要精确控制操作选择的NAS框架，避免连续性松弛带来的性能损失。
3. **端到端联合训练AG与Supernet**：架构生成器与主干网络共同优化，使搜索过程完全自动化；可借鉴至模型压缩、超网络设计等领域，实现从约束到架构的端到端映射。
4. **无需重训的部署友好性**：证明高质量的Supernet可使继承权重直接达到与重训相当的性能，为边缘设备上的实时NAS提供了实用路径；可探索与其他轻量化技术（如量化、剪枝）的结合。
5. **计算资源分布可视化分析**：通过观察采样概率随训练的变化，可直观了解不同复杂度子网的训练难度，为NAS研究提供新的诊断工具。

## 关键术语表
**One‑shot NAS**：通过单次训练超级网络（Supernet），后续直接从中抽取子网并继承权重，大幅降低神经架构搜索的计算成本。
**Weight Entanglement**：权重共享的一种形式，要求低索引的操作（如较少channel/head）必须被保留，从而避免独立权重带来的存储与优化困难。
**Probability Shift**：动态调整不同计算复杂度子网的采样概率，使训练资源向欠训练子网倾斜的策略。
**Architecture Generator (AG)**：基于LSTM的生成网络，输入目标计算约束，输出满足该约束的子网架构序列。
**Gumbel Softmax**：对离散采样进行可微近似的技巧，使one‑hot策略可通过梯度下降优化。
**Matrix Mapping**：将one‑hot策略向量转换为可微掩码矩阵的变换，用于适配权重纠缠搜索空间的前向传播与反向传播。
**Irwin‑Hall Distribution**：多个独立均匀分布随机变量之和所服从的分布；当操作数增多时，近似为正态分布，解释了均匀采样下子网复杂度分布的集中现象。
**Finite Difference Approximation**：用两次相邻迭代的损失差值近似梯度，避免计算高昂的二阶导数矩阵‑向量乘积。

## 可复现要素
- **数据集**：ImageNet（训练集1.2M图像，验证集50K图像）。
- **代码开源**：论文声明“Source codes are available at GitHub”，但具体仓库链接未在正文给出，需查阅论文附属材料或作者主页。
- **权重**：未提及预训练Supernet或搜索到子网的公开权重。
- **关键超参数**：
  - 计算资源划分步长：0.1 GFLOPs。
  - 概率向量更新频率：每100次迭代更新一次。
  - AG隐藏维度：64。
  - 前50 epoch仅训练Supernet与AG，之后联合更新概率分布。
  - 优化器：Adam，学习率1e‑3。
  - 损失平衡系数 $\lambda$：论文未明确给出，附录可能有详细说明。
  - 训练设备：8× NVIDIA Tesla A100 GPU。
