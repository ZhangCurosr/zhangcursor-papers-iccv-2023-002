---
title: "ShiftNAS-Improving-One-shot-NAS-via-Probability-Shift"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Zhang_ShiftNAS_Improving_One-shot_NAS_via_Probability_Shift_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 23:35:33"
field: "神经架构搜索"
keywords: ["One-shot NAS", "Neural Architecture Search", "Probability Shift", "Weight Entanglement", "Vision Transformer", "CNN"]
innovations: ["提出可学习的概率偏移策略动态调整采样分布以解决均匀采样导致的训练资源分配不均问题", "设计LSTM架构生成器结合矩阵映射技术实现端到端可微分子网生成"]
benchmarks: ["ImageNet"]
---

# 论文速读：ShiftNAS-Improving-One-shot-NAS-via-Probability-Shift

## 一句话总结
本文提出ShiftNAS，通过可学习的概率偏移策略解决One-shot NAS中均匀采样导致的训练资源分配不均问题，结合LSTM架构生成器实现端到端训练，无需额外搜索或重训练即可直接获取任意计算复杂度下的最优子网。

## 研究问题与动机
- **性能差距问题**：One-shot NAS中子网性能往往劣于从头训练，根源在于权重共享带来的性能损失。
- **均匀采样偏差**：现有方法假设所有候选架构同等重要，采用均匀采样，但不同复杂度的子网需要不同的训练资源。
- **训练不充分**：均匀采样导致计算资源分布近似正态分布，中等复杂度子网过度训练，而高/低复杂度子网训练不足。
- **搜索效率瓶颈**：现有方法需要额外的重训练或微调阶段，增加了计算开销。

## 核心贡献（创新点）
- **概率偏移采样策略**：提出可学习的采样概率动态调整机制，根据子网训练充分性重新分配训练资源，与均匀采样的本质区别在于主动识别并补偿训练不足的区域。
- **LSTM架构生成器（AG）**：设计可微分的架构生成器，能够根据目标计算约束准确高效地生成对应子网，解决了传统随机采样低效的问题。
- **矩阵映射技术适配权重纠缠**：针对weight-entangled search space设计矩阵映射，将one-hot策略转换为可微分mask，实现AG与supernet的联合端到端训练。
- **模型无关性验证**：在CNN和ViT多种搜索空间上验证方法有效性，证明ShiftNAS是model-agnostic的通用搜索框架。

## 方法详解
- **概率偏移机制**：
  - 将计算资源离散化为K个区间（步长0.1 GFLOPs），用可学习向量表示采样分布
  - 通过子网验证集梯度$\nabla_w L_{val}$衡量训练充分性，梯度趋近零表示收敛
  - 优化目标：$\arg\min_B E_{b\sim B}[\nabla_w L_{val}(w, \alpha|b)]$
  - 使用Gumbel Softmax进行可微分采样，采用有限差分近似梯度避免昂贵的二阶计算

- **架构生成器（AG）**：
  - 基于LSTM网络，逐操作生成子网架构序列
  - 使用Gumbel Softmax生成one-hot策略向量
  - 联合优化目标：$L = L_{task} + \lambda L_{RC}$
  - 资源约束损失：$L_{RC} = (\sum_{i=1}^{D}\sum_{j=1}^{n} b_j^i p_j'^i - C)^2$

- **矩阵映射技术**：
  - 针对weight-entangled搜索空间，设计掩码矩阵M将one-hot策略转换为可微分mask
  - 例如ViT头数选择[1,2,3]对应掩码$m^0=[1,0,0], m^1=[1,1,0], m^2=[1,1,1]$
  - 确保低索引操作优先保留，梯度可通过链式法则回传至AG

## 实验与结果
- **数据集**：ImageNet（1.2M训练图片，50K验证图片）
- **评估指标**：Top-1 Acc.、Top-5 Acc.、FLOPs、参数量
- **主要结果**：
  - ShiftFormer-T：1.3 GFLOPs下Top-1达76.0%，优于AutoFormer-tiny（74.7%）和FocusFormer-T（75.1%）
  - ShiftFormer-S：5.0 GFLOPs下Top-1达82.2%，优于AutoFormer-small（81.4%）和FocusFormer-small（81.6%）
  - ShiftCNN-S：0.24 GFLOPs下Top-1达77.2%，优于BigNAS-S（76.5%）
  - ShiftCNN-B：0.42 GFLOPs下Top-1达79.6%，优于BigNAS-M（78.9%）
  - ShiftNAS无需重训练或微调，其他方法如BigNAS需要2×以上GPU时间

## 相关工作脉络
- **Uniform NAS（SNAS等）**：采用均匀采样，假设所有架构同等重要，本文指出其导致计算资源分布呈正态，边缘区域训练不足。
- **AttentiveNAS**：提出Pareto-aware采样，但仍在固定资源分布下工作，本文进一步动态调整采样概率。
- **Focusformer**：基于资源分布聚焦Pareto前沿架构，本文方法更彻底地通过梯度信号自动学习最优分布。
- **FairNAS**：强调每个选择块参数更新次数公平，与本文关注训练充分性的角度不同。
- **BigNAS/Once-for-All**：采用weight-entanglement避免重训练，本文在其基础上改进supernet训练策略。

## 局限性与未来方向
- **仅适用于weight-entangled搜索空间**：无法直接应用于DARTS-like搜索空间，限制了方法通用性。
- **离散化粒度依赖**：计算资源离散化为K个区间，粒度选择影响精度与效率的权衡。
- **未探索更复杂的搜索空间**：仅在CNN和ViT上验证，对Transformer混合架构或3D视觉任务的效果待验证。

## 研究启发与可借鉴点
- **训练充分性度量思路**：用梯度范数衡量子网训练状态，这一简洁有效的度量可迁移至其他NAS训练策略设计。
- **端到端概率学习框架**：将采样分布作为可学习参数，通过梯度自动优化，为神经架构搜索的资源分配提供新思路。
- **矩阵映射技巧**：针对权重纠缠搜索空间的差异化掩码设计，可推广至其他共享权重的NAS框架。
- **AG与supernet联合训练**：架构生成器与主网络联合优化的端到端范式，避免了传统两阶段搜索的累积误差。

## 关键术语表
- **One-shot NAS**：一次性训练超网络后直接搜索子网的神经架构搜索方法，大幅降低搜索成本。
- **Supernet**：包含所有候选操作的大规模共享权重网络，用于评估子网性能。
- **Weight Entanglement**：权重纠缠策略，低索引操作包含高索引操作的权重，支持子网权重直接继承。
- **Gumbel Softmax**：用于离散变量可微分采样的技术，允许通过梯度优化采样策略。
- **Probability Shift**：本文提出的核心机制，根据子网训练充分性动态调整采样概率分布。
- **Architecture Generator (AG)**：基于LSTM的架构生成器，根据目标计算约束生成对应子网。
- **Irwin-Hall分布**：均匀分布随机变量之和的分布，当操作数增加时趋近正态分布。

## 可复现要素
- **数据集**：ImageNet（公开）
- **代码**：论文声明代码已在GitHub开源（具体链接见原文）
- **关键超参**：
  - 计算资源步长：0.1 GFLOPs
  - LSTM隐藏单元数：64
  - 前50epoch仅优化supernet和AG
  - 采样分布每100迭代更新一次
  - Adam优化器学习率：1e-3
- **硬件**：8× Nvidia Tesla A100 GPUs
