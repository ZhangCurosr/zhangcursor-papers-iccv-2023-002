---
title: "Towards-Attack-tolerant-Federated-Learning-via-Critical-Para"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Han_Towards_Attack-tolerant_Federated_Learning_via_Critical_Parameter_Analysis_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 21:20:33"
---

# 论文速读：Towards-Attack-tolerant-Federated-Learning-via-Critical-Parameter-Analysis

## 一句话总结
本文提出了 FedCPA，一种面向非独立同分布（non-IID）数据的联邦学习抗投毒聚合防御方法。该方法通过量化并对比各客户端模型中“最关键”与“最不重要”参数的排名稳定性，构建新的模型相似度度量，在聚合阶段自适应地降低恶意更新权重，显著提升了对定向与未定向攻击的防御能力。

## 研究问题与动机
- 现有联邦学习防御策略（如 Krum、FoolsGold、RFA 等）多依赖模型更新的欧氏距离或范数进行异常检测，在数据高度 non-IID 时，良性客户端的更新本身呈现高方差，导致恶意与良性更新在向量空间中难以区分，防御机制失效。
- 模型参数在优化过程中贡献不均，但已有防御方法普遍将整个更新向量视为同质对象，未充分挖掘参数重要性变化模式在攻击检测中的判别潜力。
- 传统中心化聚合默认所有参与者均良性，缺乏在异构数据环境下既能过滤恶意更新、又能避免对低相似度良性节点过度惩罚的鲁棒聚合机制。
- 核心目标：设计一种不依赖纯向量空间距离、而是基于参数重要性分布规律的新防御范式，使联邦学习系统在非 IID 设置下仍能有效抵御投毒攻击。

## 核心贡献（创新点）
- **实证揭示良性模型关键参数排名的一致性规律**：在多个数据集与非IID程度下，良性本地模型的 top-k 和 bottom-k 重要性参数集合表现出高度重叠与稳定排名，而中毒模型的排名扰动显著更大；与已有工作假设恶意更新在欧氏空间中远离良性簇不同，本文首次将参数重要性分布模式作为攻击判别依据。
- **提出基于临界参数集的模型相似度度量**：融合 top/bottom-k 参数集的 Jaccard 重合度与共轭集合内的 Spearman 秩相关系数；与 Multi-Krum、Norm Bound 等基于范数或两两距离的方法本质不同，该度量对非IID引起的良性更新多样性具有天然鲁棒性，且不依赖全局分布假设。
- **设计双源正常性评分与攻击容忍加权聚合框架**：构造融合上一轮全局模型相似度与同轮客户端互相似度的 Normality Score，并通过逆 Sigmoid 映射生成聚合权重；与 ResidualBase 等依赖单一中值估计的方法不同，该设计在提升攻击过滤精度的同时有效抵抗 Sybil 协同攻击与过度惩罚。
- **全面的实验验证与定量分析**：在 CIFAR-10、SVHN、TinyImageNet 上系统评估定向与非定向攻击，FedCPA 的定向攻击成功率最高降低约 4 倍；与仅针对特定攻击或低异构场景优化的基线相比，本文提供了覆盖多攻击类型、多污染率、多异构强度的统一防御基准。

## 方法详解
- **参数重要性计算**：对客户端 $i$ 在第 $t$ 轮的本地更新 $\Delta_i^t$ 与更新后参数 $\theta_i^t$，第 $n$ 个参数的关键度定义为 $p_i[n] = |\Delta_i[n] \cdot \theta_i[n]|$，同时捕获优化信号的强度与参数对最终预测的贡献度。
- **关键参数集提取**：对每个客户端的 $p_i$ 降序/升序排列，分别提取 top-k 和 bottom-k 的参数索引集合 $\Theta_i^{\text{top}}$ 与 $\Theta_i^{\
