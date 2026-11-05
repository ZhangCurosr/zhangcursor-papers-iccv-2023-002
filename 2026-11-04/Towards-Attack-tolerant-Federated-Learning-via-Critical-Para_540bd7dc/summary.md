---
title: "Towards-Attack-tolerant-Federated-Learning-via-Critical-Para"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Han_Towards_Attack-tolerant_Federated_Learning_via_Critical_Parameter_Analysis_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 16:40:24"
---

# 论文速读：Towards-Attack-tolerant-Federated-Learning-via-Critical-Para

## 一句话总结
本文提出FedCPA，一种基于关键参数分析的抗投毒联邦学习聚合策略。通过对比各客户端在top-k与bottom-k关键参数上的相似度分布，有效区分良性与恶意更新，在non-IID数据与多种投毒攻击下均显著优于现有防御基线。

## 研究问题与动机
- **核心问题**：联邦学习中恶意客户端提交的虚假更新会导致全局模型性能下降（无差别攻击）或被植入后门（定向攻击）。
- **现有方法不足**：主流防御（如Krum、Median、Trimmed Mean等）依赖欧氏距离或范数度量异常度，在non-IID数据下良性更新本身高度分散，导致恶意与良性更新在向量空间中难以区分，防御失效。
- **关键观察**：非IID场景下，良性本地模型的top-k与bottom-k关键参数（重要性 = |权重变化量 × 权重值|）具有稳定的排名分布，而投毒更新会同时剧烈扰动这两端参数的排序。
- **防御目标**：设计一种不依赖全局数据分布假设、仅基于参数重要性模式即可识别并抑制恶意更新的聚合机制。

## 核心贡献（创新点）
1. **揭示良性模型关键参数排名的稳定性规律**：发现非IID训练下良性本地模型的top/bottom关键参数排名变化幅度显著小于中等重要性参数，而恶意更新会大幅扰动两端排名。与已有工作的本质区别在于：跳出向量范数/距离的异常检测范式，转而从“参数对优化的贡献模式”维度刻画良性共性。
2. **提出基于关键参数集的模型相似度度量**：结合Jaccard相似性与Spearman秩相关系数，量化两个本地模型在top-k与bottom-k参数上的一致性。与全参对齐方法相比，该度量聚焦最敏感与最无关参数子集，对非IID噪声具有天然鲁棒性。
3. **构建带历史锚点的归一度计算与自适应聚合框架**：引入上一轮全局模型 $\phi^{t-1}$ 作为参照锚点，有效缓解Sybil攻击导致的同流合污问题，并通过逆Sigmoid映射将归一度转化为攻击容错权重。与静态截断或固定阈值方法相比，权重分配平滑且能动态压制恶意更新。
4. **系统化验证与非IID多场景泛化**：在CIFAR-10、SVHN、TinyImageNet上覆盖定向后门、标签翻转与高斯噪声攻击，证明方法在不同Dirichlet异构程度、不同恶意比例下均保持领先。

## 方法详解
- **关键参数重要性计算（Eq. 4）**：对客户端 $i$ 的本地更新 $\Delta_i^t$ 与本地参数 $\theta_i^t$，定义重要性 $p_i[n] = |\Delta_i^t[n] \cdot \theta_i^t[n]|$，同时捕捉优化强度与参数对预测的贡献度。
- **关键参数集提取**：按重要性排序后截取top-k与bottom-k索引集合 $\Theta_i^{\text{top}}$ 与 $\Theta_i^{\text{bottom}}$。
- **模型相似度度量（Eq. 5）**：$\text{sim}(\theta_i, \theta_j) = J(\Theta_i^{\text{top}}, \Theta_j^{\text{top}}) + J(\Theta_i^{\text{bottom}}, \Theta_j^{\text{bottom}}) + \rho(\cdots) + \rho(\cdots)$，其中 $J$ 为Jaccard相似度，$\rho$ 为交集上的Spearman秩相关（归一至[0,1]）。
- **归一度计算（Eq. 7）**：$\mathcal{N}(\theta_i^t) = \text{sim}(\theta_i^t, \phi^{t-1}) + \frac{1}{|\mathcal{C}^t|}\sum_{j} \text{sim}(\theta_i^t, \theta_j^t)$，融合历史全局模型锚点与当前轮次同行比对以提升抗Sybil能力。
- **攻击容错聚合（Eq. 8-9）**：对 $\mathcal{N}$ 做Min-Max归一化后，通过 $\lambda_i^t = \text{Clip}_{0\sim1}(\ln\frac{\tilde{\mathcal{N}}}{1-\tilde{\mathcal{N}}} + 0.5)$ 放大差异并映射为聚合权重；全局更新为 $\phi^{t+1} \gets \phi^t + \frac{1}{\sum \mathbf{1}(\lambda>0)} \sum \lambda_i^t \cdot \Delta_i^t$，权重趋零的恶意更新被自动过滤。

## 实验与结果
- **数据集**：CIFAR-10、SVHN、TinyImageNet（图像分类），默认 $N=20$ 客户端、Dirichlet $\beta=0.5$ 模拟非IID。
- **评估基线**：No Defense, Median, Trimmed Mean, Multi Krum, FoolsGold, Norm Bound, RFA, ResidualBase。
- **攻击场景**：定向后门攻击（污染率 $\gamma_p=0.5, 0.8$）、标签翻转攻击（$\gamma_p=0.8, 1.0$）、高斯噪声攻击（$\gamma_p=0.8$）。
- **主要结果**：FedCPA在所有数据集与攻击类型下均取得最优或接近最优ACC，并显著压低ASR。定向攻击（$\gamma_p=0.5$, CIFAR-10）中ACC达77.8%，ASR降至21.9%；定向攻击（$\gamma_p=0.8$, TinyImageNet）中ASR仅为4.8%，相比无防御的74.6%降幅约15倍；整体定向攻击ASR较最强基线降低2~4倍。
- **消融与鲁棒性**：移除bottom-k组件性能下降最显著（Table 5），印证“投毒通过唤醒低重要性参数导致过拟合”的假设；超参 $k$ 在1%~2%区间表现最优；在不同恶意比例、客户端总数与非IID程度变化下持续领先（Figure 2）。

## 相关工作脉络
1. **Byzantine-resilient aggregation (Median, Trimmed Mean)**：依赖分量级鲁棒统计量，假设攻击更新数值偏离主分布；FedCPA转而关注参数重要性结构的跨客户端共性，突破对数值幅值的依赖。
2. **距离/范数型防御（Krum, Multi
