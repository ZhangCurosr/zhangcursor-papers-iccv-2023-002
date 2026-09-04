---
title: "Unsupervised-Manifold-Linearizing-and-Clustering"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Ding_Unsupervised_Manifold_Linearizing_and_Clustering_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 23:37:08"
---

# 论文速读：Unsupervised-Manifold-Linearizing-and-Clustering

## 一句话总结
本文提出流形线性化与聚类（MLC）方法，在无标签条件下联合优化特征表示与双随机聚类隶属度矩阵，利用最大编码率减少（MCR²）目标将位于低维流形上的数据高效映射至相互正交的线性子空间族，实现高精度聚类。

## 研究问题与动机
- **线性假设失效**：经典子空间聚类依赖数据位于线性/仿射子空间并集上的几何假设，但自然图像、MNIST等真实高维数据实际分布于弯曲流形上，直接套用传统方法准确率骤降。
- **现有深度变换方法病态**：近年诸多工作尝试用深度网络学习流形到子空间的非线性映射，但已被证明许多 formulation 是病态的，会收敛至平凡表示，且主要收益来自启发式后处理而非方法本身。
- **监督损失的多样性缺失**：交叉熵等监督目标会导致“神经坍缩”现象，使同类特征坍缩为单点，丧失簇内多样性，不利于下游生成、去噪与语义解释任务。
- **无监督正交子空间学习的缺失**：MCR²虽能学到“簇内均匀展开、簇间相互正交”的理想表示，但依赖真实标签；如何在无监督场景下同时完成流形线性化与聚类，仍是开放问题。

## 核心贡献（创新点）
- **MLC无监督目标 formulations**：将MCR²推广至无监督设定，联合优化特征表示与新型双随机成对隶属度矩阵，用信息论目标替代硬标签与交叉熵。与MCR²本质区别在于无需预先知道样本归属，且通过松弛的成对矩阵避免组合优化。
- **参数复制的一次性初始化策略**：提出将自监督特征头权重直接拷贝至亲和度头的初始化方案，实现双随机隶属度的零额外训练初始化。与现有深度聚类方法（如SCAN需多次随机初始化cluster head）相比，更稳定且高效。
- **正交子空间族表示与聚类性能双重提升**：在CIFAR-10/20/100与TinyImageNet-200上验证，MLC不仅聚类准确率超越SOTA深度聚类与子空间聚类方法，且学习到的表征严格呈现数值秩分离结构，证实了流形线性化的有效性。

## 方法详解
- **核心目标（MLC, Eq.4）**：$\max_\theta R(Z;\epsilon) - R_c(Z,\Gamma;\epsilon)$，其中 $Z=f_\theta(X)$ 为单位球面 $\mathbb{S}^{d-1}$ 上的特征，$\Gamma \in \Omega$ 为双随机隶属度矩阵。$R$ 度量全局特征多样性，$R_c$ 度量各簇内特征集中度，最大化二者差值迫使簇间正交、簇内紧凑。
- **双随机隶属度建模**：摒弃 $n\times k$ 硬分配矩阵，采用 $n\times n$ 成对相似性矩阵 $\Gamma$。为避免仅靠双随机约束会退化为排列矩阵（每个样本自成一对），引入负熵正则化将 $\Gamma$ 推向均匀矩阵，并通过可微的Sinkhorn投影 $P_{\Omega,\eta}$ 强制满足双随机约束。
- **网络参数化**：共享ResNet-18骨干网络，附加feature head输出特征 $Z$；cluster head输出 $C$，计算亲和度 $\Gamma = P_{\Omega,\eta}(C^\top C)$。该参数化使 $\Gamma$ 天然满足约束，且支持mini-batch前向/反向传播。
- **两阶段初始化**：(1) 使用自监督Total Coding Rate (TCR) 目标初始化feature head，要求同一样本不同增广的特征尽量接近、不同样本特征尽量不相关；(2) 直接将feature head参数拷贝至cluster head初始化 $\Gamma$，无需额外训练。
- **训练流程**：每批次采样后施加 $A=2$ 种数据增强，分别前向得到 $Z^{(a)}, \Gamma^{(a)}$，对特征取平均后投影至球面，计算MLC损失并更新 $\theta$；训练结束后对 $\Gamma$ 执行谱聚类获得最终标签 $\hat{y}$。

## 实验与结果
- **数据集**：CIFAR-10、CIFAR-20、CIFAR-100、TinyImageNet-200，以及人工构建的不平衡版本 Imb-CIFAR-10/100。
- **基线**：子空间聚类 (EnSC, SSC-OMP)；深度聚类 (SCAN, GCC, NNM, IMC, NMCE, SPICE
