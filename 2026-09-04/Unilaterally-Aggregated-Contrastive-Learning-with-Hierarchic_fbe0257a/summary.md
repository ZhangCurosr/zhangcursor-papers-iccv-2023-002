---
title: "Unilaterally-Aggregated-Contrastive-Learning-with-Hierarchic"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Wang_Unilaterally_Aggregated_Contrastive_Learning_with_Hierarchical_Augmentation_for_Anomaly_Detection_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 23:36:37"
field: "异常检测与表示学习"
keywords: ["Anomaly Detection", "Contrastive Learning", "Self-supervised Learning", "Distribution Alignment", "Virtual Outlier"]
innovations: ["单侧聚合对比损失UniCLR同时优化inlier集中与outlier分散", "软聚合机制SA抑制数据增强引入的噪声", "层次化增强HA从浅到深渐增强度提升集中度"]
benchmarks: ["CIFAR-10", "CIFAR-100", "ImageNet-30", "MVTec-AD"]
---

# 论文速读：Unilaterally Aggregated Contrastive Learning with Hierarchical Augmentation for Anomaly Detection

## 一句话总结
本文提出了一种基于对比学习的异常检测方法 UniCon-HA，通过监督对比损失显式聚合正常样本（inliers）分布、无监督对比损失分散虚拟异常样本分布，并结合软聚合机制与层次化增强策略，在三种典型异常检测设置下均达到最优性能。

## 研究问题与动机
- **核心问题**：异常检测需要 learned representation 具备"inliers集中、outliers分散"的分布特性，但现有基于对比学习的方法（如DROC、CSI）无法同时满足这两个要求。
- **RotNet不足**：仅通过旋转分类任务学习区分四个角度，inliers分布不够紧凑，outliers分布也不够分散（图1a）。
- **DROC不足**：虽通过对比学习增大instance-level距离使outliers分散，但同时也推远了inliers，违背了inlier集中原则（图1b）。
- **CSI不足**：引入旋转分类头限制inliers到子区域，但预测器的聚合程度有限，仍未达到理想紧凑度（图1c）。
- **关键洞察**：标准对比学习中对inliers生成的多种正视图进行无差别聚合，可能因过度数据增强引入语义漂移的类异常样本，干扰正常分布学习。

## 核心贡献（创新点）
1. **UniCLR损失函数**：首次将inliers视为单一正类（聚合）、virtual outliers视为多个负类（分散），通过非对称的对比学习目标同时优化集中与扩散。
2. **软聚合机制（SA）**：根据每个增强视图与其他inliers的平均相似度动态加权，抑制因强数据增强产生的类异常样本的影响，确保inlier分布纯化。
3. **层次化增强（HA）**：受课程学习启发，在ResNet的四个residual阶段应用渐强的数据增强强度，浅层负责低语义特征聚合、深层负责高语义特征聚合，进一步提升inlier集中度。
4. **无辅助分支设计**：与CSI等方法不同，无需额外的旋转预测分类头或预训练模型，训练更简洁且可泛化至有标签/无标签多类场景。

## 方法详解
### 3.2 单侧聚合对比学习（UniCLR）
- **虚拟异常生成**：对inliers集$\mathcal{D}_{\text{in}}$应用分布偏移增强$S$（如旋转$\{90°, 180°, 270°\}$）生成virtual outliers集$\mathcal{D}_{\text{vout}}$，两者不相交。
- **对比目标**：
  - **Inliers侧（监督对比）**：所有inlier视图互为正样本，virtual outliers为负样本，推动inliers向中心聚集。
  - **Outliers侧（无监督对比）**：每个virtual outlier与其增强视图为正，其余样本为负，推动outliers向潜在空间分散。
  
  数学表达：
  $$\mathcal{L}_{\text{UniCLR}}^{i} = \begin{cases} \mathcal{L}_{\text{cons}}(\tilde{x}_i^1, \tilde{\mathcal{D}}_{\text{in}}-\{\tilde{x}_i^1\}, \tilde{\mathcal{D}}_{\text{vout}}) + \mathcal{L}_{\text{cons}}(\tilde{x}_i^2, \tilde{\mathcal{D}}_{\text{in}}-\{\tilde{x}_i^2\}, \tilde{\mathcal{D}}_{\text{vout}}), & x_i \in \mathcal{D}_{\text{in}} \\ \mathcal{L}_{\text{cons}}(\tilde{x}_i^1, \{\tilde{x}_i^2\}, \tilde{B}-\{\tilde{x}_i^2, \tilde{x}_i^1\}) + \cdots, & x_i \in \mathcal{D}_{\text{vout}} \end{cases}$$

- **Soft Aggregation (SA)**：对inliers的对比损失引入权重$w_x$：
  $$w_{x_i} = \frac{\sum_{x_j \in D_{x_i}^+ \setminus \{x_i\}} e^{z(x_i)^T z(x_j)/\tau_\omega}}{\sum_{x_k \in D_{x_i}^+} \sum_{x_j \in D_{x_i}^+ \setminus \{x_k\}} e^{z(x_k)^T z(x_j)/\tau_\omega}}$$
  距离inlier分布较远的样本获得较低权重，从而减少其干扰。

### 3.3 层次化增强（HA）
- 在ResNet的四个stage（$res_1 \sim res_4$）分别应用强度递增的增强$T_1 \sim T_4$（包括random crop、color jitter、flip等）。
- 每个阶段配备独立projection head $g_i$提取特征$z_i(x)$，在各层分别执行UniCLR聚合。
- 总损失：
  $$\mathcal{L}_{\text{all}} = \frac{1}{4} \sum_{i=1}^4 \lambda_i \mathcal{L}_{\text{UniCLR}}(\mathcal{D}_{\text{in}} \cup \mathcal{D}_{\text{vout}}; T_i)$$
- **设计直觉**：浅层捕获低级视觉特征（弱增强即可），深层学习高级语义（需强增强保证不变性）。

### 3.4 推理
- 移除所有projection heads，使用最后一层residual stage的特征。
- 检测分数采用与训练集最近邻的余弦相似度：
  $$s_i(x_i) = \max_m \text{cosine}(f(x_i), f(x_m))$$
- 无需复杂score function设计，简单有效。

## 实验与结果
### 数据集与设置
- **One-class**：CIFAR-10、CIFAR-100（20 super-classes）、ImageNet-30
- **Multi-class (unlabeled)**：CIFAR-10 vs SVHN/LSUN/ImageNet等；ImageNet-30 vs CUB-200/Dogs/Pets等
- **Multi-class (labeled)**：CIFAR-10、ImageNet-30
- **Real-world**：MVTec-AD
- **网络**：ResNet-18（从头训练，2048 epochs，SGD，lr=0.01）
- **评估指标**：AUROC

### 主要结果
| 设置 | 数据集 | 最佳方法 | AUROC | 提升 |
|------|--------|----------|-------|------|
| One-class | CIFAR-10 | UniCon-HA | 95.4% | +1.1% vs CSI (94.3%) |
| One-class | CIFAR-100 | UniCon-HA | 92.4% | +2.8% vs CSI (89.6%) |
| One-class | ImageNet-30 | UniCon-HA | 93.2% | +1.6% vs CSI (91.6%) |
| Multi-class (unlabeled) | CIFAR-10 | UniCon-HA | 98.5% (LSUN) | 多数benchmark最优 |
| Multi-class (labeled) | CIFAR-10 | UniCon-HA+OE | 99.8% (SVHN) | 接近完美 |
| Real-world | MVTec-AD (image) | UniCon-HA | 89.8% | 仅次于CutPaste (95.2%) |

- **引入OE（Outlier Exposure）**后在CIFAR-10上达到96.9%，说明OE在本框架下是有益的（与CSI中OE有害形成对比）。
- **最强结果**：有标签CIFAR-10多类设置下，UniCon-HA+OE在SVHN上达到99.8% AUROC。

### Ablation
- **SHifting变换**：旋转效果最佳（95.4%），其次Sobel/Noise/Blur（Table 5）。
- **SA与HA组合**：SA+HA（多阶段）达到最优，单独HA也能带来提升（Table 6）。

## 相关工作脉络
1. **RotNet [26]**：基于旋转预测的异常检测方法，仅保证四类可分，inlier分布不紧凑。本文与其核心区别是不依赖分类头，直接优化分布形态。
2. **DROC [49]**：首次将对比学习引入AD，但instance discrimination导致inlier分布均匀化。本文改进为单侧聚合，保留outlier分散优势的同时修复inlier集中度。
3. **CSI [51]**：结合对比学习与旋转分类头，是当时SOTA。本文证明无需分类头，通过UniCLR+SA+HA可达到更好效果，且OE从"有害"变"有益"。
4. **SupCLR [29]**：监督对比学习基础，本文将其思想适配到AD场景——所有inliers共享一个标签作为正类。
5. **CutPaste [33]**：工业缺陷检测专用方法，通过paste操作生成异常。本文指出其不适用于通用AD设置，但启发了distribution shift增强思路。
6. **Outlier Exposure [25]**：传统认为OE会破坏contrastive learning，本文通过soft aggregation化解冲突，使OE成为性能boost。

## 局限性与未来方向
- **虚拟异常的局限性**：虚拟outliers仅通过旋转生成，与实际异常分布仍有差距；未来可探索更多样化的分布偏移策略。
- **单阶段聚合**：当前仅在最后一个residual stage应用SA，可能在中间层也能获益。
- **无预训练依赖**：方法从scratch训练，计算成本较高；如何结合预训练模型是值得探索的方向。
- **仅适用于图像**：方法尚未扩展到视频、多模态等其他模态的异常检测。
- **超参敏感**：温度参数$\tau$、权重温度$\tau_\omega$、各阶段权重$\lambda_i$需调优，缺乏自动化方案。

## 研究启发与可借鉴点
1. **分布形态显式优化**：将"集中vs分散"作为representation学习的直接目标，而非间接通过 pretext task实现，这一思路可迁移到其他需要分布控制的任务（如OOD检测、域适应）。
2. **软聚合机制**：根据样本与主体分布的距离动态调整权重，可用于任何对比学习场景中过滤噪声正样本或困难样本。
3. **层次化增强策略**：从浅到深渐增强度的设计，契合了CNN的层次化特征学习规律，可作为通用正则化手段。
4. **OE的再发现**：证明在合适框架下，原本被认为有害的额外异常数据可以转化为优势，启发对已有资源的重新评估。
5. **简洁推理**：仅用最近邻余弦相似度作为检测分数，避免了复杂score函数的设计，对工程部署友好。

## 关键术语表
- **Inlier**：来自正常分布的样本，即训练集中的"好"样本。
- **Virtual Outlier**：通过对inlier应用分布偏移变换（如旋转）生成的模拟异常样本。
- **Contrastive Learning**：通过拉近正样本对、推远负样本对来学习表示的自监督学习方法。
- **Soft Aggregation (SA)**：根据样本与inlier群体平均相似度动态加权，抑制异常视图对聚合的干扰。
- **Hierarchical Augmentation (HA)**：在神经网络不同深度应用不同强度的数据增强，模拟课程学习。
- **Outlier Exposure (OE)**：在训练时引入额外异常数据集，帮助模型更好地区分inliers和outliers。
- **AUROC**：ROC曲线下的面积，衡量二分类模型区分能力的指标，越大越好。
- **UniCLR**：本文提出的单侧聚合对比损失，inliers用监督对比、outliers用无监督对比。

## 可复现要素
- **数据集**：CIFAR-10/100、ImageNet-30、MVTec-AD（均公开可用）；80 Million Tiny Images用于OE。
- **代码**：论文未提供开源代码链接。
- **关键超参**：
  - 训练epochs：2048
  - Learning rate：0.01，cosine decay
  - 温度参数$\tau$：0.5
  - 权重温度$\tau_\omega$：0.5
  - Inlier:Outlier batch比例：1:3
  - 旋转集合：$\{90°, 180°, 270°\}$
- **网络架构**：ResNet-18，从头训练，无预训练权重。
