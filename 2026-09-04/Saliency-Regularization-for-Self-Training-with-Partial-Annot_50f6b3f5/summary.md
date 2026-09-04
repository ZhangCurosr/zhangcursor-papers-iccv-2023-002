---
title: "Saliency-Regularization-for-Self-Training-with-Partial-Annot"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Wang_Saliency_Regularization_for_Self-Training_with_Partial_Annotations_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 19:17:33"
---

# 论文速读：Saliency-Regularization-for-Self-Training-with-Partial-Annot

## 一句话总结
针对多标签分类中部分标注导致未知标签加剧正负不平衡的问题，本文提出一种基于显著性正则化（SR）的自训练框架，通过在类别特制图（CSM）上强化已知正类对象区域的激活响应，并结合一致性正则化（CR）挖掘未标注信息，在低已知比例下有效缓解负向主导现象并达到SOTA性能。

## 研究问题与动机
- 部分标注多标签数据中，未知标签若被忽略或当作负样本，会显著加剧多标签固有的正负不平衡，使分类器偏向负类预测（负向主导）。
- 现有自训练方法多依赖充足已知标注（>50%）来挖掘标签相关性或实例相似度，在低已知比例（<30%）下关联结构难以准确估计。
- 传统不平衡损失（如Focal Loss、Asymmetric Loss）仅从概率调制角度调节难易样本，未从空间维度解决已知正类对象区域激活不足的问题。
- 缺乏既能在监督阶段缓解标签级正负不平衡、又无需预训练模型即可稳定生成伪标签的端到端自训练范式。

## 核心贡献（创新点）
- 构建端到端自训练框架并引入一致性正则化（CR），替代依赖预训练模型的课程学习策略，使方法可适配任意已知标签比例。**本质区别**在于摆脱了对强预训练特征提取器的依赖，实现监督与无监督模块在同一架构内的联合优化。
- 提出显著性正则化（SR），将类别特制图（CSM）的Top-k最大激活值以自适应margin形式叠加至分类logit。**本质区别**在于不同于Focal/Asymmetric Loss仅从概率调制平衡难易样本，SR通过空间激活引导同时缓解了负向主导与难易样本梯度失衡。
- 在MS-COCO、VG-200、Pascal VOC 2007及OpenImages V3上实现SOTA，且在10%极低已知比例下性能衰减最小。**本质区别**在于现有部分标注方法在低比例下普遍出现断崖式下跌，本文通过SR+CR协同维持了高/低比例场景的一致泛化能力。

## 方法详解
- **类别特制图（CSM）构建**：特征编码器输出 $\mathbf{f} \in \mathbb{R}^{W \times H \times D}$ 后接1×1卷积分类头，得到第c类的空间得分图 $\mathbf{A}_c = \theta_c^\mathsf{T}\mathbf{f}$，经ReLU与最大值归一化得到热力图 $\mathbf{M}_c$，表征该类别的判别性对象区域。
- **显著性正则化（SR）核心设计**：对 $\mathbf{A}_c$ 取Top-k最大值计算均值 $s_c = \frac{1}{k}\sum a_{c,wh}$。将 $s_c$ 以系数 $\alpha$ 叠加到全局平均池化logit $a_c$ 上，得到修正预测 $\hat{p}_c = \sigma(a_c + \alpha s_c)$。已知正类（$y_c=1$）时 $s_c>0$ 增强对象区域激活，负类（$y_c=-1$）时抑制激活，从而缓解负向主导。
- **Logit偏移与难易样本平衡**：$s_c$ 作为自适应margin动态调整梯度 $\frac{\partial L_c}{\partial a_c}$。硬正样本激活强（$s_c>0$），梯度平滑衰减；半难样本激活弱（$s_c\leq0$），损失贡献保持显著，隐式实现难易样本平衡（Proposition 1）。
- **一致性正则化（CR）与伪标签生成**：对同一图像施加弱增强 $\mathcal{A}_w(I)$ 与强增强 $\mathcal{A}_s(I)$。弱增强输出 $\hat{p}_c^w$ 结合阈值 $\tau$ 生成未知标签的硬伪标签 $\hat{y}_c = \mathbb{1}[\hat{p}_c^w \geq \tau] - \mathbb{1}[1-\hat{p}_c^w \geq \tau]$；强增强输出 $\hat{p}_c^s$ 与伪标签计算一致性损失 $\mathcal{L}_u$。SR在此过程中放大潜在正类概率，减少假负噪声。
- **整体优化目标**：$\mathcal{L} = \mathcal{L}_s + \mathcal{L}_u$，其中 $\mathcal{L}_s$ 基于 $\hat{p}_c$ 与部分真值计算BCE，$\mathcal{L}_u$ 基于强弱增强输出与伪标签计算，两模块共享同一网络参数。

## 实验与结果
- **数据集与基线**：在模拟部分标注的MS-COCO、VG-200、Pascal VOC 2007及真实部分标注OpenImages V3上评测，基线涵盖PLCL、SPLC、SST、SARB、CSL、GCN-ML等。
- **评估指标与设置**：采用平均mAP、Overall F1（OF1）、Class-wise F1（CF1）；骨干为ImageNet预训练ResNet-101，分类头随机初始化；超参 $\alpha=0.5, k=5, \tau=0.6$；优化器SGD（momentum 0.9, weight decay 1e-4），batch size 32。
- **主要结果**：
  - **MS-COCO**：平均mAP 81.0%，较上一SOTA（SARB*）提升2.6%，OF1/CF1分别提升2.2%/3.0%。
  - **VG-200**：平均mAP 49.2%，较SARB*和CSL提升3.2%。
  - **Pascal VOC 2007**：平均mAP 93.7%，在10%低已知比例下性能下降最小，验证强鲁棒性。
  - **OpenImages V3**：平均mAP 81.5%，超越CSL（80.0%）等基线。
- **消融与分布分析**：SR在10%已知比例下mAP增益最大（+5.1%）；概率分布可视化表明SR能更好区分缺失标签与真实负样本，生成更清晰的决策边界。

## 相关工作脉络
- **传统图/矩阵方法（SSGRL, GCN-ML, FastTag等）**：依赖标签共现统计邻接矩阵，在已知标签比例低时矩阵统计偏差大，难以与深度网络联合微调。
- **伪标签自训练方法（PLCL, SPLC, SST, SARB）**：多通过课程学习或相似度挖掘生成伪标签，高度依赖充足已知标注（>50%），低比例下易引入噪声。
- **不平衡损失设计（Focal Loss, Asymmetric Loss, SPLC）**：仅从概率调制角度平衡难易样本，未考虑空间维度上正类对象区域激活不足导致的负向主导。
- **选择性/类别感知损失（CSL）**：同样关注监督阶段的不平衡，但采用类别感知选择机制，本文SR从空间显著性正则化切入，两者机制正交且可互补。
- **一致性正则化半监督方法（FixMatch等）**：多用于单标签或全标注场景，本文将其适配于多标签部分标注，并通过SR辅助伪标签生成以抑制负偏差。

## 局限性与未来方向
- **局限**：仅适用于已知标签同时包含正负样本（$y_c \in \{1, -1\}$）的场景，不适用于仅有正例未知的设定；未显式建模标签间相关性（label correlation）与实例间相似度（instance similarity）。
- **未来方向**：可将SR与标签依赖图或对比学习结合，进一步挖掘低已知比例下的潜在关联；探索在纯正例部分标注场景下的适配；将框架扩展至视频多标签分类、开放词汇检测或跨模态检索任务。

## 研究启发与可借鉴点
-
