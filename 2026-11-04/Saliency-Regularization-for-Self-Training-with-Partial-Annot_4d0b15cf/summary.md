---
title: "Saliency-Regularization-for-Self-Training-with-Partial-Annot"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Wang_Saliency_Regularization_for_Self-Training_with_Partial_Annotations_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 00:54:27"
field: "多标签学习与部分标注"
keywords: ["partial annotation", "multi-label classification", "saliency regularization", "self-training", "consistency regularization", "positive-negative imbalance", "pseudo-labeling"]
innovations: ["提出显著性正则化(SR)缓解部分标注多标签分类中的负主导问题，从理论上证明其可自适应平衡easy/hard样本", "构建无需预训练模型的端到端自训练框架，结合CR与SR在不同已知比例下均有效", "以弱-强增强一致性生成伪标签，借助SR放大正标签概率以弥补未知标签"]
benchmarks: ["MS-COCO", "VG-200", "Pascal VOC 2007", "OpenImages V3"]
---

# 论文速读：Saliency-Regularization-for-Self-Training-with-Partial-Annot

## 一句话总结
本文提出一种基于显著性正则化（Saliency Regularization, SR）的自训练框架，用于解决部分标注多标签分类中因未知标签导致的正负样本失衡问题；该方法在 MS-COCO、VG-200、Pascal VOC 2007 和 OpenImages V3 上均达到 SOTA 性能。

## 研究问题与动机
- 多标签数据的部分标注场景中，大量未知标签加剧了多标签固有的正负不平衡，导致监督学习阶段已知标签（尤其是正样本）的学习不充分。
- 现有方法多假设已知标注比例充足（>50%），在低已知比例（如<30%）下难以捕捉标签间相关性和实例相似性，且未关注监督训练阶段的正负不平衡问题。
- 将未知标签忽略会导致未标注数据熵增大、泛化能力弱；将其视为负标签则引入标签噪声，现有 self-training 方法在这两种策略下均表现受限。
- 在已知标签比例较低时，模型倾向于将正样本误分类为负样本（图1中预测概率分布验证），即存在"负主导"（negative dominance）现象，空间层面的对象区域激活被抑制。

## 核心贡献（创新点）
1. **端到端自训练框架（无需预训练模型）**：提出结合一致性正则化（CR）与显著性正则化（SR）的新型自训练框架，可适配任意已知标签比例；与依赖预训练模型进行伪标签生成的方法（如PLCL、SARB）本质不同。
2. **显著性正则化（SR）缓解负主导**：通过在类别特定图（CSM）上建模显著性，强化已知正标签对应的对象区域激活，同时从理论上证明该操作可自适应调整logit空间边际，解决easy/hard样本不平衡问题（Proposition 1）；与Focal Loss/Asymmetric Loss仅通过调制因子加权的方式本质不同。
3. **一致性正则化配合SR生成高质量伪标签**：利用弱增强和强增强的一致性约束，以SR增强后的预测概率$\hat{p}_c$生成伪标签，从而召回更多真实正样本；与纯阈值pseudo-labeling方法（如PLCL的区间标注、SPLC的即时标注）本质不同，后者容易引入过多负伪标签加剧不平衡。
4. **多维度SOTA性能**：在MS-COCO平均mAP达81.0%（超第二SARB* 2.6%）、VG-200达49.2%、Pascal VOC 2007达93.7%，且在低已知比例（10%）下性能衰减最小。

## 方法详解
- **问题设定**：训练集$\mathcal{D}=\{(I^i, \mathbf{y}^i)\}$中，$\mathbf{y}^i \in \{-1, 0, 1\}^C$，其中1为已知正、-1为已知负、0为未知。
- **类别特定图（CSM）构建**：特征编码器输出$\mathbf{f} \in \mathbb{R}^{W \times H \times D}$经1×1分类头得到$\mathbf{A}_c = \theta_c^\mathsf{T}\mathbf{f}$，再经ReLU和最大归一化得CSM $\mathbf{M}_c$。
- **显著性正则化（SR）**：定义$s_c = \frac{1}{k}\sum a_{c,wh}$（取$\mathbf{A}_c$中Top-k最大值均值），将$s_c$作为logit上的自适应边际加入：$\hat{p}_c = \sigma(a_c + \alpha s_c)$，其中$a_c$为全局池化均值。$\alpha$控制显著性贡献度，$k$控制优化区域数量。
- **理论保证**：Proposition 1证明$s_c$作为logit边际可平滑调整梯度$\frac{\partial L_c}{\partial a_c}$，解决easy/hard样本不平衡；Proposition 2证明SR使不同空间位置的梯度差异化，而非传统loss均匀传播梯度。
- **一致性正则化（CR）**：基于弱增强$\mathcal{A}_w(I)$的预测$\hat{p}_c^w$，用阈值$\tau$生成伪标签$\hat{y}_c = \mathbb{1}[\hat{p}_c^w \geq \tau] - \mathbb{1}[1-\hat{p}_c^w \geq \tau]$，再用强增强$\mathcal{A}_s(I)$的输出$\hat{p}_c^{i,s}$与$\hat{y}_c$计算一致性损失。
- **总损失**：$\mathcal{L} = \mathcal{L}_s + \mathcal{L}_u$，其中$\mathcal{L}_s = \sum_i l(\hat{\mathbf{p}}^i, \mathbf{y}^i)$（用BN-BCE），$\mathcal{L}_u = \sum_i l(\hat{\mathbf{p}}^{i,s}, \hat{\mathbf{y}}^i)$。

## 实验与结果
- **数据集**：MS-COCO、VG-200、Pascal VOC 2007（随机遮蔽90%~10%标签模拟部分标注）、OpenImages V3（真实部分标注数据集）。
- **评估指标**：平均mAP（跨所有已知比例）、平均OF1/CF1，以及OP/CP/OR/CR/OF1/CF1。
- **最强结果**：MS-COCO平均mAP 81.0%（超次优SARB* 2.6%）、VG-200平均mAP 49.2%（超SARB*/CSL 3.2%）、Pascal VOC 2007平均mAP 93.7%、OpenImages V3总mAP 81.5%。
- **关键结论**：SR在低已知比例（10%）下提升最大（+5.1%），验证其对加剧的不平衡有显著缓解；CR（w/ SR）相比CR（w/o SR）有效减少负伪标签，提升正样本召回率；不同已知比例下模型性能衰减最小。

## 相关工作脉络
- **PLCL [12]**：partial-BCE + GNN建模标签相关性 + curriculum learning生成伪标签；本文不依赖GNN和预训练模型，且聚焦正负不平衡而非仅标签相关性。
- **SPLC [41]**：利用非对称损失处理标签级不平衡，但仅在监督阶段使用；本文进一步在空间显著性层面解决负主导，并结合自训练框架。
- **SARB [26] / SARB* [27]**：解耦多类别特征后在特征级做mixup；本文通过不同增强的伪标签一致性而非特征混叠来补全未知标签。
- **SST [6] / HST [5]**：需预训练模型进行语义感知特征分解；本文完全端到端，无需预训练。
- **CSL [1]**：使用Asymmetric Loss处理已知标签不平衡并基于类别分布选择负伪标签；本文以SR替代损失设计，并以一致性正则化替代阈值选择策略。
- **CL [12] / IMCL [16]**：依赖标签相关性或实例相似性挖掘未知标签；本文在低已知比例下同样有效，不受相关性建模质量影响。

## 局限性与未来方向
- 仅适用于已知标签同时包含正负标签的场景，不适用于仅有正标签的部分标注设定。
- 未建模标签间相关性（label correlation）和实例相似性（instance similarity）。
- 未来可将标签相关性建模与SR结合，或将框架扩展至仅有正标签的partial label场景。

## 研究启发与可借鉴点
- **SR的空间正则化思想可迁移**：将显著性约束转化为logit边际调节的方法，可推广至其他存在正负不平衡的检测/分割任务中，无需修改底层损失函数。
- **弱-强增强的伪标签一致性策略设计**：以弱增强生成伪标签、强增强做一致性约束的模式简洁高效，无需复杂阈值搜索或预训练模型，可与本团队的方向结合。
- **Proposition 1的证明思路**：将空间正则化等价于自适应logit边际的操作，为后续"以正则化替代调制因子平衡hard/soft样本"提供了理论依据和可复用的分析框架。
- **实验设计的全面性**：在多个数据集上测试不同已知比例（10%~90%），并给出概率分布可视化分析，为评测方法的鲁棒性提供了完整范式。

## 关键术语表
- **Saliency Regularization (SR)**：通过在类别特定图上优化Top-k显著值，将空间显著性约束转化为logit空间自适应边际的正则化方法。
- **Class-Specific Map (CSM)**：由1×1分类头对特征图逐类别卷积得到的空间热力图，表征每个类别在各像素的激活强度。
- **Self-Training (ST)**：先用已知标签训练模型，再用模型对未知标签生成伪标签，迭代优化的半监督学习范式。
- **Consistency Regularization (CR)**：对同一图像的不同增强视图施加预测一致性约束，用于挖掘未标注数据的伪标签信息。
- **Negative Dominance**：部分标注中因大量未知标签被默认视为负类，导致模型偏向负样本预测、正样本漏检的现象。
- **Partial Annotation**：图像中仅有一小部分标签被标注为正/负，其余标签标记为未知（0）的标注形式。
- **BN-BCE**：按批次内已知标签数量进行归一化的二值交叉熵损失，避免标准BCE在部分标注下梯度过小。
- **Pseudo-labeling (PL)**：利用模型对未标注数据的高置信度预测生成硬标签，用于无监督训练。

## 可复现要素
- **数据集**：MS-COCO、VG-200、Pascal VOC 2007（公开）、OpenImages V3（公开）；部分标注通过随机遮蔽生成。
- **代码/权重**：论文未提及开源代码与权重。
- **关键超参**：$\alpha = 0.5$（SR权重）、$k = 5$（Top-k值数量）、$\tau = 0.6$（伪标签阈值）； backbone为ResNet-101（ImageNet预训练），优化器SGD（momentum=0.9，weight decay=0.0001），batch size=32。
