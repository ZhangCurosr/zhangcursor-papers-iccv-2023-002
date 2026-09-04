---
title: "Space-Engage-Collaborative-Space-Supervision-for-Contrastive"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Wang_Space_Engage_Collaborative_Space_Supervision_for_Contrastive-Based_Semi-Supervised_Semantic_Segmentation_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 23:36:01"
field: "半监督语义分割"
keywords: ["半监督语义分割", "对比学习", "双空间协同", "伪标签", "原型学习"]
innovations: ["提出双空间协同伪标签策略（Mix/Cross）增强知识交换", "引入基于表示-原型相似度的新采样指示器提升对比学习效率"]
benchmarks: ["PASCAL VOC 2012", "Cityscapes", "ADE20K", "COCO-Stuff 10K"]
---

# 论文速读：Space-Engage-Collaborative-Space-Supervision-for-Contrastive

## 一句话总结
本文针对半监督语义分割中现有对比学习方法仅依赖logit空间伪标签的问题，提出**双空间协同监督（CSS）**机制，利用logit空间和表示空间的两类伪标签互相协作，并引入基于相似度的新指示器，以提升伪标签质量与表征学习效率。

## 研究问题与动机
- **单空间监督的局限性**：现有基于对比学习的半监督语义分割方法在无标签训练阶段仅依赖logit空间的预测伪标签，忽视了representation空间中的语义信息，导致知识交换不足。
- **伪标签噪声难以纠正**：logit空间的伪标签可能包含错误语义信息，且由于两个空间关注的特征区域不同，单靠logit置信度无法有效弥补表示空间学习的混淆问题。
- **采样指示器不准确**：以往方法使用logit置信度作为采样指示器，但置信度与表征空间中表示与原型的混淆程度并无直接关系，无法精准定位需要强化的困难样本。
- **对比学习在低标注数据下的瓶颈**：在极少标注（如92张）条件下，如何充分利用无标签数据并提升像素级对比学习效果是一个关键挑战。

## 核心贡献（创新点）
- **提出双空间协同监督框架（CSS）**：与现有方法仅用logit伪标签不同，本文同时从logit空间和representation空间生成伪标签并进行协同，增强两空间间的知识交换。
- **设计Mix与Cross两种伪标签策略**：Mix策略仅保留两空间一致认同的伪标签，Cross策略用一空间的伪标签监督另一空间，二者互补提升伪标签可靠性。
- **引入基于相似度（Similarity）的新指示器**：不同于以往使用logit置信度，本文用表示与类原型的余弦相似度经Softmax后作为采样指示器，更直接反映表征学习的混淆程度，从而更有效地指导困难样本挖掘。

## 方法详解
- **教师-学生框架与双头结构**：采用Mean Teacher架构，学生模型包含编码器$f(\cdot)$、分割头$g(\cdot)$和表示头$h(\cdot)$；教师模型参数由学生EMA更新。表示头将特征映射到表示空间得到像素级表示$z = h(f(x))$。
- **表示空间伪标签生成**：为每个类维护一个原型（centroid）$\hat{\rho}_c$，通过EMA迭代更新；每个表示$z'_i$的伪标签由与其最相似的原型决定：$\hat{y}^{u,rep}_i = \mathbf{1}_{\hat{c}}(\arg\max_c sim(z'_i, \hat{\rho}_c))$。
- **双空间指示器**：logit空间继续使用Softmax置信度$\hat{j}^{u,lgt}_i$；representation空间使用基于相似度的Softmax值$\hat{j}^{u,rep}_i$作为指示器，直接度量表示与原型的混淆水平。
- **两种协同伪标签策略**：
  - **Mix策略**：最终伪标签为两空间伪标签的交集，即仅在两空间预测一致时才保留。
  - **Cross策略**：用representation空间伪标签监督logit空间，反之亦然，保持两空间预测的一致性。
- **总损失函数**：$\mathcal{L} = \mathcal{L}_s + \mathcal{L}_u + \lambda_c \mathcal{L}_c$，其中$\mathcal{L}_s$为有标签数据的交叉熵损失，$\mathcal{L}_u$为基于协同伪标签的无标签交叉熵损失，$\mathcal{L}_c$为对比损失，包含正样本（同类表示与原型的相似度）和负样本（不同类表示间的相似度）。

## 实验与结果
- **数据集与基线**：在PASCAL VOC 2012、Cityscapes、ADE20K、COCO-Stuff 10K上测试；与CCT、CPS、U²PL、ST++、PRCL、PCR、PSMT等SOTA方法对比。
- **PASCAL VOC 2012（Blender设置）**：CSS(mix)在662/1323/2646/5291标注数下分别达到78.73%/79.54%/80.82%/81.06% mIoU，优于PCR（80.91% at 5291 labels）。
- **Cityscapes**：CSS(mix)在186/372/744/1488标注数下分别达到74.02%/76.93%/77.94%/79.62% mIoU，均优于对比方法。
- **ADE20K与COCO-Stuff 10K**：CSS(mix)在1/4标注比例下ADE20K达到37.01%，COCO-Stuff达到29.98%，均优于Baseline。
- **最强结果**：在PASCAL VOC 2012（5291 labels）上达81.06% mIoU，较最佳基线PCR提升约0.15%；在Cityscapes（1488 labels）上达79.62%，较PCR提升约0.51%。

## 相关工作脉络
- **Self-Training方法（如MT、U²PL）**：依赖logit空间伪标签进行无标签训练，本文通过引入representation空间伪标签扩展其监督来源。
- **对比学习语义分割（如PRCL、CCT）**：通常仅在logit空间生成伪标签并用于对比学习，本文强调双空间协同以提升表征质量。
- **原型学习方法（如PCR）**：保持logit预测与原型预测的一致性，但未利用representation空间的语义信息；本文与其正交，可结合使用。
- **跨伪监督方法（如CPS）**：通过两个独立模型保持一致性，本文则通过双空间伪标签协同实现更高效的内存一致性。
- **像素级对比学习（如Propagate Yourself）**：注重单图像内或跨图像的像素一致性，本文进一步在无标签阶段融合双空间监督信号。

## 局限性与未来方向
- **计算开销**：双空间伪标签生成与协同策略增加了训练复杂度，尤其在Cross策略下需维护更多监督信号。
- **原型初始化敏感**：Cross策略需要高质量原型，论文采取先用logit监督20个epoch初始化的策略，可能影响收敛速度。
- **未来方向**：作者提出将探索更强大的策略以增强两空间之间的知识交换，例如动态原型更新或更复杂的跨空间一致性约束。

## 研究启发与可借鉴点
- **双空间伪标签协同思想可迁移**：在其他需要多视图或多表示空间的半监督任务中，可借鉴Mix/Cross策略提升伪标签可靠性。
- **相似度指示器设计具有通用性**：将表征与原型的相似度作为采样指示器，比单一置信度更能反映学习难点，适用于各类对比学习框架。
- **实验设计严谨**：消融实验分离了伪标签策略与指示器的贡献，为后续工作提供了清晰的组件级改进依据。
- **与现有方法正交可叠加**：本文指出CSS可与PCR等方法结合，提示后续研究可在不同模块间自由组合创新。

## 关键术语表
- **Semi-Supervised Semantic Segmentation (S4)**：利用少量标注图像和大量无标注图像进行像素级语义分割的任务设定。
- **Logit Space**：分割头输出的未归一化预测空间，通常经Softmax得到类别置信度。
- **Representation Space**：通过表示头将特征映射到的隐空间，像素级表示在此空间中通过对比学习被聚合到类原型。
- **Prototype**：每个类别在表示空间中的中心向量，由该类别所有表示的均值计算并通过EMA更新。
- **Mix Pseudo-labeling**：仅保留logit空间与representation空间伪标签一致的像素作为最终监督信号的策略。
- **Cross Pseudo-labeling**：用一空间的伪标签监督另一空间预测，保持双空间一致性的策略。
- **Indicator（指示器）**：用于筛选有效训练样本的标量，本文在logit空间用置信度，在representation空间用相似度Softmax值。
- **Contrastive Loss**：通过拉近同类表示、推远异类表示以增强特征判别力的损失函数。

## 可复现要素
- **数据集**：PASCAL VOC 2012、Cityscapes、ADE20K、COCO-Stuff 10K（均为公开数据集）。
- **代码/权重**：论文未提及代码开源声明；模型基于Deeplabv3+与ResNet-101（ImageNet预训练权重公开）。
- **关键超参**：学习率0.0064（VOC/COCO）或0.0038（Cityscapes），batch size 16或8，crop size 512×512或768×768，迭代次数40k/80k，EMA系数α未明确给出，对比损失温度τ未明确给出，阈值δ_u、δ_w、δ_s未列出具体数值。
