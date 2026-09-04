---
title: "Open-vocabulary-Panoptic-Segmentation-with-Embedding-Modulat"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Chen_Open-vocabulary_Panoptic_Segmentation_with_Embedding_Modulation_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 23:35:41"
field: "开放词汇视觉分割"
keywords: ["open-vocabulary segmentation", "panoptic segmentation", "CLIP", "embedding modulation", "cross-dataset generalization"]
innovations: ["提出OPSNet框架，首次系统性解决开放词汇全景分割任务", "设计Embedding Modulation模块，通过域相似度自适应融合查询嵌入与CLIP嵌入", "提出Decoupled Supervision训练范式，利用图像级标签扩展训练概念"]
benchmarks: ["COCO panoptic segmentation", "ADE20K", "Cityscapes", "PascalContext", "ImageNet-21K"]
---

# 论文速读：Open-vocabulary Panoptic Segmentation with Embedding Modulation

## 一句话总结
论文提出 OPSNet，一个高效且强大的开放词汇全景分割框架，通过 Embedding Modulation 模块实现查询嵌入与 CLIP 嵌入的有效融合与自适应调节，在仅用 COCO 训练数据的情况下，实现了开放词汇和封闭词汇场景下的最优分割性能。

## 研究问题与动机
- **传统封闭词汇分割的局限**：现有方法（如 Mask2Former）仅在预定义类别集合内工作，无法识别新类别物体，面对真实世界中多样化对象时存在明显失败。
- **现有开放词汇方法的不足**：当前方法在封闭词汇上性能显著下降，或依赖大量额外数据进行训练（如从头训练文本-图像对齐）；部分方法需将每个候选框送入 CLIP 图像编码器提取视觉特征，效率低下。
- **泛化与性能的平衡难题**：使用预训练 CLIP 的方法虽数据高效，但难以平衡跨数据集泛化能力与训练域性能，例如 MaskCLIP 在 COCO 上表现不佳，而 OpenSeg 等方法仅能处理语义分割，无法解决实例级区分。

## 核心贡献（创新点）
1. **提出 OPSNet 框架**：首次系统性地解决开放词汇全景分割任务，实现同时在开放词汇和封闭词汇设置下的优异性能。
2. **设计 Embedding Modulation 模块**：核心创新，通过领域相似度自适应融合查询嵌入与 CLIP 嵌入，并结合 logits 去偏策略，实现有效嵌入增强与信息交换。
3. **引入 Spatial Adapter + Mask Pooling**：将 CLIP 图像编码器重参数化为保持空间分辨率的结构，通过单次前向传播完成掩码池化，显著提升效率。
4. **提出 Mask Filtering 机制**：通过 IoU Head 预测掩码质量并过滤无效候选，解决开放词汇设置下分类得分不可靠的问题。
5. **设计 Decoupled Supervision 训练范式**：利用图像级标签扩展训练概念，结合布局信息监督掩码分割，使模型仅用少量额外数据即可大幅提升泛化能力。

## 方法详解

### 基础框架
以 Mask2Former 为基础模型，移除分类头，将 learnable query 投影为 query embedding，与 CLIP 文本编码器提取的文本嵌入进行余弦相似度匹配，使用温度参数 τ=0.01 锐化 softmax 分布。

### Spatial Adapter 与 Mask Pooling
- **Spatial Adapter**：将 CLIP 图像编码器中的 attention-pooling 层重参数化为 1×1 卷积，保持空间分辨率，将特征投影到语言空间。
- **Mask Pooling**：利用预测的二值掩码对 CLIP 视觉特征进行池化，一次性提取每个对象的 CLIP 嵌入。
- **Cross-Attention**：通过交叉注意力机制用 CLIP 特征增强 query 嵌入，实现双向信息交换。

### Embedding Modulation（核心模块）
**Embedding Fusion**：
- 计算预测类别集合与训练类别集合之间的余弦相似度矩阵 H
- 计算域相似度系数 $s = \frac{1}{M}\sum_i \max_j(H_{i,j})$
- 融合嵌入：$E_m = E_q + \alpha \cdot (1-s) \cdot E_c$，其中 α=10 默认值
- 原理：域相似度越低，越依赖 CLIP 嵌入的泛化能力

**Logits Debiasing**：
- 对分类 logits 进行去偏：$\hat{z}_i = z_i / (\max_j(H_{i,j}))^\beta$，β=0.5 默认值
- 缓解对已见类别的偏好，平衡 seen 与 unseen 类别

### Mask Filtering
- 添加线性层的 IoU Head 预测掩码与 GT 的 IoU
- 使用 L2-IoU Loss 训练，利用预测 IoU 分数排序和过滤掩码
- 支持 unmatched/duplicate  proposal 回归到零

### Decoupled Supervision
- **匹配损失**：$\mathcal{L}_{match} = 1 - \frac{1}{c}\sum_{j=1}^{c} \max_i(\delta(S_{i,j})) \cdot \mathbb{1}_{j \in R^c}$
- **求和损失**：$\mathcal{L}_{sum} = ||1 - \sum_{k=1}^{K}(\sigma(M_{k,i,j}))||_2$
- 使用重新标注的 ImageNet-Val（50K 图像，1K 类别）扩展训练概念

## 实验与结果

### 数据集与评估
- **COCO**：封闭词汇全景分割主基准
- **ADE20K、Cityscapes**：跨数据集泛化评估
- **PascalContext**：开放词汇语义分割评估
- **ImageNet-21K**：大规模开放词汇预测

### 主要结果

**开放词汇全景分割（Table 2，仅 COCO 训练）**：
- OPSNet (Swin-L†)：COCO PQ=57.9，ADE20K PQ=19.0，Cityscapes PQ=41.5
- 相比 MaskCLIP-Full：COCO +21.1，ADE20K +3.9
- 相比 MaskCLIP-Base：ADE20K 提升显著（17.7 vs 9.6）

**开放词汇语义分割（Table 8）**：
- OPSNet+ (ResNet-101)：ADE20K mIoU=24.5，PascalContext=54.3，COCO=61.4
- 相比 OpenSeg (Efficient-B7)：使用 600K 字幕数据 vs 仅 50K ImageNet

**封闭词汇性能（Table 9，COCO）**：
- OPSNet (Swin-L†)：PQ=57.9，PQ_th=64.1，PQ_st=48.5
- 相比 Mask2Former (Swin-L†)：+3.3 PQ
- 超越 K-Net (+3.8)、Panoptic Segformer (+3.8)

**关键提升幅度**：
- ADE20K PQ 较 MaskCLIP-Base 提升 84%（17.7 vs 9.6）
- Cityscapes PQ 达到 41.5，为当时最佳

## 相关工作脉络

1. **Unified Segmentation (Mask2Former, K-Net)**：提供统一分割架构基础，但仅支持封闭词汇；本文扩展至开放词汇场景。
2. **Open-Vocabulary Detection/Segmentation (ViLD, Detic, OpenSeg)**：使用 CLIP/ALIGN 文本嵌入进行分类；本文区别在于同时利用 CLIP 视觉特征并实现实例级区分。
3. **MaskCLIP**：唯一针对全景分割的开放词汇方法，但 COCO 性能低（PQ=30.9）；本文通过 Embedding Modulation 显著提升封闭词汇性能。
4. **Class-Agnostic Segmentation**：移除分类头实现通用检测；本文在此基础上引入开放词汇分类能力。
5. **DenseCLIP**：将 CLIP 特征直接用于语义分割；本文聚焦全景分割，保留实例级信息。

## 局限性与未来方向
- **跨域性能仍有提升空间**：ADE20K 的 PQ 仅 19.0，相对 COCO 训练域仍有较大差距
- **CLIP 编码器的局限性**：CLIP 并非对所有领域和类别都有效，极端长尾或小众类别可能预测不准确
- **Decoupled Supervision 的数据依赖**：需要重新标注的图像级标签数据集（如 ImageNet-Val），非标准格式
- **推理效率**：虽然相比逐候选框送入 CLIP 的方法更高效，但 CLIP 编码器仍增加约 27M 参数

## 研究启发与可借鉴点
1. **Embedding Modulation 的自适应融合策略**：通过域相似度动态调节两类嵌入的权重，这一思想可迁移到其他视觉-语言联合学习任务中。
2. **Mask Pooling 高效提取视觉嵌入**：单次前向传播替代逐候选框处理，显著提升效率，可推广至其他开放词汇检测/分割任务。
3. **Logits Debiasing 思想**：利用类别相似度对分类 logits 进行去偏，平衡 seen/unseen 类别，可应用于开放词汇分类任务。
4. **Decoupled Supervision 范式**：图像级标签 + 布局约束的解耦监督策略，为多模态大规模预训练提供了高效的训练范式。

## 关键术语表
**Open-vocabulary Panoptic Segmentation**：同时分割图像中所有对象（things）和背景区域（stuff），并能识别训练集类别之外的新类别。
**CLIP**：由 OpenAI 提出的视觉-语言预训练模型，通过大量图像-文本对学习联合表示空间。
**Embedding Modulation**：本文核心模块，通过域相似度自适应融合查询嵌入与 CLIP 嵌入，并配合 logits 去偏。
**Mask Pooling**：利用预测的二值掩码对 CLIP 视觉特征进行池化，提取每个对象的视觉嵌入。
**Spatial Adapter**：将 CLIP 的 attention-pooling 重参数化为 1×1 卷积，保持空间分辨率的同时投影到语言空间。
**Decoupled Supervision**：利用图像级标签监督分类、布局信息监督掩码分割的解耦训练范式。
**Mask Filtering**：通过 IoU Head 预测掩码质量并过滤无效候选，解决开放词汇场景分类得分不可靠的问题。

## 可复现要素
- **数据集**：COCO、ADE20K、Cityscapes、PascalContext、ImageNet-Val（公开）
- **代码**：项目页面 https://opsnet-page.github.io（论文未明确说明 GitHub 链接，但提供了项目页）
- **模型权重**：论文未提及开源权重
- **关键超参**：温度参数 τ=0.01，融合系数 α=10，去偏系数 β=0.5，训练 50 轮/80K 迭代，初始学习率 0.0001
