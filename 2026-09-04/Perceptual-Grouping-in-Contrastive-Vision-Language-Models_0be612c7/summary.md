---
title: "Perceptual-Grouping-in-Contrastive-Vision-Language-Models"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Ranasinghe_Perceptual_Grouping_in_Contrastive_Vision-Language_Models_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 19:14:56"
field: "视觉-语言预训练与空间理解"
keywords: ["视觉-语言模型", "感知分组", "零样本语义分割", "对比学习", "空间感知", "鲁棒性"]
innovations: ["通过max pooling和DINO初始化等最小修改赋予CLIP感知分组能力", "在零样本语义分割和无监督分割上达到SOTA且无需任何分割标注", "证明感知分组显著提升对虚假相关性的鲁棒性"]
benchmarks: ["VOC mIoU", "ImageNet", "Waterbirds", "ADE20K", "COCO", "Cityscapes"]
---

# 论文速读：Perceptual-Grouping-in-Contrastive-Vision-Language-Models

## 一句话总结
本文发现对比式视觉-语言模型（如CLIP）在弱监督训练下缺乏对物体空间位置的理解，无法对视觉内容进行感知分组；通过三项最小修改（空间最大池化、DINO预训练初始化、Token子采样），提出的CLIPpy模型在保持零样本图像识别性能的同时，实现了无监督分割和零样本语义分割的SOTA，并显著提升了对虚假相关性的鲁棒性。

## 研究问题与动机
- **核心问题**：视觉-语言模型虽能学习通用语义表征，但对"物体在何处"缺乏理解，容易混淆前景与背景内容，无法将视觉上相关的部分分组到语义有意义的区域。
- **现有方法不足**：
  1. 现有分组方法（如GroupViT、OVS）依赖定制化架构或仅关注前景区分，忽略背景类别。
  2. 基于密集标注的方法（如MaskCLIP）需要像素级监督，丧失弱监督的可扩展性。
  3. CLIP使用CLS token聚合方式破坏了空间结构，使特征退化为"图像补丁的袋"，无法保留空间布局信息。
  4. 现有模型易受数据中虚假相关性影响，导致域偏移和反事实扰动下的性能下降。

## 核心贡献（创新点）
1. **系统性揭示CLIP的空间感知缺陷**：首次量化证明对比式VLM在零样本位置预测中严重混淆前景/背景，无法正确分组语义相关内容。
   - *区别*：不同于仅报告性能指标，本文通过可视化与消融揭示了CLS/Avg pooling对空间信息的破坏机制。
2. **提出CLIPpy——最小修改方案实现感知分组**：仅需三项调整（max pooling、DINO初始化、Token子采样）即可让CLIP获得bottom-up和top-down分组能力，无需任何分割标注或任务特定微调。
   - *区别*：相比GroupViT/OVS等定制化架构，CLIPpy直接在标准CLIP框架上做最小改动，同时支持任意数量的前景/背景类别。
3. **SOTA无监督分割与零样本语义分割**：在VOC上mIoU达52.2%，JS达47.5%，超越所有先前弱监督方法；且无需密集标注。
   - *区别*：此前SOTA依赖自监督预训练或定制训练，本文证明标准对比学习配合适当聚合即可涌现分组能力。
4. **感知分组显著提升对虚假相关性的鲁棒性**：在Waterbirds基准上域偏移下降仅2.0%（基线CLIP为32.1%），达到监督方法的鲁棒水平。
   - *区别*：首次将空间分组能力与因果鲁棒性直接关联，证明其可缓解领域偏移问题。

## 方法详解
- **架构基础**：基于ViT-B/16图像编码器与T5-base文本编码器，采用标准对比损失（image-to-text + text-to-image），温度系数τ。
- **关键修改1：空间最大池化（Max Pooling）**
  - 将图像特征[H,W,D]聚合为[D]向量时，使用全局最大池化替代CLS token或平均池化。
  - *原理*：Max pooling使梯度更新集中在单个最具判别性的空间位置，迫使模型在跨图像的共同物体位置上对齐语言概念，从而涌现空间分组。
- **关键修改2：合适的视觉预训练初始化**
  - 图像编码器使用DINO（自蒸馏自监督）初始化，而非ImageNet监督预训练；文本编码器使用Sentence-T5初始化。
  - *原理*：DINO学习细粒度空间表征，而ImageNet监督预训练偏向图像级语义分离，后者会损害分组能力。
- **关键修改3：Token子采样（Token Sub-Sampling, TSS）**
  - 训练中随机使用更高分辨率图像，并随机丢弃80%视觉token以降低计算开销。
  - *作用*：提升分割分辨率与训练稳定性，具有正则化效果。
- **推理模式**：
  - a) 分类：使用聚合后的单图像token进行零样本分类。
  - b) Bottom-up分组：对预聚合图像特征做PCA（取前8个主成分作为聚类中心），进行谱聚类。
  - c) Top-down分组：在每个空间位置计算per-region token与语言token的相似度，实现零样本语义分割。

## 实验与结果
- **数据集**：CC-12M（12M图文对）、HQITP-134M（134M图文对）；评估数据集包括ImageNet、ImageNet-v2、VOC、ADE20K、COCO、Cityscapes、Waterbirds。
- **零样本图像分类**：CLIPpy在CC-12M上ImageNet准确率为45.3%（CLIP†为46.0%），HQITP-134M上为60.3%（CLIP†为61.4%），基本持平。
- **无监督分割（Bottom-up）**：
  - VOC JS：**54.6%**（CLIP†为38.9%，提升+15.7%），SOTA。
  - ADE20K：29.5%（HQITP-134M），COCO：27.2%。
  - Cityscapes：22.3%，超越PiCIE（12.3）和STEGO（21.0）。
- **零样本语义分割（Top-down）**：
  - VOC mIoU：**52.2%**（HQITP-134M），超越GroupViT†（40.1%）和OVS（44.6%）。
  - COCO mIoU：25.5%，ADE20K mIoU：13.5%，COCO(obj)：32.0%。
- **鲁棒性（Waterbirds）**：
  - CLIP在背景不匹配时准确率下降32.1%（80.2→48.1），CLIPpy仅下降2.0%（76.9→74.9），域偏移gap仅约4%。
  - 相比需监督训练的Robust CLIP（gap 4%-8%），CLIPpy在零样本下达到相当水平。
- **消融实验**：
  - Max pooling在VOC上mIoU达50.8%，而Avg pooling仅11.6%，Cls pooling仅4.0%。
  - DINO初始化+Sentence-T5初始化组合最佳（IN 42.3%，VOC 50.8%）；ImageNet监督初始化反而损害分组（VOC 22.5%）。
  - Token子采样带来小幅提升（VOC 50.9→51.8）。

## 相关工作脉络
1. **CLIP/ALIGN**：基线对比学习VLM，使用CLS token聚合，缺乏空间感知能力，本文揭示其系统性缺陷。
2. **GroupViT [113]**：定制化ViT架构结合离散化注意力掩码实现分组，但仅聚焦前景区分；CLIPpy以最小修改在标准ViT上超越GroupViT†（VOC 50.8 vs 40.1）。
3. **OVS [114]**：同样使用弱监督与预训练策略，但需定制化训练；CLIPpy证明标准对比学习配合适当聚合即可涌现分组。
4. **DINO [12]**：自监督视觉Transformer，本文借鉴其初始化策略以实现细粒度空间表征。
5. **MaskCLIP [126]**：基于密集标注的零样本分割方法；CLIPpy无需任何分割标注即达到更高性能。
6. **Waterbirds [91]**：虚假相关性基准；CLIPpy证明感知分组可缓解此类问题，达到监督方法的鲁棒水平。

## 局限性与未来方向
- **视觉复杂度敏感**：在ADE20K等背景复杂、标签基数大的数据集上性能下降明显，分组能力受限。
- **数据偏差风险**：训练数据（CC-12M/HQITP）可能包含社会偏见，模型会继承这些偏差。
- **未来方向**：结合更大规模开放数据集（如Coyo-700M、LAION）和自监督学习进展（如STEGO）进一步提升分组能力。

## 研究启发与可借鉴点
1. **聚合策略的关键作用**：Max pooling相比CLS/Avg pooling能更好地保留空间信息，这一发现可直接迁移到其他视觉-语言任务的特征聚合设计。
2. **预训练初始化与下游能力的解耦**：ImageNet监督预训练提升分类但损害空间分组，表明不同任务需匹配不同类型的初始化先验。
3. **零样本分组-鲁棒性关联**：感知分组能力与对虚假相关性的鲁棒性存在因果联系，为设计抗偏倚模型提供新视角。
4. **Token子采样的正则化价值**：随机丢弃80% token的训练策略可在不增加计算开销下提升分割性能，适用于高分辨率视觉任务。
5. ** minimal modification范式**：证明对成熟架构的微小改动（而非全新设计）即可解锁新能力，为高效模型改进提供思路。

## 关键术语表
- **Perceptual Grouping（感知分组）**：将视觉上相关的像素组合成语义有意义区域的能力，包括自下而上（纯视觉）和自上而下（语言引导）两种模式。
- **Contrastive Vision-Language Model（对比式视语模型）**：通过对比损失对齐图像与文本嵌入的模型，如CLIP、ALIGN。
- **Zero-shot Semantic Segmentation（零样本语义分割）**：在未见类别上直接进行像素级分割，无需目标数据集的标注。
- **Jaccard Similarity（Jaccard相似度）**：评估无监督分割性能的指标，计算预测分割与真实分割的IoU均值，与类别标签无关。
- **Mean Intersection over Union（mIoU）**：零样本语义分割指标，衡量预测与标注在所有类别上的平均IoU。
- **Spurious Correlation（虚假相关性）**：模型学习与标签无关的捷径特征（如背景）导致的错误泛化行为。
- **DINO（Self-distillation with no labels）**：基于自蒸馏的视觉自监督预训练方法，能学习细粒度的空间表征。
- **Token Sub-sampling（Token子采样）**：训练中随机丢弃80%视觉token的技术，用于提升分辨率适应性并作为正则化手段。

## 可复现要素
- **数据集**：CC-12M公开可获取；HQITP-134M因版权限制不公开，但论文提供了仅在CC-12M上训练的结果以确保可复现性。
- **代码**：基于OpenAI CLIP源码修改，主要改动为架构与训练策略调整，代码变更已在论文第3节说明。
- **预训练权重**：Image backbone使用DINO（[12]）、Text backbone使用Sentence-T5（HuggingFace），均在论文中明确来源。
- **关键超参**：图像分辨率224×224（训练）、448×448（Waterbirds评估）；ViT-B/16；T5-base；32 GPU分布式训练；TSS丢弃80% token。
