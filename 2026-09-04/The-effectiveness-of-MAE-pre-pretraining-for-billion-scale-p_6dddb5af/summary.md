---
title: "The-effectiveness-of-MAE-pre-pretraining-for-billion-scale-p"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Singh_The_Effectiveness_of_MAE_Pre-Pretraining_for_Billion-Scale_Pretraining_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 21:20:26"
---

# 论文速读：The-effectiveness-of-MAE-pre-pretraining-for-billion-scale-p

## 一句话总结
本文在标准的大规模弱监督预训练（WSP）之前插入一个MAE自监督预训练阶段（pre-pretraining），证明该初始化策略能显著提升模型收敛速度与下游泛化性能；该范式在百万至数十亿参数、百万至数十亿图片的尺度下均稳定有效，并在iNaturalist-18、1-shot ImageNet-1k与零样本Food-101上刷新SOTA。

## 研究问题与动机
- Web-scale视觉基础模型普遍采用“大规模弱监督预训练→下游微调”范式，但标签噪声大、从随机权重冷启动时收敛慢，且预训练初始化质量对最终性能的影响常被低估。
- 自监督方法（如MAE）无需标签、计算高效且擅长学习结构化表征，但此前的缩放验证仅限IN1k等小规模数据与中等模型；弱监督方法（如SWAG）依赖海量带噪标签，在分类判别与零样本/低样本迁移上更强，但单独使用时检测与细粒度任务表现受限。
- 现有工作多将自监督与弱监督独立使用或简单拼接，缺乏系统性研究：自监督预训练能否作为大规模弱监督预训练的通用初始化器？其增益是否随模型与数据规模同步扩展？
- 本文旨在证明：即使在十亿级标注图片的预训练中，模型初始化阶段的质量仍起决定性作用，并提出零额外数据成本、零超参调优的两阶段范式。

## 核心贡献（创新点）
1. 提出MAE→WSP两阶段预预处理框架，用MAE自监督权重初始化随后的大规模弱监督预训练，无需额外数据或超参搜索。与已有工作的本质区别：不同于预训练后的中间微调（intermediate finetuning），该方法作用于标准预训练之前，且直接复用预训练数据本身。
2. 首次验证MAE同时具备模型规模与预训练数据规模的缩放性。与已有工作的本质区别：He et al.仅证明MAE随模型增大而改善，本文在IG-3B（30亿图片）上证明其自监督重建目标同样随数据量增加持续增益。
3. 在10类视觉识别任务（图像分类、视频识别、目标检测、低样本分类、零样本迁移等）上给出十亿级弱监督基线下的统一对比。与已有工作的本质区别：突破了此前研究聚焦单一任务或单一尺度的局限，提供跨任务、跨尺度、跨数据分布的系统性基准。
4. 揭示Web-scale训练中模型初始化对收敛效率与最终上限的关键作用。与已有工作的本质区别：直接挑战“数据/算力决定一切”的直觉，量化了pre-training pipeline中冷启动阶段的隐性价值。

## 方法详解
- **整体流程**：第一阶段使用纯图像进行MAE自监督预训练（不依赖任何标签）；第二阶段以该编码器权重初始化，使用IG-3B的hashtag伪标签进行弱监督预训练（WSP），全程沿用原有超参数，无需重新调优。
- **MAE预训练阶段**：随机遮盖75%的图像patch，仅处理剩余25%未遮盖patch以节省计算；目标像素值按patch内均值与标准差归一化，通过轻量解码器重建被遮盖区域。训练1 epoch，超参数与IN1k上MAE一致。
- **弱监督预训练阶段（WSP）**：将图像标题hashtag经WordNet synset映射生成离散伪标签，采用多标签交叉熵损失训练。默认从随机权重训练，本文改为从MAE权重加载后继续训练1 epoch。
- **零样本扩展（LiT）**：在MAE→WSP完成后冻结图像编码器，使用原始(image, caption)对训练XLM-R Large文本编码器，通过CLIP对比损失对齐图文嵌入，实现零样本分类。
- **效率优势**：MAE的patch dropping机制使第一阶段计算开销远低于全图监督训练；两阶段总FLOPs与纯WSP相当，但转移性能显著更优，效率最高提升2×。

## 实验与结果
- **预训练数据**：Instagram-3B（IG-3B），30亿唯一图像（重采样至50亿），28K类别，基于SWAG流水线自动生成标签。
- **评测基准**：ImageNet-1k、iNaturalist-18、ImageNetv2、ImageNet-ReaL、ObjectNet、Food-101、COCO、LVIS、Kinetics-400、Something-Something-v2。
- **模型尺度**：ViT-B (86M)、ViT-L (307M)、ViT-H (632M)、ViT-2B、ViT-6.5B。
- **主要结果**：
  - 图像分类：ViT-2B在iNaturalist-18达到**91.3%**（新SOTA），ViT-H在IN1k达89.3%，在IN-ReaL (90.9%)与ObjectNet (75.8%)上显著优于同规模SWAG。
  - 低样本分类：1-shot IN1k达**62.1%**（新SOTA），10-shot iNat18达80.3%，超越MSN、DINO等低样本专项方法。
  - 零样本：Food-101达**96.2%**（新SOTA），IN1k达81.9%。
  - 检测/分割（LVIS val）：ViT-H APbox 50.8、ViT-2B 51.8，大幅领先同规模SWAG (47.1) 与MAE-IN1k (51.5)。
  - 视频分类：ViT-L在K400与SSv2上均刷新同等参数规模SOTA，且预训练仅使用图像。
- **消融与效率**：仅0.1 epoch MAE即可加速WSP收敛，1 epoch饱和；相同FLOPs下MAE→WSP比纯WSP高效最高2×；在更小更干净的IN21k数据集上同样有效，验证跨分布鲁棒性。

## 相关工作脉络
- **SWAG / WSP [70]**：本文核心基线，使用相同IG-3B与超参但从随机权重训练；MAE→WSP在完全一致设置下额外获得性能提升，证明初始化而非数据/任务设计的差异是增益来源。
- **MAE [33]**：原始工作证明自监督重建随模型规模缩放；本文将其定位为“初始化器”而非“替代方案”，并补充证明其随数据集规模缩放。
- **DINOv2 [56]**：自监督强表征代表；本文指出MAE→WSP在冻结特征线性探测（IN1k 87.0-88.1%）上超越DINOv2，说明弱监督信号仍可突破纯自监督上限。
- **Scale-ViT [84]**：基于JFT-3B的强监督工作，IN1k fine-tune略领先（90.5% vs 89.7%），但本文强调JFT-3B的数据特权，而IG-3B更开放且MAE→WSP在零样本与检测上反超。
- **LiT [85] & Intermediate Finetuning [5, 50]**：LiT用于图文对齐，本文将其与MAE→WSP串行结合；相比预训练后的中间对齐，pre-pretraining作用于更早阶段，避免引入额外标注成本。
- **Vision Transformer [23]**：所有实验基于ViT架构，验证了两阶段范式与主流视觉骨干的广泛兼容性。

## 局限性与未来方向
- **数据质量瓶颈**：零样本与细粒度上限仍受预训练数据分布影响（如JFT-3B+ALIGN在IN1k零样本上仍领先），说明纯算法改进无法完全弥补高质量多模态数据的缺失。
- **任务间初始化敏感度差异**：纯WSP随模型增大检测AP不提升，必须配合MAE初始化才能打破瓶颈，提示不同下游任务对特征先验的需求机制尚需理论刻画。
- **训练阶段设置固定**：仅验证1 epoch MAE + 1 epoch WSP的组合，更长序列、异步交替或多阶段调度策略的效果未充分探索。
- **视频/时序
