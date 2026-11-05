---
title: "SegGPT-Towards-Segmenting-Everything-In-Context"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Wang_SegGPT_Towards_Segmenting_Everything_in_Context_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 00:54:56"
field: "视觉通用模型"
keywords: ["通用分割", "In-Context Learning", "Random Coloring", "Video Object Segmentation", "Few-shot Segmentation", "Vision Generalist"]
innovations: ["随机着色 in-context 训练防止多任务坍缩", "Feature Ensemble 在中间特征层集成多示例", "In-Context Tuning 冻结参数仅优化可学习 prompt 张量"]
benchmarks: ["COCO-20i", "FSS-1000", "YouTube-VOS 2018", "DAVIS 2017", "MOSE", "ADE20K"]
---

# 论文速读：SegGPT: Towards Segmenting Everything In Context

## 一句话总结
SegGPT 是首个能够根据上下文示例自动完成任意分割任务的通用分割模型，通过随机着色（random coloring）的 in-context 训练方案，将语义分割、实例分割、部件分割、视频对象分割等多样任务统一为 in-context coloring 问题，无需修改架构即可零样本/少样本推理。

## 研究问题与动机
1. 现有分割模型均为专用模型（specialist），针对特定任务、类别、粒度或数据类型训练，难以泛化到新任务或新域
2. 为不同设定重新训练模型需要大量标注数据和计算成本，不具备可持续性
3. 核心挑战一：如何将部件、语义、实例、全景、医疗、航空等不同类型数据统一纳入同一训练框架
4. 核心挑战二：如何设计区别于传统多任务学习的可泛化训练方案，使其灵活定义任务并能处理域外（out-of-domain）数据

## 核心贡献（创新点）
1. **首个单一通用分割模型**：证明单模型可自动执行多样化的分割任务，区别于传统需要为不同任务训练不同模型的方案
2. **随机着色 in-context 训练方案**：对每个样本随机重映射颜色空间，迫使模型依赖上下文语义而非固定颜色，本质区别在于防止多任务学习中常见的"颜色坍缩"（color collapse）
3. **两种 Context Ensemble 策略**：提出 Spatial Ensemble 和 Feature Ensemble 两种多示例集成方式，后者在中间特征层面交互，可支持任意数量 prompt 及视频时序建模
4. **In-Context Tuning**：冻结模型参数仅优化可学习图像张量作为 task prompt，可在不损失泛化能力的前提下适配特定数据集/场景/角色

## 方法详解
SegGPT 基于 Painter 框架，将视觉任务输出空间重定义为"图像"，统一为 masked image inpainting 问题。训练时随机遮蔽输出图像并重建缺失像素，使用 vanilla ViT-L（307M 参数）和 smooth-L1 loss。

关键设计如下：

1. **In-Context Coloring（随机着色训练）**：不同于 Painter 的预定义固定颜色空间（每个语义类别对应固定颜色），SegGPT 对每个训练样本随机采样一张共享相似上下文（如同类类别或同实例）的图像，再随机采样一组颜色并重映射，生成 in-context pair。引入 mix-context 训练：将多个相同颜色映射的图像拼接后随机裁剪，使模型聚焦于上下文关系而非特定颜色。

2. **Context Ensemble（多示例推理集成）**：
   - **Spatial Ensemble**：将多个示例拼成 $n \times n$ grid 后下采样到单示例尺寸，与 in-context coloring 直觉一致
   - **Feature Ensemble**：多个示例在 batch 维度独立计算，仅在每个 attention 层后对查询图像特征取平均，使查询图像从多个参考示例中聚合信息

3. **In-Context Tuning**：冻结整个预训练模型，初始化一个可学习图像张量作为 input context，仅优化该张量以适配特定应用（如 ADE20K 类别、特定场景或人物），推理时作为 plug-and-play prompt 使用

## 实验与结果
**训练数据**：ADE20K、COCO、PASCAL VOC、Cityscapes、LIP、PACO、视网膜血管分割（CHASE DB1/DRIVE/HRF/STARE）、航空图像（iSAID/loveDA）

**训练细节**：ViT-L 编码器（307M 参数），从 Painter checkpoint 初始化；AdamW，base LR 1e−4 cosine decay，weight decay 0.05；batch size 2048，9K iterations（warmup 1.8K），输入 448×448

**Few-shot 语义分割**：
- COCO-20^i：one-shot 56.1 / few-shot 67.9，超越 Painter（32.8/32.6），与 specialist FPTrans（56.5/65.5）相当或更优
- FSS-1000（未在其上训练）：one-shot 85.6 / few-shot 89.3，远超 Painter（61.7/62.3）

**Video Object Segmentation（未使用视频数据训练）**：
- YouTube-VOS 2018：J&F 74.7，超越 AGAME/AGSS 等专用模型
- DAVIS 2017：J&F 75.6（8 frames + Feature Ensemble）
- MOSE：J&F 45.1，与 SOTA RDE（48.8/52.9）差距不大

**In-Context Tuning**：ADE20K 语义分割 mIoU 39.6（低于 specialist RefineNet 40.7，但低于 Painter 的 49.9）；COCO 全景分割 PQ 34.4（低于 Painter 的 43.4）

**消融实验**：Feature Ensemble 在高分辨率 DAVIS 上显著优于 Spatial Ensemble（因后者下采样导致信息损失）；最优帧数为 8

## 相关工作脉络
1. **Painter [50]**：SegGPT 的直接基础框架，但 Painter 使用预定义固定颜色空间易坍缩为多任务学习，SegGPT 引入随机着色从根本上改进训练方式
2. **Pix2Seq [8, 9]**：将视觉任务统一为离散 token 序列的自回归生成，与 SegGPT 的连续像素 in-painting 范式不同
3. **OFA [49] / Unified-IO [37]**：跨任务序列到序列建模依赖特殊 token 作为硬指示，难以泛化到新任务；SegGPT 通过 in-context 示例而非硬指示实现任务泛化
4. **UViM [27]**：统一像素标注任务但为每个任务训练单独模型；SegGPT 用单模型统一所有分割任务
5. **Visual Prompting [3]**：基于 in-painting 的视觉 prompt，Segment 的早期探索；SegGPT 将其扩展至全分割家族并引入随机着色与多示例集成
6. **HODOR [1]**：无需视频数据训练的 VOS 专用模型；SegGPT 在同一设置下实现通用性，可同时处理图像和视频分割

## 局限性与未来方向
1. 随机着色增加了训练难度，导致在域内充足数据任务（ADE20K、COCO 全景）上表现不如专家模型，且明显低于其前身 Painter
2. 未在 FSS-1000 和 VOS 测试集上训练却能取得强结果，但域内微调后性能反而下降，说明随机着色对 in-domain 精细适配存在一定障碍
3. 模型规模受限于数据规模，扩大模型需配套更大规模数据，自监督学习可能是潜在解决方向
4. 本文明确不以超越专家模型 SOTA 为目标，定位在通用性而非单项精度

## 研究启发与可借鉴点
1. **随机化标签/输出映射防止坍缩**：在多任务统一训练中，通过随机着色/随机编码迫使模型学习语义关系而非依赖固定模式，可迁移至其他统一训练场景
2. **Feature Ensemble 的中间特征集成**：在 Transformer 各 attention 层对多示例查询特征取平均，比 logit-level ensemble 更高效且可扩展至任意数量 prompt，值得在其他 in-context 视觉学习任务中验证
3. **In-Context Tuning 范式**：冻结预训练主干、仅优化可学习图像张量作为 prompt，兼顾泛化性与低资源适配能力，可与本团队的提示学习/自适应方法结合
4. **统一输出空间设计**：将多样分割数据通过变换为统一图像格式纳入训练，消除手动 label merging 成本，该方法论可推广至深度估计、边缘检测等其他密集预测任务

## 关键术语表
**In-Context Learning**：通过向模型提供上下文示例（而非梯度更新）使模型执行新任务的学习范式，源于 GPT-3 的自然语言学习
**Random Coloring**：对每个训练样本随机重映射颜色空间的策略，防止模型依赖固定颜色作为任务指示而坍缩为多任务学习
**Feature Ensemble**：在 Transformer attention 层的中间特征上对多示例查询特征取平均的集成策略，支持任意数量 prompt 且可建模时序关系
**Spatial Ensemble**：将多个示例拼接成 $n \times n$ grid 后下采样为单示例尺寸的集成策略，适合低分辨率输入但高分辨率下损失信息
**In-Context Tuning**：冻结预训练模型参数，仅优化一个可学习的图像张量作为 task prompt，实现无需参数更新的任务定制
**Smooth-L1 Loss**：结合 L1 和 L2 优点的回归损失，对异常值具有鲁棒性，用于像素级颜色重建
**Few-shot Semantic Segmentation**：仅用 1 或 K 个标注样本即可对目标类别进行像素级分割的任务，分为 in-domain（测试类在训练集中）和 out-of-domain
**Video Object Segmentation (VOS)**：给定首帧或少数帧的目标掩码，在视频其余帧中追踪并连续分割该目标的稠密预测任务

## 可复现要素
- **数据集**：ADE20K、COCO、PASCAL VOC、Cityscapes、LIP、PACO、CHASE DB1、DRIVE、HRF、STARE、iSAID、loveDA、YouTube-VOS 2018、DAVIS 2017、MOSE、FSS-1000（均已公开）
- **代码**：开源，地址 https://github.com/baaivision/Painter
- **权重**：使用 Painter [50] 预训练 checkpoint 初始化（开源）
- **关键超参**：ViT-L（307M 参数）、batch size 2048、LR 1e−4 cosine scheduler、9K iterations（warmup 1.8K）、输入尺寸 448×448、AdamW optimizer、weight decay 0.05
