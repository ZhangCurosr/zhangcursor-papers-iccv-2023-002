---
title: "Towards-Deeply-Unified-Depth-aware-Panoptic-Segmentation-wit"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/He_Towards_Deeply_Unified_Depth-aware_Panoptic_Segmentation_with_Bi-directional_Guidance_Learning_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 21:20:23"
field: "多模态视觉理解"
keywords: ["depth-aware panoptic segmentation", "uniﬁed architecture", "bi-directional guidance", "geometric query enhancement", "cross-modality learning", "monocular depth estimation"]
innovations: ["提出深层统一架构，使用统一per-segment queries同时完成全景分割和实例级深度估计", "设计几何查询增强模块，通过隐式表征中介将深度特征融合进统一查询", "提出双向引导学习，利用语义-深度互促关系优化跨模态特征表示"]
benchmarks: ["Cityscapes-DVPS", "SemKITTI-DVPS"]
---

# 论文速读：Towards-Deeply-Unified-Depth-aware-Panoptic-Segmentation-wit

## 一句话总结
本文提出了一种深层统一框架，用于深度感知全景分割（Depth-aware Panoptic Segmentation, DPS），通过在架构层面使用统一查询（unified queries）并在训练阶段引入双向引导学习（bi-directional guidance learning），实现了语义分割与深度估计的深度融合，在 Cityscapes-DVPS 和 SemKITTI-DVPS 数据集上均达到 SOTA。

## 研究问题与动机
- **核心问题**：深度感知全景分割（DPS）需要同时输出实例级语义掩码和对应的深度图，现有统一架构方法（如 PolyphonicFormer、PanopticDepth）虽然在架构层面进行了统一，但学习过程仍相互独立，仅使用任务特定的损失函数分别优化，忽视了语义与几何之间的互促关系。
- **现有方法不足**：早期方法（ViP-DeepLab、PanopticDepth）采用独立分支设计；近期方法（PolyphonicFormer）使用任务特定的 queries 分别预测掩码和深度，虽实现了统一输出但缺乏跨模态特征交互；已有的单向指导方法（如 SGT）仅利用语义引导深度，未探索反向关系。
- **关键洞察**：跨越语义边界的像素往往在深度上存在显著差异，反之亦然——这一互促关系未被现有方法充分利用。

## 核心贡献（创新点）
1. **深层统一架构（Deeply Unified Architecture）**：采用相同的 per-segment queries 同时完成全景分割和实例级深度估计，而非分别学习独立的 queries，本质区别在于从查询层面实现了真正的统一而非仅输出层面的统一。
2. **几何查询增强（Geometric Query Enhancement）**：引入固定大小的隐式表征（latent representations）作为中介，通过掩码感知的交叉注意力与自注意力机制，将多尺度深度特征融合进统一查询中，实现了几何信息对查询的显式增强。
3. **双向引导学习（Bi-directional Guidance Learning）**：提出语义→深度引导（基于对比学习的 max-min 三元组策略）和深度→语义引导（基于深度连续性的语义特征对齐），实现跨模态特征的同时优化，而不仅限于单向指导。
4. **备份查询（Backup Query）**：针对低置信度掩码过滤导致的深度图空白区域，设计全局深度感知的备份查询进行补充，有效缓解了分割误差对深度估计的传递影响。

## 方法详解
- **编码器-解码器架构**：共享编码器提取特征，两个任务特定解码器生成语义特征金字塔 $\mathcal{F}$（×1/8, ×1/16, ×1/32）和深度特征金字塔 $\mathcal{F}_d$，以及像素嵌入 $\mathcal{E}_{pixel}$ 和深度嵌入 $\mathcal{E}_{depth}$。
- **统一 per-segment queries**：采用 9 层 Transformer Decoder 处理 per-segment queries $X_o^l$，通过 masked attention 与多尺度语义特征交互，最终通过 MLP + Softmax 预测分类概率，并通过 dot product 与像素嵌入生成二元掩码 $M = \sigma(\text{MLP}(X_o^l) \otimes \mathcal{E}_{pixel})$。
- **几何查询增强模块**：
  1. 初始化固定大小隐式表征 $\mathcal{R}^0$
  2. 对多尺度深度特征 $\mathcal{F}_d$ 应用 masked cross-attention（使用分割掩码约束注意力范围）
  3. 在隐式表征空间执行 self-attention 更新为 $\mathcal{R}^l$
  4. 对原始查询 $X_o^l$ 与更新后的 $\mathcal{R}^l$ 执行 cross-attention，得到几何增强查询 $X_d^l$
- **深度图聚合**：对增强查询执行 $d = D_{max} \times \sigma(\psi(X_d^l) \otimes \mathcal{E}_{depth})$（$D_{max}=80$），并按高置信度掩码聚合：$D(u,v) = \sum_{i \in \mathcal{H}} d_i(u,v) \cdot \mathbb{1}[M_i(u,v) > 0.5]$
- **备份查询**：无掩码约束地查询深度特征进行全局深度感知，dot product 生成补充深度值，覆盖空白区域
- **双向引导损失**：
  - **Semantic-to-Depth Guidance**：在 $K \times K$ 局部块内，计算块中心像素与同实例正样本的最大距离 $d_{max}^+$ 和跨实例负样本的最小距离 $d_{min}^-$，采用 triplet loss：$\mathcal{L}_{sg} = \frac{1}{N}\sum_i \max(0, \alpha + d_{max}^+(i) - d_{min}^-(i))$
  - **Depth-to-Semantic Guidance**：以中心像素为参考，计算相对深度距离与语义特征距离的联合权重：$\mathcal{L}_{dg}(i,j) = -\frac{1}{N}\sum_i\sum_j e^{-\|\hat{d}_i - \hat{d}_j\|/\tau} \cdot e^{-\|\mathcal{F}_i - \mathcal{F}_j\|_2}$，其中 $\tau=10$
- **总损失**：$\mathcal{L} = \lambda_{cls}\mathcal{L}_{cls} + \lambda_{mask}\mathcal{L}_{mask} + \lambda_{depth}\mathcal{L}_{depth} + \lambda_{sg}\mathcal{L}_{sg} + \lambda_{dg}\mathcal{L}_{dg}$，权重分别为 2、5、2.5、0.1、0.1；深度损失采用 scale-invariant loss（$\lambda=0.85$）

## 实验与结果
- **数据集**：Cityscapes-DVPS（3000 帧，含 disparity 转换的深度标注）和 SemKITTI-DVPS（19130 训练帧，稀疏语义 + 投影深度标注）
- **评估指标**：DPQ（Depth-aware Panoptic Quality），阈值 $\lambda \in \{0.1, 0.25, 0.5\}$，以及子任务的 PQ、abs_rel、RMSE 等
- **主要结果（Cityscapes-DVPS，ResNet-50）**：DPQ 达到 63.0（$\lambda=0.5$ 时为 69.3，$\lambda=0.25$ 时为 61.4，$\lambda=0.1$ 时为 52.8），超越 PanopticDepth（52.3）和 PolyphonicFormer（59.8），且 FLOPs 更低（510G vs 619G/680G）
- **主要结果（SemKITTI-DVPS，ResNet-50）**：DPQ 达到 47.9，超越所有基线
- **Swin-B  backbone**：DPQ 达到 64.3（Cityscapes）和 52.8（SemKITTI），均达到 SOTA
- **子任务提升**：单目深度估计 abs_rel 达 0.0597（超越 PolyphonicFormer 的 0.0647）；全景分割 PQ 达 69.5
- **消融结果**：逐组件验证有效，双向引导带来约 +1.8% DPQ 提升；不完全监督实验显示双向引导可弥补约 2.7% DPQ 差距
- **计算成本**：额外计算开销 < 20%，推理时无额外开销（引导学习仅在训练阶段使用）

## 相关工作脉络
1. **ViP-DeepLab [44]**：最初提出 DPS 任务的基准方法，扩展 Panoptic-DeepLab 添加深度预测头，本文在其基础上实现更深层的统一而非简单的头附加
2. **PanopticDepth [40]**：使用动态实例特定核（kernels）进行联合预测，本文采用统一查询 + 几何增强而非任务特定核
3. **PolyphonicFormer [55]**：采用 query-based 学习并设计查询链接（query linking）机制，本文通过隐式表征中介实现更简洁的几何注入，且引入双向引导
4. **SGT [23]**：语义引导的三元组深度表示学习（单向），本文扩展为双向引导并引入 max-min 策略优化边界感知
5. **MaskFormer/Mask2Former [8, 9]**：本文采用其 mask classification 范式作为分割基础，扩展至深度估计任务
6. **SDC-Depth [49]**：分解为类别特定深度预测任务，本文采用实例级统一预测而非类别分解策略

## 局限性与未来方向
- **掩码错误传播**：不准确的掩码预测会直接影响深度图质量，依赖高质量的分割预训练（Fig. 5 所示）
- **泛化能力受限**：模型可能难以推广到未见过的物体类型或场景
- **稀疏标注挑战**：在 SemKITTI-DVPS 等稀疏标注数据集上性能仍有提升空间
- **未来方向**：作者提出目标是构建完全统一的框架（fully unified framework），进一步消除任务间残余差异

## 研究启发与可借鉴点
1. **隐式表征中介设计**：使用固定大小 latent representations 作为跨模态通信的中间层，比直接 query linking 更简洁高效，可迁移到其他多任务统一架构
2. **双向引导学习范式**：不仅利用语义指导深度，也利用深度连续性指导语义特征，这种互促学习模式可推广至其他几何-语义联合任务（如法线估计、三维重建）
3. **backup query 设计**：针对低置信度过滤导致的空白区域问题，引入全局感知的备用查询进行补充，该策略可用于处理任何实例级预测中的缺失覆盖问题
4. **不完全监督鲁棒性**：实验证明双向引导在仅有一半标注时可显著提升性能，对于实际场景中标注成本高昂的情况具有重要价值
5. **超参数鲁棒性**：统一框架使超参数敏感度降低，可参考其训练策略（两阶段：50 epoch 分割预训练 + 10 epoch 联合微调）

## 关键术语表
**Depth-aware Panoptic Segmentation (DPS)**：从单张图像同时输出实例级语义掩码和对应深度图的统一任务，用于 3D 场景理解。
**Per-segment Queries**：每个查询对应一个分割实例的 learnable embedding，同时服务于语义分类和深度预测。
**Geometric Query Enhancement**：通过隐式表征和掩码感知的 cross-attention 将多尺度深度特征注入统一查询的过程。
**Bi-directional Guidance Learning**：同时利用语义→深度和深度→语义的交叉监督，优化中间特征表示的双向引导训练策略。
**Semantic-to-Depth Guidance**：基于 contrastive learning 的 max-min 三元组损失，强制同实例深度特征距离小于跨实例距离。
**Depth-to-Semantic Guidance**：利用深度连续性约束语义特征平滑性的损失，使深度相近的像素在语义特征空间也更接近。
**Latent Representation**：固定大小的 learnable embedding，作为几何信息与 per-segment queries 之间的通信中介。
**DPQ (Depth-aware Panoptic Quality)**：DPS 任务的综合评估指标，结合了分割质量和深度准确度，在多个相对深度误差阈值下评估。

## 可复现要素
- **数据集**：Cityscapes-DVPS 和 SemKITTI-DVPS（基于 Cityscapes 和 KITTI 开源数据集扩展）
- **代码**：已开源 https://github.com/jwh97nn/DeepDPS
- **模型权重**：已提供（见 GitHub 链接）
- **关键超参**：Transformer Decoder 层数 9 层；$D_{max}=80$；$\lambda_{depth}=0.85$；损失权重 $\lambda_{cls}=2, \lambda_{mask}=5, \lambda_{depth}=2.5, \lambda_{sg}=\lambda_{dg}=0.1$；引导损失中 $\tau=10$；训练分两阶段（50 epoch 预训练 + 10 epoch 微调）；输入尺寸 512×1024（Cityscapes）/ 384×1280（SemKITTI）
