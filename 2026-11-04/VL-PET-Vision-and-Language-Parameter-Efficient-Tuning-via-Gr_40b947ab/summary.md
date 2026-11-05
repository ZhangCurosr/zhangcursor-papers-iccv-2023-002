---
title: "VL-PET-Vision-and-Language-Parameter-Efficient-Tuning-via-Gr"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Hu_VL-PET_Vision-and-Language_Parameter-Efficient_Tuning_via_Granularity_Control_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 08:26:49"
field: "多模态参数高效微调"
keywords: ["Parameter-Efficient Tuning", "Vision-and-Language", "Granularity Control", "Encoder-Decoder", "Multi-modal Learning"]
innovations: ["提出粒度控制机制对PET模块修改幅度进行动态调控", "设计编码器-解码器轻量级差异化PET模块", "将粒度控制与轻量设计迁移至现有PET方法实现通用增强"]
benchmarks: ["VQAv2", "GQA", "NLVR²", "MSCOCO", "TVQA", "How2QA", "TVC", "YC2C"]
---

# 论文速读：VL-PET: Vision-and-Language Parameter-Efficient Tuning via Granularity Control

## 一句话总结
本文提出了 VL-PET 框架，通过粒度控制机制和轻量级模块化设计，有效解决视觉-语言（VL）参数高效微调中模块化修改过度及编码器-解码器功能差异被忽视的问题，在图像/视频-文本任务上显著优于 SOTA 方法。

## 研究问题与动机
- **参数高效微调在 VL 领域的潜力未被充分探索**：现有 VL PET 方法大多直接迁移 NLP/CV 的模块化修改，缺乏 VL 特定设计，且主要聚焦判别式任务，限制了 PLM 的泛化能力。
- **过度模块化修改导致性能退化**：现有方法（如 VL-Adapter）将大量可训练模块直接嵌入 PLM，影响中间输出稳定性，却未对模块化修改施加有效控制。
- **编码器与解码器的功能差异被忽视**：编码器负责 VL 对齐与建模，解码器负责条件自回归文本生成，但现有方法对两者施加相同的模块化修改，导致次优性能。

## 核心贡献（创新点）
- **提出粒度控制机制**：通过不同粒度级别（large/middleX/middleY/small）生成可控矩阵 G，实现对 PET 模块修改幅度的有效调控，与 VL-Adapter 等无控制直接插入的方式形成本质区别。
- **提出轻量级 PET 模块设计**：针对编码器在 self-attention 和 feed-forward 中集成较重的 PET 模块以增强 VL 对齐，在解码器中仅在 cross-attention 的 value 矩阵嵌入轻量模块以保留文本生成能力，区别于传统方法对所有模块一视同仁的设计。
- **构建模型无关的 VL-PET 框架**：结合多头部模块化修改与粒度控制，可实例化多种 VL-PET 模块，支持效率与效果的灵活权衡，与单一固定结构的已有方法不同。
- **验证设计对现有 PET 方法的增强效果**：将粒度控制与轻量设计迁移至 Compacter 和 VL-Adapter，显著提升其性能并减少参数量，证明设计的通用性。

## 方法详解
**统一 PET 框架**：将 PET 模块视为对冻结 PLM 模块的输出增加修改量 ΔH，即 H ← H + ΔH。

**粒度控制机制**（Sec. 3.2）：
- 引入控制矩阵 G ∈ R^{N×d}，公式为 H ← G ⊙ (H + ΔH)，其中 ⊙ 为逐元素乘积。
- 四个粒度级别：
  - **Large**：G_large = s·σ(φ(X·W_down)·W_up)，参数量 O(dr)，通过瓶颈结构生成完整控制矩阵。
  - **MiddleX**：先学习 G̃_middleX ∈ R^{N×1}，再广播至 R^{N×d}，参数 O(d)。
  - **MiddleY**：使用可学习向量 z ∈ R^{1×d}，经 sigmoid 后广播，参数 O(d)。
  - **Small**：Concat(X,H) 投影后均值池化得到标量，参数 O(d)。

**轻量级 PET 模块设计**（Sec. 3.3）：
- **编码器**：在 self-attention 和 feed-forward 模块中集成 instantiated VL-PET 模块，增强 VL 对齐与建模。
- **解码器**：仅在 cross-attention 的 **value 矩阵**中嵌入 VL-PET 模块（G=1 以节省参数），避免破坏预训练文本生成能力。
- **多头部模块化修改**：ΔH' = φ(Concat(X'·W_down^(1), ..., X'·W_down^(Nh)))·W_up，提升表达能力。
- **Layer Normalization 策略**：编码器 LN 可训练，解码器 LN 冻结，优于全训练或全冻结。

**多任务学习**：共享 PET 模块，通过统一文本生成方式联合训练多个 VL 任务，获取跨任务知识。

## 实验与结果
- **数据集**：图像-文本（VQAv2, GQA, NLVR², MSCOCO）；视频-文本（TVQA, How2QA, TVC, YC2C，来自 VALUE benchmark）。
- **基线**：Full Fine-tuning, BitFit, Prompt Tuning, Compacter, HyperFormer, LoRA, VL-Adapter, LST。
- **BART-base  backbone**：
  - VL-PET_large 平均准确率 **79.18%**，超越 VL-Adapter（76.93%，+2.92%）和 LoRA（76.60%，+3.37%），参数量更少（4.16% < 5.93%）。
  - COCO Captioning 提升尤为显著（122.03 CIDEr vs. 114.61）。
- **T5-base backbone**：
  - VL-PET_large 平均 **79.52%**，超越 VL-Adapter（76.90%，+3.41%）和 LoRA（74.30%，+7.03%）。
  - 其他 PET 方法在 T5 上未见类似提升，部分甚至退化。
- **视频-文本任务**：VL-PET_large 平均 88.63%，超越 VL-Adapter（87.95%，+0.77%）和 LoRA（83.77%，+5.80%）。
- **迁移验证**：将 VL-PET 设计应用于 Compacter 和 VL-Adapter，均获得显著提升（如 Compacter + G_large + LW 提升 2.56%，参数量从 2.70% 降至 3.28%）。

## 相关工作脉络
- **VL-Adapter [49]**：当前 SOTA PET 方法，直接迁移 NLP Adapter 至 VL encoder-decoder，对所有模块施加相同修改，忽视编码器-解码器功能差异和修改控制，本文在其基础上通过粒度控制和轻量设计实现显著提升。
- **LoRA [16]**：NLP 中的低秩适配方法，本文证明其在 VL 任务中因缺乏 VL 特异性设计而劣于 VL-PET。
- **Compacter [21]**：超复数适配器，本文验证其可通过引入粒度控制机制获得进一步增益。
- **BitFit [54] / Prompt Tuning [26]**：参数更少的 PET 方法，但在 VL 任务上性能明显落后，说明简单策略不足以弥合模态鸿沟。
- **LST [48]**：阶梯侧向微调，在 T5 上作为强基线，VL-PET 仍以更少参数超越。
- **Uniadapter [37] / MAPL [39]**：其他 VL 适配方法，多为 prompt-based 或专注判别任务，本文聚焦 generative encoder-decoder 架构上的全面改进。

## 局限性与未来方向
- 论文主要聚焦 encoder-decoder PLM（BART/T5），对于 decoder-only 或 encoder-only 架构的适配性未充分讨论。
- 粒度控制机制的超参数（如缩放因子 s、投影维度 r）需针对不同骨干网络调优，通用性有待进一步验证。
- 视频任务实验仅使用单 seed（受限于 VALUE benchmark 提交要求），统计显著性较弱。
- 未来可将粒度控制与轻量设计推广至 NLP 和 CV 等其他领域，探索跨域通用性。

## 研究启发与可借鉴点
- **编码器-解码器差异化设计**：根据模块功能分配不同强度的 PET 修改，这一思路可迁移至任何 encoder-decoder 架构的多模态或生成任务。
- **粒度控制矩阵的泛化价值**：G ⊙ (H + ΔH) 的控制范式可作为通用组件嵌入其他 PET 方法，提升其对修改幅度的调控能力。
- **多任务联合训练的有效性**：共享 PET 模块的多任务学习策略在 VL 任务中显著优于单任务，对多任务下游应用具有参考价值。
- **Cross-attention Value 矩阵的精细控制**：仅在 value 矩阵嵌入 PET 模块的设计，为如何最小化对预训练知识的干扰提供了新思路。
- **消融实验的系统性**：对粒度级别、模块位置、LN 策略、多任务学习等进行全面消融，为后续工作提供了可靠的评估基准。

## 关键术语表
- **Parameter-Efficient Tuning (PET)**：通过微调少量参数而非全参数微调来实现高效迁移学习的技术。
- **Granularity-controlled Matrix (G)**：控制 PET 模块修改幅度的权重矩阵，通过不同粒度级别生成，实现对各隐藏状态元素的重要性动态分配。
- **Lightweight PET Module Design**：针对编码器（增强 VL 对齐）和解码器（保持文本生成）的不同功能需求，差异化集成 PET 模块的策略。
- **VL-Adapter**：将 NLP 中的 Adapter 直接迁移至 VL encoder-decoder 的现有 SOTA 方法，缺乏 VL 特异性设计。
- **Multi-head Modular Modification**：将 PET 模块以多头部方式并行应用，增强表达能力并提升稳定性。
- **Cross-attention Value Matrix**：解码器中用于融合编码器视觉-语言表征的 value 投影矩阵，本文在此处嵌入轻量 PET 模块。

## 可复现要素
- **数据集**：VQAv2, GQA, NLVR², MSCOCO, TVQA, How2QA, TVC, YC2C（VALUE benchmark），均为公开数据集。
- **代码**：已开源，链接 https://github.com/HenryHZY/VL-PET。
- **模型权重**：论文未提及公开权重。
- **关键超参**：缩放因子 s（针对不同 PLM 专门设置）、投影维度 r、头部数 N_h（最优为 4）、粒度级别选择。
