---
title: "MeViS-A-Large-scale-Benchmark-for-Video-Segmentation-with-Mo"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Ding_MeViS_A_Large-scale_Benchmark_for_Video_Segmentation_with_Motion_Expressions_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 00:52:35"
---

# 论文速读：MeViS-A-Large-scale-Benchmark-for-Video-Segmentation-with-Mo

## 一句话总结
本文构建了大规模动作表达视频分割基准数据集 MeViS，聚焦以纯运动/动作为主的自然语言指令在复杂视频中进行目标定位与分割；通过系统性评测现有 RVOS 方法与自研基线 LMPM，揭示了当前方法在运动表达引导下的显著性能瓶颈，并为后续研究提供了数据、指标与基础方案。

## 研究问题与动机
- 现有 RVOS 数据集（如 DAVIS17-RVOS、Refer-Youtube-VOS、A2D Sentence）的表达高度依赖静态属性（颜色、类别、位置），目标往往单帧即可识别，严重弱化了对视频时序运动信息的利用。
- 真实场景中大量目标需依赖跨帧动作轨迹才能区分（如“飞走的鸟”“回头的小长颈鹿”），静态特征在复杂多目标共存的画面中失效。
- 现有 SOTA 方法普遍随机采样少量帧，缺乏对全片长程运动与全局时序上下文的显式建模，面对运动表达任务时性能骤降。
- 缺乏专门针对 motion expression 的大规模评测基准与规范化的标注/验证协议，阻碍了该子方向的发展。

## 核心贡献（创新点）
1. **提出 MeViS 大规模运动表达数据集**：收录 2,006 个视频、8,171 个目标与 28,570 条运动表达，首次以动作/轨迹描述为核心线索，区别于以往依赖静态特征的 RVOS 数据集。
2. **引入 multi-object expression 设定**：单条自然语言可同时指代多个目标（平均 1.59 个/表达），更贴近真实应用场景并显著提升任务复杂度。
3. **设计基线模型 LMPM**：融合语言引导提取、跨帧 motion perception 与相似度阈值匹配机制，可稳定输出单目标或多目标轨迹，弥补现有方法对长程动态建模的不足。
4. **建立完整的 benchmark 与诊断协议**：评测 5 种主流 RVOS 方法，结合时序上下文增量实验与跨数据集交叉验证，量化揭示了静态表达与运动表达之间的性能鸿沟。

## 方法详解
- **Language-Guided Extractor**：使用 $N_1$ 个语言 query 对视频 T 帧进行图文交互，过滤无关背景，输出轻量级 object embeddings（借鉴 VITA [15]），避免使用计算密集的 object feature maps。
- **Motion Perception**：在所有帧的 object embeddings 之间执行跨帧自注意力（inter-frame self-attention），捕获短促动作与长程连续动作的全局时序上下文，使表征具备运动演变信息。
- **Transformer Decoder 轨迹预测**：以 $N_2$ 个语言 query 为 query，将经 Motion Perception 的 object embeddings 作为 key/value，解码生成各候选目标的完整轨迹嵌入。
- **阈值匹配输出（Matching）**：计算语言特征与预测轨迹的相似度，通过阈值 $\sigma$ 筛选匹配项，支持动态数量输出，而非仅取 top-1。
- **实现细节**：主干为 Tiny Swin Transformer，文本编码器为冻结的 RoBERTa；Motion Perception 6 层，Decoder 3 层；训练 150,000 迭代，AdamW，学习率 5e-5；推理时输入最短边 resize 至 448；超参 $\sigma=0.8$，$N_1=20$，$N_2=10$。

## 实验与结果
- **数据集划分**：训练 1,712 / 验证 140 / 测试 154 个视频；平均每个视频 4.28 个目标，平均视频时长 13.16 秒，目标平均持续 10.88 秒。
- **评估指标**：$\mathcal{J}$（区域 IoU）、$\mathcal{F}$（轮廓 F-measure），综合报告 $\mathcal{J}\&\mathcal{F}$。
- **必要性实验（表 2/3）**：
  - 纯图像方法 VLT 在 DAVIS17-RVOS 与 Refer-Youtube-VOS 上分别达 60.4% / 63.1% J&F，但在 MeViS 上仅 27.8%；加入全局时序上下文（+TC）后提升至 35.5%，证明时序建模对运动表达至关重要。
  - 在 Ref-COCO/+/g 图像数据上训练的模型，测试到 MeViS 上仅 22.5%（VLT）/ 27.0%（ReferFormer），显著低于在其他视频 benchmark 的表现，印证静态 vs. 运动表达的 distribution gap。
- **LMPM Ablation（表 4）**：仅语言 query + 5 帧随机采样（31.0%）→ +Motion Perception（+5.3% 至 36.3%）→ +阈值匹配多目标输出（+0.9% 至 37.2%）。
- **Benchmark 结果（表 5）**：URVOS、LBDT、MTTR、ReferFormer、VLT+TC 在 MeViS 验证集上 J&F 落于 27.8%~35.5%，远低于其在传统 benchmark 的 >60%；本文 LMPM 取得最佳 37.2% J&F（$\mathcal{J}=34.2$，$\mathcal{F}=40.2$），但仍存在较大提升空间。
- **可视化分析**：成功处理“移至前景”“离开屏幕”等长程动作；失败集中于目标消失后重出现（全局时序理解不足）及复杂纠缠运动中的多目标丢失。

## 相关工作脉络
1. **Referring Image Segmentation（RIS）**：VLT [10]、LAVT [59]、CRIS [52] 等依赖单帧图文融合，擅长静态属性定位，但在 MeViS 上因缺失时序建模而性能暴跌，凸显静态方法向视频延伸的局限。
2. **早期 RVOS 数据集**：A2D Sentence [13]、DAVIS-RVOS [21] 以演员/短程动作为主，表达含明显类别或静态线索，单目标主导，与 MeViS 的开放世界多目标运动设定差异显著。
3. **Refer-Youtube-VOS / URVOS [44]**：虽提供 full-video 表达，但仍频繁混入颜色、位置等静态线索；单表达严格对应单目标，缺乏多目标表达的自然性。
4. **Transformer-based RVOS**：ReferFormer [55]、MTTR [2] 通过随机采样少量帧进行时空特征提取，未显式建模全片长程运动，面对 MeViS 跨越多帧的动作描述时易遗漏关键信息。
5. **视频实例 Token 化表示**：VITA [15]、TubeLink [25] 提出用 object embedding/token 替代密集特征图以提升效率，本文 LMPM 沿用该设计降低计算开销并强化 instance 区分度。
6. **跨模态预训练对齐**：CLIP [43] 等提供图文联合表征基础，但本文强调需在时序维度显式建模 motion expression 与视频轨迹的动态对应关系，而非仅依赖静态跨模态相似度。

## 局限性与未来方向
- **数据集局限**：视频仍主要来自现有高质量 VOS 数据集，领域与场景多样性受限；多目标表达的比例分布与复杂程度未作细粒度统计。
- **方法局限**：LMPM 作为 baseline 在目标完全消失后重出现（Figure 6c）及多目标动作纠缠（Figure 6d）场景下仍会出现轨迹断裂或漏检，长程全局理解能力有待加强。
- **计算效率**：跨帧 self-attention 在较长视频上仍有一定显存与算力开销，未做极致轻量化或稀疏化优化。
- **未来方向（论文自述）**：改进视听运动建模、设计更鲁棒的长/短程及复杂动作处理模块、降低冗余检测、探索迁移学习与开放世界概念对齐、优化跨模态融合策略。

## 研究启发与可借鉴点
1. **数据构建规范可迁移**：A1-A4 标注准则与 V1“单帧不可识别即剔除”的验证原则，可作为构建时序导向多模态数据集的通用质量控制流程。
2. **Object Embedding + 跨帧注意力的轻量化范式**：用 instance-level embedding 替代密集 feature map 并结合 temporal attention，兼顾计算效率与轨迹建模能力，适用于中后期视频理解、长视频 grounding 等任务。
3. **阈值匹配支持动态输出**：摒弃固定 top-1 决策，改用相似度阈值 $\sigma$ 筛选，天然适配 one-expression-to-multiple-targets 分布，可推广至开放词汇视频定位、多实例指代分割等方向。
4. **实验诊断设计严谨**：通过 TC 增量模块与跨数据集交叉验证清晰剥离静态/动态表达的贡献，实验逻辑可直接复用于后续 benchmark 论文的消融与对比设计。

## 关键术语表
- **MeViS (Motion expressions Video Segmentation)**：本文提出的大规模 benchmark 数据集，以动作/轨迹描述为核心语言线索，用于系统评估 motion-guided 视频分割能力。
- **RVOS (Referring Video Object Segmentation)**：给定视频序列与自然语言描述，对全片指代目标进行逐帧分割的任务。
- **LMPM (Language-guided Motion Perception and Matching)**：本文提出的 MeViS 基线方法，通过语言引导提取、跨帧运动感知与相似度阈值匹配实现多目标轨迹预测。
- **Temporal Context (TC)**：模型利用多帧信息捕捉目标随时间演变状态的能力；本文通过显式 Attention 模块引入，用于弥补单帧静态特征的不足。
- **Multi-object Expression**：单条自然语言同时指代多个目标对象的表达设定，为 MeViS 独有特性，平均每条表达对应 1.59 个目标。
- **Object Embedding**：轻量级实例级特征向量，相比密集 object feature map 计算更高效且具备 instance 专属性，常用于 Token 化视觉目标表示。
- **$\mathcal{J} \& \mathcal{F}$**：视频分割综合评估指标，分别为区域 IoU 相似度（$\mathcal{J}$）与轮廓 F-measure（$\mathcal{
