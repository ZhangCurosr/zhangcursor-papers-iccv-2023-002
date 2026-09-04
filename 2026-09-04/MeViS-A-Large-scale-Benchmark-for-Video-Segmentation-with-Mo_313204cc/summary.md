---
title: "MeViS-A-Large-scale-Benchmark-for-Video-Segmentation-with-Mo"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Ding_MeViS_A_Large-scale_Benchmark_for_Video_Segmentation_with_Motion_Expressions_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 19:14:24"
field: "视频理解与分割"
keywords: ["视频分割", "参照视频目标分割", "运动表达", "多模态理解", "基准数据集", "时序上下文"]
innovations: ["提出以运动表达式为主导的大规模 MeViS 数据集，强调动作描述并支持多目标表达", "设计 LMPM 基线方法，通过物体嵌入与跨帧自注意力聚合全局运动上下文", "系统性证明现有 RVOS 方法在运动依赖场景下存在显著性能缺口"]
benchmarks: ["MeViS", "Refer-Youtube-VOS", "DAVIS17-RVOS", "A2D Sentence", "Ref-COCO"]
---

# 论文速读：MeViS-A-Large-scale-Benchmark-for-Video-Segmentation-with-Motion-Expressions

## 一句话总结
本文提出了 MeViS（Motion expressions Video Segmentation）大规模数据集，聚焦于以运动表达式为主导的语言引导视频目标分割任务；同时提供基线方法 LMPM，并验证了现有 RVOS 方法在处理运动依赖表达时的显著不足。

## 研究问题与动机
- 现有参照视频目标分割（RVOS）数据集（如 DAVIS17-RVOS、Refer-Youtube-VOS）的表达多依赖静态属性（颜色、类别等），目标对象往往可在单帧内识别，运动信息被弱化。
- 图像级参照分割方法在现有视频数据集上即可取得高分，说明当前评测未能体现视频的时序运动特性，缺乏对运动表达式的系统性研究。
- 真实场景中存在大量"多目标+运动描述"的复杂指代表达，而现有数据集大多仅支持单目标一对一表达，难以反映实际困难。

## 核心贡献（创新点）
- **提出 MeViS 数据集**：收录 2,006 个视频、8,171 个目标与 28,570 条运动表达式，规模与表达复杂度均超过既有 RVOS 数据集。
- **强调运动优先的表达设计**：通过严格的标注与验证规则（A1–A4、V1），确保目标主要由动作/位移描述，消除可被单帧静态特征识别的样本。
- **引入多目标表达式**：单条表达可指代任意数量目标（平均 1.59 个/表达），突破既有 RVOS 数据集的单目标限制。
- **提出基线方法 LMPM**：设计 Language-Guided Extractor + Motion Perception + 匹配阈值筛选的完整流程，首次在 MeViS 上建立有竞争力的基线。
- **系统性基准评测与归因分析**：证明现有 SOTA 方法（27.8%–31.0% J&F）在 MeViS 上大幅落后于其他基准（>60%），并给出时序上下文必要性的实验证据。

## 方法详解
LMPM（Language-guided Motion Perception and Matching）分为四个阶段：

1. **Language-Guided Extractor**：以 $N_1 = 20$ 条语言查询在全视频 $T$ 帧内检测潜在目标，采用物体嵌入（object embedding）而非特征图表示目标，降低计算开销并增强实例鲁棒性（受 VITA 启发）。
2. **Motion Perception**：对所有帧的物体嵌入执行跨帧自注意力，聚合全局时序上下文，使短暂无常动作与跨越全片的长期动作均可被捕获。
3. **Transformer Decoder**：以 $N_2 = 10$ 条语言查询为 query、经 Motion Perception 的物体嵌入为 key/value，解码语言相关信息并预测物体轨迹。
4. **相似度匹配与阈值筛选**：将语言特征与预测轨迹做相似度计算，仅保留超过阈值 $\sigma = 0.8$ 的轨迹作为最终输出，从而天然支持单目标与多目标表达。

网络实现细节：使用 Tiny Swin Transformer 作为主干，RoBERTa 作为冻结文本编码器；训练 150,000 次迭代，学习率 $5 \times 10^{-5}$（AdamW）；Motion Perception 使用 6 层，Decoder 使用 3 层。

## 实验与结果
- **数据集划分**：训练集 1,712 视频、验证集 140、测试集 154；评估指标为 $\mathcal{J}$（IoU）、$\mathcal{F}$（轮廓 F 分数）及平均 $\mathcal{J\&F}$。
- **现有方法对比（验证集）**：URVOS 27.8%、LBDT 29.3%、MTTR 30.0%、ReferFormer 31.0%；远低于其在 Refer-Youtube-VOS / DAVIS17-RVOS 上 >60% 的水平，证明 MeViS 挑战性显著。
- **LMPM 表现**：达到 37.2% $\mathcal{J\&F}$（$\mathcal{J}=34.2$、$\mathcal{F}=40.2$），为当前最优基线。
- **时序上下文必要性**：将 VLT / ReferFormer 扩展加入全局时序模块（TC）后，MeViS 上分别提升约 5% $\mathcal{J\&F}$；而图像方法 VLT 无时序设计时仅得 27.8%。
- **跨数据集迁移**：在 Ref-COCO 系列上训练的模型测试到 MeViS 时性能明显低于 Refer-Youtube-VOS 与 DAVIS17-RVOS，说明存在"静态表达 vs 运动表达"的分布鸿沟。
- **消融（LMPM）**：移除 Motion Perception 下降 5.3%；移除阈值匹配（仅输出最高分轨迹）下降 0.9%，两者缺一不可。
- **失败案例分析**：目标消失后重新出现、多目标长期纠缠导致部分目标丢失，表明全局时序建模与复杂动作分解仍是开放难题。

## 相关工作脉络
- **Referring Image Segmentation**（VLT、LAVT、MAttNet）：侧重单帧图像理解与跨模态匹配，缺乏时序建模，难以应对 MeViS 的运动依赖性。
- **Referring Video Object Segmentation**（URVOS、ReferFormer、MTTR、LBDT）：多采用随机帧采样或轻量时空特征，未显式建模跨越整片的长程运动，在 MeViS 上表现不足。
- **A2D Sentence / J-HMDB Sentence**：聚焦演员-动作对应，但表达与目标数量有限且偏单目标，运动复杂度与场景多样性不及 MeViS。
- **DAVIS-RVOS / Refer-Youtube-VOS**：主流 RVOS 基准，表达式常含颜色/位置等静态线索，静态识别即可解，无法推动运动感知算法发展。
- **VITA**：以物体嵌入进行视频实例分割的先驱工作，本文 LMPM 借鉴其嵌入表示以提升效率与鲁棒性。
- **MOSE / OVIS**：复杂场景视频分割数据集，侧重遮挡与开世界目标，但与本文"运动表达驱动"的研究切入点不同。

## 局限性与未来方向
- **目标间断/复现难处理**：当前方法在目标短暂消失后重新出现时易产生轨迹断裂或歧义。
- **多目标长程纠缠仍具挑战**：当多个目标运动模式相似或交叉时，匹配与分离可靠性下降。
- **随机帧采样的信息遗漏**：现有方法多依赖少量随机帧，难以覆盖运动描述中所指的关键时刻。
- **开放世界与跨域泛化**：数据集仍来自已有 VOS 源数据，面向开世界场景的迁移能力有待验证。
- 作者提出未来方向包括：改进视-语运动建模、复杂运动类型统一表征、冗余检测优化、跨模态融合、迁移学习与领域自适应、开世界表达理解等。

## 研究启发与可借鉴点
- **运动优先的数据标注规范**：A1–A4 与 V1 规则体系可迁移至其他时序理解任务，有效剔除静态捷径（shortcut），逼迫模型真正学习运动语义。
- **物体嵌入 + 跨帧自注意力的轻量时序聚合**：相比逐像素特征传播，LMPM 的 object-embedding 路线在保持性能的同时显著降低计算负担，适合部署到长视频场景。
- **阈值匹配替代单一 argmax 决策**：用相似度阈值筛选多目标输出，避免强制单目标假设，可作为多目标视频分割/定位任务的通用设计。
- **跨模态鸿沟量化评测**：通过 Image→Video 跨集迁移实验揭示静态/运动表达 gap，可为其他多模态时序任务提供评测范式参考。
- **失败案例驱动的方向诊断**：本文对"消失-重现"与"纠缠丢失"的可视化分析，为后续研究提供了明确的难点清单，可直接用于 Ablation 设计与数据集扩充。

## 关键术语表
- **MeViS**：Motion expressions Video Segmentation，本文提出的以运动表达式为主导的大规模视频分割基准数据集。
- **RVOS（Referring Video Object Segmentation）**：参照视频目标分割，基于自然语言描述在视频序列中分割并追踪目标对象的任務。
- **LMPM（Language-guided Motion Perception and Matching）**：本文提出的基线方法，通过语言引导提取、运动感知与阈值匹配完成多目标分割。
- **Object Embedding**：以低维向量表示目标实例的嵌入方式，相比特征图更轻量且具备实例特异性。
- **Motion Perception**：基于跨帧自注意力的时序上下文聚合模块，使物体嵌入能够捕获全片范围的短期与长期运动。
- **$\mathcal{J\&F}$**：区域相似度 $\mathcal{J}$ 与轮廓 F 分数 $\mathcal{F}$ 的平均值，MeViS 与既有 RVOS 基准的主要评测指标。
- **Multi-object Expression**：单条语言表达式可同时指代多个目标的表达形式，是 MeViS 区别于既有数据集的关键特性之一。
- **Temporal Context (TC)**：全片或跨帧的时序上下文信息；本文通过额外模块引入后显著提升模型在 MeViS 上的表现。

## 可复现要素
- **数据集**：MeViS 已公开发布，下载地址 https://henghuiding.github.io/MeViS（仅限非商业研究用途）。
- **代码/权重**：论文未明确提供开源链接，仅提供了基线方法 LMPM 的详细实现参数。
- **关键超参**：$N_1 = 20$、$N_2 = 10$、匹配阈值 $\sigma = 0.8$、训练迭代 150,000、学习率 $5 \times 10^{-5}$（AdamW）；主干为 Tiny Swin Transformer，文本编码器为冻结 RoBERTa；输入最短边 Resize 至 448 像素。
