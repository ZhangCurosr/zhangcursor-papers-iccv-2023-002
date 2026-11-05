---
title: "UniFormerV2-Unlocking-the-Potential-of-Image-ViTs-for-Video"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Li_UniFormerV2_Unlocking_the_Potential_of_Image_ViTs_for_Video_Understanding_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 08:25:47"
field: "视频理解与视频Transformer"
keywords: ["Video Understanding", "Vision Transformer", "Action Recognition", "Spatiotemporal Modeling", "Transfer Learning"]
innovations: ["将UniFormer高效设计作为即插即用模块嵌入预训练图像ViT，解决图像-视频域间隙问题", "提出局部时序MHRA与查询式交叉MHRA，分别高效建模局部时序冗余和全局长程时空依赖", "构建Kinetics-710统一后预训练基准，以更低成本实现多Kinetics子集性能提升"]
benchmarks: ["Kinetics-400", "Kinetics-600", "Kinetics-700", "Something-Something V2", "Moments in Time", "ActivityNet", "HACS"]
---

# 论文速读：UniFormerV2-Unlocking-the-Potential-of-Image-ViTs-for-Video

## 一句话总结
论文提出 UniFormerV2，通过将高效 UniFormer 设计作为即插即用模块引入预训练图像 ViT，实现图像 ViT 在视频理解任务中的潜力释放；该方法在 8 个主流视频基准上达到 SOTA，首次使 Kinetics-400 top-1 准确率达到 90.0%。

## 研究问题与动机
- 图像 ViT 在视频任务上的性能提升遭遇显著的"图像-视频"域间隙：在场景相关任务（如 Kinetics）上表现良好，但在复杂时序相关任务（如 Something-Something）上远逊于 CNN 方法。
- 视频专用模型（如 UniFormerV1）虽能无缝迁移到视频域，但因其独特架构需要大量图像预训练，难以扩展和规模化。
- 当前基于 ViT 的视频方法依赖昂贵的端到端预训练（如 VideoMAE 需数千 epoch），或存在局部/全局建模不均衡的问题，缺乏兼顾效率与效果的新范式。
- 开源强大图像 ViT（如 CLIP-ViT、MAE-ViT）的快速涌现，为"直接复用预训练图像表示+轻量视频适配"提供了新机遇。

## 核心贡献（创新点）
- **提出 UniFormerV2 框架**：将 UniFormer 的高效设计重构为即插即用模块嵌入预训练图像 ViT，而非从零训练专用视频骨干。与 VideoMAE 等需数千 epoch 预训练的方法本质不同，该方法直接利用已有强大图像 ViT。
- **设计局部 UniBlock（Local UniBlock）**：在空间 ViT 块之前插入局部时序 MHRA（LT-MHRA），将时空亲和分解为局部时序 + 全局空间两部分，有效降低局部时序冗余并保留 ViT 预训练的空间能力。与 ST-Adapter 等仅做卷积适配的方法不同，本文的 LT-MHRA 作为显式关系聚合器建模局部时序依赖。
- **设计全局 UniBlock（Global UniBlock）**：引入基于查询的交叉 MHRA（Cross-MHRA），将可学习 query 作为视频表示，以 $O(L)$ 复杂度替代原始 UniFormer 全局 MHRA 的 $O(L^2)$ 计算，同时捕捉长程时空依赖。与 DETR 式多 query 设计不同，本文仅用一个 query 与 KV 交互。
- **构建多阶段融合架构（Multi-Stage Fusion Block）**：自适应聚合多尺度时空表征，并提出 Kinetics-710 统一后预训练基准（合并 K400/600/700 并去重，训练视频从 1.14M 降至 0.66M），仅需 5 轮微调即可在各 Kinetics 子集上提升 >1%。
- **系统性 SOTA 验证**：在 8 个基准（K400/600/700、MiT、SSV1/V2、ActivityNet、HACS）上全面超越现有方法，首次实现 Kinetics-400 top-1 准确率 90.0%。

## 方法详解
- **整体框架**：输入视频经 3D 卷积（3×16×16）投影为 $L$ 个时空 token（$L = T \times H \times W$），时空下采样因子为 16（空间）和 2（时间），得到 $\mathbf{X}^{in} \in \mathbb{R}^{L \times C}$。随后在 ViT 多层中交替插入局部和全局 UniBlock，最后通过多阶段融合生成视频表征。
- **局部 UniBlock**（对应浅层，侧重时序建模）：
  - 时序局部 MHRA（LT-MHRA）：在 BatchNorm 后执行，亲和矩阵为 $A_n^{LT}(\mathbf{X}_i, \mathbf{X}_j) = a_n^{i-j}$，其中 $j$ 属于时间管 $t \times 1 \times 1$，仅捕获局部时序关系。
  - 全局空间 MHRA（GS-MHRA）：在 LayerNorm 后执行，使用原始 ViT 块的全局空间自注意力（单帧 $1 \times H \times W$ 内的 QK 注意力）。
  - 公式：$\mathbf{X}^T = \text{LT\_MHRA}(\text{Norm}(\mathbf{X}^{in})) + \mathbf{X}^{in}$，$\mathbf{X}^S = \text{GS\_MHRA}(\text{Norm}(\mathbf{X}^T)) + \mathbf{X}^T$，$\mathbf{X}^L = \text{FFN}(\text{Norm}(\mathbf{X}^S)) + \mathbf{X}^S$。
- **全局 UniBlock**（对应深层，侧重长程依赖）：
  - 首先应用动态位置嵌入（DPE，3×3×3 depth-wise 时空卷积）注入 3D 位置信息。
  - 查询式交叉 MHRA：引入一个零初始化的可学习 query token $\mathbf{q} \in \mathbb{R}^{1 \times C}$，通过交叉注意力将全部 $L$ 个时空 token 压缩为一个视频 token：$A_n^C(\mathbf{q}, \mathbf{X}_j) = \frac{\exp\{Q_n(\mathbf{q})^T K_n(\mathbf{X}_j)\}}{\sum_{j'} \exp\{Q_n(\mathbf{q})^T K_n(\mathbf{X}_{j'})\}}$，再经线性投影融合。
  - 公式：$\mathbf{X}^C = \text{DPE}(\mathbf{X}^L) + \mathbf{X}^L$，$\mathbf{X}^{ST} = \text{C\_MHRA}(\text{Norm}(\mathbf{q}), \text{Norm}(\mathbf{X}^C)) + \mathbf{X}^C$，$\mathbf{X}^G = \text{FFN}(\text{Norm}(\mathbf{X}^{ST})) + \mathbf{X}^{ST}$。
- **多阶段融合**：四个候选策略中，最简单有效的 Sequential 方式被采用——前一级的视频 token 作为当前级的 query。最终融合结果 $\mathbf{Z} = \alpha \mathbf{F} + (1-\alpha) \mathbf{F}^C$，其中 $\mathbf{F}^C$ 取自最后一级局部块的 class token，$\alpha$ 为 Sigmoid 学习参数加权融合。
- **Kinetics-710 后预训练**：合并 K400/600/700 训练集（去除重复视频和泄露测试视频），共 0.66M 视频、710 类，用 8 帧输入训练，之后仅需 5 轮微调即可在各 Kinetics 子集上显著提升。

## 实验与结果
- **数据集与基准**：8 个大规模视频基准——场景相关（Kinetics-400/600/700）、异构（Moments in Time V1）、时序相关（Something-Something V1/V2）、未裁剪（ActivityNet、HACS）。
- **最强结果**：
  - **Kinetics-400**：UniFormerV2-L（336↑，64 帧）达 **90.0% top-1**（首次突破 90%），显著优于 X-CLIP-L（87.7%）、MTV-H（89.1%）等；同等参数量下优于 MViTv2-L（86.1%）1.2%。
  - **Kinetics-600**：UniFormerV2-L（64 帧）达 **90.1%**。
  - **Kinetics-700**：UniFormerV2-L（64 帧）达 **82.7%**。
  - **Something-Something V2**：UniFormerV2-L（32 帧）达 **73.0%** top-1；V1 达 **62.7%**（SOTA）。
  - **Moments in Time**：UniFormerV2-L（8 帧）达 **47.0%** top-1。
  - **ActivityNet**：UniFormerV2-L（16 帧）达 **94.7%** top-1（超越前 best +4.5%）。
  - **HACS**：UniFormerV2-L（16 帧）达 **95.5%** top-1（超越前 best +3.6%）。
- **消融结论**：全局块对场景相关任务关键（K400），局部块对时序相关任务关键（SSV2）；深层特征对长程依赖更重要；Sequential 融合最简单有效；K710 后预训练节省约 33% 训练成本且效果更优。
- **泛化性**：无论基于 SL（DeiT III）、CL（CLIP/DINO）还是 MIM（MAE/BeiT）预训练的 ViT，UniFormerV2 均持续超过 TimeSformer，验证设计的通用性。

## 相关工作脉络
- **TimeSformer [4]**：视频任务中早期的全注意力 ViT，对长时序有效但对局部时序建模不足；UniFormerV2 在同等 FLOPs 下大幅超越（SSV2: 81.0% vs 69.5%）。
- **ST-Adapter [47]**：将时序深度卷积作为适配器嵌入 ViT；UniFormerV2 的 LT-MHRA 将其视为显式关系聚合器，不含激活函数并加 BatchNorm，SSV2 提升约 1%。
- **UniFormerV1 [35]**：提出 MHRA 统一卷积与注意力；但需从头预训练导致扩展困难；UniFormerV2 复用预训练 ViT 并重新设计局部/全局模块，解决可扩展性瓶颈。
- **VideoMAE [54]**：基于 Masked Image Modeling 的自监督视频预训练，需 1600 epoch 从头训练；UniFormerV2 直接利用公开预训练 ViT，在 SSV2 上与 VideoMAE-L 相当（73.0% vs 74.3%），但无需昂贵预训练。
- **X-CLIP [46]**：利用额外语言知识增强视频识别；UniFormerV2-L 在无需语言辅助的情况下达到 87.7%（vs X-CLIP-L 87.1%），证明纯视觉路径的有效性。
- **MViTv2 [37]**：视频专用多层级 ViT，需长时图像预训练；UniFormerV2-L 以仅 5% 更少 FLOPs 取得更高 K400 精度（88.8% vs 86.1%）。

## 局限性与未来方向
- **计算开销仍随帧数增长显著**：64 帧推理 FLOPs 达 75.3T（L 模型），在实时应用中仍有压力；论文未深入讨论更高效的时间采样策略。
- **依赖高质量预训练 ViT**：方法性能上限受限于所选 backbone 的预训练质量，对未充分预训练的 ViT 增益有限。
- **仅关注分类任务**：实验主要评估动作分类，对于视频定位、时序检测、视频检索等下游任务的泛化能力未系统验证。
- **Cross-MHRA 单 query 的容量**：单个 query 压缩全部时空信息的表征能力可能受限，未来可探索多 query 变体或自适应 query 数量。
- **未探索与大规模自监督视频预训练的结合**：如与 MAE/BEiT 等视频预训练范式结合的可能性尚未探索。

## 研究启发与可借鉴点
- **"分解时空亲和"的设计思路**：将原始 UniFormer 的联合局部时空亲和拆解为局部时序 + 全局空间两部分，使得可以分别复用预训练 ViT 空间能力和 UniFormer 高效时序建模，该思路可迁移到其他需要在预训练视觉模型中注入时序能力的场景（如图像-视频对比学习）。
- **Query-based 跨注意力聚合**：用单个 learnable query 经 cross-attention 压缩全部 token 的设计，既保留了 $O(L)$ 复杂度又实现了全局池化效果，可替代传统 GlobalAveragePool 用于视频表征学习，且更具表达能力。
- **Kinetics-710 去重合并策略**：合并多个同类数据集并严格去重/去除泄露视频的做法，提供了一种低成本提升预训练质量的实用范式，可推广至其他视频分类基准的联合训练。
- **多阶段 Sequential 融合**：简单地将前一级输出作为下一级 query 的 sequential fusion 策略，在保持低计算量的同时实现多尺度信息传递，是轻量级视频建模的有效设计，值得在其他视频 Transformer 架构中尝试。
- **BN before local MHRA / LN before global MHRA** 的混合归一化设计：针对不同模块的特性选择归一化方式，这一细节在实践中对训练稳定性有重要贡献，值得在其他 ViT 视频适配工作中参考。

## 关键术语表
- **MHRA（Multi-Head Relation Aggregator）**：UniFormer 的核心组件，统一卷积和自注意力为多头关系聚合器，通过亲和矩阵建模 token 间关系， shallow 层用于局部建模，deep 层用于全局建模。
- **LT-MHRA（Local Temporal MHRA）**：UniFormerV2 局部块中的时序关系聚合器，亲和矩阵局限于时间管 $t \times 1 \times 1$，仅建模局部时序依赖，避免全量时空计算的开销。
- **Cross-MHRA**：全局块中的查询式交叉关系聚合器，以可学习 query 为 Q、所有时空 token 为 K/V，以 $O(L)$ 复杂度将视频全部帧 token 压缩为单一视频表征。
- **DPE（Dynamic Position Embedding）**：3×3×3 depth-wise 时空卷积形式的动态位置编码，为 token 注入 3D 位置信息，增强时空感知能力。
- **Kinetics-710**：本文构建的统一后预训练基准，合并 Kinetics-400/600/700 的训练集（去重后 0.66M 视频、710 类），比单独训练各子集节省约 33% 成本。
- **Global UniBlock**：UniFormerV2 中位于局部块之上的模块，通过 Cross-MHRA 和 DPE 捕获长程时空依赖，主要服务于场景相关任务的全局理解。
- **Multi-Stage Fusion Block**：将各层全局块输出的视频 token 按 Sequential/Parallel/Hierarchical 等方式融合，最终与 class token 加权求和得到判别性视频表示。
- **ST-Adapter**：基于 Adapter 思想的视频 ViT 适配方法，将时序深度卷积作为适配器插入预训练 ViT；本文在其基础上改进为显式关系聚合器。

## 可复现要素
- **代码与权重**：代码已开源，地址 https://github.com/OpenGVLab/UniFormerV2；论文未明确声明权重是否开源（通常此类 SOTA 方法会提供部分权重，建议查阅 GitHub 仓库）。
- **数据集**：Kinetics-400/600/700、Moments in Time V1、Something-Something V1/V2、ActivityNet、HACS 均为公开数据集；Kinetics-710 由本文合并构建（论文未提供独立下载链接，可从各子集自行构建）。
- **关键超参**：输入分辨率 224，稀疏采样（sparse sampling）；K400 用 8×4×3 测试策略，SSV 用 16×1×3；最佳模型基于 CLIP-ViT-B/L，后加 K710-0.66M 预训练；K710 训练用 8 帧，fine-tune 5 轮；插入全局块的层数：大部分数据集用最后 4 层，SSV1/V2 分别用最后 8/16 层。
