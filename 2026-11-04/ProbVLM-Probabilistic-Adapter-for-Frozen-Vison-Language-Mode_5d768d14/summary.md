---
title: "ProbVLM-Probabilistic-Adapter-for-Frozen-Vison-Language-Mode"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Upadhyay_ProbVLM_Probabilistic_Adapter_for_Frozen_Vison-Language_Models_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 08:22:55"
field: "多模态表示学习"
keywords: ["视觉语言模型", "概率嵌入", "不确定性估计", "跨模态检索", "后处理适配器", "广义高斯分布", "主动学习"]
innovations: ["首个冻结VLM概率适配器，通过类内/类间对齐损失将确定性嵌入转为GGD分布", "提出可微类间对齐近似，捕获跨模态语义歧义", "验证不确定性在主动学习和模型选择中的下游效用"]
benchmarks: ["COCO", "Flickr-30k", "CUB", "Oxford-Flowers 102"]
---

# 论文速读：ProbVLM-Probabilistic-Adapter-for-Frozen-Vison-Language-Mode

## 一句话总结
本文提出 ProbVLM，一种后处理概率适配器，能够将冻结的大型视觉-语言模型（VLM）的确定性嵌入转换为概率分布，通过类内和类间对齐目标高效估计校准的嵌入不确定性，无需从头训练大模型或使用大规模数据集。

## 研究问题与动机
- **确定性嵌入的局限性**：CLIP 等 VLM 将图像/文本映射到单一向量，无法反映输入固有的歧义性（如同一概念可有多个图像或描述）。
- **现有概率方法依赖从头训练**：概率嵌入方法（如 PFE、PCME）需要从头训练深度网络，无法利用已存在的冻结预训练 VLM。
- **跨模态不确定性估计缺失**：现有不确定性估计方法主要针对单模态数据，缺乏针对跨模态（图像-文本）预训练模型的有效不确定性量化方法。
- **下游应用需求**：准确的校准不确定性可用于主动学习样本选择和模型选择等实际任务，但现有 VLM 缺乏此类能力。

## 核心贡献（创新点）
1. **首个冻结 VLM 概率适配器**：提出 ProbVLM，是首个将冻结大型 VLM 的确定性嵌入转换为概率嵌入的后处理方法，无需大规模训练。
   → 与 PCME 等从头训练方法相比，直接复用预训练权重，仅需轻量 MLP 适配器。
2. **联合类内/类间对齐损失**：设计广义高斯分布（GGD）建模异方差不确定性，结合类内重建损失和近似化的类间对齐损失。
   → 区别于 PFE 仅处理单模态高斯分布，本文捕获跨模态语义歧义。
3. **校准不确定性量化框架**：同时估计偶然不确定性（通过 GGD 方差）和认知不确定性（通过 Dropout 多次前向传播）。
   → 不同于 ECE、TTDA 等基线，本文方法在检索任务上实现单调性能-不确定性关联。
4. **扩散模型可视化新范式**：利用 Stable Diffusion 从采样嵌入分布生成图像，解释概率嵌入的语义含义。
   → 首次将预训练扩散模型用于概率嵌入分布的视觉解释。
5. **下游任务验证**：证明校准不确定性可有效用于主动学习（样本选择）和预训练模型选择。
   → 相比随机采样和现有模型选择方法，基于不确定性的策略显著提升性能。

## 方法详解
**整体框架**：ProbVLM 由两个轻量级 MLP 适配器组成（$\Psi_\mathcal{V}$ 和 $\Psi_\mathcal{T}$），分别附加于冻结的视觉和文本编码器之后，预测嵌入分布的参数。

**类内对齐（Intra-modal Alignment）**：
- 使用广义高斯分布（GGD）$\mathcal{G}(\mathbf{z}; \hat{\mathbf{z}}, \hat{\alpha}, \hat{\beta})$ 建模嵌入分布，其中 $\hat{\mathbf{z}}$ 为预测均值（应接近冻结编码器输出），$\hat{\alpha}$ 为尺度参数，$\hat{\beta}$ 为形状参数（捕获重尾特性）。
- 重建损失：$L_{\text{rec}}(\zeta) = \left(\frac{|\hat{\mathbf{z}}-\mathbf{z}|}{\hat{\alpha}}\right)^{\hat{\beta}} - \log\frac{\hat{\beta}}{\hat{\alpha}} + \log\Gamma(\frac{1}{\hat{\beta}})$

**类间对齐（Cross-modal Alignment）**：
- 理想目标：$p(\mathbf{z}_v = \mathbf{z}_u) = \iint \mathcal{G}_\mathcal{V}(\mathbf{z}_v)\mathcal{G}_\mathcal{T}(\mathbf{z}_u)\delta(\mathbf{z}_v - \mathbf{z}_u)d\mathbf{z}_v d\mathbf{z}_u$
- 由于 Bivariate Fox H-function 计算复杂，采用近似：$p(\mathbf{z}_v = \mathbf{z}_u) \approx \frac{1}{2}\int (\mathcal{G}_\mathcal{V}(\mathbf{z})\delta(\mathbf{z}-\mathbf{z}_\mathcal{T}) + \mathcal{G}_\mathcal{T}(\mathbf{z})\delta(\mathbf{z}-\mathbf{z}_\mathcal{V}))d\mathbf{z}$
- 类间损失 $L_{\text{cross}}$ 为上述近似的负对数似然，对称衡量图像-文本分布的交集概率。

**总损失**：$L_{\text{ProbVLM}} = L_{\text{rec}}^\mathcal{V} + L_{\text{rec}}^\mathcal{T} + \lambda_{\text{cross}} L_{\text{cross}}$

**不确定性量化**：
- 偶然不确定性：$\hat{\sigma}^2_{\text{aleatoric}} = \frac{\hat{\alpha}^2 \Gamma(3/\hat{\beta})}{\Gamma(1/\hat{\beta})}$
- 认知不确定性：通过 Dropout 多次前向传播（M=10）估计：$\hat{\sigma}^2_{\text{epistemic}} = \frac{1}{M}\sum_{m=1}^{M}(\hat{\mathbf{z}}_m - \bar{\hat{\mathbf{z}}})^2$
- 总不确定性：$\hat{\sigma}^2_{\text{total}} = \hat{\sigma}^2_{\text{epistemic}} + \hat{\sigma}^2_{\text{aleatoric}}$

**扩散模型可视化**：从预测分布采样嵌入向量 $\{\hat{\mathbf{z}}_i\}$，通过 Stable Diffusion 的条件文本编码器 $\Omega(\cdot;\theta_\Omega^*)$ 生成图像集 $\mathcal{J}$。

## 实验与结果
**数据集**：COCO、Flickr-30k（跨模态检索）、CUB（细粒度鸟类）、Oxford-Flowers 102（FLO，细粒度花卉）。

**评估指标**：
- Recall@k（R@k）用于检索性能
- 校准指标：Spearman 秩相关 S（期望-1）、$R^2$（线性拟合优度）、综合指标 $-SR^2$（期望 1.0）

**主要结果（CLIP ViT-B/32，训练于 COCO，评估于 COCO）**：
| 方法 | i2t S | i2t $R^2$ | i2t $-SR^2$ | t2i S | t2i $-SR^2$ |
|------|-------|-----------|-------------|-------|-------------|
| **ProbVLM** | **-0.99** | **0.93** | **0.93** | -0.70 | 0.35 |
| PFE* | -0.79 | 0.59 | 0.47 | 0.79 | -0.59 |
| PCME* | -0.89 | 0.75 | 0.68 | 0.30 | 0.16 |
| TTDA | -0.79 | 0.69 | 0.55 | -0.19 | 0.07 |

- ProbVLM 在四个数据集上均取得最优校准分数，COCO i2t 任务达到近乎完美校准（$-SR^2 = 0.99$，CLIP on FLO）。
- 与基线使用相同嵌入，仅不确定性估计不同，证明校准质量显著提升。

**消融实验**：
- 类间损失系数 $\lambda_{\text{cross}}$ 需非零以学习有意义的跨模态不确定性；过大则损害重建保真度。
- 仅用 50% 数据即可实现满意校准，展示数据高效性。
- 输入遮蔽程度增加时，不确定性单调上升，符合预期。

**应用实验**：
- **主动学习**：基于不确定性选择 top-k 样本微调 CLIP，显著优于随机采样。
- **模型选择**：通过均值不确定性排序源模型，在 CUB/Flickr/COCO 上准确选出最优模型（FLO 上选到第二名）。

## 相关工作脉络
- **CLIP / BLIP**：大规模 VLM 提供高质量确定性嵌入，但不建模不确定性；本文在此基础上添加概率适配器。
- **PFE（Probabilistic Face Embedding）**：首个后处理概率嵌入工作，但仅处理单模态图像；本文扩展至跨模态。
- **PCME（Probabilistic Cross-Modal Embeddings）**：使用软对比损失训练概率嵌入，但需从头训练；本文保持冻结编码器。
- **TTDA（Test-Time Data Augmentation）**：通过测试时扰动估计不确定性，无需训练但精度有限；本文有监督适配器更优。
- **BayesCap（ECCV 2022）**：针对冻结网络的贝叶斯身份帽，但仅处理单模态；本文首次面向 VLM 跨模态场景。
- **概率嵌入理论（Oh et al., ICLR 2019）**：奠定 GGD 建模基础，本文将其推广至预训练 VLM 适配。

## 局限性与未来方向
- **适配器容量限制**：简单 3 层 MLP 可能无法充分建模复杂分布，更深层架构值得探索。
- **扩散可视化计算开销**：Stable Diffusion 采样耗时，实时应用受限。
- **仅估计嵌入不确定性**：未扩展到 VLM 分类、生成等任务的端到端不确定性。
- **跨域泛化**：训练于 COCO/Flickr，测试于 CUB/FLO，域偏移下的校准稳定性需进一步验证。
- **未来方向**：结合自监督预训练扩展至更多模态、优化扩散采样效率、应用于开放词汇检测等下游任务。

## 研究启发与可借鉴点
1. **后处理适配器范式**：冻结大模型 + 轻量适配器的思路可迁移至其他预训练模型（如 AudioVLM、多模odal LLM），避免全量微调。
2. **GGD 异方差建模**：广义高斯分布比标准高斯更适合重尾分布，可借鉴于医学图像分割、单目深度估计等不确定性敏感任务。
3. **类间对齐近似技巧**：将复杂积分近似为两个方向似然的平均，为跨模态分布对齐提供可扩展方案。
4. **Dropout 认知不确定性估计**：低成本获得 epistemic uncertainty，适用于资源受限场景。
5. **扩散模型作为解释工具**：将 embedding 分布映射回输入空间，为黑盒模型提供可解释性分析。

## 关键术语表
- **ProbVLM**：概率视觉语言模型适配器，将冻结 VLM 的确定性嵌入转换为概率分布。
- **广义高斯分布（GGD）**：建模重尾异方差的概率分布，由高斯（β=2）和拉普拉斯（β=1）特例推广。
- **偶然不确定性（Aleatoric）**：数据固有噪声导致的不确定性，不可通过更多数据减少。
- **认知不确定性（Epistemic）**：模型参数未知导致的不确定性，可通过更多训练数据减少。
- **类内对齐（Intra-modal Alignment）**：确保适配器输出分布均值忠实于冻结编码器嵌入。
- **类间对齐（Cross-modal Alignment）**：确保匹配图像-文本对的分布重叠度高。
- **校准（Calibration）**：不确定性估计与实际性能下降之间的单调关联程度。
- **Latent Diffusion**：潜扩散模型（如 Stable Diffusion），用于从高维 embedding 生成图像。

## 可复现要素
- **数据集**：MS-COCO、Flickr-30k、CUB-200-2011、Oxford-Flowers 102（均公开）
- **代码**：已开源（https://github.com/ExplainableML/ProbVLM）
- **权重**：CLIP（ViT-B/32, ViT-B/16）、BLIP 预训练权重；Stable Diffusion 开源
- **关键超参**：MLP 维度 256、训练 100 epoch、学习率 1e-4、Dropout 0.1、$\lambda_{\text{cross}}$ 需调优（论文未给具体值，附录有说明）
- **测试时前向次数**：M=10（用于认知不确定性估计）
