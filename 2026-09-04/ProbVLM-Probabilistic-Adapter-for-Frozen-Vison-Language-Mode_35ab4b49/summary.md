---
title: "ProbVLM-Probabilistic-Adapter-for-Frozen-Vison-Language-Mode"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Upadhyay_ProbVLM_Probabilistic_Adapter_for_Frozen_Vison-Language_Models_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 19:15:41"
field: "多模态表示学习与不确定性量化"
keywords: ["Vision-Language Models", "Probabilistic Embeddings", "Uncertainty Estimation", "Cross-Modal Retrieval", "Post-hoc Adapter", "Calibration"]
innovations: ["首个用于冻结大规模VLM的后验概率适配器", "联合内模态重建与跨模态对齐的损失函数以校准不确定性", "利用不确定性指导主动学习与预训练模型选择"]
benchmarks: ["MS-COCO", "Flickr-30k", "CUB", "Oxford-Flowers 102"]
---

# 论文速读：ProbVLM: Probabilistic Adapter for Frozen Vison-Language Models

## 一句话总结
本文提出**ProbVLM**，一种后验（post-hoc）概率适配器，能将**冻结的大规模视觉-语言模型**（如CLIP、BLIP）的确定性嵌入映射转化为**异方差概率分布**，从而在不重新训练基础模型的情况下，量化并校准跨模态嵌入空间中的固有歧义与不确定性。

## 研究问题与动机
1. **核心问题**：大规模VLM（如CLIP）通过确定性映射将图像/文本映射到单一嵌入向量，无法反映物理世界中“一图多义”、“一 caption 多解”的固有歧义性，导致嵌入空间不确定性未被量化。
2. **现有方法不足**：
    *   传统概率嵌入方法（如Probabilistic Embeddings）需从零开始训练深度网络，依赖海量数据和算力，无法直接利用已存在的预训练VLM。
    *   现有针对冻结模型的不确定性估计方法多局限于单模态（如仅图像），缺乏对视觉-语言双模态对齐场景的有效扩展。
    *   直接使用时序对比学习等确定性损失无法捕捉嵌入分布的异方差不确定性（heteroscedastic uncertainty）。

## 核心贡献（创新点）
1. **首个冻结VLM后验概率适配器**：提出ProbVLM，首次将冻结的大规模VLM（CLIP/BLIP）的确定性嵌入转换为校准的概率分布，无需重新训练基础编码器。
    *   *区别*：不同于PFE（Probabilistic Face Embedding）等仅处理单模态的方法，也不同于需从头训练的PCME等概率嵌入方法，ProbVLM通过轻量级适配器保留预训练知识并引入概率建模。
2. **联合内外模态对齐的训练目标**：设计包含**内模态重建损失**（Intra-modal Alignment）和**跨模态对齐损失**（Cross-modal Alignment）的组合优化目标，使预测分布的均值忠于原始嵌入，同时捕获跨模态歧义。
    *   *区别*：跨模态损失项直接优化图像与文本概率分布之间的重叠似然，而非简单的对比距离，更贴合检索任务中的语义对齐需求。
3. **高效且可校准的不确定性量化**：利用预测的广义高斯分布（GGD）参数估计**偶然不确定性**（aleatoric uncertainty），并结合推理时Dropout进行**模型不确定性**（epistemic uncertainty）估计，提供校准良好的总体不确定性。
    *   *区别*：在跨模态检索任务上，其不确定性估计表现出优异的校准性（calibration），性能随不确定性增加单调下降，优于多种基线。
4. **下游任务验证与新可视化方法**：证明估计的不确定性可用于**主动学习**样本选择和**预训练模型选择**，并创新性地结合**Latent Diffusion Model**（Stable Diffusion）可视化嵌入分布所对应的图像语义变化。

## 方法详解
1. **框架**：冻结预训练VLM（$\Phi_\mathcal{V}, \Phi_\mathcal{T}$），在其后接入两个轻量级MLP适配器（$\Psi_\mathcal{V}, \Psi_\mathcal{T}$）。给定嵌入$\mathbf{z}$，适配器输出GGD分布参数$(\hat{\mathbf{z}}, \hat{\alpha}, \hat{\beta})$，其中$\hat{\mathbf{z}}$为均值，$\hat{\alpha}, \hat{\beta}$为尺度与形状参数。
2. **内模态对齐损失**（$L_{rec}$）：
    *   目标：使适配器输出的分布均值$\hat{\mathbf{z}}$尽可能接近原始确定性嵌入$\mathbf{z}$，同时预测异方差噪声$(\hat{\alpha}, \hat{\beta})$。
    *   形式：最大化广义高斯分布（GGD）的似然，等价于最小化负对数似然损失$L_{rec}$（公式3）。GGD能灵活建模重尾分布。
3. **跨模态对齐损失**（$L_{cross}$）：
    *   目标：使配对的图像-文本嵌入的概率分布$\mathcal{G}_\mathcal{V}(\mathbf{z})$和$\mathcal{G}_\mathcal{T}(\mathbf{z})$在概念相似处尽量重叠。
    *   形式：近似计算两分布重叠的似然$p(\mathbf{z}_\mathcal{V} = \mathbf{z}_\mathcal{T})$（公式6、7）。具体为计算文本嵌入$\mathbf{z}_\mathcal{T}$在视觉分布$\mathcal{G}_\mathcal{V}$下的似然，加上视觉嵌入$\mathbf{z}_\mathcal{V}$在文本分布$\mathcal{G}_\mathcal{T}$下的似然，取负对数作为损失。
4. **总损失**：$L_{ProbVLM} = L_{rec}^\mathcal{V} + L_{rec}^\mathcal{T} + \lambda_{cross} L_{cross}$（公式8）。
5. **不确定性量化**：
    *   **偶然不确定性**：由GGD方差公式 $\hat{\sigma}^2_{aleatoric} = \frac{\hat{\alpha}^2 \Gamma(3/\hat{\beta})}{\Gamma(1/\hat{\beta})}$ 计算。
    *   **模型不确定性**：推理时开启适配器Dropout进行M次前向传播，计算均值预测的方差 $\hat{\sigma}^2_{epistemic}$。
    *   **总不确定性**：$\hat{\sigma}^2_{total} = \hat{\sigma}^2_{epistemic} + \hat{\sigma}^2_{aleatoric}$。
6. **分布可视化**：从预测的文本嵌入分布中采样向量，输入至冻结的**Stable Diffusion**（通过其CLIP文本编码器）进行解码，生成对应分布下的图像集合以直观展示语义变化。

## 实验与结果
*   **数据集**：MS-COCO, Flickr-30k（主流跨模态检索）；CUB (Caltech-UCSD Birds), Oxford-Flowers 102 (FLO)（细粒度检索，使用描述句子）。
*   **评估指标**：检索性能用Recall@k (R@k)；不确定性校准度用Spearman秩相关(S)与$R^2$的乘积$-SR^2$（理想值为1.0）。
*   **基线方法**：PFE*（固定均值，学习协方差）、PCME*（适配概率对比损失）、TTDA（测试时数据增强）。
*   **主要结果**：
    *   **校准性**：在全部四个数据集和两种VLM（CLIP, BLIP）上，ProbVLM均显著优于所有基线。例如，在COCO上CLIP的i2t任务，ProbVLM的$-SR^2$达**0.93**（接近理想值1.0），而最佳基线PFE*仅为0.47。在FLO上甚至达到0.99。
    *   **消融实验**：跨模态损失系数$\lambda_{cross}$需非零以保证良好校准；仅用**50%** 数据即可获得满意校准效果，证明其数据高效性。
    *   **消融**：随着图像/文本被遮蔽比例增加，预测的不确定性单调上升，符合直觉。
*   **应用**：
    *   **主动学习**：在FLO数据集上，基于ProbVLM不确定性选择top-k样本进行微调，显著优于随机采样。
    *   **模型选择**：在目标域无标签数据上，利用不确定性可准确选出性能最佳的源域微调模型（Table 2）。

## 相关工作脉络
1.  **Vision-Language Models (CLIP, BLIP等)**：提供强大的跨模态对齐能力，但输出为确定性嵌入，未建模不确定性。本文直接在其冻结参数上使用适配器。
2.  **Probabilistic Embeddings (Oh et al., Chun et al. PCME)**：将输入映射为分布以建模歧义，但通常需从头训练。本文区别在于**后验适配**冻结VLM，而非从头训练。
3.  **Probabilistic Face Embedding (PFE)**：在单模态人脸嵌入上学习概率分布的开创性工作。本文将其思想扩展至**视觉-语言双模态**，并设计了跨模态对齐损失。
4.  **Uncertainty Estimation in Pretrained Models (e.g., BayesCap)**：针对冻结模型估计不确定性的方法多集中于单模态分类任务。本文将其成功应用于**多模态交叉检索**场景，并量化两种不确定性。
5.  **Latent Diffusion for Visualization**：使用预训练扩散模型（Stable Diffusion）逆向映射或可视化嵌入空间，本文创新性地用于**可视化概率嵌入分布**所蕴含的语义多样性。

## 局限性与未来方向
1.  **依赖预训练模型**：方法效果高度依赖于底层VLM（如CLIP, BLIP）的质量与嵌入空间特性，在不同架构或更小的VLM上泛化能力待验证。
2.  **适配器容量限制**：当前使用简单3层MLP，对于极其复杂或分布偏移大的数据，其容量可能不足。
3.  **跨模态损失近似**：使用的跨模态对齐损失是理论精确积分的近似，虽 scalable 但可能有信息损失。
4.  **未来方向**：可探索将ProbVLM集成到VLM的微调过程（而非完全冻结）；扩展至更多模态（如视频、3D）；改进近似精度或探索更高效的分布拟合方式。

## 研究启发与可借鉴点
1.  **后验适配器范式**：对于任何已冻结的强大预训练模型，可通过在其输出后附加轻量级概率头来高效获得不确定性估计，无需昂贵重训练。此范式可迁移至其他预训练模型（如分割、检测模型）。
2.  **内/外模态联合对齐**：结合“保持原特征保真度”的内模态损失和“促进语义对齐”的跨模态损失，可有效利用成对数据学习校准的不确定性。此思路可用于其他多模态表示学习。
3.  **不确定性用于下游任务**：证明VLM的嵌入不确定性可直接服务于**主动学习**和**模型选择**等实用任务，为利用基础模型的不确定性信号提供了新思路。
4.  **分布可视化新途径**：结合Latent Diffusion Model对概率嵌入分布进行采样和逆映射可视化，提供了一种直观解读高维分布语义变化的新方法。

## 关键术语表
*   **Post-hoc Adapter**：后接在预训练模型之后的轻量级可训练模块，用于在不完全重训原模型的情况下赋予其新功能（如输出分布）。
*   **Heteroscedastic Uncertainty**：异方差不确定性，指噪声的方差随输入不同而变化，而非固定常数，由概率分布的参数（如$\hat{\alpha}$）建模。
*   **Generalized Gaussian Distribution (GGD)**：广义高斯分布，由尺度$\alpha$和形状$\beta$参数化，能灵活建模从尖峰到重尾的各种分布形态，此处用于参数化嵌入分布。
*   **Aleatoric Uncertainty**：偶然不确定性，源于数据固有的噪声或模糊性，不可通过收集更多数据消除，由模型输出的分布参数建模。
*   **Epistemic Uncertainty**：模型不确定性，源于模型对未见数据的知识不足，可通过收集更多数据或改进模型来减少，此处通过Monte Carlo Dropout近似。
*   **Cross-Modal Alignment**：跨模态对齐，使来自不同模态（如图像和文本）的语义相关内容在嵌入空间中相互靠近的过程。
*   **Calibration**：校准，指模型预测的不确定性与其实际表现准确性之间的匹配程度，高校准度意味着不确定性高的样本确实更容易出错。
*   **Latent Diffusion Model**：潜在扩散模型（如Stable Diffusion），在压缩的潜在空间中进行去噪扩散过程以生成高分辨率图像的预训练大模型。

## 可复现要素
*   **数据集**：MS-COCO, Flickr-30k, CUB, Oxford-Flowers 102 (论文使用，但下载渠道需自行查找)。
*   **代码**：**已开源**，位于 https://github.com/ExplainableML/ProbVLM。
*   **预训练模型**：CLIP (ViT-B/32, ViT-B/16, ResNet50), BLIP；Stable Diffusion (用于可视化)。
*   **关键超参**：适配器为3层MLP，维度[embedding_dim, 256, 256, embedding_dim]；训练100个epoch，学习率1e-4；Dropout概率0.1（训练时）；$\lambda_{cross}$ 需调优（论文图5显示非零且不宜过大）。
