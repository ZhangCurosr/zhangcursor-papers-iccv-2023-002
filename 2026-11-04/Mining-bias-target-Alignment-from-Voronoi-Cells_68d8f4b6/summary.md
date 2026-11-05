---
title: "Mining-bias-target-Alignment-from-Voronoi-Cells"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Nahon_Mining_bias-target_Alignment_from_Voronoi_Cells_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 00:52:36"
field: "计算机视觉去偏"
keywords: ["debiasing", "bias-agnostic", "Voronoi cells", "mutual information regularization", "deep learning fairness"]
innovations: ["基于Voronoi边界距离自动检测bias-target对齐时机", "条件互信息正则化仅在misaligned样本上去除bias信息", "bias-agnostic方法性能媲美监督方法且在多偏置场景达SOTA"]
benchmarks: ["Biased MNIST", "Multi-Color MNIST", "CelebA", "9-class ImageNet", "ImageNet-A"]
---

# 论文速读：Mining-bias-target-Alignment-from-Voronoi-Cells

## 一句话总结
本文提出了一种 **bias-agnostic 的去偏方法**，通过在 vanilla 模型训练过程中监测**误分类样本到目标类 Voronoi 边界的距离**，找到 bias-target 对齐信息最强的时刻，再利用该信息对样本加权并重训去偏模型，同时通过互信息正则项在 bottleneck 层去除 bias 对齐信息。在多个去偏基准上，该方法性能**与监督方法相当**，且在多偏置场景下实现了新的 SOTA。

## 研究问题与动机
- **深度网络易学习 spurious correlations（偏见）**：例如依靠背景颜色而非形状分类，导致泛化能力受限（如海面上降落的飞机被误分类）。
- **现有去偏方法多需 bias 标签**（监督方法），但标注成本高昂或存在噪声；而 unsupervised/bias-agnostic 方法虽不依赖 bias 标签，但缺乏可靠的 bias 检测机制。
- **现有 bias-agnostic 方法假设"早期特征即 bias"**（如 LfF），但无法保证最先学到的特征一定是偏见特征，且难以扩展到多偏置场景。
- **本文动机**：提出一种不依赖 bias 标签、能从训练数据中**自动检测 bias-target 对齐时机**的方法，并通过距离度量量化 misalignment。

## 核心贡献（创新点）
- **基于 Voronoi 边界的 bias 对齐时机检测**：提出通过最大化误分类样本到目标类 Voronoi 边界的平均距离来自动确定提取 bias-target 对齐信息的最佳 epoch $e^*$，而非假设"早期特征= bias"。
- **bias-target 对齐信息驱动的样本重加权**：根据对齐/不对齐信息对 loss 进行重加权，bias-misaligned 样本获得更高权重，鼓励模型学习泛化性更强的特征。
- **条件互信息正则化去除 bias 对齐信息**：在 bottleneck 层引入 auxiliary head（IRH），最小化 bias-misaligned 样本上 latent representation 与 bias label 的条件互信息，而非直接去除全部 bias 信息，避免影响 aligned 样本。
- **bias-agnostic 方法性能媲美监督方法**：在 Biased MNIST、Multi-Color MNIST、CelebA、9-class ImageNet 等基准上，该方法在多数设置下达到 unsupervised SOTA，并在部分场景下超越监督方法（如 9-class ImageNet）。

## 方法详解
### 整体框架
分两阶段：(1) **Bias 提取阶段**：训练 vanilla 模型，计算各 epoch 的目标类质心与 Voronoi 边界，找到 $e^*$；(2) **去偏训练阶段**：利用 $e^*$ 时的对齐信息重加权 loss，并在 bottleneck 层施加条件互信息正则。

### 关键设计
1. **Bottleneck 隐表示与 Voronoi 边界**：在 bottleneck 层，对每类目标类 $t$，计算正确分类样本的隐表示均值作为质心 $C_{e,t}$；定义 $H_{e,i,j}$ 为 $C_{e,i}$ 与 $C_{e,j}$ 的中垂超平面，将隐空间划分为 Voronoi cells。

2. **Bias 对齐时机检测（公式 2-4）**：
   - 定义 $d^*(a_{e,n})$ 为误分类样本 $n$ 到其预测类质心与真实类质心之间 Voronoi 边界的 $\ell_2$ 距离，并除以两类质心的平均范数以补偿 weight-decay 导致的距离衰减。
   - 选择 $e^* = \arg\max_e \frac{\sum_i d^*(a_{e,n})}{|\mathcal{D}_e^\perp|}$，即误分类样本平均距离最大化的 epoch。

3. **Loss 重加权（公式 5-6）**：
   - 根据 $e^*$ 时的对齐状态，定义 $\rho_t = \frac{|\mathcal{D}_{e^*,t}^\parallel|}{|\mathcal{D}_{e^*,t}^\parallel \cup \mathcal{D}_{e^*,t}^\perp|}$ 为类 $t$ 中 aligned 样本的比例。
   - 样本权重 $r_n = \frac{1}{\rho_{b_n^*}}$（若 aligned）或 $\frac{1}{1-\rho_{b_n^*}}$（若 misaligned），使得 misaligned 样本获得更高权重。

4. **Bias 对齐信息去除（公式 7-8）**：
   - 在 bottleneck 后插入 auxiliary head（IRH），输出 bias 预测 $b_{e,n}$。
   - 计算条件互信息 $\mathcal{I}^\perp$，仅针对 bias-misaligned 样本（$\delta_{b_n^*,\hat{y}_n}=0$），衡量 IRH 输出 $B$ 与 ground-truth bias $B^*$ 的依赖性。
   - 总 loss 中加入 $\lambda_{\mathcal{I}^\perp} \cdot \mathcal{I}^\perp$（论文取 $\lambda=2$），梯度反向传播至 backbone，推动 misaligned 样本远离 bias attractor。

## 实验与结果
- **数据集**：Biased MNIST（4 种 $\rho$）、Multi-Color MNIST、CelebA（blond 属性，gender 为 bias）、9-class ImageNet + ImageNet-A。
- **基线对比**：与 supervised（Rubi、EnD、BiasCon+BBal、FairKL）、unsupervised（ReBias、HEX、LfF、EIIL、PGI、DebiAN、LearnedMixin、SoftCon、Group DRO）全面对比。
- **主要结果**：
  - **Biased MNIST**：在 $\rho=0.99, 0.995, 0.997$ 时达到 unsupervised SOTA（97.7%、95.5%、92.7%），在 $\rho=0.999$ 时因噪声较大略低于 BiasCon+BBal（58.7% vs 94.0%），但降低学习率后可提升至 72.6%。
  - **Multi-Color MNIST**：实现 unsupervised SOTA（unbiased accuracy 73.1%），在双偏置冲突样本（$R^\perp \cap L^\perp$）上达 24.1%，较 LfF（5.1%）提升 **+8%**。
  - **CelebA**：unbiased 90.2%（unsupervised SOTA，较 LfF +4.8%），bias-conflicting 84.5%（较 LfF +2.0%），综合排名第三（次于监督方法 BiasCon+BBal、LNL）。
  - **9-class ImageNet**：ResNet-18 直接提取时达 96.4%（unsupervised SOTA），ImageNet-A 达 34.2%，与 FairKL（34.5%）相当。
- **消融实验**：在 Biased MNIST ($\rho=0.99$) 上，加权 loss 贡献 +7.1%，IRH 贡献 +1.7%，条件互信息（仅 misaligned）进一步贡献 +0.5%。

## 相关工作脉络
- **LfF [30]**：假设 bias 特征"最早被学习"，放大早期预测并重新加权 difficult samples；本文不依赖该假设，而是通过 Voronoi 距离**主动寻找** bias 对齐最强的时刻，且能扩展到多偏置场景。
- **ReBias [5] / HEX [37]**：针对 texture bias 的 bias-tailored 方法（需先验知识）；本文方法为 bias-agnostic，不针对特定 bias 类型。
- **DebiAN [27] / EIIL [13] / PGI [1]**：通过优化公平性度量（Equalized Odds、IRM）或不变性原则推断 bias groups；本文不依赖 fairness 度量，而是从几何距离角度刻画 bias-target alignment。
- **BiasCon+BBal [19] / FairKL [6]**：监督方法，需 bias 标签；本文在不使用 bias 标签的情况下达到 comparable 甚至更优性能（尤其多偏置场景）。
- **DFA [26] / LWBC [25]**：通过 latent representation 解耦或 committee 训练提取 difficult samples；本文通过 Voronoi 边界距离**精确量化** alignment 时机，避免"过早提取"问题。
- **Information Removal at Bottleneck [33]**：本文 IRH 模块的设计灵感来源，但本文首次将其与**条件互信息**结合，仅针对 misaligned 样本去除 bias 信息。

## 局限性与未来方向
- **极端高相关场景下性能下降**：当 bias-target 相关性极高（如 $\rho=0.999$）时，bias-misaligned 样本极少且梯度噪声大，导致 $e^*$ 检测不稳定；论文建议调小学习率缓解。
- **完全偏置数据集失效**：若 $\rho=1$（所有样本均 bias-aligned），则无 misaligned 样本可供提取，方法失效。
- **学习率敏感**：bias 对齐信息的提取对学习率较敏感，需针对极端场景进行调参。
- **未来方向**：探索自适应学习率策略以应对极端偏置场景；将方法扩展至更多样化的 bias 类型（如空间位置、颜色、texture 混合）。

## 研究启发与可借鉴点
- **Voronoi 边界距离作为 bias 检测指标**：通过隐空间中误分类样本到目标类 Voronoi 边界的距离来量化 bias-target alignment，该几何视角可迁移至其他需要检测"何时学到偏见"的场景。
- **条件互信息正则化**：仅针对 misaligned 样本最小化互信息，而非全部样本，这一设计避免了破坏 aligned 样本的分类信息，可推广至其他去偏或公平性任务。
- **两阶段范式（提取→去偏）**：先用 vanilla 模型提取对齐信息，再用该信息指导去偏模型的训练，这一解耦设计便于模块替换（如更换 backbone 或距离度量），适合团队协作分工。
- **实验设计借鉴**：在 Multi-Color MNIST 上按 four subsets 交叉分析，能清晰揭示方法在多偏置场景下的行为；该评估方式值得在其他多偏置任务中复用。
- **可结合的方向**：将 Voronoi-based alignment 检测与自监督预训练结合，或在医疗、自动驾驶等高风险领域验证 bias-agnostic 去偏的实际效果。

## 关键术语表
- **Bias-agnostic**：不依赖 bias 标签或 bias 先验知识，直接从训练数据中推断 bias 的去偏方法。
- **Voronoi boundary**：在高维隐空间中，两个类别质心之间的中垂超平面，用于划分各类别的最近邻区域。
- **Bias-target alignment/misalignment**：样本的 bias 特征与目标类别是否一致；aligned 表示 bias 与目标一致（易学习），misaligned 表示 bias 与目标冲突（难学习）。
- **Information Removal Head (IRH)**：插入在 bottleneck 层的 auxiliary classification head，用于估计 latent representation 中 bias 信息的可提取性。
- **Conditional Mutual Information**：在给定 bias-misalignment 条件下，衡量 latent representation 与 bias label 之间的信息共享量，用作正则项去除 bias。
- **Weighted Loss Reweighting**：根据样本的 bias 对齐状态分配不同权重，misaligned 样本获得更高权重以鼓励模型学习泛化特征。
- **Spurious Correlation**：训练数据中与目标任务无关但统计上相关的特征（如背景颜色与物体类别的关联），易被模型错误利用。
- **Distortion Robustness**：模型在面对分布外（OOD）或对抗性扰动时保持性能的能力，去偏方法旨在提升此项指标。

## 可复现要素
- **数据集**：Biased MNIST、Multi-Color MNIST、CelebA、9-class ImageNet、ImageNet-A；论文未明确声明开源状态，CelebA 和 ImageNet 子集通常可从公开渠道获取。
- **代码/权重**：论文未提供开源代码或预训练权重（ICCV 2023 通常附 project page，但本文未提及）。
- **关键超参**：
  - $\lambda_{\mathcal{I}^\perp} = 2$（条件互信息正则系数）
  - IRH 学习率 = 0.1
  - Biased MNIST：80 epochs，初始 lr=0.1（epoch 40/60 衰减），weight decay=$10^{-4}$
  - Multi-Color MNIST：500 epochs
  - CelebA/9-class ImageNet：ResNet-18，沿用 [19,30] 的训练设置
- **硬件**：Nvidia GeForce RTX 3090 Ti (24GB)
- **实现**：Python + PyTorch 1.13
