---
title: "Learning-Neural-Eigenfunctions-for-Unsupervised-Semantic-Seg"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Deng_Learning_Neural_Eigenfunctions_for_Unsupervised_Semantic_Segmentation_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 19:12:55"
field: "无监督视觉分割"
keywords: ["无监督语义分割", "谱聚类", "神经特征函数", "ViT", "端到端聚类"]
innovations: ["将谱聚类参数化为神经网络特征函数，避免显式特征分解", "双图核融合语义特征与底层像素信息", "Gumbel-Softmax量化神经特征函数输出为离散聚类分配"]
benchmarks: ["Pascal Context", "Cityscapes", "ADE20K"]
---

# 论文速读：Learning-Neural-Eigenfunctions-for-Unsupervised-Semantic-Seg

## 一句话总结
论文将谱聚类重构为端到端参数化神经网络方法，通过神经网络特征函数（Neural Eigenfunctions）在预训练ViT特征空间上直接学习谱嵌入并量化为聚类分配，实现了高效、可扩展的无监督语义分割，在Pascal Context、Cityscapes和ADE20K上均超越MaskCLIP、ReCo等基线。

## 研究问题与动机
1. **传统谱聚类缺乏语义理解能力**：在原始像素上操作，对色彩变换敏感，无法识别语义相似性。
2. **谱分解计算代价高昂**：大型数据集上矩阵特征分解复杂度为 $O((NHW/P^2)^3)$，不可扩展。
3. **非参数化导致泛化困难**：传统谱聚类是transductive的，对新测试样本需额外重新求解特征分解，无法端到端推理。
4. **现有改进方法仍存瓶颈**：DSM [30] 利用预训练ViT改善了语义性，但跨图像同步步骤复杂，且仍需对每幅图单独进行谱分解，效率与灵活性受限。

## 核心贡献（创新点）
1. **将谱聚类参数化为神经网络特征函数逼近**：用轻量NN $\psi$ 学习核 $\kappa$ 的主特征函数，避免显式特征分解，支持任意新样本直接推断。
2. **设计结合高层语义与底层像素的双图核**：以预训练ViT patch特征构建余弦相似度最近邻图捕获语义，以下采样像素构建 $L^2$ 距离最近邻图捕获几何边界，加权融合为统一核函数。
3. **通过Gumbel-Softmax量化输出为离散聚类分配**：约束神经特征函数输出为 $K$ 维one-hot向量，省去后验K-means等显式聚类步骤，实现端到端可微的聚类流程。
4. **建立轻量高效的端到端谱聚类范式**：仅在最后一层attention block输出上增加2个linear self-attention block + 线性头，训练耗时约半天（RTX 3090），测试时零额外开销。

## 方法详解
### 整体框架
方法以预训练ViT提取的patch特征为输入，构建图核 $\kappa$，通过神经网络 $\psi$ 学习其特征函数，经Gumbel-Softmax量化后双线性上采样至原图分辨率，argmax得到聚类掩码，最后用多数投票或Hungarian matching对齐语义类别。

### 图核构造（Section 3.3）
- **语义图**：对预训练ViT输出的patch特征矩阵 $\mathbf{F} \in \mathbb{R}^{NHW/P^2 \times C}$，构建余弦相似度 $k$-NN 图（$k=256$），邻接矩阵：
  $$\mathbf{A}_{u,v} = \frac{\mathbf{F}_u \mathbf{F}_v^\top}{\|\mathbf{F}_u\| \|\mathbf{F}_v\|}, \quad v \in k\text{-NN}(u, \mathbf{F}, \text{cosine})$$
- **像素图**：将原图双线性下采样至 $H/P \times W/P$，拼接空间坐标得 $\tilde{\mathbf{X}} \in \mathbb{R}^{NHW/P^2 \times 5}$，构建 $L^2$ 距离 $k$-NN 图（限于同一图像内邻居），邻接矩阵 $\tilde{\mathbf{A}}_{u,v} = 1$ 若 $v \in \tilde{k}\text{-NN}(u, \tilde{\mathbf{X}}, L^2)$，否则为0。
- **融合核**：归一化邻接矩阵加权求和：
  $$\kappa = \mathbf{D}^{-1/2} \mathbf{A} \mathbf{D}^{-1/2} + \alpha \tilde{\mathbf{D}}^{-1/2} \tilde{\mathbf{A}} \tilde{\mathbf{D}}^{-1/2}, \quad \alpha = 0.3$$

### 神经特征函数学习（Section 3.4）
采用NeuralEF [12] 目标函数，近似核 $\kappa$ 的 $K$ 个主特征函数：
$$\max_\psi \sum_{j=1}^K \mathbf{R}_{j,j} - \beta \sum_{j=1}^K \sum_{i=1}^{j-1} \widehat{\mathbf{R}}_{i,j}^2, \quad \text{s.t. } \mathbb{E}_{p(\mathbf{f})}[\psi(\mathbf{f}) \circ \psi(\mathbf{f})] = \mathbf{1}$$
其中 $\mathbf{R}$ 为核与输出的协方差矩阵估计，$\widehat{\mathbf{R}}$ 使用 stop-gradient 版本打破特征函数间对称性。蒙特卡洛估计：
$$\mathbf{R} \approx \frac{1}{B^2} \boldsymbol{\Psi} \cdot \kappa(\mathbf{F}^B, \mathbf{F}^B) \cdot \boldsymbol{\Psi}^\top, \quad \widehat{\boldsymbol{\Psi}} = \text{stop\_grad}(\boldsymbol{\Psi})$$
网络 $\psi$ 结构：固定ViT backbone + 2个linear self-attention block + 线性头（输出维度 $K$，权重列正交约束）+ $L^2$ batch norm。

### 量化与推理（Section 3.5-3.6）
训练时：Gumbel-Softmax（温度从1余弦退火至0.3）+ $L^2$ batch norm，使输出趋近one-hot。测试时：移除两者，直接对 $\psi(f(\mathbf{x}))$ 取softmax logits，argmax得聚类分配，双线性上采样至原分辨率后多数投票匹配类别。

## 实验与结果
### 数据集与设置
- **Pascal Context**（60类）：标准裁剪到 $320\times320$ 评估；滑窗协议下含背景类评估。
- **Cityscapes**（27类）：同上；滑窗协议下19类评估。
- **ADE20K**（150类）：滑窗协议评估。
- 预训练模型：timm库 ViT-S/B/L（ImageNet-21k预训练 + ImageNet微调），分辨率 $384\times384$，patch大小 $P=16$。
- 超参：$K=256$（ADE20K用512），$\beta=0.08$，$\alpha=0.3$，学习率 $10^{-3}$，Adam，40 epochs，batch=16，单卡RTX 3090约半天。

### 主要结果
**Pascal Context（Table 1）**：
- Ours (ViT-S): **Acc 70.4%, mIoU 38.8%**；vs MaskCLIP 25.5 / ReCo 27.2 / K-means(ViT-S) 28.9，提升显著。
- Ours*（取线性头前特征）: 39.6%，略优于Ours。

**Cityscapes（Table 2）**：
- Ours (ViT-S): **Acc 83.4%, mIoU 28.2%**；vs MaskCLIP 10.0 / ReCo 19.3 / K-means 22.4。
- Ours* (ViT-L): **30.7%**，为Cityscapes最高。

**滑窗协议（Table 3，ViT-S）**：
- Pascal Context: Ours mIoU **41.4%**（Supervised DeepLabv3+ 48.5%）；Cityscapes: Ours mIoU **46.7%**（Supervised 77.3%）。

**ADE20K（Table 4，ViT-S）**：
- Ours mIoU **21.6%**（K-means 19.2%），Ours* **23.6%**。

**零样本迁移（Table 8）**：在ImageNet上训练5 epoch，迁移至Pascal Context / Cityscapes：mIoU 15.2% / 18.5%，与MaskCLIP/ReCo参考值接近但显著低于全量训练。

**CLIP预训练骨干（Table 9）**：ViT-L/14@336px + CLIP → Pascal Context mIoU **44.0%**，超过ImageNet预训练的ViT-B（37.2%）和ViT-L（38.7%），表明对比预训练更适配本方法。

## 相关工作脉络
1. **DSM [30]**：首篇将谱聚类与ViT预训练特征结合的方法，构建patch-wise语义亲和矩阵+像素图，仍需每图单独谱分解与跨图同步；本文将其参数化消除分解步骤，并支持端到端测试推断。
2. **MaskCLIP [48] / ReCo [39]**：基于CLIP的zero-shot语义分割，依赖精心设计的文本prompt与自训练；本文无需文本、无需跨模态对齐，直接利用视觉特征进行聚类，且无需自训练即可超越前者。
3. **NeuralEF [12]**：提出用神经网络逼近核函数的特征函数，定义新的正交化目标打破特征函数对称性；本文首次将其引入 dense prediction 的谱聚类任务，并结合Gumbel-Softmax实现离散输出。
4. **SpIN [34]**：早期深度谱推断网络，学习目标不严谨，仅学到特征函数张成的子空间而非函数本身，需额外昂贵策略补救；NeuralEF解决了此问题，本文直接采用后者。
5. **SSL聚类方法（IIC [21], PiCIE [7], STEGO [16]）**：通过聚类目标、对比学习或特征蒸馏学习分割特征；本文证明在强预训练ViT特征上，经典谱聚类经参数化改造后仍是强基线，且无需设计专门表征学习目标。
6. **传统谱聚类 [38]**：在原始像素上执行归一化割；本文继承其理论根基但完全重构为参数化、特征空间操作、端到端可微版本。

## 局限性与未来方向
1. **仍需地面真值进行聚类-类别对齐**：多数投票/Hungarian matching需标注数据，限制了纯零样本部署能力；作者建议引入文本prompt引导聚类是潜在方向。
2. **ADE20K细粒度性能受限**：预训练ViT缺乏patch-level细粒度监督，导致相似语义被合并（如台灯与路灯）；建议结合patch-wise自监督损失微调骨干。
3. **零样本迁移性能差距大**：在ImageNet上训练的模型迁移到复杂场景图像时mIoU大幅下降，因源域与目标域分布差异显著，需更现实的预训练数据。
4. **K值选择依赖经验**：过大K增加计算开销且NeuralEF可能无法收敛到小特征值对应的特征函数；需折中选取。
5. **未与DSM直接对比**：因DSM的跨图像同步实现复杂，作者仅理论推测两者性能相近但本方效率更高。

## 研究启发与可借鉴点
1. **NeuralEF + Gumbel-Softmax 的"谱聚类参数化"范式**：可将任意核函数（如对比学习产生的相似度矩阵）转化为可微端参量化版本，直接适用于视频分割、点云分割、图数据等需批量聚类的dense预测任务。
2. **双图核设计思想**：高层语义（特征余弦NN图）+ 底层细节（像素 $L^2$ NN图）的加权融合策略可迁移至其他需要兼顾语义一致性与边界精度的任务（如图像分割、异常检测）。
3. **K > 真实类别数**：消融表明 $K=256$ 显著优于 $K=64/128$，说明过参数化聚类维度有助于捕获细粒度结构；可为其他聚类方法的超参选择提供先验。
4. **CLIP预训练优于ImageNet分类预训练**：表9显示CLIP骨干在该框架下mIoU持续提升，提示对比学习特征更适合谱嵌入分解，值得在相关任务中系统验证。
5. **"Ours*"启发**：取出线性头前的更高维特征做后验K-means反而略优，说明神经特征函数的瓶颈层存在信息压缩；可探索保留中间层特征的混合策略。

## 关键术语表
**谱聚类（Spectral Clustering）**：基于图拉普拉斯矩阵特征向量进行数据分块的经典聚类方法，理论上等价于最小化图归一化割。
**神经特征函数（Neural Eigenfunction）**：用神经网络逼近积分核特征函数的技术（NeuralEF），避免显式矩阵分解，支持泛化到新样本。
**图核（Graph Kernel）**：由图邻接矩阵（归一化后）定义的相似度核，本文融合语义图与像素图，作为特征函数学习的输入。
**Gumbel-Softmax**：可微的类别重参数化技巧，使离散one-hot输出能在反向传播中传递梯度，训练结束后移除以获得硬分配。
**DSM（Deep Spectral Methods）**：Melas-Kyriazi等提出的将ViT特征融入传统谱聚类的无监督分割基线，本文在其基础上参数化改造。
**Transductive vs. Inductive**：Transductive方法仅对训练数据建模，无法直接推断新样本；本文方法转为inductive（参数化），支持任意新数据一次前向即可得到分割。
**多数投票（Majority Voting）**：将测试图上每个像素所属聚类在所有训练图中统计出现频率，以最高频聚类为其语义标签。
**Linear Self-Attention**：Katharopoulos等提出的 $O(N)$ 复杂度注意力机制，本文用于构建轻量神经特征函数网络。

## 可复现要素
- **数据集**：Pascal Context、Cityscapes、ADE20K（均为公开数据集）。
- **代码**：已开源，见 https://github.com/thudzj/NeuralEigenfunctionSegmentor。
- **预训练模型**：timm库提供的ViT-S/B/L（ImageNet-21k预训练 + ImageNet微调），分辨率 $384\times384$。
- **关键超参**：patch大小 $P=16$，$K=256$（ADE20K用512），$\alpha=0.3$，$\beta=0.08$，$k=256$（语义图NN），学习率 $10^{-3}$，40 epochs，batch=16，Gumbel温度从1退火至0.3，Adam优化器，无weight decay。
- **评估协议**：标准裁剪至 $320\times320$（Pascal/Cityscapes）；滑窗协议（参照Segmenter [40]）；背景类处理依数据集而定。
- **后处理**：可选CRF细化（论文称贡献边际）。
