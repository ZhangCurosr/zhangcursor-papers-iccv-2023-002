---
title: "Stochastic-Segmentation-with-Conditional-Categorical-Diffusi"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Zbinden_Stochastic_Segmentation_with_Conditional_Categorical_Diffusion_Models_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 00:56:19"
field: "视觉分割与不确定性量化"
keywords: ["随机语义分割", "类别扩散模型", "不确定性建模", "医学图像分割", "条件扩散"]
innovations: ["首次将类别扩散模型应用于条件语义分割，原生建模多专家标注分布", "在LIDC多标注数据集上以9M参数达到SOTA，超越多种已有方法", "结合Dino-ViT预训练特征，在Cityscapes高分辨率上超越多个重基线"]
benchmarks: ["LIDCv1", "LIDCv2", "Cityscapes-val"]
---

# 论文速读：Stochastic-Segmentation-with-Conditional-Categorical-Diffusi

## 一句话总结
论文提出了条件类别扩散模型（CCDM），首次将扩散模型从连续像素空间扩展到离散类别空间，用于对多专家标注的不确定性分布进行建模；在LIDC随机语义分割数据集上达到SOTA，同时在Cityscapes标准数据集上以更小参数量超过多个强基线。

## 研究问题与动机
- **单一确定性输出的局限性**：传统语义分割网络只生成一张"最优"分割图，但在医学诊断、自动驾驶等安全关键领域，专家标注本身存在分歧（aleatoric uncertainty），单一结果会给出虚假的信心感。
- **条件标签分布建模困难**：需学习 $q(\mathbf{x}_0 \mid I)$——给定图像时类别标签的条件分布，该分布具有多模态性、输出空间高维且标注数据有限。
- **已有扩散分割方法的局限**：现有基于扩散的分割方法（如SegDiff、MedSegDiff）均沿用原始DDPM的连续高斯扩散框架，需通过阈值化或量化等启发式手段得到类别输出，未能充分利用类别扩散模型的优势。

## 核心贡献（创新点）
1. **提出CCDM：首个将多项式扩散（multinomial diffusion）应用于条件语义分割的方法**，在离散类别空间直接建模去噪过程，避免连续↔离散的域切换。
2. **首次在多标注随机分割任务上实现SOTA**：在LIDC数据集（多专家 lung nodule 标注分歧场景）上以9M参数超越多种已有方法，是首个基于扩散模型的随机分割方法。
3. **在Cityscapes单标注任务上展现竞争力**：以30M参数的轻量模型优于DeepLabv3/HRNet等重基线，结合Dino-ViT预训练特征进一步提升至65.8 mIoU（256×512）。
4. **设计了与现有连续扩散分割方法的明确区分**：与SegDiff/AB/CIMD等连续扩散方法不同，CCDM原生处理类别变量，反向过程输出为 $D \times L$ 的概率张量而非连续噪声。

## 方法详解
- **类别扩散前向过程**：设像素标签 $x_t \in \mathcal{L}=\{1,\ldots,L\}$，前向过渡为逐像素类别分布：$q(x_t \mid x_{t-1}) = \mathcal{C}\!\left(x_t;\; \frac{\beta_t}{L}\mathbf{1} + (1-\beta_t)\mathbf{e}_{x_{t-1}}\right)$，其中 $\beta_t$ 为噪声调度系数。经 $t$ 步后可直接采样：$q(x_t \mid x_0) = \mathcal{C}\!\left(x_t;\; \frac{1-\bar{\alpha}_t}{L}\mathbf{1} + \bar{\alpha}_t \mathbf{e}_{x_0}\right)$。
- **后验计算**：由贝叶斯规则可得 $q(x_{t-1} \mid x_t, x_0) = \mathcal{C}(x_{t-1};\; \pi(x_t, x_0))$，其中 $\pi$ 由式(7)给出。
- **网络输出与反向过程**：网络 $f_\theta$ 接收当前标签图 $\mathbf{x}_t$、时间步 $t$ 和图像 $I$，输出 $\hat{\mathbf{P}}_0 \in [0,1]^{D\times L}$ 即每个像素对 $L$ 个类别的预测概率。通过式(12)将 $\hat{\mathbf{P}}_0$ 转换为 $t-1$ 步的预测参数 $\hat{\mathbf{p}}_{t-1}$，进而采样得到 $\mathbf{x}_{t-1}$。
- **条件机制**：图像 $I$ 以两种方式注入网络：①将原始像素值作为额外通道拼接至输入；②使用预训练 Dino-ViT 提取特征，拼接至 U-Net 编码器第三层（$\frac{1}{8}$ 空间分辨率）。
- **训练目标**：最大化 ELBO，等价于逐时间步采样 $t$ 并计算预测分布与后验分布之间的逐像素 KL 散度之和。
- **推理**：从均匀噪声 $\mathbf{x}_T$ 出发，沿反向链逐步去噪，最后一步取 argmax 输出确定性分割图；多次推理生成多样样本以估计分布。

## 实验与结果
- **数据集与基准**：
  - **LIDC**（多标注）：1,018例CT扫描，抽取肺结节中心切片，分辨率128×128，含两种划分LIDCv1（15,096张，6:2:2）和LIDCv2（12,816张，7:15:15）。评估指标：GED（越小越好）、HM-IoU（越高越好），采样数 $n=16/32/50/100$。
  - **Cityscapes**（单标注）：2,975训练图+500验证图，19类。评估指标：mIoU，分辨率128×256和256×512。
- **LIDC关键结果**：
  - LIDCv1 GED₁₆：**0.212**（SOTA，仅次于MoSE的0.218，但HM-IoU更优）；LIDCv1 HM-IoU₃₂：**0.631**。
  - LIDCv2 GED₁₆：**0.239**，HM-IoU₁₆：**0.598**（均为SOTA）。
  - 以9M参数超越42M参数的MoSE和24M参数的CIMD。
- **Cityscapes关键结果**（256×512）：
  - CCDM+Dino，10 samples融合：**mIoU 65.8%**，超过UPerNet+Swin-Tiny的66.0%。
  - 单样本即达64.0%，超越DeepLabv3/HRNet等多组基线。
- **消融**：表3显示UNet层数和通道数增加可显著提升mIoU；Dino特征对高分辨率（256×512）提升更大（+3.7pp vs +2.3pp）。

## 相关工作脉络
1. **Probabilistic U-Net / HProb. Unet / JProb. Unet**：基于cVAE的随机分割范式，用低维潜变量编码分割变体；CCDM摒弃隐变量假设，直接在类别标签空间建模扩散分布，输出更丰富多样。
2. **PhiSeg / SSN / cFlow / MoSE / PixelSeg / CAR**：各自采用层次化cVAE、正态流、自回归PixelCNN、混合专家等不同分布建模策略；CCDM是其中首个将扩散模型引入该方法系的尝试。
3. **SegDiff / MedSegDiff（连续扩散分割）**：沿用DDPM高斯扩散框架，通过对连续输出阈值化得到类别；CCDM原生处理类别变量，无需域转换。
4. **AB（Analog Bits）**：将类别编码为比特位并用连续扩散建模；CCDM直接对 $L$ 类one-hot概率扩散，更简洁自然。
5. **CIMD（Collectively Intelligent Medical Diffusion）**：最近提出的连续扩散医学分割方法；本文与其对比展示了类别扩散在医学随机分割上的优势。

## 局限性与未来方向
- **采样速度慢**：250步反向过程导致推理耗时，需依赖DDIM等加速技巧；论文将其列为正交研究方向。
- **高分辨率扩展困难**：类别扩散在高分辨率上训练不稳定，需级联或多阶段方案（类似Latent Diffusion思路），但当前对类别数据尚无成熟的latent扩散框架。
- **需多标注数据**：当前方法假设图像有 $L$ 个专家标注以拟合标签分布，单标注场景下退化为确定性预测（Cityscapes实验需融合多样本）。

## 研究启发与可借鉴点
1. **类别扩散框架可迁移至其他离散输出任务**：如图像修复（discrete mask inpainting）、超分辨率（discrete level quantization）中同样存在类别建模需求，CCDM的 $f_\theta$ 条件注入方式可直接复用。
2. **Dino-ViT 特征与扩散模型结合的策略**：将预训练ViT特征拼接至U-Net中间层而非仅做输入拼接，对高分辨率任务增益显著，可作为通用增强手段。
3. **推理时的样本融合策略**：Cityscapes实验中"概率均值融合"优于"多数投票"，说明直接用预测概率分布而非硬投票有利于利用多样性信息。
4. **不确定性度量与下游决策的结合**：本方法生成的多样本可直接用于构建预测置信度估计，适合与主动学习、在线校正等场景结合。

## 关键术语表
- **CCDM（Conditional Categorical Diffusion Model）**：本文提出的模型，以输入图像为条件、在类别空间进行去噪扩散，生成多样化分割图。
- **Aleatoric Uncertainty（偶然不确定性）**：源于数据本身固有的歧义性（如多专家标注分歧），不同于认知不确定性。
- **GED（Generalised Energy Distance）**：衡量生成样本分布与真实标注分布差异的距离指标，值越小越好。
- **HM-IoU（Hungarian-Matched IoU）**：先用匈牙利算法匹配生成样本与ground truth，再计算IoU，避免样本顺序敏感问题。
- **Multinomial Diffusion**：Hoogeboom等人提出的将DDPM推广到类别变量的扩散框架，前向过程以固定概率 $\beta_t$ 随机重置类别。
- **Polyak Averaging**：模型参数的指数移动平均技术，训练后期稳定收敛，此处 $\alpha=0.99995$。
- **Cosine Schedule**：$\beta_t$ 按余弦曲线从接近0变化到1的噪声调度策略，比线性调度效果更好。

## 可复现要素
- **数据集**：LIDC（公开）；Cityscapes（公开）。
- **代码**：论文未明确提及GitHub链接（需从官方作者页查找）。
- **关键超参**：$T=250$ 步；$\beta_t$ 余弦调度；Adam优化器；学习率 $1\text{e-}4 \to 1\text{e-}6$ 多项式衰减；Batch Size 64（LIDC）/16（Cityscapes高分辨率）；Polyak平均 $\alpha=0.99995$（LIDC）/$0.999$（Cityscapes）。
- **预处理**：LIDC随机翻转+旋转（0/90/180/270°）；Cityscapes resize+color jitter+随机翻转+ImageNet归一化。
- **Dino-ViT**：使用预训练Dino-ViT-S提取 $\frac{1}{8}$ 分辨率特征，非微调。
