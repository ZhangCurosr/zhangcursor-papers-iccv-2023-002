---
title: "WaterMask-Instance-Segmentation-for-Underwater-Imagery"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Lian_WaterMask_Instance_Segmentation_for_Underwater_Imagery_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-05 01:57:30"
field: "水下视觉/实例分割"
keywords: ["水下实例分割", "UIIS数据集", "图注意力", "边界感知分割", "多尺度特征融合", "图像退化恢复"]
innovations: ["DSGAT图注意力模块重建退化图像细节", "MFRM多级特征融合模块精细预测前景与边界掩码", "BMS边界掩码策略与BLL边界学习损失联合优化分割边界"]
benchmarks: ["UIIS（自建水下实例分割数据集）", "COCO mask AP评测协议"]
---

# 论文速读：WaterMask: Instance Segmentation for Underwater Imagery

## 一句话总结
本文提出首个通用水下图像实例分割数据集 **UIIS**（4,628张图像，7个类别，像素级标注），并设计首个专用于水下场景的实例分割模型 **WaterMask**，通过图注意力恢复退化细节、多级特征融合优化预测、边界学习策略精细化掩码，在 UIIS 上显著优于 Mask R-CNN 等基线。

## 研究问题与动机
- **缺乏通用多类别数据集**：现有水下标注数据仅针对特定目标（如鱼类 DeepFish、管道泄漏点），或仅支持检测/语义分割任务，无法支撑水下多类别实例分割研究。
- **水下成像质量退化**：波长相关衰减、散射导致图像细节丢失，加之浮游生物引起的"海雪"噪声，使直接移植自然图像实例分割方法（如 Mask R-CNN）效果显著下降。
- **多下采样操作丢失边界细节**：经典实例分割网络经历过多层下采样， underwater 对象的形状与边界信息难以恢复，鱼群、珊瑚等聚合实例的边界重叠更加剧了难度。
- **实例尺度分布极端**：UIIS 中小于 14×14 像素的微小实例占 11.7%，大于 128×128 的大实例占 22.8%，对分割网络提出双重挑战。

## 核心贡献（创新点）
1. **构建首个通用水下实例分割数据集 UIIS**：与已有专用数据集（如 DeepFish、WishFish）本质不同，覆盖 7 类多类别对象，支持多类别实例分割任务的系统研究。
2. **提出 Difference Similarity Graph Attention Module（DSGAT）**：利用图注意力在网络深层最高分辨率特征 $P_2$ 上重建因质量退化和下采样丢失的细节；与已有注意力机制（如 CBAM）的本质区别在于其显式建模图像块间的空间-相似性图关系，而非通道/空间维度的独立加权。
3. **提出 Multi-level Feature Refinement Module（MFRM）**：将 DSGAT 重建的高分辨率细粒度特征与 FPN 中层级特征（$P_3$–$P_6$）逐级融合，迭代生成前景掩码与边界掩码；与标准 Mask R-CNN ROIHead 的本质区别在于细粒度残差流的多级补充机制，且在两次迭代中不增加额外参数。
4. **提出 Boundary Mask Strategy（BMS）+ Boundary Learning Loss（BLL）**：通过拉普拉斯边界判定将低分辨率前景掩码与高分辨率边界掩码按位置融合，并用 BLL 强化边界像素分类权重；与 BMask R-CNN 的区别在于 BMS 显式利用不同分辨率特征的互补结构，而非仅附加边界感知头。
5. **全面验证与泛化实验**：在 UIIS 上系统性评测，并将 WaterMask 迁移至 Cascade R-CNN 框架，证明模块的框架无关性。

## 方法详解
**整体框架**（图 4）：Backbone（ResNet）+ FPN 输出 $P_2$–$P_5$ → **DSGAT** 重建 $P_2$ → **MFRM** 融合多尺度特征 → 输出前景掩码 $M_2$ 与边界掩码 $M_3$ → **BMS** 融合为最终输出 $M_{out}$。

**DSGAT（§4.1）**：
- 输入 FPN 最高分辨率特征 $P_2 \in \mathbb{R}^{h \times w \times c}$，经 $s \times s$ 卷积下采样得到 $P_2' \in \mathbb{R}^{h/s \times w/s \times d}$，每行视作图节点 $\vec{h_i}$，对应原图 $4s \times 4s$ 像素块。
- 图边仅连接距当前节点欧氏距离最远的 $k$ 个邻居节点（而非全连接），以降低计算复杂度。
- 注意力系数：
$$a_{ij} = \frac{\exp(\sigma(l^\top [W\vec{h_i}\ \| \ W\vec{h_j}]))}{\sum_{n \in \mathcal{N}_i} \exp(\sigma(l^\top [W\vec{h_i}\ \| \ W\vec{h_n}]))}$$
- 节点特征更新：$\vec{h_i'} = \delta(\sum_{n \in \mathcal{N}_i} a_{in} W \vec{h_n})$，输出经反卷积上采样得到残差流 $P_{res}$。
- 超参：patch 大小 $12 \times 12$（stride $s=3$），邻居数 $k=11$。

**MFRM（§4.2）**：
- 经 $14 \times 14$ RoIAlign 提取初始实例特征 $F_1$，经两个 $3 \times 3$ 卷积层生成。
- 迭代两次：每次从 $P_2^*$ 用 RoIAlign 提取细粒度局部特征，与上一阶段输出拼接，经 $1 \times 1$ 卷积（通道减半）+ $2 \times$ 上采样，得到 $F_2$（前景）与 $F_3$（边界）。
- 总参数量无额外增加。

**BMS（§4.3）**：
- $F_2, F_3$ 经 $1 \times 1$ 卷积得 $M_2, M_3$。
- 边界判定使用离散 Laplacian 算子 $\nabla^2 p$（卷积核 $b \times b$，中心权 $b^2-1$，其余 $-1$）：
$$B(M) = \begin{cases} 1, & |\nabla^2 p(M)| \leq \mu b^2 \\ 0, & \text{otherwise} \end{cases}, \quad \mu=0.15$$
- 最终掩码：
$$M_{out} = f_{2\times}(M_2) \odot B_{2\times} + M_3 \odot (1 - B_{2\times})$$

**BLL（§4.4）**：
- 聚焦边界像素的二叉交叉熵加权损失：
$$\mathcal{L}_B = \frac{\sum_{i}^{H \times W} R_{2\times}^i \cdot BCE(M_3^i, G_3^i)}{\sum_{i}^{H \times W} R_{2\times}^i}, \quad R_{2\times}^i = f_{2\times}(B(M_2) \vee B(G_2))$$
- 总掩码损失：$\mathcal{L}_{mask} = \mathcal{L}_B + \sum_{k \in [1,2]} \lambda_k \mathcal{L}_{BCE}(M_k, G_k)$，其中 $\lambda_1=0.25, \lambda_2=0.65$。

## 实验与结果
- **数据集**：UIIS，4,628 张 RGB 图像，7 类（Fish, Reefs, Aquatic plants, Wrecks/ruins, Human divers, Robots, Sea-floor），训练/验证拆分 3,937 / 691 张。
- **评估指标**：标准 COCO mask AP 系列（mAP, AP$_{50}$, AP$_{75}$, AP$_S$, AP$_M$, AP$_L$）及各类别 AP（AP$_f$鱼类, AP$_h$人类, AP$_r$沉船）。
- **基线对比（ResNet-101-FPN，3× schedule）**：
  - WaterMask R-CNN mAP = **27.2**，较 Mask R-CNN（23.4）提升 **+3.8 mAP**。
  - Cascade WaterMask R-CNN mAP = **27.1**，较 Cascade Mask R-CNN（25.5）提升 **+1.6 mAP**。
- **关键提升**：在严格 IoU=0.80 条件下，WaterMask R-CNN 比 Mask R-CNN 提升 **+6.1 mAP**（图 6），说明边界定位精度显著改善。
- **SOTA 对比（Table 3）**：WaterMask R-CNN‡ mAP=27.2，超过 BMask R-CNN（+5.1）、PointRend（+1.3）、QueryInst（持平），弱于 SOLOv2（27.3）和 R³-CNN（27.5）但综合各项指标具有竞争力；Cascade WaterMask R-CNN 在鱼类/人类 AP 上分别领先 Mask2Former 达 5.9 / 3.9 mAP。
- **消融（Table 4）**：w/o DSGAT（−1.4 mAP）、w/o MFRM（−2.5 mAP）、w/o BMS（−3.1 mAP）、w/o BLL（−1.7 mAP），四项均有贡献，BMS 贡献最大。

## 相关工作脉络
1. **Mask R-CNN（He et al., 2017）**：本文的框架基线；差异在于 WaterMask 在 RoIHead 之前引入 DSGAT 重建高分辨率细节，而非直接使用 FPN 输出。
2. **BMask R-CNN（Cheng et al., 2020）**：同样关注边界，但通过附加边界感知头实现；WaterMask 的 BMS 从结构上利用不同分辨率特征的互补，无需额外边界分支。
3. **Cascade Mask R-CNN（Cai & Vasconcelos, 2018）**：采用多级精炼检测器提升 mAP；本文将其作为验证框架泛化性的载体，而非直接对比改进。
4. **QueryInst / Mask2Former（Transformer-based）**：基于可学习 query 的全端到端分割；WaterMask 属于两阶段范式，优势在于对小样本/边界敏感场景的高精度拟合。
5. **DeepFish（Garcia-D'Urso et al., 2022）**：仅针对鱼类实例分割；UIIS 的 7 类别覆盖更全面，支持多类别场景。
6. **水下语义分割数据集（Islam et al., 2020）**：面向像素级语义；UIIS 首次提供像素级实例标注，填补实例级空白。

## 局限性与未来方向
- **数据集规模有限**：4,628 张图像对于实例分割任务偏小，作者承认需扩展更大规模、更具挑战性的复杂水下图像。
- **小物体性能略有下降**：DSGAT 重建可能在 3× schedule 下对小目标（AP$_S$）造成轻微干扰（0.3 mAP 落后于 R³-CNN）。
- **DSGAT 内存开销**：当 patch 设为 $8 \times 8$（stride=2）时显存超出限制，当前最优 stride=3 仍受显存约束，限制了更高分辨率特征的直接利用。
- **未来方向**：扩展 UIIS 规模、覆盖更多复杂水下场景；探索轻量级 DSGAT 变体以改善小目标表现；适配 Transformer 范式进行全局上下文建模。

## 研究启发与可借鉴点
1. **DSGAT 的图注意力细节恢复思路**可迁移至其他低质量退化图像分割任务（如雾天、模糊、医学影像），通过构建空间-相似性图聚合互补信息。
2. **BMS 的多分辨率掩码融合策略**（拉普拉斯边界判定 + 位置替换）设计简洁且参数零增长，可在任何需要精细边界的分割网络中作为后处理模块复用。
3. **MFRM 的迭代特征补充机制**（两次迭代不增加参数量）为高分辨率特征利用率不足的问题提供了轻量级解决方案，值得在视频分割、遥感分割中探索。
4. **BLL 的边界像素加权训练策略**（根据前景预测与 GT 边界区域的并集动态加权）可与任何边界感知损失（如 LOE、BCE+dice）组合使用，增强边界回归效果。
5. **UIIS 的数据集构建流程**（UCIQE/UIQM 质量筛选 + 多人协商标注 + COCO 格式）为领域特定数据集建设提供了可复现的工程范式。

## 关键术语表
- **UIIS（Underwater Image Instance Segmentation）**：首个通用水下实例分割数据集，包含 4,628 张图像、7 类别像素级标注，格式为 COCO 标准。
- **DSGAT（Difference Similarity Graph Attention Module）**：基于图注意力的细节恢复模块，在 FPN 最高分辨率特征层上通过相似性图聚合互补退化细节。
- **MFRM（Multi-level Feature Refinement Module）**：多级特征精炼模块，将 DSGAT 重建的细粒度特征与 FPN 中层级特征融合，迭代生成前景与边界预测特征。
- **BMS（Boundary Mask Strategy）**：边界掩码策略，通过 Laplacian 算子判定边界位置，融合低分辨率前景掩码与高分辨率边界掩码。
- **BLL（Boundary Learning Loss）**：边界学习损失，对边界像素区域赋予更高 BCE 权重，引导网络专注边界分类。
- **AP$_S$ / AP$_M$ / AP$_L$**：分别表示小（<32²）、中（32²~96²）、大（>96²）实例的平均精度。
- **FPN（Feature Pyramid Network）**：多尺度特征金字塔，输出 $P_2$–$P_5$ 四级特征用于不同尺度目标检测与分割。
- **RoIAlign**：Region of Interest Align 操作，精确提取 ROI 对应特征并消除量化误差。

## 可复现要素
- **数据集**：UIIS 已公开，4,628 张图像，7 类别，COCO 格式。
- **代码与权重**：开源地址 https://github.com/LiamLian0727/WaterMask；基于 PyTorch + MMDetection。
- **关键超参**：DSGAT patch 大小 $12 \times 12$（stride $s=3$），邻居数 $k=11$；BMS Laplacian 核尺寸训练时 $7 \times 7$、测试时 $9 \times 9$，$\mu=0.15$；BLL 权重 $\lambda_1=0.25, \lambda_2=0.65$；学习率 $2.5 \times 10^{-3}$，SGD 优化，每张 GPU 2 张图。
- **硬件**：NVIDIA A5000 GPU；另使用 MindSpore Lite 部署推理。
- **训练策略**：1× 和 3× schedule（3× 含 multi-scale training）。
