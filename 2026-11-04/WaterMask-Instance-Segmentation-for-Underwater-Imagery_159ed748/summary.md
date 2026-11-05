---
title: "WaterMask-Instance-Segmentation-for-Underwater-Imagery"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Lian_WaterMask_Instance_Segmentation_for_Underwater_Imagery_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 08:27:26"
field: "水下视觉感知"
keywords: ["水下图像", "实例分割", "图注意力", "边界优化", "数据集构建", "WaterMask", "UIIS"]
innovations: ["首个通用水下实例分割数据集UIIS（4628张，7类别）", "DSGAT图注意力模块恢复水下退化细节", "BMS+BLL边界感知分割策略提升聚集实例边界精度"]
benchmarks: ["UIIS", "COCO mask AP"]
---

# 论文速读：WaterMask-Instance-Segmentation-for-Underwater-Imagery

## 一句话总结
本文提出了首个通用水下图像实例分割数据集 **UIIS**（4628 张图像、7 类别、像素级标注）以及首个专为水下场景设计的实例分割模型 **WaterMask**；通过图注意力特征恢复、多尺度特征融合与边界优化策略，在水下质量退化与实例聚集场景下实现显著性能提升。

## 研究问题与动机
1. **缺乏通用数据集**：现有水下标注数据多为单类目标（如鱼类）或仅面向检测/语义分割任务，无法支撑多类别实例分割研究。
2. **水下成像退化严重**：波长相关衰减、散射、海雪噪声等导致图像质量劣化，边界模糊、细节丢失。
3. **实例聚集与遮挡**：鱼群、珊瑚礁等实例高度聚集且相互遮挡，传统方法因过多下采样难以精确分割边界。
4. **尺度差异大**：数据集中存在大量极小（<14×14，占 11.7%）与极大（>128×128，占 22.8%）实例，分割难度高。

## 核心贡献（创新点）
1. **首个通用水下实例分割数据集 UIIS**：4628 张 RGB 图像、7 类别（鱼、珊瑚礁、水生植物、沉船/废墟、潜水员、机器人、海底地形），采用 COCO 格式标注。
2. **差异相似图注意力模块 DSGAT**：在特征金字塔最高分辨率层 $P_2$ 上构建图节点，通过最远 k 邻接 + 图注意力聚合相似区域的退化残差细节，以残差形式补充丢失信息。
3. **多级特征细化模块 MFRM**：将 DSGAT 生成的全局细粒度特征与 $P_3$–$P_6$ 局部粗粒度特征逐阶段融合，输出 $F_2$（前景掩码）与 $F_3$（边界掩码）。
4. **边界掩码策略 BMS + 边界学习损失 BLL**：用 Laplacian 算子从低分辨率掩码提取边界，将高分辨率边界掩码 $M_3$ 替换至 $M_2$ 的边界区域；BLL 对边界像素赋更高权重，迫使网络更关注模糊边界的分类。
5. **广泛的实验验证与 SOTA 对比**：在 Mask R-CNN 和 Cascade R-CNN 框架下均显著提升，较 Mask R-CNN 最高提升 3.8 mAP；在 fish/human/wreck 等关键类别上大幅领先 Transformer 基线。

## 方法详解
**整体框架**：WaterMask 基于 Mask R-CNN 架构，替换/增强其 mask head 部分，由 DSGAT、MFRM、BMS 三模块串联构成。

1. **DSGAT（差异相似图注意力模块）**
   - 输入：特征金字塔最高分辨率层 $P_2 \in \mathbb{R}^{h \times w \times c}$
   - 先经 $s \times s$ 卷积（stride s）下采样为 $P_2' \in \mathbb{R}^{h/s \times w/s \times d}$，每个像素对应原图 $4s \times 4s$ 区域作为图节点 $h_i \in \mathbb{R}^d$
   - 构建图：每个节点仅连接与其欧氏距离最远的 $k$ 个节点（$k=11$）
   - 图注意力计算：
     $$a_{ij} = \frac{\exp(\sigma(l^\top [Wh_i \| Wh_j]))}{\sum_{n \in \mathcal{N}_i} \exp(\sigma(l^\top [Wh_i \| Wh_n]))}$$
   - 特征更新：$\vec{h_i'} = \delta(\sum_{j \in \mathcal{N}_i} a_{ij} W \vec{h_j})$，$\delta$ 为 ELU
   - 反卷积 + 上采样得到残差特征 $P_{res} \in \mathbb{R}^{h \times w \times d}$，与原始 $P_2$ 相加后送入后续模块

2. **MFRM（多级特征细化模块）**
   - 用 $14 \times 14$ RoIAlign 从 $P_3$–$P_6$ 提取局部粗粒度特征，经两个 $3 \times 3$ 卷积得到初始实例特征 $F_1$
   - 迭代两次：每次从 $P_2^*$（DSGAT 输出）用 RoIAlign 提取同尺度细粒度特征，与上一阶段输出拼接，经 $1 \times 1$ 卷积（通道减半）+ $2 \times$ 上采样得到 $F_2$、$F_3$
   - 通道数不增加，无额外计算负担

3. **BMS（边界掩码策略）**
   - $F_2$、$F_3$ 分别经 $1 \times 1$ 卷积得到不同分辨率掩码 $M_2$、$M_3$
   - 用 Laplacian 算子 $\nabla^2 p$ 从 $M_2$ 提取边界：
     $$p_{ij} = \begin{cases} b^2 - 1, & i=j=\frac{b-1}{2} \\ -1, & \text{otherwise}\end{cases}$$
   - 边界判定：$B(M) = 1$ 当 $|\nabla^2 p(M)| \leq \mu b^2$（$\mu=0.15$），否则 0
   - 最终输出：$M_{out} = f_{2\times}(M_2) \odot B_{2\times} + M_3 \odot (1 - B_{2\times})$

4. **边界学习损失 BLL**
   - 对 $M_3$ 的边界区域加权 BCE：
     $$\mathcal{L}_B = \frac{\sum_i R_{2\times}^i \cdot BCE(M_3^i, G_3^i)}{\sum_i R_{2\times}^i}$$
   - 其中 $R_{2\times}^i = f_{2\times}(B(M_2) \lor B(G_2))$ 为应关注的边界区域
   - 总 mask 损失：$\mathcal{L}_{mask} = \mathcal{L}_B + \lambda_1 \mathcal{L}_{BCE}(M_1, G_1) + \lambda_2 \mathcal{L}_{BCE}(M_2, G_2)$，$\lambda_1=0.25,\ \lambda_2=0.65$

## 实验与结果
- **数据集划分**：训练集 3937 张，验证集 691 张；评估指标采用标准 COCO mask AP（mAP、AP50、AP75、APS/APM/APL、APf/APh/APr）
- **基线对比（Mask R-CNN）**：
  - ResNet-50-FPN，1× schedule：WaterMask 23.3 vs Mask R-CNN 21.7（+1.6 mAP）
  - ResNet-101-FPN，1× schedule：WaterMask 25.6 vs Mask R-CNN 22.3（+3.3 mAP）
  - ResNet-50-FPN，3× schedule：WaterMask 26.4 vs Mask R-CNN 23.5（+2.9 mAP）
  - ResNet-101-FPN，3× schedule：WaterMask 27.2 vs Mask R-CNN 23.4（**+3.8 mAP**，最强结果）
- **Cascade 框架**：Cascade WaterMask R-CNN 达 27.1 mAP，较 Cascade Mask R-CNN 提升 1.6 mAP
- **SOTA 对比**：较 Mask2Former（25.7 mAP）提升 1.4 mAP；在 fish 类别（APf）上领先 5.9，在 human 类别（APh）上领先 3.9
- **Ablation**：DSGAT +1.4，MFRM +2.5，BMS +3.1，BLL +1.7（相对 baseline 22.5 mAP）

## 相关工作脉络
1. **Mask R-CNN**：本文基础框架，通过 DSGAT/MFRM/BMS 针对水下退化与聚集特性进行 head 级改造，而非简单堆叠参数。
2. **Cascade Mask R-CNN**：多阶段 bbox refinement 结构，本文证明 WaterMask 头可无缝接入，泛化性好。
3. **BMask R-CNN**：同样利用边界信息改进 mask 定位，但本文面向水下退化场景，DSGAT 通过图注意力显式恢复退化细节，边界损失设计也更具针对性。
4. **QueryInst / Mask2Former**：Transformer-based 实例分割方法，本文作为强基线对比；WaterMask 在复杂水下场景中于常见类别上显著超越。
5. **PointRend**：将分割视为渲染问题自适应选点，本文未采用此思路，而是通过多尺度特征融合与边界约束直接优化边界精度。

## 局限性与未来方向
1. **数据集规模有限**：4628 张图像对于深度学习任务偏小，未来需扩展更多样化、更复杂场景的数据。
2. **微小目标略有下降**：DSGAT 与 ResNet-101 组合在 APS 上略低于 baseline（-0.3），对极小实例的恢复仍有优化空间。
3. **计算资源依赖**：DSGAT 中 patch size 过小时显存超限（s=2 不可行），实际部署需权衡分辨率与显存。
4. **未来方向**：扩大数据集规模与难度、探索更高效的结构设计、向实时水下视觉应用延伸。

## 研究启发与可借鉴点
1. **图注意力用于退化细节恢复**：DSGAT 将特征图建模为图并利用远距节点聚合相似退化残差，思路可迁移至模糊图像、医学影像等细节易丢失的场景。
2. **多尺度特征迭代融合无需增参**：MFRM 通过通道减半 + 上采样实现特征 refinement，在几乎零额外参数下提升 mask 质量，值得在其他分割任务中借鉴。
3. **边界加权损失的设计**：BLL 通过动态边界区域掩码对 BCE 加权，迫使网络聚焦模糊边界分类，可推广至任何边界敏感任务（如遥感分割、显微图像）。
4. **跨框架泛化验证**：WaterMask 同时适配 Mask R-CNN 与 Cascade R-CNN，说明模块设计具有通用性，为后续工作提供可插拔 head 的设计范式。

## 关键术语表
**UIIS**：Underwater Image Instance Segmentation，首个通用水下图像实例分割数据集，含 4628 张图像、7 类别、像素级 COCO 格式标注。
**DSGAT**：Difference Similarity Graph Attention Module，通过图注意力机制在最高分辨率特征层聚合相似图像块的退化残差细节。
**MFRM**：Multi-level Feature Refinement Module，迭代融合 DSGAT 细粒度特征与 FPN 粗粒度特征，生成用于前景和边界预测的多尺度实例特征。
**BMS**：Boundary Mask Strategy，利用 Laplacian 算子提取低分辨率掩码边界，将高分辨率掩码的边界区域融合至最终输出。
**BLL**：Boundary Learning Loss，对边界像素赋予更高权重的加权 BCE 损失，使网络更关注模糊边界区域的分类。
**mAP**：mean Average Precision，实例分割常用综合评价指标，对多个 IoU 阈值和类别取平均。
**APs/APm/APl**：分别表示小/中/大尺度目标的平均精度，用于评估模型对不同尺寸实例的分割能力。

## 可复现要素
- **数据集**：UIIS 已公开（论文摘要给出代码地址，数据集随代码一同提供）
- **代码**：https://github.com/LiamLian0727/WaterMask
- **实现框架**：PyTorch + MMDetection；另提供 MindSpore Lite 版本
- **训练配置**：SGD optimizer，初始学习率 2.5e-3，每 GPU 2 张图，1×/3× schedule（3× 含多尺度训练）
- **关键超参**：DSGAT patch size 12×12（stride s=3），邻居数 k=11；BMS Laplacian 核大小训练 7×7/测试 9×9；μ=0.15，λ1=0.25，λ2=0.65
- **硬件**：NVIDIA A5000 GPU
