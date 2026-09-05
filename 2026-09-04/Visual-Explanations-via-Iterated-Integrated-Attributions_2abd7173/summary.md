---
title: "Visual-Explanations-via-Iterated-Integrated-Attributions"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Barkan_Visual_Explanations_via_Iterated_Integrated_Attributions_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-05 01:56:46"
field: "计算机视觉可解释性"
keywords: ["可解释AI", "视觉归因", "集成梯度", "Vision Transformer", "CNN解释"]
innovations: ["提出通用迭代积分归因(IIA)框架统一解释CNN和ViT", "引入梯度Rollout(GR)将梯度信息融入ViT注意力解释"]
benchmarks: ["ImageNet", "IN-Seg", "COCO", "VOC2012"]
---

# 论文速读：Visual-Explanations-via-Iterated-Integrated-Attributions

## 一句话总结
本文提出迭代积分归因（IIA）方法，通过在输入图像和网络内部表示（激活图/注意力矩阵）上进行迭代积分，为 CNN 和 ViT 模型生成精确、聚焦的解释图，在多项客观/主观评测中全面超越现有最先进方法。

## 研究问题与动机
1. **深度学习模型可解释性不足**：CNN/ViT 在图像分类、检测等任务取得 SOTA，但缺乏可解释性，难以说明预测推理过程。
2. **现有方法局限**：梯度-based 方法（如 Grad-CAM）仅利用最后一层激活；路径积分方法（如 IG）仅在输入空间做线性插值，忽略中间表示信息；ViT 解释方法（如 Rollout）简单平均注意力分数导致信号模糊。
3. **缺乏统一框架**：现有方法多针对特定架构设计，缺乏同时适用于 CNN 和 ViT 的通用解释框架。

## 核心贡献（创新点）
1. **提出通用迭代积分归因（IIA）框架**：通过在所有网络层的激活/注意力图上进行迭代积分，统一解释 CNN 和 ViT 模型，区别于 IG 仅对输入做单一路径积分。
2. **引入梯度 Rollout（Gradient Rollout, GR）**：作为 ViT 的积分函数，将注意力矩阵与其梯度进行 Hadamard 乘积，比原始 Rollout 更精准地捕捉 token 重要性。
3. **设计 IIa2 和 IIa3 变体**：IIA2 对输入和最后一层进行双积分；IIA3 进一步增加倒数第二层，形成三积分，充分利用多层特征聚合信息。
4. **全面实验验证**：在 ImageNet 及三个分割数据集上，覆盖 5 种主流架构（ResNet101、DenseNet201、ConvNeXt-Base、ViT-B、ViT-S），在所有 11 项解释指标和 4 项分割指标上均取得最优。

## 方法详解
**核心公式与思想**：

1. **通用设定**：给定输入 $\mathbf{x}$ 和 $L$ 层网络 $h^l$，目标生成 2D 解释图 $\mathbf{m} \in \mathbb{R}^{p_0 \times q_0}$ 量化每个输入像素对预测的归因。

2. **迭代积分定义**（公式 9）：
$$
\mathbf{m}_{\mathbf{b}}^l = \int_0^1 \int_0^1 \dots (\mathbf{u}^l - \mathbf{r}^l) \circ \int_0^1 \mathbf{q}^l d a_l \dots d a_0
$$
其中 $\mathbf{q}^l$ 是前 $l$ 层中间表示及其梯度的函数，指示向量 $\mathbf{b}$ 控制哪些层参与插值。

3. **CNN 实现**：
   - 对激活图 $\mathbf{u}^l = h^l(\mathbf{h}^{l-1})$ 进行插值：$\mathbf{v}^l = \mathbf{r}^l + (a_l)^{b_l}(\mathbf{u}^l - \mathbf{r}^l)$
   - 参考表示 $\mathbf{r}^l = \min(\mathbf{u}^l)$（逐通道最小值）
   - 积分函数设为 $\mathbf{q}^l = \mathbf{v}^l \circ \frac{\partial f(\mathbf{h}^L)}{\partial \mathbf{v}^l}$（激活 × 梯度，Hadamard 乘积）

4. **ViT 实现**：
   - 对注意力矩阵进行插值，参考为零张量（softmax 输出非负）
   - 引入 **Gradient Rollout (GR)**：将注意力矩阵替换为 注意力 × 梯度
   - 融合所有 heads 的 [CLS] token 信息生成 2D 解释图

5. **IIA2 vs IIA3**：
   - IIA2：$b_0=1, b_L=1$，对输入和最后一层插值
   - IIA3：$b_0=1, b_{L-1}=1, b_L=1$，额外加入倒数第二层，捕捉更丰富的语义特征

6. **计算复杂度**：理论为 $\mathcal{O}(n^\beta Q)$，但通过 GPU 批处理可使实际运行时间与 IG/GC 相当；当 $B \geq n^M$ 时，IIA 甚至可快于 IG。

## 实验与结果
**数据集**：ImageNet (50K), IN-Seg (4,276 images), COCO (5,000), VOC (1,449)

**评估模型**：ViT-Base, ViT-Small, ResNet101, DenseNet201, ConvNext-Base

**解释指标**：NEG, POS, INS, DEL, ADP, PIC, SIC, AIC（共 8 项，含更高/更低更好方向）

**分割指标**：PA, mAP, mIoU, mF1

**主要结果**：
- **IIA3 在全部 5 种模型架构、所有数据集、所有指标上均取得最优**（Table 1-4）
- CNN 解释测试（RN+IN）：IIA3 在 NEG 达到 56.89，POS 仅 15.78（越低越好），INS 达 48.53
- ViT 解释测试（ViT-B+IN）：IIA3 在 NEG 达到 58.31，POS 为 15.02
- 分割测试（CN+IN-Seg）：IIA3 的 mF1 达 42.32，超越第二名 LIFT（35.91）
- **IIA3 > IIA2 > IIA2(L-1)**：三积分优于双积分；当空间分辨率相同时（ViT），最后一层优于倒数第二层

**关键对比**：
- 相比 GC/GC++：IIA3 在 RN+IN 的 NEG 提升约 0.47（56.89 vs 56.42）
- 相比 T-Attr/GAE（ViT SOTA）：IIA3 在 ViT-B+IN 的 NEG 提升约 2.64（58.31 vs 55.67）

## 相关工作脉络
1. **Grad-CAM 系列**（GC/GC++/LayerCAM/LIFT）：基于梯度与激活图加权，但仅利用单一层信息；IIA 通过迭代积分融合多层信息。
2. **路径积分方法**（IG/BIG/GIG）：仅在输入空间线性插值，忽略中间激活；IIA 扩展到内部表示空间的迭代积分。
3. **ViT 注意力可视化**（Rollout/T-Attr/GAE）：Rollout 简单聚合注意力；T-Attr/GAE 使用 Deep Taylor Decomposition；IIA 引入 GR 结合梯度信息，且统一支持 CNN/ViT。
4. **扰动基方法**（Abrlation-CAM/Saliency）：依赖随机扰动或无梯度操作；IIA 显式利用梯度信息实现更精准的归因。

## 局限性与未来方向
1. **组合爆炸**：$2^L$ 种 $\mathbf{b}$ 组合，本文仅探索双/三积分；完整迭代积分的计算开销与最优层选择仍待研究。
2. **分辨率权衡**：消融显示 IIA2(L-1) 在 POS/DEL 优于 IIA2，因倒数第二层分辨率更高；如何在高分辨率与强语义聚合间平衡需进一步探索。
3. **LIFT 平图现象**：定性实验中观察到 LIFT 生成"平"的解释图（缺乏区分度），值得深入分析其失效机制。
4. **跨领域泛化**：当前实验集中于图像分类与分割，对检测、生成等任务的解释能力未验证。

## 研究启发与可借鉴点
1. **迭代积分思想可扩展**：将 IG 的单一路径积分推广为多层表示的迭代积分，思路简洁且效果显著，可迁移至 NLP/多模态模型的归因研究。
2. **GR（梯度 Rollout）设计**：将梯度信息与注意力矩阵结合的方式对 ViT 解释具有通用价值，可与其他 ViT 分析方法（如 LRP）结合。
3. **实验设计**：同时评估解释质量（POS/NEG/INS/DEL 等）和分割性能（mIoU/mF1），且提供 Target/Predicted 两类设置，评测体系全面，值得借鉴。
4. **可复现性**：代码已开源（https://github.com/iia_iccv23/iia），超参配置遵循各基线作者推荐设置，便于后续研究复现与对比。

## 关键术语表
**Iterated Integrated Attributions (IIA)**：通过在输入图像和网络内部表示（激活图/注意力矩阵）上进行迭代积分来生成解释图的方法。
**Gradient Rollout (GR)**：IIA 在 ViT 中使用的积分函数，将注意力矩阵与其梯度的 Hadamard 乘积替代原始 Rollout 中的注意力矩阵。
**Integrated Gradients (IG)**：沿输入与参考图像的线性路径积分梯度，是 IIA 的特例（仅对输入做单次积分）。
**Attention Rollout (AR)**：逐层累积 Transformer 注意力分数的方法，IIA 的 GR 是其梯度增强版本。
**POS/NEG**：正向/负向扰动测试的 AUC，衡量高/低归因区域对预测的影响程度。
**INS/DEL**：插入/删除测试的 AUC，衡量逐步添加/移除高归因像素对预测分数的影响。
**IIA2 / IIA3**：分别对 2 层（输入+末层）和 3 层（输入+倒数第二层+末层）进行迭代积分的 IIA 变体。
**$\mathbf{b}$ 指示向量**：控制哪些网络层参与插值积分的 0/1 向量，决定 IIA 的具体形态。

## 可复现要素
- **数据集**：ImageNet ILSVRC 2012（公开）、IN-Seg（公开）、COCO 2017（公开）、VOC 2012（公开）
- **代码**：开源，https://github.com/iia_iccv23/iia
- **权重**：预训练模型链接在 GitHub 仓库提供
- **关键超参**：插值步数 $n=10$；IIA2 使用 $b_0=1, b_L=1$；IIA3 使用 $b_0=1, b_{L-1}=1, b_L=1$
- **基线配置**：遵循各基线方法作者推荐的超参设置
