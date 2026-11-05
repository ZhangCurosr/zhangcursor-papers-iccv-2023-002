---
title: "Partition-and-Debias-Agnostic-Biases-Mitigation-via-A-Mixtur"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Li_Partition-And-Debias_Agnostic_Biases_Mitigation_via_a_Mixture_of_Biases-Specific_Experts_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 00:53:34"
field: "计算机视觉-公平性与去偏"
keywords: ["bias mitigation", "agnostic biases", "mixture of experts", "debiased classification", "counterfactual learning", "fairness"]
innovations: ["提出 agnostic biases 场景并设计 Partition-and-Debias 框架，通过跨深度多专家隐式处理多种未知偏差", "引入反事实对比训练与难度加权损失，实现目标-偏差特征的隐式解耦", "设计多样性正则化与自适应门控融合机制，提升多专家协同去偏效果"]
benchmarks: ["Biased MNIST", "BAR", "Modified IMDB", "MIMIC-CXR+NIH", "CelebA worst-group"]
---

# 论文速读：Partition-and-Debias-Agnostic-Biases-Mitigation-via-A-Mixtur

## 一句话总结
本文提出了一种针对"无偏置认知（agnostic biases）"场景的图像分类去偏方法 PnD，通过在网络不同深度部署多个偏差特定专家（biases-specific experts）并配合门控模块，隐式地将多种未知偏差空间划分到不同子空间中联合去偏，解决了真实场景中同时存在多种未知偏差类型且数量未知的难题。

## 研究问题与动机
- **核心问题**：真实图像中往往同时存在多种偏差属性（如 CelebA 中年幼样本常同时关联 female、attractive、lipstick 三个偏差），且偏差类型与数量均未知，传统单偏差假设无法覆盖。
- **现有方法不足**：多数去偏方法隐含假设图像只含一种已知/未知偏差，或需要预定义偏差类型；即使使用多偏差数据集（如 OccamNet），仍受限于对偏差类型的强假设。
- **特征分布观察**：不同偏差类型的特征会聚类于网络不同深度（纹理特征浅层更丰富，位置/尺度特征深层更突出），因此需要在多个深度分别处理。
- **评估缺口**：在 Biased MNIST（7 种偏差）、MIMIC-CXR+NIH（数据源偏差）等复杂场景下，现有 SOTA 方法的 worst-group 准确率大幅下降，而 PnD 显著提升。

## 核心贡献（创新点）
1. **提出 agnostic biases 新场景**：首次系统定义并研究偏差类型与数量均未知的现实去偏设定，填补了现有工作仅考虑单偏差/已知偏差类型的空白。
2. **Partition-and-Debias 框架**：通过在每个 ResNet 残差块后插入偏差特定专家 + 门控模块，实现跨深度的偏差空间划分与联合去偏，无需预知偏差数量。
3. **两阶段去偏训练机制**：初始训练阶段利用 GCE 损失强化偏差检测、加权 CE 损失聚焦难样本；反事实训练阶段通过对比学习显式解耦目标特征与偏差特征。
4. **多样性正则化与自适应门控**：引入 KL 散度损失促进各专家关注不同深度/偏差维度；门控模块按专家置信度加权融合，避免浅层专家噪声干扰最终预测。

## 方法详解
- **双编码器结构**：去偏编码器 $\mathcal{D} = \{D^{(i)}\}_{i=1}^M$ 与偏差编码器 $\mathcal{B} = \{B^{(i)}\}_{i=1}^M$ 并行从输入图像中提取目标特征 $\mathbf{z}_d^{(i)}$ 与偏差特征 $\mathbf{z}_b^{(i)}$，$M=4$（ResNet-18 的四个 residual block）。
- **偏差特定专家 $E^{(i)}$**：每个专家包含去偏分类器 $C_d^{(i)}$ 与偏差分类器 $C_b^{(i)}$，输入为拼接特征 $\mathbf{z}^{(i)} = [\mathbf{z}_d^{(i)}; \mathbf{z}_b^{(i)}]$。
- **初始训练损失**：
  - 偏差检测损失 $\mathcal{L}_{bias} = \sum_{i=1}^M \text{GCE}(\hat{\mathbf{y}}_b^{(i)}, \mathbf{y})$，利用 GCE 强化易学偏差特征；
  - 去偏分类损失 $\mathcal{L}_{debias} = \sum_{i=1}^M w^{(i)} \times \text{CE}(\hat{\mathbf{y}}_d^{(i)}, \mathbf{y})$，其中权重 $w^{(i)} = \frac{\text{CE}(\hat{\mathbf{y}}_b^{(i)}, \mathbf{y})}{\text{CE}(\hat{\mathbf{y}}_d^{(i)}, \mathbf{y}) + \text{CE}(\hat{\mathbf{y}}_b^{(i)}, \mathbf{y})}$ 使模型关注难样本（去偏易、偏差检测难的样本）；
  - 多样性损失 $\mathcal{L}_{div} = \sum_{i=2}^M \exp(-\text{KL}(\hat{\mathbf{y}}_b^{(i)}, \hat{\mathbf{y}}_b^{(i-1)}))$ 鼓励不同专家关注不同偏差维度。
- **反事实训练**：在 mini-batch 中随机配对其他样本的偏差特征与目标特征，构造正样本（相同目标、不同偏差）与负样本（不同目标、相同偏差），通过对比损失 $\mathcal{L}_{con}$ 使模型对目标特征保持不变、对偏差特征敏感。
- **门控融合**：最终预测 $\hat{\mathbf{y}}_d = \sum_{i=1}^M p^{(i)} \times \hat{\mathbf{y}}_d^{(i)}$，其中 $p^{(i)}$ 为门控模块输出的 softmax 权重，总损失 $\mathcal{L} = \mathcal{L}_{cls} + \mathcal{L}_{gate} + \mathcal{L}_{div} (+ \beta \mathcal{L}_{con})$。

## 实验与结果
- **数据集**：Biased MNIST（7 种偏差，bias ratio 0.75/0.95）、BAR（1 种偏差，纯偏置训练集）、Modified IMDB（2 种偏差：gender、glasses）、MIMIC-CXR+NIH（1 种数据源偏差）、CelebA（real-world，4 个偏差属性 worst-group 评估）。
- **基线**：ResNet-18、LfF、DFA、OccamNet、DebiAN、UBNet。
- **主要结果**（bias ratio=0.95）：
  - Biased MNIST（7 偏差）：**PnD 70.43%** vs OccamNet 66.85%（+3.58pp），全面领先；
  - Modified IMDB（2 偏差）：PnD **74.34%**（all groups），优于 ResNet-18 74.31%；
  - MIMIC-CXR+NIH：PnD **60.73%** vs DebiAN 60.00%（+0.73pp）；
  - BAR（单偏差，纯偏置训练）：PnD 69.83% 略低于 DebiAN 69.88%，因需要无偏数据；
  - CelebA worst-group 平均：PnD **60.00%** vs PAD 60.00%（并列第一），all groups 83.40%。
- **关键结论**：PnD 在多偏差场景下优势显著；随偏差数量从 1 增至 7 保持性能稳定；门控权重显示第 3/4 专家贡献最大，印证了"深层偏差更易影响最终预测"的假设。

## 相关工作脉络
- **已知偏差方法（LFf、DFA、OccamNet）**：依赖偏差可编辑或目标特征更简单的强假设，仅处理单一偏差类型；本文放松假设，无需预定义偏差。
- **未知偏差方法（UBNet、DebiAN）**：UBNet 从浅层提取"无偏"特征但忽略多偏差；DebiAN 仅能消除一个主导二元偏差；本文同时处理多类未知偏差。
- **MoE 在 CV/NLP 中的应用**：传统 MoE 用于域划分或加速推理；本文首次将 MoE 适配至去偏任务，并在网络不同深度部署专家。
- **表征解耦与去偏（End、BiasSwap）**：依赖显式属性标签或数据增强；本文通过对比学习与多样性正则化隐式解耦。
- **定位差异**：本文是首个面向"类型+数量均未知"的 agnostic biases 场景的无监督去偏框架，不依赖任何偏差标注或可编辑假设。

## 局限性与未来方向
- **参数量增加**：多专家结构引入了额外参数，未来可通过专家稀疏化（如 Switch Transformers）缓解。
- **依赖无偏训练数据**：在纯偏置数据集（如 BAR）上性能略低于 DebiAN，作者承认缺少归纳偏置时完全去除多未知偏差困难。
- **反事实样本构造的随机性**：当前随机配对策略可能引入噪声，可探索更合理的正负样本选取机制。
- **扩展到视频/多模态**：当前仅在图像分类任务验证，可探索时序场景下的动态偏差演化。

## 研究启发与可借鉴点
1. **跨深度专家并行去偏策略**：将 MoE 思想嵌入网络中间层而非仅末端，可作为多任务/多域去偏的通用设计范式。
2. **难度加权 CE 损失**：利用偏差检测 loss 与去偏 loss 的比值动态调整样本权重，为"难样本聚焦"提供了简洁有效的实现。
3. **反事实对比学习**：通过配对跨样本的特征构造正负对，无需额外标注即可强制目标-偏差解耦，可迁移至其他去偏或公平性任务。
4. **多样性正则化（KL 散度）**：以梯度方式鼓励专家间输出分布差异，适用于任何需要专家异构性的混合模型。
5. **worst-group 评估设计**：在 CelebA 上使用四类偏差属性的 worst-group 平均指标，为公平性 benchmark 提供了可复用的评估模板。

## 关键术语表
- **Agnostic Biases**：指偏差类型与数量均未知的现实场景，区别于已知单偏差或仅类型未知的设定。
- **Partition-and-Debias (PnD)**：将网络按深度划分多个子空间，各空间内独立处理对应偏差的框架。
- **Biases-Specific Expert**：嵌入于网络某深度的专家模块，包含去偏分类器与偏差分类器，专注处理该深度聚集的偏差特征。
- **GCE Loss (Generalized Cross Entropy)**：对难样本更敏感的交叉熵变体，用于增强偏差检测器的学习效果。
- **Counterfactual Training**：通过跨样本特征配对构造反事实样本，利用对比损失显式解耦目标与偏差特征的训练阶段。
- **Adaptive Gating**：基于各专家去偏输出的 softmax 加权融合机制，使高置信度专家主导最终预测。
- **Worst-Group Accuracy**：按偏差属性划分的分组中准确率最低的组，用于评估去偏后的公平性。

## 可复现要素
- **数据集**：Biased MNIST、BAR 公开；Modified IMDB、MIMIC-CXR+NIH 为论文构建（代码中含合成脚本）；CelebA 公开。
- **代码/权重**：代码已开源 https://github.com/Jiaxuan-Li/PnD（PyTorch）；论文未提供预训练权重。
- **关键超参**：bias ratio=0.95（默认）、ResNet-18 backbone、图像resize至 160×160（BAR 为 224×224 随机裁剪+水平翻转）、3 次运行取均值；$\alpha$、$\beta$ 为损失权重，论文未明确给出数值（需在源码中确认）。
