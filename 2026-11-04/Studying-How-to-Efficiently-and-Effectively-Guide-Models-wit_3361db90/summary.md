---
title: "Studying-How-to-Efficiently-and-Effectively-Guide-Models-wit"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Rao_Studying_How_to_Efficiently_and_Effectively_Guide_Models_with_Explanations_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 00:56:46"
---

# 论文速读：Studying How to Efficiently and Effectively Guide Models with Explanations

## 一句话总结
本文在大规模真实多标签数据集上系统评估了模型引导（Model Guidance）的各项设计选择，提出将 EPG 分数转化为可微的 Energy loss，并证明仅使用廉价粗粒度的边界框标注（甚至仅 1% 样本）即可高效提升模型对目标特征的聚焦能力与分布外泛化性。

## 研究问题与动机
- DNN 常依赖训练数据中的虚假相关性（如背景、共现物体）而非类别相关特征，导致在分布偏移或复杂场景下泛化能力下降。
- 现有模型引导方法多在简单或合成数据集上验证，缺乏统一、公平的对比基准，难以提炼普适的有效设计原则。
- 传统引导严重依赖昂贵的语义分割掩码标注，实际落地与大规模训练的成本极高。
- 多标签场景下，同时优化多个类别的归因会导致显著的计算开销，亟需高效的训练策略。

## 核心贡献（创新点）
- **系统性大规模评测框架**：首次在 PASCAL VOC 2007 与 MS COCO 2014 上统一评估 attribution 方法、模型架构、引导深度与损失函数的组合效应，填补领域内缺乏公平对比的空白。
- **提出 Energy loss**：将原本仅用于评估的 EPG 分数改造为完全可微的定位损失，避免 $L_1$/PPCE 强制 attribution 均匀分布的弊端，使模型更聚焦物体特异性特征。
- **低成本高效引导策略**：证明仅使用边界框（而非分割掩码）及极少量标注（1% 训练集）即可实现显著的定位性能提升，大幅降低人工标注成本。
- **分布偏移鲁棒性验证**：在 Waterbirds-100 数据集上展示模型引导能有效抑制背景虚假相关性，显著提升最坏分组准确率（Worst-group accuracy）。

## 方法详解
- **联合优化目标**：$\mathcal{L} = \mathcal{L}_{\mathrm{class}} + \lambda_{\mathrm{loc}} \mathcal{L}_{\mathrm{loc}}$，其中 $\mathcal{L}_{\mathrm{class}}$ 为二元交叉熵，$\mathcal{L}_{\mathrm{loc}}$ 为定位损失，$\lambda_{\mathrm{loc}}$ 控制分类与定位目标的权衡。
- **归因方法评估**：对比 IxG、IntGrad（配合 -DNN 加速单次反向）、GradCAM 及 B-cos（模型固有可解释性），并在输入层、中间层及最终卷积层进行正则化。
- **定位损失函数设计**：
  - $L_1$ loss：最小化 GT 掩码与归一化正归因的 $L_1$ 距离，倾向于均匀填充掩码区域。
  - PPCE loss：掩码内逐像素 BCE，仅激励掩码内高响应，不对掩码外施加约束。
  - RRR* loss：基于任意归因方法的平方正则化，仅惩罚掩码外的归因值。
  - **Energy loss**：$\mathcal{L}_{\mathrm{loc}, k} = -\mathrm{EPG}_k = -\frac{\sum_{h,w} M_{k,hw} A_{k,hw}^{+}}{\sum_{h,w} A_{k,hw}^{+}}$，最大化标注区域内的归因能量占比，不强制均匀性。
- **多标签效率策略**：每批次随机采样一个 GT 类别进行定位优化，避免多次反向传播带来的巨大开销。
- **模型选型机制**：采用 F1、EPG、IoU 三指标的 Pareto 前沿选取最佳 checkpoint，而非依赖单一指标，更全面反映准确率与可解释性的权衡。

## 实验与结果
- **数据集与基线**：PASCAL VOC 2007、MS COCO 2014（多标签分类）、Waterbirds-100（分布偏移）；模型涵盖 Vanilla ResNet-50、-DNN ResNet-50、B-cos ResNet-50/ViT-S/DenseNet-121。
- **核心数值发现**：
  - Energy loss 在 EPG 指标上全面 Pareto 占优（R1）；$L_1$ loss 在 IoU 指标上最优（R2）。
  - B-cos 在输入层引导时表现最佳（EPG 达 71.7 @ 79.4% F1），但所有模型在最终层引导均稳健有效（R4、R5）。
  - 引导可提升分类精度（R6）：在 Waterbirds-100 上，Energy loss 使最坏分组准确率从 43.4% 提升至 56.1%，整体准确率提升至 71.2%。
  - 仅用 1% 标注图像（VOC 中 25 张）即可提升 EPG 达 23.0 p.p.，F1 仅下降 0.3 p.p.（R7）。
  - Energy loss 对边界框膨胀误差极度鲁棒；$L_1$ 在 50% 膨胀时性能显著退化（R8）。
  - 在深层（Mid2）引导可获得与输入层相近的 EPG 增益，训练提速约 1.7x（R9）。

## 相关工作脉络
- **Ross et al. (RRR, IJCAI 2017)**：开创“right for the right reasons”引导范式，但仅在简单数据集上使用 GradCAM+$L_2$ 正则；本文将其扩展至多损失、多架构与大规模真实数据对比。
- **Gao et al. (GRADIA/RES)**：引入 $L_1$ 与 PPCE 等定位损失；本文系统对比并指出 $L_1$ 强制均匀性的缺陷，提出 Energy loss 作为更优替代。
- **Chefer et al. (Transformer Interpretability, CVPR 2021 / NeurIPS 2022)**：聚焦 ViT 注意力优化；本文同时涵盖 CNN 与 Vision Transformer，并重点考察标注成本与训练效率。
- **Bohle et al. (B-cos Networks, CVPR 2022)**：提出模型固有可解释网络；本文验证其在引导任务中的优越性，并提供可微实现以支持端到端训练。
- **Petryk et al. (Language-guided, CVPR 2022)**：使用语言规约抑制虚假相关性；本文证明廉价边界框引导可达到类似甚至更优的纠偏效果，且无需自然语言标注。

## 局限性与未来方向
- 多标签场景采用单类别随机采样策略，可能忽略类别间特征交互与竞争关系，限制复杂场景下的引导效果。
- 边界框仍包含部分背景区域，细粒度定位上限受限于标注精度，无法完全替代像素级分割掩码。
- 实验主要集中在标准视觉分类基准，尚未充分验证于 VQA、实例分割、医疗影像等下游或高维视觉任务。
- 未来方向可包括：自适应多类别联合引导、结合弱监督/无监督定位预训练、向多模态（视觉-语言）任务迁移，以及探索更高效的跨层归因正则化机制。

## 研究启发与可借鉴点
- **评估指标转损失函数**：将不可微的评估
