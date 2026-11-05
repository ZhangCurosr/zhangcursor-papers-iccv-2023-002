---
title: "The-effectiveness-of-MAE-pre-pretraining-for-billion-scale-p"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Singh_The_Effectiveness_of_MAE_Pre-Pretraining_for_Billion-Scale_Pretraining_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 08:25:00"
field: "大规模视觉预训练"
keywords: ["MAE", "pre-pretraining", "weakly supervised pretraining", "vision transformer", "self-supervised learning", "billion-scale pretraining"]
innovations: ["提出MAE预预训练框架，在WSP前增加自监督初始化阶段，无需额外数据或超参调优", "首次验证MAE随训练数据集规模扩展，补全了MAE的缩放行为理解", "揭示十亿级预训练中模型初始化的关键作用，2B模型超越6.5B基线"]
benchmarks: ["ImageNet-1k", "iNaturalist-18", "COCO", "LVIS", "Kinetics-400", "Food-101", "ObjectNet", "ImageNet-ReaL"]
---

# 论文速读：The-effectiveness-of-MAE-pre-pretraining-for-billion-scale-p

## 一句话总结
本文提出在大规模弱监督预训练（WSP）之前增加一个自监督MAE预预训练阶段（MAE→WSP），以随机初始化模型为起点，先利用MAE在无标签数据上预预训练，再用十亿级弱标签数据完成标准预训练，显著提升了视觉模型的收敛速度与下游迁移性能，在多个任务上取得新的最优结果。

## 研究问题与动机
1. 视觉识别的主流"预训练-微调"范式中，模型通常从随机权重开始，直接用大规模弱监督数据（如Instagram的十亿级图片）训练，初始化的质量未被系统研究。
2. 自监督学习（如MAE）与弱监督预训练（WSP）长期以来被独立使用，缺乏两者的有效结合策略。
3. MAE虽已被证明能随模型规模扩展，但其是否也能随**训练数据集规模**扩展尚未被验证，限制了其作为通用初始化策略的应用。
4. 在十亿参数级模型与十亿级图像规模下，模型初始化对最终性能的影响程度仍不明确。

## 核心贡献（创新点）
1. **提出MAE→WSP预预训练框架**：先在无标签数据上用MAE自监督预预训练，再用弱监督数据进行标准预训练，无需额外数据或超参调优，即可系统性地提升下游性能与收敛速度。
2. **发现MAE可扩展至十亿级数据集**：首次验证MAE不仅随模型规模扩展（He et al. 已证明），也能随训练数据集规模扩展，补全了对MAE缩放行为的理解。
3. **揭示模型初始化在大尺度预训练中的关键作用**：即使在使用十亿级标签的Web规模预训练中，预训练初始化仍能带来显著性能提升，挑战了"数据量足以覆盖初始化差异"的隐含假设。
4. **在10个视觉任务上验证跨尺度有效性**：覆盖百万到十亿参数模型、百万到十亿图片数据集，在图像分类、视频识别、目标检测、低样本/零样本分类等任务上一致提升。
5. **取得多项新SOTA**：iNaturalist-18（91.3%）、1-shot ImageNet-1k（62.1%）、Food-101零样本（96.2%）等任务上刷新记录，且2B模型超越6.5B纯WSP模型。

## 方法详解
- **架构**：统一使用Vision Transformer（ViT），涵盖ViT-B（86M）、ViT-L（307M）、ViT-H（632M）、ViT-2B（1.9B）和ViT-6.5B，分辨率224×224（部分评估使用518×518）。
- **预预训练阶段（MAE）**：采用MAE框架，随机遮蔽75%图像patch，通过较小的decoder重建遮蔽区域的像素值，最小化像素重建误差。模型仅处理25%未遮蔽patch，保证计算效率。在IG-3B数据上训练1个epoch，复用IN1k预训练的MAE超参，无需额外调优。
- **弱监督预训练阶段（WSP）**：从IG-3B的Instagram图像及其关联的hashtag文本中提取伪标签（映射到WordNet synset），使用多标签分类交叉熵损失进行预训练。复用SWAG [70] 的训练超参，同样训练1个epoch。
- **整体流程（MAE→WSP）**：先用MAE在IG-3B图像上无标签预预训练1 epoch，然后用同一批数据（带hashtag标签）进行WSP预训练1 epoch。整个流程不引入额外数据，也不需调整超参。
- **零样本评估**：采用LiT方法，冻结MAE→WSP预训练的图像编码器，训练文本编码器（基于XLM-R Large）以CLIP loss对齐图像-文本嵌入，进行零样本推理。

## 实验与结果
- **数据集**：预训练使用Instagram-3B（IG-3B，28K类别，30亿唯一图片，重采样至50亿）。评估涵盖10个任务，包括ImageNet-1k、iNaturalist-18、ImageNetv2、ImageNet-ReaL、ObjectNet、Food-101、COCO、LVIS、Kinetics-400、Something-Something-v2。
- **图像分类**：ViT-H（MAE→WSP）在IN1k上达89.3%，ViT-2B在iNat18上达91.3%（较MAE提升+4.5%），在IN-ReaL和ObjectNet上均取得最高报告性能。
- **视频分类**：ViT-L在K400和SSv2上均取得最高报告性能，尽管预训练仅使用图像数据。
- **目标检测（LVIS）**：ViT-H达50.8 AP^box，ViT-2B达51.8 AP^box，显著优于同规模SWAG（47.1）。
- **目标检测（COCO）**：ViT-H达57.7 AP^box，ViT-2B达58.0 AP^box，优于SWAG的55.7。
- **低样本分类**：ViT-H在IN1k 1-shot达62.1%（新SOTA），10-shot达83.7%；ViT-2B在iNat18 10-shot达80.3%。
- **零样本分类**：ViT-2B在Food-101上达96.2%（新SOTA），在IN1k上达81.9%。
- **收敛效率**：预预训练仅需0.1 epoch即能带来改善，1 epoch后增益饱和；相同FLOPs预算下，MAE→WSP比WSP最多快2×到达相同性能。
- **规模化行为**：2B参数的MAE→WSP模型超越6.5B参数的纯WSP模型；检测任务上，仅靠WSP增大模型规模不提升性能，但加入MAE预预训练后检测性能随模型增大持续提升。

## 相关工作脉络
1. **SWAG [70]**：纯弱监督预训练基线，使用相同IG-3B数据和ViT-H架构，MAE→WSP在其基础上增加预预训练阶段，检测AP^box提升3.7（LVIS）和2.0（COCO）。
2. **MAE [33]**：纯自监督预训练方法，擅长检测和高FLOPs任务，但在零样本和低样本分类上落后WSP超10个百分点；MAE→WSP结合了双方的优势。
3. **DINOv2 [56]**：自监督学习冻结表征的代表性工作，MAE→WSP的线性探针性能（ViT-H: 87.0，ViT-2B: 88.1）超越DINOv2。
4. **Scale-ViT [84]**：使用JFT-3B预训练，在IN1k上领先MAE→WSP约1%，但MAE→WSP在鲁棒性（IN-ReaL、ObjectNet）和零样本Food-101上更优，凸显预训练数据选择的重要性。
5. **Florence [82] / CoCa [81]**：使用FLD-900M或JFT-3B+ALIGN等大语种-图像对预训练，在零样本IN1k上领先，但MAE→WSP以纯图像预训练在Food-101上取得96.2%的最优结果。
6. **中间微调（Intermediate Finetuning）[5, 50]**：在标准预训练后增加对齐下游任务的微调阶段；本文的预预训练处于预训练**之前**，定位完全不同。

## 局限性与未来方向
1. **额外计算开销**：MAE预预训练阶段增加1个epoch的训练成本，虽然比WSP更高效（MAE仅处理25% patch），但对算力极度敏感的场景仍需权衡。
2. **零样本性能受限于预训练数据**：在零样本IN1k上落后于使用ALIGN/JFT预训练的方法，说明预训练数据的语义丰富度仍是关键瓶颈。
3. **检测性能提升幅度有限**：虽然MAE→WSP显著改善了WSP的检测缩放行为，但与使用Objects365/FLOD-9M等专用检测数据的方法相比仍有差距。
4. **预训练数据规模与质量的影响需进一步解耦**：MAE→WSP与Scale-ViT的性能差异可能源于数据（IG-3B vs JFT-3B）而非方法本身，需要更公平的数据对齐实验。
5. **对不同架构的泛化性未充分验证**：本研究聚焦ViT架构，CNN或其他架构下的效果尚待探索。

## 研究启发与可借鉴点
1. **模型初始化在大尺度预训练中的价值被低估**：即使在使用数十亿标签的数据上，一个简单有效的初始化阶段仍能带来系统性提升，后续研究可在初始化策略上继续挖掘。
2. **MAE的1 epoch预预训练性价比极高**：仅需极少的额外计算即可显著改善下游性能，且增益在0.1 epoch时即出现、1 epoch饱和，适合快速迭代。
3. **小规模预预训练数据同样有效**：在IN21k（14M图片）上进行MAE预预训练同样能提升WSP性能，意味着可复用现有公开MAE权重作为初始化，降低数据获取门槛。
4. **低样本/零样本场景下的组合策略**：MAE与WSP在不同协议下各有优劣，MAE→WSP的自然融合为低样本学习提供了更鲁棒的表征，值得在少样本医学图像等场景验证。
5. **预训练数据选择对零样本能力的影响**：食物类零样本任务的优异表现提示，特定领域的预训练数据质量可能比模型规模更关键，启发数据优先的研究思路。

## 关键术语表
**Pre-pretraining（预预训练）**：在标准预训练之前增加的初始训练阶段，用于改善模型初始化质量，而非对齐下游任务。
**MAE（Masked Autoencoder）**：自监督预训练方法，随机遮蔽75%图像patch并通过decoder重建像素值，以学习视觉表征。
**WSP（Weakly Supervised Pretraining）**：利用互联网图像的弱标签（如hashtag）进行多标签分类预训练，介于监督与自监督之间。
**LiT（Locked Image Text tuning）**：冻结图像编码器、训练文本编码器以CLIP loss对齐图文嵌入的零样本迁移方法。
**Linear Probing（线性探针）**：冻结预训练模型权重，仅训练线性分类头评估表征质量的评估协议。
**ViT-Det**：基于Vision Transformer的检测器架构，将ViT作为 backbone 接入Cascade Mask R-CNN框架。
**Scaling Law（缩放定律）**：模型性能随模型规模或数据规模单调提升的规律，本文验证了MAE同时满足两种缩放。
**Adapter / VPT**：保持主干网络冻结、仅训练少量新增参数的参数高效迁移学习技术。

## 可复现要素
- **预训练数据集**：Instagram-3B（IG-3B），由SWAG pipeline生成，28K类别，30亿唯一图片；论文未公开该数据集，但其处理流程与SWAG一致。
- **评估数据集**：ImageNet-1k、iNaturalist-18、COCO、LVIS、Kinetics-400、Something-Something-v2、Food-101等均为公开数据集。
- **代码/权重**：论文未明确声明开源代码与模型权重。
- **关键超参**：MAE遮蔽比例75%，预预训练1 epoch；WSP使用交叉熵多标签损失，训练1 epoch；分辨率224×224（评估时使用518×518）；Patch大小ViT-B/L为16，ViT-H/2B/6.5B为14。
- **检测框架**：Cascade Mask R-CNN + ViTDet架构，超参来自[47]并做适配。
- **零样本评估**：LiT方法，文本编码器基于XLM-R Large初始化，训练1 epoch。
