---
title: "Unified-Visual-Relationship-Detection-with-Vision-and-Langua"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Zhao_Unified_Visual_Relationship_Detection_with_Vision_and_Language_Models_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 23:36:46"
field: "视觉关系检测"
keywords: ["visual relationship detection", "vision-language model", "unified label space", "human-object interaction", "scene graph generation", "open-vocabulary detection"]
innovations: ["利用 VLM 图文对齐嵌入统一多数据集异构标签空间", "自底向上级联训练范式 + 预测 box index 的关系解码设计", "首次证明统一 VRD 模型可扩展至大模型且性能媲美特定模型"]
benchmarks: ["HICO-DET", "V-COCO", "Visual Genome"]
---

# 论文速读：Unified Visual Relationship Detection with Vision and Language Models

## 一句话总结
论文提出了 UniVRD，一种基于视觉-语言模型（VLMs）的自底向上统一视觉关系检测方法，首次利用预训练 VLM 的图文对齐嵌入空间统一多个异构 VRD 数据集的标签空间，在 HICO-DET 上达到 38.07 mAP（SOTA），并证明统一模型可扩展至大型模型时性能持续提升。

## 研究问题与动机
1. **现有 VRD 模型局限于单一数据集**：当前方法大多仅使用单个数据集训练，受限于图像域和文本词表范围，泛化性和可扩展性较差。
2. **多数据集标签空间异构**：不同 VRD 数据集（HICO-DET、V-COCO、Visual Genome）的物体类别和谓词标签存在同义（如 'read' vs 'look at'）、从属（'person' vs 'woman/man'）和重叠（'wine glass' vs 'glass'）关系，手动合并极其困难。
3. **二阶语义加剧对齐难度**：视觉关系是对象间二阶语义，同一对象或谓词在不同语境下可能呈现不同形式（时态、单复数），使得简单的标签映射难以捕捉语义等价性。
4. **VLMs 提供语义统一新路径**：CLIP、LiT 等 VLM 预训练后的图文嵌入高度对齐，语义相近的关系自然靠近，为异构标签空间统一提供了无需人工规则的可行方案。

## 核心贡献（创新点）
1. **首个利用 VLM 统一多数据集 VRD 的自底向上框架**：与先前工作（使用静态 word embedding 固定小集合）的本质区别在于利用 VLM 的预训练图文嵌入空间进行语义级统一，无需手动合并标签。
2. **级联（Cascade）训练范式**：不同于直接端到端训练，先在带边界框标注的数据上训练对象检测器，再训练关系解码器，且根据数据量选择是否冻结/微调检测器；这与传统单阶段方法形成本质区别。
3. **自然语言替代类别整数定义统一标签空间**：将类别名和关系三元组转换为文本 prompt（如 "a person riding a horse"）经文本编码器得到 text query；与 DETR 类方法使用固定类别整数 ID 的本质区别在于支持开放词汇和语义聚合。
4. **预测边界框索引而非坐标以提升效率**：关系解码器通过余弦相似度匹配 instance embeddings 获得 subject/object 的 box index，避免冗余预测；与 single-stage 方法直接预测框坐标的本质区别在于减少重复推理。
5. **证明统一模型可扩展性**：首次在 VRD 任务上证明统一模型随模型规模增大持续改进，且性能可与数据集特定模型持平；与以往多数据集统一工作（如 MSeg、Zhao et al.）出现的性能下降现象形成鲜明对比。

## 方法详解
**整体架构**：由对象检测器 + 关系解码器两阶段级联组成，均采用 set prediction 范式（Hungarian 匹配）。

1. **对象检测器（Encoder-only ViT）**：
   - 使用预训练 VLM（CLIP/LiT）的图像编码器，移除 pooling 和最终 projection 层，每个 token 输出经线性层得到 instance embedding $z_i$。
   - Box 坐标通过 FFN $f_{\text{box}}(z)$ 预测；分类通过 instance embedding 经线性层 $f_{\text{cls}}$ 投影后与 text query 做 cosine similarity，输出 logits 过 focal sigmoid CE loss（处理非 disjoint 标签）。
   - 不添加 background 类别，避免对未完整标注的正样本施加惩罚。

2. **关系解码器（Transformer Decoder）**：
   - 输入为预学习的 relation queries（数量 M=100）和对象检测器输出的 instance embeddings $\mathcal{Z}$。
   - 采用类似 Perceiver Resampler 的设计，将 latent query 的 KV 与 $\mathcal{Z}$ 的 KV 拼接。
   - 关系嵌入 $r_j$ 经两个 FFN 分别投影为 subject 嵌入和 object 嵌入，通过余弦相似度从 $\mathcal{Z}$ 中检索最佳匹配的 box index $s_j, o_j$（公式 1），从而获得 subject box $b_{s_j}$ 和 object box $b_{o_j}$。
   - 关系分类使用关系三元组转换的 text query（如 "a person riding a horse"），过 focal softmax CE loss。

3. **级联训练策略**：
   - Stage 1：初始化对象检测器，使用含边界框标注的数据（HICO-DET、COCO、Objects365、VG），**冻结文本编码器**。
   - Stage 2：训练关系解码器，使用含关系标注的数据（HICO-DET、V-COCO、VG）；小数据时冻结检测器防止过拟合，大数据时微调整体性能更好。

4. **损失函数**：
   - 对象检测损失（公式 3）：$\mathcal{L}_{\text{OD}} = \frac{1}{N}\sum_i \mathcal{L}_{\text{cls}}(z_i, \mathcal{T}_{\text{obj}}'; \hat{y}_i) + \mathcal{L}_{\text{box}}(b_i; \hat{b}_i)$
   - 关系检测损失（公式 5）：$\mathcal{L}_{\text{VRD}} = \frac{1}{M}\sum_j \mathcal{L}_{\text{cls}}(r_j, \mathcal{T}_{\text{rel}}'; \hat{c}_j) + \mathcal{L}_{\text{ind}}(r_j; \hat{s}_j, \hat{o}_j)$，其中 $\mathcal{L}_{\text{ind}}$ 用 focal softmax CE 替代 CE 做 index 预测（公式 4）。

5. **数据增强**：
   - **Mosaics**：以 0.4/0.3/0.3 概率使用单图、2×2、3×3 图像网格，融合多数据集样本。
   - **Text Prompting**：对象类别随机采样 80 个 CLIP prompt 模板之一，保证同图同类别 prompt 一致；关系三元组统一用模板 "a {subject} {predicate}-ing a {object}"。

6. **推理**：组合对象检测结果和关系预测得到三元组，按关系 text query embedding 计算相似度得分，Top-K 后做 per-class PNMS（阈值 0.7）。

## 实验与结果
**数据集**：
- HOI 检测：HICO-DET（37,536 训练/9,515 测试，600 类 HOI triplet）、V-COCO（2,533 训练/2,876 验证/4,946 测试，24 actions/80 objects）
- SGG：Visual Genome（108,077 图像，Top-150 物体/Top-50 谓词，7:3 划分）
- 辅助 OD 数据集：COCO、Objects365

**评估指标**：HICO-DET 报告 Full/Rare/Non-Rare mAP；V-COCO 报告 mAP（Scenarios 1&2）；VG 报告 mR@50/mR@100 和 mAP。

**主要结果**：
- **HICO-DET SOTA**：UniVRD (LiT: ViT-H/14) 达到 **38.07 mAP**（Full），较当前最佳自底向上方法提升 **14.26 mAP**（60% 相对提升）；超越所有单阶段方法（如 CDN 31.44、GEN-VLKT 33.75）。
- **可扩展性**：ViT-B/32 → R26+B/1 → ViT-L/14 → ViT-H/14 规模增大，Full mAP 从 29.93 持续提升至 38.07，证明统一模型在数据充足时规模越大收益越大。
- **V-COCO**：统一模型相比特定模型 mAP_S#1 提升约 5.21–5.39，验证了跨数据集知识迁移的有效性（V-COCO 训练集仅约 5,000 张）。
- **VG SGG**：UniVRD (ViT-L/14) 达 mR@50=12.6，mR@100=14.5，超越 HOI 专用方法（AS-Net†: 6.1/7.2, HOTR†: 9.4/12.0）。
- **统一 vs 特定**：HICO-DET 和 VG 上统一模型性能与数据集特定模型相当；VG recall 有轻微下降（归因于 HOI 数据增加导致非 HOI 关系召回略微降低）。
- **消融关键发现**：Mosaics（-1.76）、CLIP prompt for OD（-1.85）、cascade 训练 vs one-stage（-5.05）、per-class PNMS vs vanilla PNMS（+1.24）均为重要增益。

## 相关工作脉络
1. **Word Embedding 语义 VRD**（DRG、FCMNet、ACP、PD-Net 等）：使用静态 word2vec/GloVe 将关系映射到向量空间，但仅限于固定小集合且无图像对齐；UniVRD 使用 VLM 预训练的多模态嵌入，语义表达力和泛化性更强。
2. **传统自底向上 HOI 检测**（InteractNet、GPNN、iCAN、No-Frills 等）：仅使用单一数据集，受限于数据集标签体系；UniVRD 首次将多个 OD 和 VRD 数据集统一训练。
3. **单阶段 HOI 检测**（UnionDet、DIRV、HOTR、QPIC、CDN、GEN-VLKT 等）：端到端直接预测 HOI triplet，计算效率不同；UniVRD 自底向上设计使其能复用大量 OD 预训练数据。
4. **多数据集统一检测**（MSeg、Zhao et al. ECCV2020、Zhou et al.）：在目标检测/分割中合并多数据集标签，但通常出现性能下降；本文首次证明 VRD 统一模型可媲美特定模型。
5. **VLM 用于 VRD**（RLIP、PEVL、CPT）：侧重 few-shot/transfer learning 和 pretext task 设计；本文核心创新是利用 VLM 嵌入空间做**标签空间统一**，属首次探索。
6. **Scene Graph Generation 去偏方法**（TDE、CogTree、DLFE、IETrans+Rwt、DT2-ACBS）：针对 VG 长尾偏见设计；本文明确说明这些增强手段正交可叠加，UniVRD 本身不显式建模层次结构。

## 局限性与未来方向
1. **极端长尾关系类别处理不足**：论文自述当前方法未特殊处理极端 biased/long-tailed 关系类别，可结合 data transfer 或 resampling 策略进一步优化 SGG 表现。
2. **关系层级结构未显式建模**：当前方法在单次预测中同时推理对象和谓词，未建模层次关系；未来可引入更强 VQA-VLM（如 PaLI）建模层级。
3. **统一模型在 VG recall 上有轻微下降**：表明 HOI 数据增加可能轻微损害非 HOI 关系召回，需进一步探索平衡策略。
4. **计算资源需求高**：最大模型（ViT-H/14, 992 GFLOPs）需要大量 TPUv3 训练，限制了普及性。

## 研究启发与可借鉴点
1. **VLM 嵌入统一异构标签空间的设计范式**：可迁移至其他多任务/多数据集视觉理解任务（如通用目标检测、open-vocabulary segmentation），用 text embedding 替代人工标签合并。
2. **级联训练（Cascade Training）稳定性策略**：Stage 1 先训检测器（冻结文本编码器）→ Stage 2 再训解码器的两阶段范式，对多模块依赖模型训练有普适参考价值。
3. **预测 box index 而非坐标的关系匹配设计**：避免冗余预测，提升效率；该设计可推广至其他需要关联多个检测实例的任务（如关键点配对、实例匹配）。
4. **Image-conditioned relation retrieval（One-shot Transfer）**：无需重新训练即可用图像嵌入替代文本嵌入进行关系检索，为视觉检索/跨模态查询提供了灵活接口，可结合本团队方向探索 open-vocabulary relationship retrieval。

## 关键术语表
**UniVRD**：Unified Visual Relationship Detection 的缩写，本文提出的统一视觉关系检测框架。
**VLM（Vision-Language Model）**：视觉-语言模型（如 CLIP、LiT），在大规模图文对上联合训练， learns aligned image-text embeddings。
**Text Query / Text Prompt**：将类别名或关系三元组转换为自然语言描述后输入文本编码器得到的 embedding，替代传统类别整数 ID。
**Relation Query**：关系解码器中预学习的 latent query，用于从 instance embeddings 中检索主体和对象。
**Cascade Training**：先训练对象检测器再训练关系解码器的两阶段训练策略，确保模块间依赖关系的稳定优化。
**Per-class PNMS**：在每个关系类别内分别执行非极大值抑制，而非全局 PNMS，提升去重效果。
**Mosaics Augmentation**：将多张图像拼接为 2×2 或 3×3 网格的增强方法，融合多数据集样本并增加尺度变化。
**One-shot Transfer（Image-based Retrieval）**：不修改模型即可用图像 embedding 替代文本 embedding 进行关系检索的能力。

## 可复现要素
- **数据集**：HICO-DET、V-COCO、Visual Genome、COCO、Objects365 — 均已公开
- **代码**：论文声明将公开发布于 GitHub（URL 在摘要中引用为 footnote 1），但截至论文发表时可能尚未上线
- **关键超参**：relation queries 数量 M=100；per-class PNMS 阈值 0.7；Stage 2 学习率 $1.0 \times 10^{-4}$；batch size=64；cosine learning rate decay；per-example global norm gradient clipping
- **实现框架**：JAX + Scenic library；硬件：TPUv3
- **基础模型**：CLIP（ViT-B/32、ViT-B/16、ViT-L/14）和 LiT（ViT-B/32、R26+B/1、ViT-H/14）
