---
title: "Unified-Visual-Relationship-Detection-with-Vision-and-Langua"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Zhao_Unified_Visual_Relationship_Detection_with_Vision_and_Language_Models_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 08:26:13"
---

# 论文速读：Unified-Visual-Relationship-Detection-with-Vision-and-Langua

## 一句话总结
本文提出UniVRD，一种基于视觉-语言模型(VLM)的自底向上统一视觉关系检测方法，首次利用VLM的图文对齐嵌入空间将多个异构数据集的标签语义自动归一化，使单一模型在保持与数据集专用模型相当性能的同时，显著提升跨数据集泛化能力与模型可扩展性。

## 研究问题与动机
1. 现有VRD模型通常仅基于单一数据集训练，受限于特定图像域与词汇表，泛化性与可扩展性不足。
2. 多数据集合并面临标签空间高度异构的挑战：类别间存在同义（如`read` vs `look at`）、从属（如`person` vs `woman/man`）或重叠（如`wine glass` vs `glass`）关系，且涉及二阶主体-谓词-客体语义，手动构建统一分类体系成本极高。
3. 传统语言先验方法仅依赖静态词向量，覆盖的语义类别固定且有限，无法充分利用开放世界的细粒度语义知识。
4. VLM（如CLIP、LiT）在海量图像-文本对上联合预训练，天然具备高度对齐的嵌入空间，为通过语义距离自动统一异构标签提供了可行路径。

## 核心贡献（创新点）
1. **首个基于VLM的多数据集统一VRD框架**：与以往依赖人工合并taxonomy或固定词向量的工作不同，本文直接利用VLM对齐的图文嵌入将异构类别映射到连续语义空间，实现无需人工对齐的标签统一。
2. **Encoder-only ViT适配检测+索引匹配关系解码**：移除池化层与背景类，每个token直接作为实例嵌入；关系解码器通过余弦相似度取索引而非回归坐标，避免冗余预测并复用检测器表征。
3. **开放词汇文本提示分类机制**：用自然语言提示（如`a photo of a person`、`a person riding a horse`）经文本编码器生成查询，替代离散类别整数，天然支持跨数据集类别泛化与语义归一化。
4. **揭示统一模型的缩放增益**：首次证明在VRD任务上统一模型不仅能媲美专用模型，且在模型规模放大时持续受益（HICO-DET提升14.26 mAP），打破了“统一训练必降性能”的固有认知。

## 方法详解
1. **对象检测器**：采用预训练VLM的ViT图像编码器，移除池化与最终投影层，将每个token输出投影为实例嵌入。分类头通过线性层得到分类嵌入，FFN预测边界框坐标；不引入background类，配合负样本频率采样避免对未完整标注样本施加惩罚。
2. **关系解码器**：在检测器后接入轻量Transformer解码器，包含M个可学习关系查询。采用类Perceiver Resampler设计，将查询KV与实例嵌入KV拼接以提升效率。关系嵌入经两个FFN分别投影为主体/客体嵌入，与实例嵌入集合计算余弦相似度，取argmax获得主体/客体框索引（$s_j, o_j$），再检索对应边界框。
3. **文本嵌入分类**：将物体类别填入CLIP提示模板，将三元组填入进行时模板（`a {subject} {predicate}-ing a {object}`），经文本编码器生成文本查询。分类损失采用Focal Sigmoid Cross-Entropy（处理非互斥标签），关系分类则用Focal Softmax CE替代。
4. **数据增强**：Mosaics增强按0.4/0.3/0.3概率随机拼接1×1、2×2、3×3图像网格，扩大尺度覆盖并融合多源数据集；物体提示在80种CLIP模板中随机采样，关系提示使用固定模板且不做提示集成。
5. **级联训练范式**：两阶段训练。Stage 1仅用bbox标注数据（含纯检测数据集）微调检测器，冻结文本编码器。Stage 2加入关系标注微调解码器；数据量小时冻结检测器防过拟合，数据充足时解冻检测器可显著提升性能。匈牙利匹配构建集合预测损失。
6. **推理与One-shot迁移**：组合检测框与索引输出三元组，计算关系嵌入与文本查询相似度得分，执行类内Pair-wise NMS（阈值0.7）。模型支持直接将图像嵌入替换文本查询，无需重训即可实现图像条件关系检索。

## 实验与结果
- **数据集与设置**：HOI评估使用HICO-DET（6
