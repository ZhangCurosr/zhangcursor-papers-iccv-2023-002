---
title: "SegGPT-Towards-Segmenting-Everything-In-Context"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Wang_SegGPT_Towards_Segmenting_Everything_in_Context_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 21:18:42"
field: "通用视觉分割"
keywords: ["in-context learning", "generalist segmentation", "few-shot segmentation", "video object segmentation", "random coloring", "feature ensemble"]
innovations: ["随机颜色映射方案防止多任务坍缩，增强泛化能力", "特征级上下文集成策略支持任意数量提示和时序建模", "上下文微调机制实现零参数更新的领域适配"]
benchmarks: ["COCO-20i", "PASCAL-5i", "FSS-1000", "YouTube-VOS 2018", "DAVIS 2017", "MOSE", "ADE20K", "COCO Panoptic"]
---

# 论文速读：SegGPT-Towards-Segmenting-Everything-In-Context

## 一句话总结
SegGPT 是一个通用分割模型，通过 in-context learning 框架将多种分割任务（实例、语义、部分、轮廓、文本、视频等）统一为"in-context 着色"问题，仅需单个模型即可在不同数据集和场景中实现零样本或少样本分割。

## 研究问题与动机
1. **现有分割模型均为 specialist**：每种任务（前景分割、交互分割、语义分割、实例分割、全景分割等）需独立训练，适配新类别或新数据类型（如视频）成本高昂。
2. **异构数据难以统一训练**：部分分割、语义分割、实例分割、全景分割、人像、医学图像、航拍图像等数据类型差异巨大，传统多任务学习方案缺乏灵活性。
3. **泛化能力不足**：现有视觉通用模型（如 Pix2Seq、OFA、UViM）依赖特殊 token 或硬编码指示符，难以迁移到新任务或处理 out-of-domain 目标。

## 核心贡献（创新点）
1. **首个通用分割模型**：SegGPT 首次证明单个模型可自动完成多种分割任务；与已有工作的本质区别在于不依赖特定颜色或特殊 token 作为任务指示符，而是通过上下文示例引导模型完成任意任务。
2. **随机颜色映射方案（Random Coloring Scheme）**：通过对每个数据样本随机重映射颜色，迫使模型依赖上下文推理而非颜色本身；与 Painter 固定颜色空间的本质区别在于防止模型坍缩为多任务学习解。
3. **上下文集成策略（Context Ensemble）**：提出空间集成（Spatial Ensemble）和特征集成（Feature Ensemble）两种多示例推理方法；与 NLP prompt ensemble 的预测 logits 融合不同，Feature Ensemble 在中间特征层交互，支持任意数量提示和时序建模。
4. **上下文微调（In-Context Tuning）**：冻结预训练模型参数，仅优化可学习图像张量作为任务提示；与 Visual Prompting（CLIP 类模型固定像素填充）的本质区别在于提供原生图像接口，适配通用分割而非分类。

## 方法详解
- **整体框架**：基于 Painter [50] 框架，将视觉任务输出空间重新定义为"图像"，统一为图像修复（inpainting）问题；使用 vanilla ViT-L（307M 参数）和 smooth-ℓ₁ 损失，无架构修改。
- **In-Context Coloring 训练**：
  - 随机采样与输入图像共享相似上下文（如相同语义类别或实例）的图像，从中采样颜色集合并映射到随机颜色，生成 in-context pair。
  - 引入 mix-context 训练：将多个共享相同颜色映射的图像拼接后随机裁剪缩放，形成混合上下文训练样本。
  - 不同数据类型的上下文定义：语义分割随机采样类别，实例分割随机采样实例数，同一图像的不同变换视为上下文图像。
- **Context Ensemble 推理**：
  - **Spatial Ensemble**：将多个示例拼接到 n×n 网格后下采样至单示例尺寸，语义信息在上下文提取中几乎无额外成本。
  - **Feature Ensemble**：在 batch 维度组合多个示例独立计算，每层 attention 后对 query 图像特征求平均，使 query 聚合所有参考示例信息。
- **In-Context Tuning**：冻结整个预训练模型，初始化可学习图像张量作为输入上下文，仅优化该张量；调优后可作为 plug-and-play 提示用于特定数据集/场景/人物。

## 实验与结果
- **训练数据集**：ADE20K（150类别）、COCO（80 things + 53 stuff）、PASCAL VOC、Cityscapes、LIP（人体）、PACO（部件属性）、CHASE DB1/DRIVE/HRF/STARE（视网膜血管）、iSAID/loveDA（航拍图像）。
- **评估任务与关键结果**：
  - **Few-shot 语义分割（COCO-20i/PASCAL-5i）**：SegGPT one-shot 56.1 / 83.2，few-shot 67.9 / 89.8，显著优于 Painter（32.8/64.5 和 32.6/64.6）及多数 specialist 模型。
  - **Few-shot 语义分割（FSS-1000，未训练）**：one-shot mIoU 85.6，few-shot 89.3，与在 FSS-1000 上训练的 specialist 模型（如 DACM 90.8/91.7）接近。
  - **视频对象分割（YouTube-VOS 2018，未用视频训练）**：G=74.7，Js=75.1，Fs=80.2；DAVIS 2017 J&F=75.6；MOSE J&F=45.1。
  - **ADE20K 语义分割（In-Context Tuning）**：mIoU=39.6，较 Painter 下降 10.3 点。
  - **COCO 全景分割（In-Context Tuning）**：PQ=34.4，较 Painter 下降 9.0 点。
- **最强结果**：PASCAL-5i few-shot 89.8 mIoU；FSS-1000 few-shot 89.3 mIoU（未训练）；YouTube-VOS 2018 G=74.7。
- **消融结论**：Feature Ensemble 在高分辨率 DAVIS 上优于 Spatial Ensemble（避免下采样信息丢失）；DAVIS 上 8 帧为最优平衡点。

## 相关工作脉络
1. **Painter [50]**：SegGPT 的直接基础，采用固定颜色空间的 in-context 着色框架；SegGPT 通过随机颜色映射解决多任务坍缩问题，泛化性更强。
2. **Pix2Seq [8,9] / OFA [49] / Unified-IO [37]**：将视觉任务输出定义为离散 token 序列；与 SegGPT 本质区别在于 SegGPT 使用连续图像空间且无硬编码任务指示符。
3. **UViM [27]**：统一 panoptic 分割、深度估计、colorization 等像素标注任务，但需为每个任务训练独立模型；SegGPT 仅需单一模型。
4. **Visual Prompting [2,3]**：在 CLIP 等模型上通过固定像素填充实现视觉提示；SegGPT 提供原生图像接口并面向通用分割，而非分类任务。
5. **HODOR [1]**：无需视频训练的视频分割 specialist 模型；SegGPT 是 generalist，可同时处理图像和视频分割。
6. **FPTrans [57] / HSNet [39] / VAT [21]**：少样本语义分割 specialist 方法；SegGPT 无需针对 shot 数单独训练，单个模型支持 one-shot 和 few-shot。

## 局限性与未来方向
1. **随机颜色映射增加训练难度**：导致在数据充足的 in-domain 任务（ADE20K、COCO 全景分割）上性能下降（较 Painter 分别低 10.3 和 9.0 点）。
2. **模型规模受限**：当前 ViT-L 规模不足以充分挖掘 generalist 潜力， scaling up 需要更大数据支持。
3. **视频分割依赖前序帧缓存**：Feature Ensemble 虽能建模时序关系，但需维护历史 mask 队列，推理效率有待优化。
4. **未来方向**：扩大模型规模、探索自监督学习以获取更大数据、将 in-context learning 拓展至更多视觉任务。

## 研究启发与可信借鉴点
1. **随机映射策略可迁移**：随机颜色/值重映射思想可用于其他视觉任务的 in-context 学习（如深度估计、关键点检测），防止模型依赖固定输出分布。
2. **Feature Ensemble 设计值得借鉴**：多层特征级交互比 NLP 的 prompt ensemble（logits 融合）更高效，可推广至多示例视觉推理场景。
3. **In-Context Tuning 范式**：冻结主干+优化可学习图像提示的方式，为通用模型的领域适配提供了低参数量、保持泛化性的新途径，可与本团队的方向（如领域自适应、少样本学习）结合。
4. **数据采样策略的统一视角**：将不同分割任务统一为"颜色采样+上下文配对"的格式，简化了多源数据整合流程，无需手动合并标签体系。
5. **消融洞察**：Spatial vs Feature Ensemble 在不同分辨率数据上的表现差异，提示后续工作需根据数据特性选择集成策略。

## 关键术语表
- **In-Context Learning**：给定少量示例 prompt 后，模型无需更新参数即可完成新任务的学习范式，源自 GPT-3。
- **Random Coloring Scheme**：对每个训练样本随机重映射输出颜色，迫使模型依赖上下文而非固定颜色进行推理。
- **Context Ensemble**：多示例推理策略，包括 Spatial Ensemble（网格拼接下采样）和 Feature Ensemble（特征层平均聚合）。
- **In-Context Tuning**：冻结预训练模型，仅优化可学习图像张量作为任务提示的轻参数适配方法。
- **Smooth ℓ₁ Loss**：结合 L1 和 L2 优势的损失函数，对离群点更鲁棒，用于像素级回归任务。
- **Few-Shot Semantic Segmentation**：仅用少量标注样本（如 1 或 k 个）学习的语义分割任务。
- **Video Object Segmentation (VOS)**：给定首帧或少数帧的 mask，对视频其余帧进行目标分割的任务。
- **Panoptic Segmentation**：统一语义分割（stuff）和实例分割（things）的全景分割任务。

## 可复现要素
- **数据集**：ADE20K、COCO、PASCAL VOC、Cityscapes、LIP、PACO、CHASE DB1、DRIVE、HRF、STARE、iSAID、loveDA、YouTube-VOS 2018、DAVIS 2017、MOSE、FSS-1000（均为公开数据集）。
- **代码**：已开源，地址 https://github.com/baaivision/Painter。
- **权重**：使用 Painter [50] 预训练 checkpoint 初始化，论文未单独提供 SegGPT 权重下载链接。
- **关键超参**：ViT-L（307M 参数）、batch size=2048、learning rate=1e-4（cosine scheduler）、weight decay=0.05、训练 9K iter（warmup 1.8K）、输入尺寸 448×448、DAVIS 最优帧数=8。
