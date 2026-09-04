---
title: "T-oo-Large-Data-Reduction-for-Vision-Language-Pre-Training"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Wang_Too_Large_Data_Reduction_for_Vision-Language_Pre-Training_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 21:19:13"
field: "多模态预训练与数据效率"
keywords: ["Vision-Language Pre-Training", "Data Compression", "Dataset Pruning", "Multi-modal Learning", "Codebook", "Image-Text Alignment"]
innovations: ["首个面向大规模多模态数据的通用压缩算法，压缩率75%-90%", "码本语义聚类+原文本/生成文本拼接策略缓解图文不对齐与模型崩溃", "跨架构泛化验证：在CLIP/ViLT/BLIP及4大数据集上实现同等或更优下游性能"]
benchmarks: ["MSCOCO Retrieval", "Flickr30K", "VQA", "NLVR²", "RefCOCO+", "COCO Captioning", "MSRVTT Video Retrieval", "ImageNet Zero-shot Classification"]
---

# 论文速读：T-oo-Large-Data-Reduction-for-Vision-Language-Pre-Training

## 一句话总结
论文提出 **TL;DR** 算法，通过基于码本的编码器-解码器 captioner 选择代表性样本并生成补充标题，将大规模视觉语言预训练（VLP）数据集压缩至 10%-25%，在 7 个下游任务上达到甚至超越全量数据集训练效果，显著降低训练成本。

## 研究问题与动机
1. **严重图文不对齐**：大规模 VLP 数据集（如 CC3M、LAION400M）中存在大量低匹配度图文对，ITM 分数趋近于零的样本损害多模态表示学习。
2. **高冗余性**：重复/相似内容过多，仅 25% 精选数据即可优于全量数据集训练效果（Figure 1 实验）。
3. **训练成本高昂**：如 CoCa 在 2048 CloudTPUv4 上训练约需 5 天，存储和计算开销难以负担。
4. **现有方法局限**：Dataset Distillation、Data Pruning 等多针对单模态图像任务，依赖类别标签，无法直接迁移至多模态场景。

## 核心贡献（创新点）
1. **首个面向大规模多模态数据的通用压缩方法**：TL;DR 可将 CC3M 压缩 76%（2.82M→0.67M）、YFCC15M 压缩 83%（15M→2.5M），压缩率远超 Data Pruning（20-30%）。
2. **码本引导的语义聚类采样**：将图像特征通过可学习码本投影到语义空间后聚类，相比直接使用 Image/Text Embedding 聚类可获得更好的下游性能（TR@1 72.8 vs 70.6/69.0）。
3. **原文本与生成文本拼接缓解模型崩溃**：纯生成文本会导致 Contrastive Loss 塌陷（captioning collapse），拼接策略既保留多样性又消除低质量图文对。
4. **跨架构/跨数据集泛化验证**：在 CLIP、ViLT、BLIP 三种主流 VLP 架构和 CC3M/CC12M/YFCC15M/LAION40M 四大数据集上均取得可比或更优结果。

## 方法详解
TL;DR 分两阶段：

**Stage 1：码本 captioner 训练**
- 视觉编码器（ViT-B/16）提取 L=196 个 token 特征；
- 可学习码本（K=3000 个嵌入向量）通过最近邻查找将特征量化为 index vector；
- 文本解码器（BERT-LM Head）自回归生成 caption；
- 损失函数：语言建模损失（LM loss）+ 对称承诺损失（symmetric commitment loss）；
- 码本初始化：用词频最高的 K 个关键词/短语的文本嵌入初始化（优于 Xavier 和 object tags 初始化）。

**Stage 2：数据压缩**
1. **样本选择**：对每个图像的 index vector 执行 K-Means 聚类（N=3000 簇），每簇均匀采样 M%（默认 25%）。
2. **标题精炼**：将生成文本 $T_g$ 与原文本 $T_o$ 拼接为 $T = T_o + T_g$，保留唯一性的同时消除图文不对齐。
3. 最终得到压缩数据集 $D_c$，用于 VLP 预训练。

## 实验与结果
- **数据集**：CC3M（2.82M）、CC12M（10.8M）、YFCC15M（15M）、LAION40M(128)（40M）。
- **模型架构**：CLIP、ViLT、BLIP。
- **下游任务**：MSCOCO/Flickr30K 检索、VQA、NLVR²、RefCOCO+、COCO Captioning、No-Caps、MSRVTT 视频检索、ImageNet 零样本分类（共 7 类任务）。

**主要结果**（BLIP 架构，CC3M 数据）：
| 指标 | 全量 CC3M | TL;DR-CC3M (25%) | 对比 |
|------|-----------|-------------------|------|
| COCO TR@1 | 70.9 | **72.8** (+1.9) | 超越 |
| COCO IR@1 | 54.3 | **54.8** (+0.5) | 超越 |
| Flickr30K TR@1 | 86.3 | **87.5** (+1.2) | 超越 |
| VQA test-dev | 71.5 | **73.1** (+1.6) | 超越 |
| NLVR² test-P | 76.2 | **78.0** (+1.8) | 超越 |
| RefCOCO+ val | 72.4 | **75.1** (+2.7) | 超越 |
| COCO Caption B@4 | 36.8 | **37.6** (+0.8) | 超越 |
| COCO Caption CIDEr | 121.6 | **123.8** (+2.2) | 超越 |

- **跨数据集**：TL;DR-YFCC15M（2.5M）在多数任务上优于全量 YFCC15M（15M）；TL;DR-LAION40M（8M）在 6/7 任务上超越全量基线。
- **训练时间**：CC12M 从 65h 降至 14h，YFCC15M 从 90h 降至 15h，LAION40M 从 120h 降至 48h。

## 相关工作脉络
1. **Dataset Distillation**（如 MMT、Cafe）：针对小尺度单模态图像（CIFAR），依赖类别标签，在 ImageNet-1K 上仅 33.8% 准确率，不适用于多模态无监督场景。
2. **Data Pruning**（如 Sorscher et al. 2022）：基于梯度/难度筛选 20-30% 样本，需类别标签，压缩率低，本文压缩率达 75-90%。
3. **Neural Data Server**（NDS）：面向单任务检索，需访问所有下游数据，本文方法 task-agnostic，无需下游先验。
4. **CiT**（Xu et al. 2023）：动态训练数据策展，但依赖额外辅助模型和检索流程；本文无需大型辅助模型。
5. **VQ-GAN/DALL-E 2/Stable Diffusion 生成数据**：Table 7 对比显示，即使最先进的生图模型生成的 0.3M 数据仍显著落后于真实数据（TR@1: 52.4 vs 58.3）。

## 局限性与未来方向
1. **最高压缩比需人工设定**：采样比例（如 25%）和聚类数（3K）为手动调参，尚未实现端到端自适应优化。
2. **视觉多样性损失影响分类任务**：ImageNet 零样本分类性能略低于全量训练（CLIP/BLIP 架构），因压缩降低了视觉多样性。
3. **LAION 压缩率相对较低**：因 LAION400M 已做过 CLIP 相似度过滤，图文不对齐问题较轻，方法增益受限。
4. **未来方向**：探索更高压缩比、结合文本到图像生成模型提升多样性、拓展至视频-语言数据集。

## 研究启发与可借鉴点
1. **码本初始化策略可迁移**：用高频关键词/短语的文本嵌入初始化码本，比随机初始化更稳定，可借鉴于其他向量量化多模态学习任务。
2. **"拼接而非替换"的标题精炼思路**：纯生成标题易导致对比学习塌陷，拼接策略简单有效，可用于其他多模态去噪场景。
3. **语义聚类优于视觉聚类**：codebook 将图像投影到语义空间后再聚类，比直接使用 BLIP image embedding 效果更好，提示多模态聚类应重视语义对齐。
4. **跨架构验证设计**：在 dual-stream/one-stream/encoder-decoder 三种架构上统一验证，结论说服力强，值得借鉴。
5. **与团队方向结合机会**：可将 TL;DR 压缩后的数据集用于后续大模型微调，或结合团队的数据高效学习研究方向扩展至视频/3D 领域。

## 关键术语表
- **TL;DR**：论文提出的 Vision-Language 数据压缩算法名（Too Large; Data Reduction），意为"太长不看"。
- **Codebook-based Captioner**：基于可学习码本的编码器-解码器 caption 生成器，将图像特征量化为离散码后进行文本生成。
- **Image-Text Matching (ITM) Score**：图文匹配分数，衡量图文对对齐程度，低分数样本表明存在严重 misalignment。
- **Symmetric Commitment Loss**：向量量化中用于约束 encoder 和 codebook 同步更新的对偶损失。
- **Model Collapse / Captioning Collapse**：在对比学习中，当生成文本过于单一时模型陷入退化解的现象。
- **Cross-modality Alignment**：跨模态对齐，指图像与文本在共享语义空间中的匹配能力。
- **Visual Grounding**：视觉定位，指根据文本描述在图像中定位目标区域的能力。
- **Data Pruning**：数据剪枝，从大集中选择最具代表性的子集以加速训练。

## 可复现要素
- **数据集**：CC3M、CC12M、YFCC15M、LAION40M 均为公开数据集。
- **代码/权重**：论文未声明代码或压缩后数据集是否开源（论文未提及）。
- **关键超参**：码本大小 K=3000，token 长度 L=196（ViT-B/16），聚类数 N=3000，采样比例 25%，训练 20 epochs，batch size=1260，学习率 warm-up 至 3e-4，AdamW optimizer，weight decay=0.05。
- **硬件**：数据压缩用 8×NVIDIA A100 GPU，预训练用 2 nodes×16 GPUs。
