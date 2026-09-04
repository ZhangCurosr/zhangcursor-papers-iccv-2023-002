---
title: "Scenimefy-Learning-to-Craft-Anime-Scene-via-Semi-Supervised"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Jiang_Scenimefy_Learning_to_Craft_Anime_Scene_via_Semi-Supervised_Image-to-Image_Translation_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 21:18:25"
---

# 论文速读：Scenimefy-Learning-to-Craft-Anime-Scene-via-Semi-Supervised

## 一句话总结
本文提出了一种半监督图像到图像翻译框架 Scenimefy，通过语义约束微调 StyleGAN 生成并筛选高质量伪配对数据，结合条件对抗损失与新颖的 Patch-wise 对比风格损失（StylePatchNCE），实现了复杂真实场景到精细动漫场景的高保真渲染，同时贡献了一个高分辨率新海诚风格纯净动漫场景数据集。

## 研究问题与动机
1. **任务难点三重叠加**：真实场景含复杂多对象层级关系，动漫风格依赖有机手绘纹理（如草地笔触、云层晕染），且真实与动漫域之间存在巨大间隙，导致现有方法难以兼顾语义一致性与风格化程度。
2. **纯无监督 I2I 的局限性**：CycleGAN 等依赖循环一致性假设，在复杂背景场景下易产生结构失真或伪影；现有动漫风格化方法（如 CartoonGAN、AnimeGAN）多采用手工边缘平滑损失，难以生成细腻的手绘质感。
3. **数据质量瓶颈**：现有动漫数据集多混杂人脸或前景人物，风格分布与纯背景场景差异显著，直接微调预训练生成模型易引发过拟合与语义漂移。
4. **细粒度纹理丢失**：White-box、CTSS 等方法将场景抽象为平滑色块，丢失了动漫中重要的局部笔触与多尺度纹理细节。

## 核心贡献（创新点）
1. **半监督双分支 I2I 翻译框架**：在标准无监督分支外引入监督分支，利用弱配对伪数据提供软监督信号，从根本上缓解了纯无监督设定下结构对齐困难的问题；与纯对抗训练方法的本质区别在于以动态衰减权重平衡伪数据引导与真实域分布学习。
2. **语义约束的 StyleGAN 微调策略**：冻结底层保留空间布局，结合 CLIP 全局语义与 PatchNCE 局部对比双重先验进行少样本域适应；与 StyleGAN-NADA 等人脸/属性级适配方法不同，本文专门针对复杂自然场景的结构层级与纹理多样性进行了联合约束设计。
3. **分割引导的伪数据质量控制机制**：利用 Mask2Former 预测语义掩码，以像素级交叉熵阈值与类别丰富度作为自动过滤标准；区别于人工筛选或单纯依赖生成器置信度，该方案从语义一致性与信息密度两个正交维度提升伪配对数据可用性。
4. **StylePatchNCE 多尺度对比风格损失**：将生成器编码器特征经轻量 MLP 投影后，在同/不同位置 patch 间施加对比约束，显式驱动精细动漫笔触生成；与 CartoonGAN 等手工规则损失的本质区别在于数据驱动、支持多粒度且无需预设风格先验。
5. **高质量纯净动漫场景数据集**：收集并清洗了 5958 张 1080×1080 新海诚风格关键帧，主动剔除含人物主体的干扰画面，填补了场景级动漫渲染基准数据的空白。

## 方法详解
整体流程分为三阶段：
1. **伪配对数据生成**：初始化预训练于真实场景的 StyleGAN2 源生成器 $G_s$，冻结其低分辨率早期层以保留布局，在动漫数据集上微调得到目标生成器 $G_t$。共享潜码 $w$ 生成伪配对 $\{x^p=G_s(w), y^p=G_t(w)\}$。微调损失包含：
   - 全局语义约束：$L_{global} = D_{CLIP}(x^p, y^p) + \lambda_{lpips} lpips(x^p, y^p)$，利用 CLIP 余弦距离与感知损失对齐类别级内容。
   - 局部细节约束：借鉴 CUT 的 $L_{patch}$，使用预训练 CLIP 编码器提取随机裁切 patch 的嵌入，拉近同位置正样本、推远异位置负样本。
   - 总损失：$L_{fine\ tune} = L_{GAN}^t(G_t, D) + \lambda_{global}L_{global} + \lambda_{patch}L_{patch}$。
2. **分割引导数据筛选**：使用 Mask2Former 对 $x^p$ 与 $y^p$ 同步预测语义掩码，计算像素级交叉熵损失 $L_{BCE}$，剔除 $L_{BCE}>5.0$ 的样本；同时排除仅含单一检测类别的图像以保障语义丰富度，最终保留 30,000 对高质量数据。
3. **半监督 I2I 翻译**：
   - **监督分支**：基于条件 GAN 损失 $L_{cGAN}$ 学习跨域映射，并引入新颖的 `StylePatchNCE` 损失：$L_{StylePatchNCE} = \sum_{l=1}^{L}\sum_{i\neq j}L_{patch}^{style}(\tilde{v}_l^i, v_l^i, v_l^j)$
