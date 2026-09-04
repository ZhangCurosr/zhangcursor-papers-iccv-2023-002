---
title: "Learning-Image-Adaptive-Codebooks-for-Class-Agnostic-Image-R"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Liu_Learning_Image-Adaptive_Codebooks_for_Class-Agnostic_Image_Restoration_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 19:12:52"
---

# 论文速读：Learning-Image-Adaptive-Codebooks-for-Class-Agnostic-Image-R

## 一句话总结
提出 **AdaCode**，通过学习多组类别特定基码本并结合输入图像自适应生成权重图，实现类无关（class-agnostic）的图像重建与恢复，突破了传统单码本方法需按类别单独训练且对复杂自然场景表征不足的瓶颈。

## 研究问题与动机
1. **类别依赖限制泛化**：现有基于离散码本（如 VQGAN、FeMaSR）的生成先验性能优异，但通常需针对不同图像类别（人脸、建筑、自然场景等）分别训练，无法直接处理任意自然图像。
2. **单码本表达力不足**：自然图像常混合多种纹理与结构，单一全局码本只能对潜空间做一种固定划分，易在重建/恢复中引入伪影或过度平滑。
3. **缺乏内容自适应机制**：不同区域视觉语义差异显著，亟需一种能动态感知输入内容、灵活组合多种离散化视角的自适应先验建模方案。

## 核心贡献（创新点）
1. **多基码本自适应融合框架（AdaCode）**：提出以“类别特化基码本+可学习权重图”替代单一全局码本，实现输入图像依赖的离散表征动态组合。与 VQGAN/FeMaSR 本质区别在于用多视角潜空间划分替代单一划分，显著提升对混合语义内容的表达力。
2. **三阶段解耦训练策略**：将先验学习（基码本预训练、表征学习）与下游恢复精炼分阶段执行，在保证参数规模相当的前提下稳定训练。与以往直接复用单码本的做法不同，该设计有效缓解了离散编码突变带来的不连续性。
3. **码本级对比-风格对齐损失**：在 Stage III 引入 InfoNCE 对比损失与风格损失，直接在离散潜特征空间对齐退化特征与 GT 特征。与传统仅依赖像素/感知损失的方法相比，增强了复原结果的语义一致性与纹理真实感。

## 方法详解
整体采用**三阶段串行训练**，核心模块包括编码器 $E$、解码器 $G$、判别器 $D$、$K$ 个基码本 $Z_k$ 及权重预测模块。

- **Stage I 基码本预训练**：使用 SegFormer 对 HR 数据集进行 150 类语义分割，聚类为 5 个超类（Architectures, Indoor Objects, Natural Scenes, Street Views, Portraits），分别在各子集上独立训练 VQGAN。优化目标包含图像级损失（$\mathcal{L}_1$、感知损失 $\mathcal{L}_{per}$、对抗损失 $\mathcal{L}_{adv}$，$\lambda=0.1$）与码本级损失：
  - VQ 损失：$\mathcal{L}_{VQ} = \|sg[\hat{z}] - z_q\|_2^2 + \beta \|\hat{z} - sg[z_q]\|_2^2$（$\beta=0.25$，直通估计器）
  - 语义正则：$\mathcal{L}_{sem} = \|CONV(\hat{z}) - \Phi(y'_k)\|_2^2$（$\Phi$ 为 VGG19 特征提取器）
- **Stage II AdaCode 表征学习**：固定所有 $Z_k$ 与解码器 $G$。输入经 $E$ 得 $\hat{z}$，各码本独立量化为 $z_{q_k}$。权重预测模块（4 个 RSTB + 卷积层）生成 $K$ 通道权重图 $w$，按 $z = \sum_{i} w_i \times z_{q_i}$ 融合为自适应特征。损失为 $\mathcal{L}_{stage2} = \mathcal{L}_1 + \mathcal{L}_{per} + \lambda \mathcal{L}_{adv} + \mathcal{L}_{VQ}(E, G)$。
- **Stage III 下游恢复**：引入 FeMaSR 风格的编码器（特征提取+残差短路），冻结 $G$。利用固定模型提取 GT 特征 $z_{gt}$，将恢复任务转化为退化特征 $z$ 向 $z_{gt}$ 对齐。新增码本级损失：
  $\mathcal{L}_{code} = \mathcal{L}_{InfoNCE}(z_{gt}, z) + \mathcal{L}_{style}(z_{gt}, z) + \beta \|\hat{z} - sg[z_{gt}]\|_2^2$
  总损失 $\mathcal{L}_{stage3} = \mathcal{L}_1 + \mathcal{L}_{per} + \lambda \mathcal{L}_{adv} + \mathcal{L}_{code}$。

## 实验与结果
- **数据集**：训练集为 DIV2K、Flickr2K、DIV8K 与 FFHQ（共 198,061 张 $512\times512$ patch）；测试集涵盖 OST（重建）、Set5/Set14/BSD100/Urban100/Manga109（超分）、DIV2K/DIV8K（修复）。
- **评估基线**：重建对比 VQGAN、VQGAN-aux；超分对比 KX-Net、Real-ESRGAN、FeMaSR；修复对比 GPEN、MAT、FeMaSR。
- **核心数值**：
  1. **重建**：AdaCode 达 **25.76 dB / 0.7705 SSIM**，远超 VQGAN（21.35/0.5664）与 VQGAN-aux（21.92/0.6030），码本规模保持 $1792\times256$。
  2. **超分**：×2 倍率在 Set5、BSD100、Urban100、Manga109 上全面领先或持平最强基线；×4 倍率 Set5 达 **25.868 dB / 0.7731 SSIM**，LPIPS 最低，整体超越 Real-ESRGAN 与 FeMaSR。
  3. **修复**：DIV2K 上 **30.115 PSNR / 0.0516 LPIPS**，DIV8K 上 **32.701 PSNR / 0.0372 LPIPS**，FID **1.1657**，三项指标均为当前最优。
- **结论**：多基码本自适应融合在计算开销相近的情况下，显著提升了类无关图像重建与恢复的保真度与感知质量。

## 相关工作脉络
1. **VQGAN [7] / VQVAE [39]**：奠定离散码本用于图像生成与重建的基础，但仅使用单一
