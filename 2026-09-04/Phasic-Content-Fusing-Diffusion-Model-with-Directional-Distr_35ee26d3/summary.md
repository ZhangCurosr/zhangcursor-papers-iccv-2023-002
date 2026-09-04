---
title: "Phasic-Content-Fusing-Diffusion-Model-with-Directional-Distr"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Hu_Phasic_Content_Fusing_Diffusion_Model_with_Directional_Distribution_Consistency_for_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 19:15:46"
---

# 论文速读：Phasic Content Fusing Diffusion Model with Directional Distribution Consistency for Few-Shot Model Adaption

## 一句话总结
本文针对极少量样本（<10张）下生成模型易过拟合且内容退化的问题，提出了一种分阶段内容融合扩散模型，通过显式分解去噪阶段的学习目标、设计方向分布一致性损失（DDC）以及迭代跨域结构引导策略（ICSG），在内容保留、风格迁移与分布稳定性上显著优于现有Few-shot生成方法。

## 研究问题与动机
- **核心问题**：Few-shot图像生成（训练样本极少）时，生成模型极易过拟合，且生成结果出现内容丢失、风格迁移失败。
- **现有方法不足**：
  1. 基于GAN的Few-shot方法（如IDC、RSSA）受限于架构与生成机制，难以充分保留源域内容；其一致性损失仅约束样本对距离，易导致训练过程中分布发生旋转，造成不稳定。
  2. 直接将GAN的Few-shot损失应用于扩散模型，在去噪后期（t较小、学习局部细节阶段）会发生风格迁移失败，导致风格捕捉失准（如图1所示）。
  3. 扩散模型在few-shot场景下同样存在过拟合风险，且缺乏针对“内容/风格”与“局部细节”分阶段学习的显式解耦机制。

## 核心贡献（创新点）
- **分阶段内容融合训练策略（PCF）**：将扩散模型训练显式分解为t-large（学习内容与风格）和t-small（学习目标域局部细节）两阶段，设计自适应加权与内容注入模块避免阶段间学习干扰。*与已有工作的本质区别：首次结合扩散模型去噪阶段的先验特性设计分阶段内容锚定机制，而非像GAN方法那样在全阶段统一施加一致性损失。*
- **方向分布一致性损失（DDC）**：引入从源域特征中心指向目标域特征中心的方向向量进行分布约束，显式优化分布平移与结构保持，避免传统成对距离约束引发的分布旋转。*与已有工作的本质区别：将隐式的“分布形状相似”转化为显式的“方向引导+中心对齐”，理论更清晰且训练更稳定高效。*
- **迭代跨域结构引导策略（ICSG）**：在推理阶段通过目标域扩散模型迭代生成风格对齐的参考特征，结合低通滤波结构约束，消除源域风格对结构保留的干扰。*与已有工作的本质区别：改进了ILVR直接引用源域参考图的做法，通过跨域风格增强迭代实现结构与风格的解耦约束。*

## 方法详解
- **分阶段训练机制**：定义偏移Sigmoid函数 $m(t) = \frac{1}{1 + e^{-(t - T_s)}}$ 与衰减加权函数 $w(t) = 1 - (t/T)^\alpha$，在大t阶段放大内容与风格相关损失权重，小t阶段聚焦目标域局部细节。
- **分阶段内容融合模块（PCF）**：基于UNet编码器提取源域图像特征 $E(x^A)$ 与加噪图像特征 $E(x_t^A)$。利用 $m(t)$ 自适应混合内容特征与高斯噪声：$\hat{E}(x^A) = m(t)E(x^A) + (1-m(t))z$，再经卷积块与 $E(x_t^A)$ 融合后送入解码器预测噪声，从而在去噪初期强化内容注入。
- **方向分布一致性损失（DDC）**：采用CLIP作为特征编码器 $E$，计算域间方向向量 $w = \frac{1}{m}\sum_{i=1}^m E(x_i^B) - \frac{1}{n}\sum_{i=1}^n E(x_i^A)$。损失函数为 $\mathcal{L}_{DDC} = \| E(x^A) + w, E(x_0^{A\to B}) \|^2$，显式约束生成分布保持源域结构且中心迁移至目标域。
- **综合损失函数**：$\mathcal{L} = m(t)(1-w(t))(\lambda_{DDC}\mathcal{L}_{DDC} + \lambda_{style}\mathcal{L}_{style}) + w(t)\mathcal{L}_{dif}$，其中 $\mathcal{L}_{style}$ 为生成图与目标域图像的Gram矩阵风格差异，$\mathcal{L}_{dif}$ 为标准DDPM扩散噪声预测损失。
- **迭代跨域结构引导（ICSG）**：推理时，将源图像 $x$ 经目标域扩散模型迭代得到风格对齐特征 $y_{t-1}^B$，通过低通滤波 $\phi_N$ 施加结构约束：$x_{t-1} = x'_{t-1} + \phi_N(y_{t-1}^B) - \phi_N(x'_{t-1})$。结合Style Enhancement（SE）模块迭代M次（往返加噪-去噪）强化目标风格，实现结构保真与风格迁移的协同。

## 实验与结果
- **数据集与设置**：源域为 FFHQ 与 LSUN Church；目标域为 Sketches、Cartoon、Van Gogh绘画、Haunted houses。评估设置包括 5-shot 与 10-shot。
- **评估基线**：FreezeD、MineGAN、IDC、RSSA 及直接 fine-tune
