---
title: "Latent-OFER-Detect-Mask-and-Reconstruct-with-Latent-Vectors"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Lee_Latent-OFER_Detect_Mask_and_Reconstruct_with_Latent_Vectors_for_Occluded_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 19:12:55"
---

# 论文速读：Latent-OFER: Detect, Mask, and Reconstruct with Latent Vectors for Occluded Facial Expression Recognition

## 一句话总结
提出 Latent-OFER，通过 ViT-SVDD 仅用无遮挡样本自监督检测未知遮挡区域，再利用融合对称先验的混合 ViT-CNN 网络重建面部表情细节，并借助 CNN 类激活图筛选表情相关 ViT 潜向量与 CNN 特征协同预测，显著提升遮挡环境下的 FER 精度。

## 研究问题与动机
- 现实场景中面部常被随机或物体遮挡，传统 FER 模型在遮挡比例增加时准确率急剧下降（尤其当 Grad-CAM 关键区域被遮挡时）。
- 现有遮挡鲁棒特征提取方法难以应对未知遮挡类型与位置；子区域分析依赖面部关键点，遮挡后关键点定位失效；无遮挡图像辅助方法需配对纯净图像作为特权信息，脱离实际部署条件。
- 收集带多样遮挡与表情标注的数据集成本高、耗时长，现有 OFER 方法多依赖全标注遮挡图像训练。
- 需要一种无需遮挡标注、能感知未知物体遮挡、同时重建保留表情语义且能与识别模块协同的端到端解决方案。

## 核心贡献（创新点）
1. **ViT-SVDD 补丁级遮挡检测器**：仅用无遮挡 ViT 潜向量训练深部 SVDD 单机分类器，自动学习正常特征流形半径，可零样本泛化至训练未见的遮挡物体。与依赖全标注异常数据的监督检测器本质不同，实现了自监督、未见遮挡的免标注检测。
2. **自组装层（Self-assembly Layer）与语义一致性损失**：重建过程引入人脸左右对称先验，按跨相关相似度加权融合历史生成、已知区域最相似及对称位置 patch；同时加入语义一致性损失约束重建图像与 GT 图像在预训练 FER 网络上的表情概率分布一致。与以往仅追求视觉自然度的生成重建方法本质不同，显式保留了表情判别语义。
3. **表情相关 ViT 潜向量提取器**：利用 CNN 空间/通道注意力生成 CAM，以空间注意力 Top-50% 区域为键检索完整 ViT 潜向量空间中的对应值，仅将表情关键潜向量与 CNN 特征融合。与直接使用全部 ViT token 或仅依赖 CNN 特征的方法本质不同，避免了无关外观信息引入类间差异。

## 方法详解
- **整体流程**：输入遮挡人脸 → ViT-SVDD 对 ViT patch 进行单机距离度量 → 超出半径的 patch 被 mask → 混合重建网络补全遮挡区域 → CNN 生成 CAM 检索 ViT 表情相关潜向量 → CNN 特征 + 筛选后 ViT 潜向量联合输入 FER 分类器。
- **ViT-SVDD 检测**：将图像划分为 ViT patch，用 ImageNet 预训练 ViT 编码得到潜向量。优化目标（式1）为最小化潜向量到超球中心 \(c\) 的二次距离并加 Frobenius 范数权重正则：\(\min \frac{1}{n}\sum_{i=1}^n \|\Phi(x_i;\mathcal{W})-c\|^2 + \frac{\lambda}{2}\sum_{l=1}^L \|w^l\|_F^2\)。推理时计算各 patch 到 \(c\) 的距离，超过自适应半径则判定为遮挡并生成 mask。
- **混合重建网络**：U-Net 架构融合 ViT（强全局建模、适配多种姿态/遮挡形状）与 CNN（强局部纹理）。编码器内嵌自组装层：对第 \(i\) 个遮挡 patch \(p_i\)，融合三类候选（对称位置 patch \(p_s\)、未遮挡区最相似 patch \(p_k\)、上一轮生成 patch \(p_{i-1}\)），权重由归一化余弦相似度 \(S(p,p_x)=\frac{\langle p,p_x\rangle}{\|p\|\|p_x\|}\)（式2）计算得到（式3）。该设计使生成结果同时具备局部纹理连贯性与面部解剖对称性。
- **损失函数**：总损失 \(L = \lambda_{re}L_{re} + \lambda_c L_c + \lambda_{sc} L_{sc} + \lambda_d(L_d + L_{d_f})\)（式5）。其中语义一致性损失 \(L_{sc} = \sum_{c=1}^{7} p_c(z_{gt}) \log(p_c(z_{rec}))\)（式4），由预训练 FER 网络输出 7 类表情概率分布计算交叉熵，迫使重建图像保留原始表情属性。
- **潜向量筛选与识别**：CNN 分支输出 CAM 后记录空间注意力权重，取权重超过全体均值 50% 的位置作为 key，从完整 ViT 潜向量序列中读取对应 value 作为表情相关潜向量；若 patch 检测失败导致潜向量不可信，空间注意力不会聚焦于该区域，其潜向量自然不被检索与使用。最终 CNN 特征与筛选后的 ViT 潜向量拼接输入分类头。

## 实验与结果
- **数据集**：RAF-DB、AffectNet、KDEF（clean）；Syn-AffectNet/Syn-RAF-DB/Syn-KDEF（合成遮挡）；Occlusion-AffectNet、Occlusion-RAF-DB、FED-RO（真实遮挡）。
- **检测模块对比**（Table 1）：ViT-SVDD 准确率 98.3%、Precision 94.1%、Recall 98.7%，显著优于 One-class SVM（91.1%）与 Patch-SVDD（85.2%）。
- **重建质量与 FER 关联**（Table 8/9）：自组装层 PSNR 26.65、SSIM 0.880；在 Syn-RAF-DB 上基于重建结果的 FER 准确率 77.3%，超越 MAE（72.6%）与 CSA（75.6%）。
- **Clean 数据集**（Table 3）：AffectNet 63.9%、RAF-DB 89.6%、KDEF 88.3%，接近同期最优 FER 模型水平。
- **真实/合成遮挡数据集**（Table 4/5/6）：FED-RO 达 71.8%（新 SOTA）；Occlusion-AffectNet 66.1%（较 OADN +2.1%p）；Occlusion-RAF-DB 84.2%（较 RAN/Wang's +1.5%p）。
- **复杂度**（Table 7）：156 GFlops、373M 参数，在同等量级模型中取得最高精度。
- **消融**（Table 10）：完整模块组合在 Syn-KDEF 上达 86.7%；仅 CNN 特征 75.4%，仅 ViT 全潜向量 76.5%，仅提取表意潜向量 78.5%，验证了重建与表达性潜向量筛选的双重增益。

## 相关工作脉络
1. **单机异常/遮挡检测**：Patch-SVDD [64]、One-class SVM [39] 等面向像素或层级结构。本文将其迁移至 ViT patch 潜空间并配合重建流水线，专注免标注、泛化未见遮挡的检测场景。
2. **遮挡鲁棒特征提取**：如 Wang et al. [53] 的自监督对比学习。本文从“提取抗噪特征”转向“主动重建+靶向潜向量融合”，在遇到复杂未知遮挡时提供更多信息补充路径。
3. **图像修复/补全**：MAE [17]、CSA [28]、GAN-based WGAN AE [30] 等侧重感知自然度。本文针对 FER 下游任务，引入对称先验与表情语义一致性损失，避免重建后表情特征被抹平。
4. **无遮挡图像辅助方法**：Pan et al. [35]、Xia et al. [61] 需成对纯净图像作特权信息。本文完全在单张遮挡图像上运行，更贴合无先验的真实部署。
5. **Transformer 特征利用**：多数 ViT-FER 工作直接 pooling 全 token。本文提出 CAM-guided 检索机制，实现“只取判别性 token”的轻量高效特征对齐。

## 局限性与未来方向
- **检测模块强依赖**：ViT-SVDD 漏检/误检会直接影响重建质量与潜向量可用性；虽然空间注意力可缓解影响，但极端漏检仍会造成性能波动。
- **重建模块跨域泛化受限**：当前仅用单一数据集（KDEF 合成遮挡）训练检测器、用 RAF-DB/AffectNet 训练重建器，面对跨数据集分布偏移时重建一致性可能下降。
- **未来方向**：引入多数据集联合训练或统一对齐策略提升重建模块的跨域适应性；探索更轻量的检测-重建联合预训练范式以进一步降低部署成本。

## 研究启发与可借鉴点
1. **自监督单机 patch 检测范式**：ViT-SVDD 的“仅用正常样本学超球流形→余距判别”设计可复用到医学图像病灶掩码、工业缺陷检测等免异常标注场景。
2. **对称先验+跨相关加权的自组装策略**：将结构化先验（人脸对称性）与上下文相似性结合生成 patch，思路可扩展至其他具有几何规律的结构化图像修复（如手办、建筑立面）。
3. **语义一致性损失迁移**：用下游任务预训练模型约束重建输出的概率分布，保证“看得自然”同时“利于识别”，对人脸老化还原、低光照增强后识别等恢复型预处理链路具有直接参考价值。
4. **CAM 引导的 token 检索机制**：利用 CNN 热力图作为 key 从 ViT 序列中召回特定 value，避免了全量 fine-tune 的开销，可作为 ViT-CNN 双分支特征融合的通用接口设计。

## 关键术语表
**OFER (Occluded Facial Expression Recognition)**：遮挡人脸表情识别，研究面部被物体、手部等遮挡时仍准确判断 7 类基本情绪的任务。
**ViT-SVDD**：部署在 Vision Transformer patch 潜向量上的深部支持向量数据描述单机分类器，仅用无遮挡样本训练，用于零标注检测未知遮挡区域。
**Self-assembly layer**：自组装层，利用人脸水平对称性、未遮挡区最相似 patch 与已生成 patch 三者，按余弦相似度加权融合以修复遮挡区域的结构化重建模块。
**Semantic consistency loss**：语义一致性损失，通过预训练 FER 分类器的 7 类表情概率分布计算交叉熵，约束重建图像在表情语义上与原始 GT 保持一致。
**Expression-relevant latent vectors**：表情相关潜向量，经 CNN 空间注意力 CAM 筛选出的 Top-50% 关键位置对应的 ViT token 向量，仅用于下游特征融合。
**Hybrid reconstruction network**：混合重建网络，在 U-Net 框架内同时集成 ViT 全局建模分支与 CNN 局部纹理分支，并嵌入自组装层与多目标损失。
**CAM (Class Activation Map)**：类激活图，由 CNN 注意力机制生成的空间重要性热力图，用于定位对表情判别最有价值的图像区域。
**Synthetic occlusion**：合成遮挡，通过随机裁剪粘贴手、杯子等物体覆盖原图生成的人工遮挡数据，用于检测与重建模块的训练。

## 可复现要素
- **代码**：已开源 https://github.com/leeisack/Latent-OFER
- **数据集**：RAF-DB、AffectNet、KDEF、Occlusion-AffectNet、Occlusion-RAF-DB、FED-RO（公开）；Syn-AffectNet/Syn-RAF-DB/Syn-KDEF 由作者基于上述数据集合成
- **实现环境**：PyTorch，GTX-3090 GPU
- **关键超参**：\(\lambda_{re}=1,\ \lambda_{c}=0.01,\ \lambda_{sc}=1,\ \lambda_{d}=0.002\)；ViT patch 尺寸 16×16；CAM 空间注意力阈值 Top-50%；检测模块使用 KDEF 随机 copy-paste 合成遮挡自监督训练
- **权重公开情况**：论文仅公开代码仓库，未提供预训练模型权重下载链接

<!--META
{"keywords": ["Occluded Facial Expression Recognition", "Vision Transformer", "One-class Classification", "Image Inpainting", "Latent Vector Fusion", "Semantic Consistency Loss"], "field": "遮挡人脸表情识别", "innovations": ["提出 ViT-SVDD 单机分类模块，仅用无遮挡样本自监督检测未知物体遮挡", "设计混合 ViT-CNN 重建网络与自组装层，结合人脸对称先验与语义一致性损失保留表情判别细节", "通过 CNN 类激活图检索表情相关 ViT
