---
title: "Texture-Generation-on-3D-Meshes-with-Point-UV-Diffusion"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Yu_Texture_Generation_on_3D_Meshes_with_Point-UV_Diffusion_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 08:24:13"
field: "3D 图形生成"
keywords: ["Texture Generation", "Diffusion Model", "3D Mesh", "UV Mapping", "Point Cloud Diffusion", "Coarse-to-Fine"]
innovations: ["首次将扩散模型专门用于网格纹理生成，提出 Coarse-to-Fine 两阶段 Point-UV Diffusion 框架", "引入 Style Guidance 与 Hybrid Condition 策略，分别缓解数据集颜色偏差与级联模型误差传播", "支持文本/单视图图像条件生成，推理速度比测试时优化方法快约 20 倍"]
benchmarks: ["ShapeNet (chair/table/car/bench)", "FID", "KID", "LPIPS"]
---

# 论文速读：Texture-Generation-on-3D-Meshes-with-Point-UV-Diffusion

## 一句话总结
本文提出 Point-UV Diffusion，一种粗到细的两阶段扩散模型框架，首次在给定任意拓扑 3D 网格上自动生成高保真、几何一致的 UV 纹理贴图，并通过风格引导和混合条件策略有效解决数据分布偏差与 UV 碎片化导致的不连续问题。

## 研究问题与动机
1. **核心问题**：如何为给定 3D 网格自动合成高质量、高频细节丰富且与几何结构兼容的纹理。
2. 体素/点云方法受显存限制只能生成低分辨率纹理，难以满足视觉需求。
3. Texture Fields 等隐式表示虽支持高分辨率，但结果普遍过于平滑，缺失高频细节。
4. Texturify 等基于四面体参数化的方法会破坏输入网格的原始几何拓扑，无法忠实保留结构。
5. 现有方法多局限于 GAN/VAE，且直接将 2D 扩散模型应用于 UV 贴图会产生严重接缝与不连续伪影。

## 核心贡献（创新点）
1. **首次将扩散模型专门应用于网格纹理生成任务**。与 Texture Fields / Texturify 等以往方法相比，本文方法能同时保持几何结构保真与高频细节生成。
2. **提出 Coarse-to-Fine 两阶段生成框架**。先通过 3D 点扩散生成全局一致的低频颜色基底，再在 UV 空间进行 2D 扩散增强细节，有效缓解 UV 映射碎片化带来的不连续问题，而直接 2D 扩散模型在此场景下会出现明显伪影。
3. **引入 Style Guidance 机制缓解数据集颜色分布偏差**。通过对训练数据颜色做 PCA + K-means 聚类获得风格标签，使模型推理时可主动选择风格以生成多样化纹理，而先前方法多受限于数据集主导色。
4. **设计 Hybrid Condition + Condition-Truncated Sampling 策略**。训练时以概率 p_hybrid 使用基于区域颜色池化的 smooth texture 替代 coarse texture，推理时前 40% 时间步联合两条件、后 60% 仅用 smooth map，从而降低 Fine Stage 对 Coarse Stage 输出质量的依赖。
5. **证明方法可扩展至文本与单视图图像条件生成**，且推理仅需 30 秒，相比 Text2Mesh 的 10 分钟测试时优化方法速度快约 20 倍。

## 方法详解
- **整体 Pipeline**：Coarse Stage（3D 点扩散）→ UV 投影 + KNN 插值 → Fine Stage（2D UV 扩散）。
- **Coarse Stage**：
  - 在网格表面进行 FPS 采样得到 K = 4096 个点，构成颜色张量 z₀ 与坐标 z_coor。
  - 前向扩散添加噪声得到 z_t，反向去噪网络基于 PVCNN，输入为 [z_coor, z_t, f_g, t]，其中 f_g = E_φ(x_shape) 为全局形状嵌入，x_shape = [x_normal, x_mask, x_coor]。
  - **Style Guidance**：将 z₀ 展平后用 PCA 降维并 K-means 聚类得风格标签 z_style，作为额外条件输入网络，缓解数据集颜色偏好。
  - 着色点投影到 UV 空间，对未着色像素做 KNN 插值生成粗糙纹理图 x_coarse。
- **Fine Stage**：
  - 2D U-Net（含 self-attention）以 [x_shape, x_t, x_coarse, t] 为输入，输出高清纹理图 x̂₀。
  - **Hybrid Condition**：训练中以 p_hybrid = 0.3 的概率使用 x_smooth 替代 x_coarse；x_smooth 通过对 mask 连通区域做平均颜色池化得到，能在训练/推理间缩小分布差距。
  - **Condition-Truncated Sampling**：推理时前 t_c = 40% 时间步同时使用 x_coarse 和 x_smooth，之后仅保留 x_smooth，引导模型从低频到高频逐步细化。
  - **渲染损失**：随机选 4 个视角，用预测 UV 图和 GT UV 图各渲染一张 1024×1024 图像，裁剪 224×224 patch 计算 L₁ loss，辅助提升视觉质量。

## 实验与结果
- **数据集**：ShapeNet 四类（chair / table / car / bench），用 xatlas 生成 UV 贴图，UV 分辨率 512×512，扩散步数 T = 1024，cosine noise schedule（β ∈ [0.0001, 0.02]）。
- **评估基线**：Texture Fields [29]、Texturify [38]、PVD-Tex [49]（将点 voxel 扩散适配为纹理生成器）。
- **定量结果（FID / KID，越低越好）**：

| 类别 | 方法 | FID ↓ | KID ↓ |
|------|------|-------|-------|
| Chair | Ours | **9.88** | **0.22** |
| Chair | PVD-Tex | 15.52 | 0.62 |
| Car | Ours | **26.89** | **0.68** |
| Car | PVD-Tex | 59.47 | 3.74 |
| Table | Ours | **9.63** | **0.15** |
| Bench | Ours | **23.09** | **0.15** |

- **提升幅度**：Chair 类别 FID 较次优方法提升约 36%，Car 类别 FID 降低 55%；KID 在所有类别均优于基线。
- **多样性评估**：以 LPIPS 衡量同 mesh 生成 5 张纹理的 pairwise 相似度，Ours 获得 0.083，显著高于 Texture Fields（0.005）和 PVD-Tex（0.029）。
- **用户研究**：Ours 获 49.8% 偏好率，优于 Texturify（15.9%）和 PVD-Tex（29.2%）。
- **条件生成**：成功实现文本 / 单视图图像条件纹理生成；与 Text2Mesh 相比，在冻结其几何变形分支后仍能在低分辨率网格上产出更丰富的细节。

## 相关工作脉络
1. **Texture Fields [29]**：隐式函数学习纹理表示，擅长高分辨率但结果偏平滑；本文在此基础上引入扩散模型并显式建模 UV 空间细节。
2. **Texturify [38]**：四面体网格卷积生成面片颜色，可生成细部但破坏输入网格拓扑；本文直接作用于 UV 贴图，完整保留几何结构。
3. **PVD [49] / PVD-Tex**：点 voxel 扩散面向点云生成，未针对 UV 碎片化问题做设计；本文的 Coarse-to-Fine 策略专门解决 3D→2D UV 映射带来的不连续挑战。
4. **Text2Mesh [24]**：测试时 CLIP 驱动的网格风格化优化方法，耗时约 10 分钟且需冻结几何分支；本文提供 30 秒前向生成方案。
5. **DreamFusion / Magic3D [31, 21]**：基于 score distillation 从 2D 扩散生成 NeRF / 网格几何；本文聚焦给定网格的纹理生成，不涉及几何创建。
6. **Text2Shape [6]** / **Clip-Mesh [26]**：文本驱动形状/纹理生成，但依赖特定监督信号；本文在无额外 3D 标注条件下实现 unconditional 与 conditional 纹理生成。

## 局限性与未来方向
- 受限于当前 3D 数据集的规模与多样性，生成的纹理深度与真实感尚不及成熟 2D 图像生成模型。
- 高度依赖 UV 映射质量；当网格 UV 展开产生大量碎片化切割时（如 car 类别），纹理会出现接缝伪影。
- 未来方向：构建更大规模、更高多样性的 3D 纹理数据集；结合更先进的 UV 参数化算法以提升跨 patch 一致性。

## 研究启发与可借鉴点
1. **Coarse-to-Fine 扩散范式可迁移**：对于存在多尺度依赖的生成任务（如高分辨率图像、点云着色），先低频后高频的两阶段扩散设计值得借鉴，能有效解耦全局一致性与局部细节生成。
2. **Style Guidance 解决数据分布偏差**：通过聚类分析训练集颜色/风格分布并注入风格条件，是一种简单有效的多样性增强手段，可推广至其他受 bias 影响严重的生成任务。
3. **Hybrid Condition + 截断采样缓解级联模型误差传播**：训练时随机注入弱条件、推理时逐步丢弃强条件，该策略可推广至其他级联扩散模型以提升稳定性。
4. **渲染损失连接 2D 生成与 3D 一致性**：在 UV 扩散中引入多视角渲染 L₁ loss，将 2D 图像质量与 3D 视觉一致性间接耦合，值得在其他 3D-to-2D 生成任务中复用。

## 关键术语表
- **Point-UV Diffusion**：本文提出的两阶段扩散模型，结合 3D 点扩散与 2D UV 扩散完成网格纹理生成。
- **Coarse-to-Fine Pipeline**：先生成低频颜色基底再逐步细化高频细节的两阶段生成策略。
- **Style Guidance**：通过 PCA + K-means 对训练集颜色聚类获得风格标签，并将其作为条件输入扩散模型以提升多样性。
- **Hybrid Condition**：Fine Stage 训练中随机以 smooth texture（区域颜色池化结果）替代 coarse texture，降低对上游输出质量的依赖。
- **Condition-Truncated Sampling**：推理时分段切换条件图，前期联合双条件生成低频结构，后期仅用 smooth map 专注高频细节。
- **UV Mapping**：将 3D 网格表面参数化为 2D 坐标系的映射，是纹理生成的标准表示，但会使连续表面在 2D 中碎片化。

## 可复现要素
- **数据集**：ShapeNet（chair / table / car / bench），公开可用。
- **UV 工具**：xatlas [46]，开源。
- **代码与权重**：https://cvmilab.github.io/Point-UV-Diffusion
- **关键超参**：FPS 采样点数 K = 4096；UV 分辨率 512 × 512；扩散步数 T = 1024；cosine noise schedule（β ∈ [0.0001, 0.02]）；p_hybrid = 0.3；condition-truncated time t_c = 0.4（即前 40% 时间步）。
