---
title: "TexFusion-Synthesizing-3D-Textures-with-Text-Guided-Image-Di"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Cao_TexFusion_Synthesizing_3D_Textures_with_Text-Guided_Image_Diffusion_Models_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 08:24:54"
field: "3D generative modeling"
keywords: ["3D texture synthesis", "text-guided generation", "diffusion models", "latent diffusion", "multiview consistency", "TexFusion", "SDS-free"]
innovations: ["提出 SIMS 多视图交替去噪聚合机制，避免 SDS 蒸馏并提升跨视角一致性", "基于视角质量的潜纹理聚合策略与神经颜色场后处理解耦潜空间/RGB空间", "两阶段多分辨率级联细化，兼顾全局语义一致性与局部细节"]
benchmarks: ["FID (SD2-depth proxy ground truth)", "User study on Amazon Mechanical Turk"]
---

# 论文速读：TexFusion-Synthesizing-3D-Textures-with-Text-Guided-Image-Diffusion-Models

## 一句话总结
本文提出 TexFusion，一种基于文本引导的图像扩散模型（Stable-Diffusion）的新方法，用于为给定 3D 网格几何体高效合成高质量、视图一致且全局连贯的 3D 纹理；它避免了 SDS 蒸馏的慢速与过饱和问题，只需约 3 分钟即可完成纹理生成。

## 研究问题与动机
1. **目标**：给定 3D 网格与文本提示，直接合成贴合几何表面、语义一致的高质量纹理，用于 VR/AR、游戏资产等下游应用。
2. **现有 SDS 方法的缺陷**：DreamFusion 类方法依赖 Score Distillation Sampling 对神经辐射场进行迭代优化，需要高 classifier-free guidance 权重，导致颜色过饱和、多样性下降，且每样本需 30 分钟以上的优化时间。
3. **直接多视图独立采样的缺陷**：在各相机视角下分别使用 2D 扩散模型采样会缺乏跨视角 3D 一致性，早期视角的错误难以在后续视角中被修正（类似 Janus 面问题）。
4. **训练数据依赖局限**：传统 3D 纹理合成方法往往需要大量带纹理的 3D 形状训练数据，且仅适用于单一类别，泛化能力弱。

## 核心贡献（创新点）
1. **提出 Sequential Interlaced Multiview Sampler (SIMS)**：在多视图序列中去噪过程中交替聚合潜纹理，使相邻视角的去噪方向相互通信，从而在采样阶段即保持 3D 一致性，与 TEXTure 等方法在“全视角顺序生成后再投影”的设计形成本质区别。
2. **避免 SDS 蒸馏，改用标准扩散采样**：直接在 2D 图像扩散模型的采样轨迹上操作，无需对 3D 表示进行反向传播优化，显著提升采样速度（3 分钟 vs. 30+ 分钟）。
3. **潜纹理聚合的质量导向策略**：基于图像空间导数（Jacobian 幅度）衡量视角质量，优先采用视角更“正对且近距离”的视图更新潜纹理，缓解最近邻采样引入的混叠问题。
4. **神经颜色场后处理实现潜空间到 RGB 的转换**：通过优化 instantNGP 参数化的神经颜色场，将解码后多视角 RGB 图像的轻微不一致性平滑为一致的 UV 纹理贴图，解决 LDM 解码在透视变换下的不等价性问题。
5. **多分辨率级联细化**：先在宽 FOV 下获取全局语义一致的低分辨率潜纹理，再上采样至高分辨率并在部分噪声状态下重新执行 SIMS，兼顾全局一致性与局部细节。

## 方法详解
**整体流程**：输入文本提示 $y$、网格 $\mathcal{M}$、相机位姿 $\{C_n\}$；以 SD2-depth（Stable Diffusion 2-depth）为骨干，在潜空间执行 SIMS，得到 3D 一致的潜纹理图 $\mathbf{z}_0$；再用 neural color field 蒸馏为 RGB 纹理。

- **SIMS（Section 4.1）**：
  - 初始化潜纹理 $\mathbf{z}_T \sim \mathcal{N}(0, I)$。
  - 对每个扩散步 $i$，按相机顺序 $n=1..N$：渲染 $\mathbf{x}_{i,n}' = \mathcal{R}(\mathbf{z}_{i,n})$，使用 DDIM 采样得到 $\mathbf{x}_{i-1,n}$；再将 $\mathbf{x}_{i-1,n}$ 逆渲染回 UV 空间得到 $\mathbf{z}_{i-1,n}'$。
  - 对于已访问像素与未访问像素混合的区域，向已访问区域注入噪声使其回到步 $i$ 的噪声尺度，再参与下一视角采样（公式 5）。
  - 聚合规则基于视角质量：采用图像空间导数的负雅可比幅度 $Q_{i,n}$ 衡量视角质量；维护 visited mask $M_i$ 与质量缓存 $Q_i$，按公式 6 决定是否覆盖。

- **潜纹理渲染细节（Section 4.2.1）**：
  - 使用 NVDIFFRAST 进行光栅化渲染，图像尺寸设为 $64\times64$，潜向量维度 $D=4$。
  - 使用最近邻采样而非双线性插值/mipmap，避免早期扩散步中插值改变像素方差，破坏扩散模型输入分布。
  - 背景像素添加标准差为 $\sigma_i$ 的高斯噪声，使模型专注于前景。

- **从潜纹理到 RGB 纹理（Section 4.2.3）**：
  - 使用 SD 解码器 $\mathcal{D}$ 将最终潜图像解码为 RGB 图像 $\xi_n = \mathcal{D}(\mathbf{x}_{0,n})$，由于潜向量对应 $8\times8$ 像素块，解码结果存在视角不一致性。
  - 构建 multi-resolution hash encoding（instantNGP）+ 浅层 MLP 的神经颜色场 $f_\phi(\text{hash}(xyz))$，在 3D 空间坐标上优化。
  - 损失函数：L2 loss + VGG perceptual loss，Adam 优化，学习率 0.01，约 100 轮收敛。
  - 最后在高分辨率纹理图上查询 $f_\phi$ 得到 RGB 贴图。

- **几何处理与多分辨率细化（Section 4.2.4）**：
  - 网格归一化到边长为 1 的立方体内并居中。
  - 第一阶段：宽 FOV 相机，低分辨率潜纹理，仅执行 SIMS 得到 $\mathbf{z}_{0,lr}$。
  - 第二阶段：窄 FOV 相机，按 FOV 比值的平方确定新纹理分辨率；将 $\mathbf{z}_{0,lr}$ 最近邻上采样并随机加噪至 $T=500$ 步，重新运行 SIMS，最后经神经颜色场得到最终纹理。

## 实验与结果
- **数据集**：作者自建的 86 个 mesh-text 配对（35 个网格，每网格 2-4 个提示），未公开大规模基准。
- **基线**：TEXTure [62]（同为 SD2-depth 驱动的文本引导纹理方法）；同时提到 SDS 类方法（DreamFusion、Magic3D、Latent-NeRF）作为速度对比参考。
- **FID**：以 SD2-depth 在相同深度提示下合成的图像为代理 ground truth，在 8 个标准视角下计算 FID：TEXTure 79.47，TexFusion 59.78（越低越好）。
- **用户研究（Amazon Mechanical Turk，3 人投票）**：TexFusion 在自然颜色（75.58% vs. 24.42%）、更少伪影（68.60% vs. 31.40%）、对齐提示（56.98% vs. 43.02%）上显著优于 TEXTure；细节偏好方面 TEXTure 以 65.12% vs. 34.88% 领先。
- **速度**：单 GPU 约 3 分钟/纹理；TEXTure 约 5 分钟；SDS 方法通常 40 分钟以上。
- **结论**：TexFusion 在保持更高速度、更好全局一致性与更少伪影的前提下，取得更优 FID，达到文本引导纹理合成的最新水平。

## 相关工作脉络
1. **DreamFusion / Magic3D / Latent-NeRF**：基于 SDS 将 2D 扩散先验蒸馏为 3D 神经辐射场；TexFusion 明确避开 SDS，改为多视图采样聚合，解决过饱和与速度慢的问题。
2. **TEXTure**：同期并发工作，也在纹理图上做多视图去噪，但采用“顺序执行完整去噪再投影”的自回归策略；TexFusion 采用交替聚合（interlaced），在每一步都同步跨视角信息，一致性更强。
3. **AUV-Net / Texturify / EG3D / GET3D**：传统/学习式 3D 纹理合成方法，依赖特定 UV 对齐、单一类别训练数据或缺乏文本条件；TexFusion 无需真实 3D 纹理训练数据，可直接泛化到任意网格与文本。
4. **MultiDiffusion**：将多个扩散路径融合用于全景图生成；与 TexFusion 在“聚合多源预测”思想上相近，但 MultiDiffusion 仅面向 2D 图像合成，不涉及 3D 一致性。
5. **SDS 类工作（SDS[59], SCJ[84] 等）**：使用梯度信号优化 3D 表示；TexFusion 不使用梯度，直接采样，因此规避了 SDS 所需的高 guidance 权重带来的色彩饱和问题。

## 局限性与未来方向
- **自述局限**：输出纹理锐度仍有提升空间；当前方法非实时（约 3 分钟/纹理），不适合交互场景。
- **改进尝试的副作用**：非随机 DDIM（$\sigma_{t_i}=0$）可增强材质细节，但在平滑表面易产生伪影；ControlNet 高分辨率深度条件可能在小面片网格上触发不必要的面部边界细节。
- **未来方向**：结合更快的扩散采样器（如 DPM-solver、Pseudo-numerical methods）加速；可与后续 TEXTure 改进工作（自动修复低质量区域、per-prompt finetuning）结合；探索实时纹理生成管线。

## 研究启发与可借鉴点
1. **交替聚合优于顺序自回归**：在基于 2D 先验生成 3D 一致内容时，每个采样步都与全局表示交互，可比纯顺序多视图生成更稳定；该思路可迁移至 3D 形状生成、多视图联合修复等任务。
2. **视角质量导向的聚合掩码设计**：用图像空间导数（Jacobian 幅度）衡量视角“直接程度”，作为融合权重依据，既简单又符合渲染理论，可用于其他逆渲染/纹理融合场景。
3. **潜空间与 RGB 空间的解耦后处理**：先在高维潜空间保证 3D 一致性，再通过轻量神经场将解码后的多视角 RGB 蒸馏为 UV 贴图，既发挥 LDM 生成质量，又规避透视不等价性；该方法可复用至其他 latent diffusion + UV texture 流水线。
4. **多分辨率级联策略的适配**：先用低分辨率保证全局语义布局，再上采样并局部细化；这一粗到细范式可与各类扩散模型采样结合，减少长程不一致。
5. **非随机 DDIM / ControlNet 作为可调细节旋钮**：在保持主干不变的前提下，通过替换采样器或引入额外条件网络（ControlNet）增强局部细节控制，为同一框架提供多种风格/精度权衡。

## 关键术语表
- **SDS (Score Distillation Sampling)**：利用 2D 扩散模型的分数梯度来优化 3D 表示，使渲染图像在扩散先验下获得高概率。
- **SIMS (Sequential Interlaced Multiview Sampler)**：TexFusion 核心采样器，在多相机视角序列中交替执行去噪与潜纹理聚合，保持跨视角 3D 一致性。
- **SD2-depth**：Stable Diffusion 2.0 的深度条件版本，可同时接受文本与深度图提示，用于本文的扩散骨干。
- **Neural Color Field**：基于 instantNGP 哈希编码与浅层 MLP 的参数化函数，将 3D 空间坐标映射到 RGB 值，用于融合多视角解码结果。
- **Classifier-free Guidance**：在训练时随机丢弃条件，使模型同时学习条件与无条件分布，采样时通过加权组合增强条件强度。
- **Janus Face Problem**：3D 生成中因模型缺乏跨视角一致性而导致的“双面/多面”异常结构现象。
- **InstantNGP**：基于多分辨率哈希编码的神经隐式表示加速技术，用于高效空间查询。
- **UV Parameterization**：将 3D 曲面展开到 2D UV 空间的参数化表示，用于定义纹理贴图坐标。

## 可复现要素
- **数据集**：作者自建 35 mesh / 86 prompt 数据集，论文未提供公开下载链接（详细分布在 supplementary）。
- **代码/权重**：论文未声明开源代码或发布模型权重；骨干使用公开可用的 Stable Diffusion 2-depth（SD2-depth）。
- **关键超参**：渲染分辨率 $64\times64$，潜维度 $D=4$；神经颜色场优化学习率 0.01、约 100 轮；多分辨率细化时第二部分起始噪声步 $T=500$；相机 FOV 随物体尺寸自适应调整（详见附录）。
- **依赖**：NVDIFFRAST、Stable Diffusion 2-depth、instantNGP、XATLAS（可选 UV 展开工具）。
