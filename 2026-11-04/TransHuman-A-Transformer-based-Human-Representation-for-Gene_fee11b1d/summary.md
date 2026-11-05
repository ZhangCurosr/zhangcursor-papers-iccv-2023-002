---
title: "TransHuman-A-Transformer-based-Human-Representation-for-Gene"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Pan_TransHuman_A_Transformer-based_Human_Representation_for_Generalizable_Neural_Human_Rendering_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 22:12:03"
field: "可泛化神经人体渲染"
keywords: ["NeRF", "human rendering", "transformer", "generalizable", "SMPL", "canonical space"]
innovations: ["在规范空间下用Transformer处理painted SMPL以捕捉全局人体关系", "提出DPaRF将token绑定可变形局部辐射场实现观测空间稳健编码"]
benchmarks: ["ZJU-MoCap", "H36M"]
---

# 论文速读：TransHuman: A Transformer-based Human Representation for Generalizable Neural Human Rendering

## 一句话总结
本文提出 TransHuman 框架，首次在规范空间（canonical space）下利用 Transformer 处理 painted SMPL 以捕捉人体各部位间的全局关系，并通过可变形局部辐射场（DPaRF）将表征变形回观测空间，实现了高质量的可泛化神经人体渲染，在 ZJU-MoCap 和 H36M 上取得显著 SOTA。

## 研究问题与动机
1. **可泛化神经人体渲染的核心挑战**：如何仅凭稀疏参考视图，零样本泛化到新人物/新姿态，生成高保真自由视点视频。
2. **现有 SPC 方法存在观测空间优化导致的姿态失配**：NHP 等先前方法在变化观测空间下训练，训练与推理姿态分布不一致，泛化能力受限。
3. **SPC 局部感受野有限，对遮挡不完整输入敏感**：人体自遮挡严重导致 painted SMPL 大量缺失，3D 卷积无法捕获全局语义关系。
4. **离散体素采样存在语义歧义**：直接在观测空间做 grid voxelization 会导致时序语义变化与空间语义纠缠。

## 核心贡献（创新点）
1. **提出 Transformer-based Human Encoding（TransHE）**：首次在 painted SMPL 顶点上应用 Transformer 捕捉全局人体部分关系，区别于 SPC 的局部扩散机制。
2. **设计规范空间人体分组策略（Canonical Body Grouping）**：在静态 T-pose 下进行 k-means 聚类分组，避免观测空间分组导致的语义歧义（时序不一致与空间纠缠）。
3. **引入规范学习方案（Canonical Learning）**：在固定规范坐标下学习位置编码与全局关系，消除训练-推理姿态失配，区别于现有方法直接使用观测坐标的做法。
4. **提出可变形局部辐射场（DPaRF）**：为每个 token 绑定一个坐标系统随姿态变形的局部 NeRF，通过 k 近邻聚合获得稳健查询点表征，区别于全局 trilinear 采样。

## 方法详解
**框架总览**：TransHuman 由 TransHE → DPaRF → FDI 三模块串联，最终通过 NeRF 体积渲染输出颜色。

1. **TransHE（Transformer-based Human Encoding）**
   - **painted SMPL 构建**：将参考图像 CNN 特征投影到预拟合 SMPL 顶点 $V^o \in \mathbb{R}^{6890 \times 3}$，得到 $F \in \mathbb{R}^{6890 \times d_1}$。
   - **规范人体分组**：在 canonical T-pose 下对顶点做 k-means，得到分组字典 $\mathcal{D}^c$，通过平均池化聚合为 $N_t$ 个 token：$\widehat{F} = \mathcal{G}_{\mathcal{D}^c}(F) \in \mathbb{R}^{N_t \times d_1}$。
   - **规范学习**：使用规范坐标 $\widehat{V}^c$ 经 NeRF 位置编码 $\gamma_1$ 后，输入 ViT：$\widehat{F}' = \mathcal{T}(\widehat{F}, \gamma_1(\widehat{V}^c))$，输出带全局关系的 token。

2. **DPaRF（Deformable Partial Radiance Fields）**
   - **坐标系统变形**：每个 token $i$ 绑定一个 canonical 坐标系 $W_i^c$（以 $\widehat{V}_i^c$ 为原点），观测空间下旋转为 $W_i^o = \widehat{R}_i W_i^c$，其中 $\widehat{R}_i$ 为该 token 内顶点的平均旋转矩阵（由 SMPL 24 关节 blend weights 计算）。
   - **坐标编码**：查询点 $\mathbf{p}$ 在 token $i$ 的局部坐标系下编码为 $\overline{\mathbf{p}}_i = W_i^o(\mathbf{p} - \widehat{V}_i^o)$，拼接条件特征得 $\mathbf{h}_i = [\widehat{F}_i'; \gamma_2(\overline{\mathbf{p}}_i)]$。
   - **k-近邻聚合**：对距离最近的 $N_k$ 个 DPaRF 加权求和：$\mathbf{h} = \sum_{i=1}^{N_k} \text{softmax}(\cdot) \mathbf{h}_i$。

3. **FDI（Fine-grained Detail Integration）**
   - **外观特征构建**：将 CNN 深特征与原始 RGB 投影特征拼接后 FC 融合，得到 $\mathbf{a}^j$。
   - **Cross-Attention 融合**：以 $\mathbf{h}^j$ 为 query、$\mathbf{a}^j$ 为 key/value，经 cross-attention 得 $\mathbf{f}^j$，再沿视角维平均池化得最终条件特征 $\mathbf{f}$。

4. **Volume Rendering**
   - 密度 $\sigma(\mathbf{p}) = MLP_\sigma(\mathbf{f})$，颜色 $\mathbf{c}(\mathbf{p}) = MLP_\mathbf{c}(\mathbf{f}, \gamma_3(\mathbf{d}))$。
   - 标准 NeRF 可微体积渲染积分：$\mathbf{C}(\mathbf{r}) = \int_{z_n}^{z_f} T(z)\sigma(z)\mathbf{c}(z)dz$。

5. **训练损失**：$\mathcal{L} = \mathcal{L}_{MSE} + 0.1 \cdot \mathcal{L}_{PER}$（像素级 MSE + 感知损失）。

## 实验与结果
**数据集**：ZJU-MoCap（10 人物，23 相机）、H36M（8 人物，4 相机）。

**评估设置**：姿态泛化（pose generalization）、身份泛化（identity generalization）、单帧泛化（one-shot）、跨数据集泛化（cross-dataset）。

**关键结果（Identity Generalization on ZJU-MoCap）**：
| 方法 | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|------|--------|--------|---------|
| NHP [19] | 24.94 | 0.905 | 0.157 |
| GP-NeRF [6] | 24.55 | 0.902 | 0.159 |
| **Ours** | **26.15** | **0.918** | **0.098** |
| Ours†（GP-NeRF split + overfit trick） | **27.55** | **0.933** | **0.090** |

**姿态泛化（Pose Generalization）**：PSNR 27.25，较第二名 GP-NeRF 提升 **+2.20 PSNR**，LPIPS 降低 **45%**。

**跨数据集（ZJU→H36M）**：PSNR 20.48 vs NHP 18.84，提升 **+1.64 PSNR**。

**消融结论**：
- 规范分组 + 规范位置编码：PSNR 26.15 vs 观测分组 25.28（-0.87）。
- 坐标编码关键：去除后 PSNR 从 26.15 降至 25.80。
- k 近邻聚合（$N_k=7$）带来稳定提升。
- RGB 细粒度特征贡献 +0.10 PSNR。
- 与 SPC 直接对比：26.15 vs 25.14，差距显著。

**效率**：参数量 6.08M（vs GP-NeRF 9.52M），推理显存 6.2GB（vs 10.3GB），同等时间内 PSNR 优于 GP-NeRF 0.84 dB。

## 相关工作脉络
1. **NHP (Neural Human Performer, NeurIPS 2021)**：首个将 SMPL painted features 经 SPC 扩散用于可泛化人体渲染的方法，本文在其基础上用 Transformer 替代 SPC 以解决局部感受野与语义歧义问题。
2. **GP-NeRF (ECCV 2022)**：引入 coarse geometry prior 加速渲染的 SPC 改进版，本文在相同 split 下重跑并去除其 overfitting trick，仍大幅领先。
3. **IBRNet / PixelNeRF**：通用场景可泛化 NeRF 基线，本文将其思想迁移至动态人体，核心差异在于引入 SMPL 先验与规范空间 Transformer 表征。
4. **Neural Body (CVPR 2021)**：per-subject 动态人体重建，需密集视角与逐人优化，本文面向 feed-forward 泛化设置。
5. **SPC (Sparse ConvNet, CVPR 2015)**：3D 稀疏卷积基础网络，NHP/GP-NeRF 均依赖其做特征扩散；本文以 Transformer 全局建模取代局部卷积扩散。
6. **KeyNeRF (ECCV 2022)**：基于关键点相对空间编码的可泛化 NeRF，本文与其不同在于直接操作 painted SMPL 顶点而非 keypoints。

## 局限性与未来方向
1. **SMPL 拟合误差未联合优化**：当前依赖预拟合 SMPL，泛化设置下 joint optimization 尚未探索。
2. **受限于有序多视角捕获设置**：目前训练需多相机同步数据，未支持 unconstrained capture（如稀疏/无序视角）。
3. **Token 数量选择需权衡**：$N_t$ 过大引入误导信息（消融显示 500/1000 反而下降），隐含对表达容量的约束。
4. **未涉及服装形变建模**：SMPL 刚性假设对宽松衣物遮挡区域表征仍有限。

## 研究启发与可借鉴点
1. **规范空间全局建模范式可迁移**：将 Transformer 置于 canonical space 而非 observation space 的设计思路，可推广至动物、手、表情等含刚性-非刚性混合结构的神经渲染任务。
2. **局部坐标系变形编码（DPaRF）机制**：绑定局部坐标系统并随姿态旋转的思想，可应用于其他参数化先验（如 SHAPENET 物体、动物 Mesh）的可泛化 NeRF。
3. **语义分组避免空间纠缠**：在 canonical pose 下做 k-means 分组的策略，对任意网格/点云表征的全局关系学习具有通用参考价值。
4. **RGB 细粒度残差补充**：将原始 RGB 投影与深特征拼接再 cross-attention 融合，是一种轻量高效弥补 CNN 下采样细节丢失的通用技巧。
5. **公平复现意识**：作者指出 prior 工作 split 不一致且含 overfitting trick，本文统一 split 重跑后仍大幅领先，提示社区需重视基线复现的公平性。

## 关键术语表
**Canonical Space（规范空间）**：SMPL 模型的 T-pose 静止参考坐标系，用于消除训练-推理姿态变化带来的语义不一致。
**Observation Space（观测空间）**：人物实际拍摄姿态下的 3D 坐标空间，随时间变化。
**Painted SMPL**：将参考图像的 CNN 特征投影到 SMPL 网格顶点上得到的带纹理特征网格。
**DPaRF（Deformable Partial Radiance Field）**：为每个 body token 绑定的局部 NeRF，其坐标系统随姿态旋转，支持查询点在当前姿态下的精确编码。
**TransHE（Transformer-based Human Encoding）**：在规范空间下对分组后的 painted SMPL token 运行 Transformer，学习人体各部位间的全局语义关系。
**FDI（Fine-grained Detail Integration）**：通过 cross-attention 将人类表示（几何先验）与像素对齐外观特征融合，注入光照/纹理等细粒度信息。
**Pose Misalignment**：训练时优化姿态分布与推理时目标姿态不一致导致的泛化性能下降现象。
**Semantic Ambiguity**：观测空间中同一体素内包含来自不同语义部位的顶点，且随姿态变化而改变，导致 Transformer 学习困难。

## 可复现要素
- **数据集**：ZJU-MoCap（公开）、H36M（公开）；官方 human split 已提供（NHP / GP-NeRF 两种）。
- **代码/权重**：项目页 https://pansanity666.github.io/TransHuman/，论文未明确说明 GitHub 开源状态，但附有项目页面（通常含代码与预训练模型）。
- **关键超参**：CNN=ResNet-18（前3层），$N_v=3$，$N_t=300$（ViT-Tiny），$N_k=7$，每射线采样 64 点，分辨率 512×512，$\lambda_{PER}=0.1$。
