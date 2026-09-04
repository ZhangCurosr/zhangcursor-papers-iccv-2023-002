---
title: "TransHuman-A-Transformer-based-Human-Representation-for-Gene"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Pan_TransHuman_A_Transformer-based_Human_Representation_for_Generalizable_Neural_Human_Rendering_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 21:20:56"
field: "可泛化神经人类渲染"
keywords: ["generalizable neural human rendering", "NeRF", "transformer", "SMPL", "canonical space", "dynamic human reconstruction", "free-viewpoint video"]
innovations: ["在规范空间中用Transformer处理涂绘SMPL以捕获人体全局关系，消除姿态错位", "提出DPaRF将Token绑定到可变形局部辐射场，实现观测空间的鲁棒查询点编码"]
benchmarks: ["ZJU-MoCap", "H36M"]
---

# 论文速读：TransHuman-A-Transformer-based-Human-Representation-for-Gene

## 一句话总结
本文提出TransHuman框架，通过Transformer在静态规范空间处理涂绘SMPL并捕获人体部件间的全局关系，解决了传统SPC-Based方法在变化观测空间中优化导致的姿态错位与局部感受野受限问题，实现了高效且高保真的可泛化神经人类渲染，在ZJU-MoCap上Pose Generalization设置下取得27.25 PSNR的新SOTA。

## 研究问题与动机
1. **动态人类渲染的挑战**：生成高保真自由视角动态人类视频对AR/VR、游戏、远程存在感等应用至关重要，但人体具有动态可变形特性，相比静态场景更复杂，需利用人体先验知识。
2. **现有通用渲染方法的局限**：Generalizable Neural Human Rendering任务要求给定稀疏参考视图即可前馈泛化到新个体，但此前方法（NHP [19]、GP-NeRF [6]）均采用SPC-Based表示，存在两大缺陷：(i) 在变化的观测空间中优化导致训练-推理姿态错位；(ii) 3D卷积局部感受野有限，对因自遮挡导致的涂绘SMPL不完整敏感。
3. **核心需求**：需要一种能在静态规范空间中捕获人体全局关系、并能通过变形映射回观测空间进行鲁棒查询点编码的新型人类表示。

## 核心贡献（创新点）
1. **提出TransHuman框架**：首个基于Transformer的可泛化神经人类渲染框架，在规范空间中处理涂绘SMPL并捕获人体部件间全局关系，实现显著优于现有方法的性能。
2. **TransHE（Transformer-based Human Encoding）**：通过规范体部分组策略避免语义歧义，并在规范空间中进行位置编码学习全局关系，与SPC-Based方法直接在观测空间优化的本质区别在于消除了姿态错位并具备全局视角。
3. **DPaRF（Deformable Partial Radiance Fields）**：将每个Token绑定到可变形局部辐射场，通过坐标系统变形在观测空间中编码查询点，与SPC的三线性采样本质区别在于无需离散3D体素且能处理不完整输入。
4. **FDI（Fine-grained Detail Integration）**：在人体表示指导下通过交叉注意力整合像素对齐的细粒度外观特征，弥补粗粒度人体先验的不足，提升纹理和光照细节。

## 方法详解
**整体架构**：TransHuman由三部分组成——TransHE、DPaRF、FDI，配合体积渲染实现端到端训练。

1. **TransHE（Transformer-based Human Encoding）**：
   - 将参考图像的深度特征投影到预拟合SMPL顶点得到涂绘SMPL $F \in \mathbb{R}^{6890 \times d_1}$。
   - **规范体部分组**：先在T-pose规范空间 $V^c$ 上用k-means聚类获得分组字典 $\mathcal{D}^c$（仅计算一次），再将观测空间的特征按此字典平均池化聚合为 $N_t$ 个Token $\widehat{F} \in \mathbb{R}^{N_t \times d_1}$，避免观测空间中因姿态变化导致的时空语义歧义。
   - **规范学习**：使用规范位置 $\widehat{V}^c$ 作为位置编码输入Lightweight ViT-Tiny $\mathcal{T}(\cdot)$，输出带全局关系的Token $\widehat{F}'$。

2. **DPaRF（Deformable Partial Radiance Fields）**：
   - **坐标系统变形**：每个Token $i$ 在规范空间初始化坐标系统 $W_i^c$，在观测空间中通过平均旋转矩阵 $\widehat{R}_i$（由SMPL 24个关节的blend权重计算）变形为 $W_i^o = \widehat{R}_i W_i^c$。
   - **坐标编码**：查询点 $\mathbf{p}$ 在第 $i$ 个DPaRF下的坐标为 $\overline{\mathbf{p}}_i = W_i^o(\mathbf{p} - \widehat{V}_i^o)$，融合为 $\mathbf{h}_i = [\widehat{F}_i'; \gamma_2(\overline{\mathbf{p}}_i)]$。
   - **K近邻聚合**：将查询点分配给最近的 $N_k$ 个DPaRF，按距离softmax加权聚合得到最终人体表示 $\mathbf{h}$。

3. **FDI（Fine-grained Detail Integration）**：
   - **细粒度外观特征**：将CNN投影特征与原始RGB特征拼接后经FC层融合为 $\mathbf{a}^{1:N_v}$。
   - **粗到细整合**：以人体表示 $\mathbf{h}^{1:N_v}$ 为Query、外观特征 $\mathbf{a}^{1:N_v}$ 为Key/Value进行交叉注意力，得到 $\mathbf{f}^{1:N_v}$，再沿视角维度平均池化为最终条件特征 $\mathbf{f}$。

4. **体积渲染**：
   - 密度 $\sigma(\mathbf{p}) = MLP_\sigma(\mathbf{f})$，颜色 $\mathbf{c}(\mathbf{p}) = MLP_\mathbf{c}(\mathbf{f}, \gamma_3(\mathbf{d}))$。
   - 可微体积渲染：$\mathbf{C}(\mathbf{r}) = \int_{z_n}^{z_f} T(z)\sigma(z)\mathbf{c}(z)dz$。
   - 训练损失：$\mathcal{L} = \mathcal{L}_{MSE} + 0.1 \mathcal{L}_{PER}$（像素级MSE + 感知损失）。

## 实验与结果
**数据集**：ZJU-MoCap（10个主体，23同步相机）和H36M（4相机，8个代表性主体）。

**评估基线**：逐个体优化方法（NV、NT、NHR、NB）和通用方法（NHP、GP-NeRF、PixelNeRF、KeyNeRF、PVA）。

**主要结果**（Identity Generalization设置，PSNR↑/SSIM↑/LPIPS↓）：
- **Pose Generalization**：TransHuman 27.25/0.936/0.087，较第二名GP-NeRF（25.05/0.909/0.159）提升 **+2.20 PSNR、45% LPIPS**。
- **Identity Generalization**：TransHuman 26.15/0.918/0.098，较NHP（24.94/0.157）显著提升；甚至在逐个体方法直接训练目标主体的不公平设定下仍超越其+3.27 PSNR。
- **One-shot Generalization**（仅1参考视图）：24.11/0.891/0.142，超越NHP（23.20/0.182）。
- **Cross-dataset Generalization**（ZJU-MoCap训练→H36M测试）：20.48/0.856/0.169，超越NHP（18.84/0.222）。

**效率对比**：参数量6.08M，推理时间17min（对比GP-NeRF 9min），但PSNR高1.60；快速版Ours-16pts仅需9min推理时间，仍比GP-NeRF高0.84 PSNR，且训练/推理内存更低。

## 相关工作脉络
1. **NHP [19]**：通用神经人类渲染开山之作，采用SPC-Based表示在观测空间中扩散涂绘SMPL特征，是本文的主要对比基线，本文通过规范空间Transformer克服其姿态错位和局部感受野限制。
2. **GP-NeRF [6]**：近期最快的通用方法，引入几何先验加速推理，但存在使用测试视图过拟合的不合理技巧，本文重新公平复现并全面超越。
3. **NeRF [29]**：神经辐射场基础工作，为密度/颜色预测的MLP和体积渲染提供核心框架，本文将其条件化扩展到动态人类。
4. **SPC [23]**：稀疏卷积网络，被先前工作用于3D特征扩散，本文用Transformer替代以获取全局关系和避免离散体素采样。
5. **SMPL [26]**：参数化人体模型，提供预拟合顶点和姿态参数，是本文表示的人体先验基础。
6. **IBRNet [42] / PixelNeRF [51]**：通用场景的NeRF泛化方法，启发本文将泛化思想从静态场景扩展到动态人类。

## 局限性与未来方向
1. **SMPL拟合依赖**：当前方法依赖预拟合的SMPL参数，未探索在通用设置下联合优化拟合SMPL，未来可探索端到端的姿态-形状估计与渲染联合优化。
2. **训练数据限制**：仅在多视角、受控捕捉设置下训练，未来可扩展到无约束捕捉场景（如单目视频、非结构化数据）。
3. **token数量敏感**：实验显示token数过大（>300）不带来进一步提升，可能存在最优区间，值得进一步理论分析。
4. **参考视图依赖**：虽然支持few-shot设置，但多视图仍显著提升性能，单视图极限性能有待探索。

## 研究启发与可借鉴点
1. **规范空间全局建模范式**：将"在规范空间建模全局关系→变形回观测空间"的思路可迁移到其他动态3D任务（如动物、面部、刚体场景的NeRF泛化）。
2. **Token化取代体素化**：用Transformer Token替代SPC的3D体素，既避免语义歧义又提升计算效率，为3D点云/网格的特征学习提供新范式。
3. **可变形局部坐标系统**：DPaRF中为每个Token绑定独立可变形坐标系统的思想，可推广到非刚性场景的局部辐射场建模。
4. **跨数据集泛化验证**：在H36M上验证跨数据集泛化能力的设计值得借鉴，为方法的鲁棒性提供更全面的评估。
5. **粗到细的两阶段特征融合**：先用人体先验提供几何约束的粗表示，再用交叉注意力融合细粒度外观特征的FDI设计，可作为类似任务的特征整合模板。

## 关键术语表
**TransHuman**：本文提出的基于Transformer的可泛化神经人类渲染框架。
**SPC (SparseConvNet)**：稀疏卷积网络，用于在3D体素空间中扩散特征，是先前方法的表示基础。
**SMPL**：Skinned Multi-Person Linear模型，参数化人体模型，提供顶点和姿态先验。
**TransHE**：Transformer-based Human Encoding模块，在规范空间中通过Transformer捕获人体部件间全局关系。
**DPaRF**：Deformable Partial Radiance Fields，为每个Token绑定可变形局部辐射场，在观测空间中编码查询点。
**FDI**：Fine-grained Detail Integration模块，通过交叉注意力整合像素对齐的细粒度外观特征。
**ZJU-MoCap**：浙江大学多视角动态人体捕捉数据集，含10个主体23相机同步视频，本文主要评测基准。
**H36M**：Human3.6M数据集，4相机多人运动捕捉数据集，用于跨数据集泛化验证。

## 可复现要素
- **数据集**：ZJU-MoCap和H36M均为公开数据集，论文使用官方划分（遵循NHP [19]发布的人类分割）。
- **代码**：项目页面为 https://pansanity666.github.io/TransHuman/，论文未明确声明代码开源仓库链接。
- **关键超参**：Token数 $N_t = 300$，K近邻数 $N_k = 7$，参考视图数 $N_v = 3$，射线采样点数64，ResNet-18（前3层）提取特征，ViT-Tiny作为Transformer骨干。
- **论文未提及**：具体训练轮数、学习率、优化器设置、GPU型号、权重开源状态。
