---
title: "Rethinking-Amodal-Video-Segmentation-from-Learning-Supervise"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Fan_Rethinking_Amodal_Video_Segmentation_from_Learning_Supervised_Signals_with_Object-centric_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 08:23:25"
field: "视频非完整分割"
keywords: ["amodal segmentation", "video segmentation", "object-centric learning", "BEV representation", "multi-view fusion", "self-supervised learning"]
innovations: ["首次将监督信号引入视频非完整分割，结合object slots多视图融合", "提出BEV特征翻译模块替代光流，引入3D几何先验", "设计基于ObjAttention的多视图融合时间模块，实现高效跨帧信息聚合"]
benchmarks: ["Movi-B", "Movi-D", "KITTI"]
---

# 论文速读：Rethinking-Amodal-Video-Segmentation-from-Learning-Supervised-Signals-with-Object-centric-Representation

## 一句话总结
本文提出EoRaS框架，首次将监督信号与以对象为中心的多视角特征融合相结合来解决视频非完整（amodal）分割问题，通过BEV特征翻译和object slots注意力机制，在不依赖光流的前提下有效推断被遮挡物体的完整形状。

## 研究问题与动机
- 视频非完整分割要求从可见部分推断物体完整mask，是极具挑战性的视觉任务。
- 现有自监督方法（如SaVos）依赖光流进行帧间信息传播，在摄像机运动场景下2D warp会导致信号失真。
- 纯监督方法仅依赖当前帧的形状先验（shape prior），对复杂形状和严重遮挡泛化能力有限。
- 高质量视频目标分割（VOS）已能生成精确的可见mask，可作为强监督信号，但此前未被充分利用。

## 核心贡献（创新点）
1. **首次将监督信号引入视频非完整分割任务**：利用当前帧可见mask作为监督，并结合跨视角特征融合，突破了纯自监督方法的局限。
2. **提出基于多视图融合层的时间模块（Multi-view Fusion Layer）**：引入可学习的object slots作为信息容器，通过ObjAttention层实现不同视角特征的高效聚合，避免递归计算的开销。
3. **首次将BEV（Bird's-Eye View）特征翻译引入非完整分割任务**：利用相机内参矩阵将前视特征投影至BEV空间，无遮挡的3D视图为当前帧特征补充形状信息，替代了光流机制。
4. **端到端架构实现SOTA性能**：在Movi-B、Movi-D和KITTI数据集上均取得最优结果，且开集分割实验验证了良好的跨域泛化能力。

## 方法详解
**整体架构**：EoRaS由四模块组成：特征编码→BEV翻译→多视图融合→解码。

**特征编码模块**：使用ImageNet预训练的FPN50提取输入帧的前视特征$f_t^k$（公式1）。

**BEV翻译网络**：基于相机内参矩阵$K$，将3D空间点$(x,y,z)$投影到图像平面$(u,v)$，通过双线性插值获取特征并构建3D体积特征$V_{3D}$（公式2），再经CNN压缩高度维度生成BEV特征$b_k^t$（公式3）。

**多视图融合层**：设计ObjAttention层，由self-attention、cross-attention和MLP堆叠而成（公式4-6）。定义Shape Receiver（SR）和Shape Provider（SP）角色：首先用上一帧slots $S_{t-1}$作为SR、当前帧BEV特征作为SP融合（公式7），再用更新后slots作为SP、当前帧前视特征作为SR进一步融合（公式8），最后用slots更新前视特征（公式9）。

**双向预测**：为解决冷启动问题，引入反向流（backward prediction），拼接正反向特征送入解码器。

**解码网络**：DeConv网络输出全貌掩码$\hat{M}_t^k$和可见掩码$\hat{V}_t^k$（公式10）。

**损失函数**：使用Focal Loss联合监督全貌和可见mask（公式11-12），$\mathcal{L} = \mathcal{L}_{full} + \lambda \cdot \mathcal{L}_{vis}$，实验中$\lambda=1$。

## 实验与结果
**数据集**：合成数据集Movi-B（CLEVR规则形状）、Movi-D（Google Scanned Objects真实形状，遮挡更严重）；真实数据集KITTI（弱监督设置，仅标注car类别）。

**评估指标**：全貌mIoU（mIoU_full）和被遮挡区域mIoU（mIoU_occ）。

**主要结果（Table 1）**：
- Movi-B：mIoU_full=79.22，mIoU_occ=47.89，较SaVos-sup提升8.50%（全貌）和14.28%（遮挡区域）
- Movi-D：mIoU_full=69.44，mIoU_occ=36.96，较SaVos-sup提升8.83%和14.32%
- KITTI：mIoU_full=87.07，mIoU_occ=52.00，较SaVos-sup提升约4%（全貌）和15%（遮挡区域）

**消融实验（Table 2）**：在Movi-B上，时间模块带来约2.3%遮挡区域提升，BEV模块带来约1.06%提升，双向预测带来1.38%提升。

**鲁棒性分析（Table 3）**：slot数量从8到256变化时性能几乎不变，证明对超参不敏感。

**开集分割（Table 5）**：在Movi-B预训练后直接迁移到Movi-D/KITTI，性能显著优于AISFormer和SaVos-sup，验证泛化能力。

## 相关工作脉络
1. **SaVos [35]**：自监督视频非完整分割的代表作，利用光流聚合帧间信息，但受摄像机运动影响大；本文在相同设定下将其转为监督版本（SaVos-sup）进行公平对比，EoRaS大幅超越。
2. **AISFormer [30]**：图像级非完整分割SOTA，使用Transformer mask head；本文作为视频方法在时序信息利用上更优，超越AISFormer约2%。
3. **Slot Attention [19]**：无监督对象中心学习的先驱；本文借鉴其思想，但改为有监督设定，将object slots作为信息容器而非恢复器。
4. **BEV相关方法 [6,17,20,21,23,26,27]**：主要用于自动驾驶场景建图；本文首次将BEV特征翻译引入非完整分割任务，以轻量方式注入3D信息。
5. **PCNET [36] / Convex基线**：自监督图像级方法；本文在测试时即使不使用GT visible mask也能超越这些方法，体现架构优势。
6. **BiLSTM变体 [12]**：论文设计了同等backbone的BiLSTM对比实验，证明所提multi-view fusion layer优于传统时序聚合方式。

## 局限性与未来方向
- 训练阶段需要可见mask作为监督信号（虽可在测试时丢弃，但真实场景中VOS结果的精度直接影响性能上限）。
- BEV翻译仅利用单帧信息，未建模帧间时序关系，3D重建精度受限于单目深度估计能力。
- BEV假设物体水平放置，垂直堆叠的物体在BEV空间中仍会遮挡，无法完全消除遮挡。
- KITTI等真实数据集标注稀疏，限制了模型在复杂真实场景下的充分训练。
- 未来可探索端到端结合VOS模型获取可见mask、引入时序BEV建模、扩展至多类别场景等方向。

## 研究启发与可借鉴点
1. **BEV特征翻译替代光流**：在视频分割任务中，通过单帧BEV投影引入3D几何先验，避免了光流的误差累积，该方法可迁移到其他时序视觉任务（如视频目标检测、跟踪分割）。
2. **Object slots作为跨视图信息容器**：以可学习query形式汇聚多视角特征的模式具有通用性，可推广至图像/视频的多视角融合、跨模态对齐等任务。
3. **监督信号+无监督几何先验的混合范式**：本文证明同时利用当前帧监督信号和跨帧跨视角特征可显著提升性能，这一思路可用于解决其他存在"局部可见、全局未知"问题的任务（如视频补全、动态场景重建）。
4. **轻量级BEV生成方式**：相比三维重建方法，本文的BEV翻译仅需内参矩阵+CNN，计算开销小且易于训练，可作为标准模块集成到更多3D感知任务中。
5. **开集分割评测策略**：在简单数据集预训练后直接迁移到复杂数据集的评测方式，为模型泛化性评估提供了清晰的benchmark范式。

## 关键术语表
**Amodal Segmentation（非完整分割）**：推断被遮挡物体的完整形状，区分于仅预测可见部分的instance segmentation。
**Object-centric Learning（对象中心学习）**：将场景中的每个对象表示为独立的embedding/query，以结构化方式建模场景要素。
**BEV（Bird's-Eye View）**：鸟瞰视角，从上方俯视场景的空间表示，常用于自动驾驶环境建模。
**Object Slots**：可学习的固定数量query向量，作为信息容器从输入特征中聚合对象级表示。
**ObjAttention**：由self-attention、cross-attention和MLP堆叠的对象注意力层，支持shape信息在provider和receiver之间双向传递。
**Shape Provider / Shape Receiver**：在多视图融合中，SP提供形状线索，SR吸收形状线索的角色分工。
**Focal Loss**：针对类别不均衡设计的损失函数，通过调制因子降低易分类样本的权重，增强难例学习。
**GTVM（Ground Truth Visible Mask）**：测试时使用真实可见mask辅助预测，本文通过交集后处理或作为额外输入验证方法鲁棒性。

## 可复现要素
- **数据集**：Movi-B、Movi-D（Kubric合成）、KITTI（真实场景）；Movi系列公开，KITTI公开
- **代码开源**：是，https://github.com/kfan21/EoRaS
- **关键超参**：slot数量$n_s=8$、ObjAttention层数$N=2$、Focal Loss的$\gamma=2$、$\lambda=1$（可见mask损失权重）、batch size=4、学习率Movi数据集$1e^{-5}$/KITTI$1e^{-4}$、AdamW优化器、weight decay=$5e^{-4}$、learning rate decay=0.95、4×Tesla T4 GPU训练50 epoch
