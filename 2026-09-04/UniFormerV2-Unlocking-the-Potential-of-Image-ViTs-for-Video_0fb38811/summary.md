---
title: "UniFormerV2-Unlocking-the-Potential-of-Image-ViTs-for-Video"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Li_UniFormerV2_Unlocking_the_Potential_of_Image_ViTs_for_Video_Understanding_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 21:21:19"
field: "视频理解与动作识别"
keywords: ["Video Understanding", "Vision Transformer", "Action Recognition", "Spatiotemporal Modeling", "Pretrained ViT"]
innovations: ["Local UniBlock时空解耦设计实现高效时序建模", "基于查询的交叉MHRA降低全局聚合计算复杂度", "UniFormerV2解锁预训练图像ViT的视频潜力并达到多基准SOTA"]
benchmarks: ["Kinetics-400", "Kinetics-600", "Kinetics-700", "Something-Something V2", "Moments in Time", "ActivityNet", "HACS"]
---

# 论文速读：UniFormerV2-Unlocking-the-Potential-of-Image-ViTs-for-Video

## 一句话总结
论文提出UniFormerV2，通过高效的UniFormer设计（局部时间关系聚合与全局跨注意力聚合）解锁预训练图像ViT的视频理解潜力，在8个主流视频基准上达到SOTA，首次实现Kinetics-400上90.0% top-1准确率。

## 研究问题与动机
1. **图像ViT到视频任务的领域差距**：预训练图像ViT在场景相关任务上表现良好，但在时间相关任务上远逊于CNN方法，原因在于时空学习能力的不足。
2. **专用视频模型的可扩展性受限**：如UniFormerV1虽能无缝迁移到视频域，但作为独特架构缺乏图像预训练起点，需耗费大量成本进行图像预训练。
3. **开源强大图像ViT的涌现**：CLIP、DINO、MAE等开源图像ViT提供了丰富的监督信号，如何高效利用这些预训练权重成为关键问题。
4. **效率与性能的平衡需求**：现有方法要么依赖昂贵的自监督预训练（如VideoMAE需1600个epoch），要么使用私有数据/模型集成，缺乏经济友好的方案。

## 核心贡献（创新点）
1. **Local UniBlock（局部时间关系聚合）**：在空间ViT块前插入局部时间MHRA，有效降低时间冗余并继承预训练ViT的空间表征能力，与UniFormerV1的联合时空局部关系本质不同，实现了时空解耦的高效建模。
2. **Global UniBlock（基于查询的交叉MHRA）**：引入可学习查询向量将时空tokens聚合为视频token，计算复杂度从$O(L^2)$降至$O(L)$，相比UniFormerV1的全局MHRA大幅降低计算开销。
3. **Multi-Stage Fusion Block（多阶段融合架构）**：自适应集成多尺度时空表征，通过序列融合方式整合各阶段全局token，保留了不同层级的语义信息。
4. **Kinetics-710统一预训练基准**：构建去重后的K400/600/700合并训练集（0.66M视频 vs 原1.14M），训练成本降低33%同时提升性能，实现仅需5-epoch微调即可在三个数据集上获得>1%提升。

## 方法详解
**整体框架**：
- 输入视频经3D卷积（3×16×16）投影为$L$个时空tokens（时间下采样因子2，空间下采样因子16）
- 构建Local UniBlock和Global UniBlock，最终通过Multi-Stage Fusion整合

**Local UniBlock设计**（公式3-7）：
$$\mathbf{X}^T = \text{LT\_MHRA}(\text{Norm}(\mathbf{X}^{in})) + \mathbf{X}^{in}$$
$$\mathbf{X}^S = \text{GS\_MHRA}(\text{Norm}(\mathbf{X}^T)) + \mathbf{X}^T$$
$$\mathbf{X}^L = \text{FFN}(\text{Norm}(\mathbf{X}^S)) + \mathbf{X}^S$$

- LT MHRA：局部时间关系聚合，亲和力矩阵$A_n^{LT}(\mathbf{X}_i, \mathbf{X}_j) = a_n^{i-j}$，仅在时间管$t \times 1 \times 1$内建模
- GS MHRA：来自预训练ViT的全局空间自注意力
- 关键区别：UniFormerV1采用联合时空局部亲和力$a_n^{i-j}$（$j$属于$3D$管$\Omega_i^{t \times h \times w}$），需从头学习；本文分解为局部时间+全局空间，可继承图像预训练

**Global UniBlock设计**（公式8-13）：
$$\mathbf{X}^C = \text{DPE}(\mathbf{X}^L) + \mathbf{X}^L$$
$$\mathbf{X}^{ST} = \text{C\_MHRA}(\text{Norm}(\mathbf{q}), \text{Norm}(\mathbf{X}^C))$$
$$\mathbf{X}^G = \text{FFN}(\text{Norm}(\mathbf{X}^{ST})) + \mathbf{X}^{ST}$$

- C MHRA：交叉注意力风格的关系聚合，可学习查询$\mathbf{q}$与所有时空tokens交互
- 计算复杂度从$O(L^2)$降至$O(L)$
- 查询token初始化为零以保证训练稳定性
- DPE（Dynamic Position Embedding）动态整合3D位置信息

**Multi-Stage Fusion Block**（四种策略）：
- Sequential：前一阶段视频token作为当前阶段查询
- Parallel：拼接所有token后线性投影
- Hierarchical KV：前一阶段token作为当前阶段context的一部分
- Hierarchical Q：前一阶段token作为当前阶段query的一部分
- 最终分类器：$\mathbf{Z} = \alpha \mathbf{F} + (1-\alpha) \mathbf{F}^C$，$\alpha$为可学习参数

## 实验与结果
**数据集**：
- 场景相关：Kinetics-400/600/700、Moments in Time V1
- 时间相关：Something-Something V1/V2
- 未修剪：ActivityNet、HACS

**主要结果**：
| 基准 | 方法 | Top-1 | 提升/特点 |
|------|------|-------|----------|
| Kinetics-400 | UniFormerV2-L (64帧) | **90.0%** | 首个突破90%的方法，参数量仅35% |
| Kinetics-400 | UniFormerV2-L (32帧) | 89.7% | 优于CoCa (88.9%) |
| Kinetics-600 | UniFormerV2-L (64帧) | **90.1%** | SOTA |
| Kinetics-700 | UniFormerV2-L (64帧) | **82.7%** | SOTA |
| Moments in Time | UniFormerV2-L | 47.8% | 优于MTV-H (+2.2%) |
| Sth-Sth V2 | UniFormerV2-L (32帧) | 73.0% | 接近MViTv2-L (74.3%) |
| Sth-Sth V1 | UniFormerV2-L (32帧) | **62.7%** | SOTA |
| ActivityNet | UniFormerV2-L (32帧) | **94.7%** | 提升4.5% |
| HACS | UniFormerV2-L (16帧) | **95.5%** | 提升3.6% |

**关键结论**：
- Global UniBlock对场景相关任务关键，Local UniBlock对时间相关任务关键
- 时序下采样+双倍输入帧可扩大时间感受野
- 基于CLIP-ViT的UniFormerV2在全部8个基准上达到SOTA
- K710预训练相比单独训练节省33%成本且性能更优

## 相关工作脉络
1. **UniFormerV1 [35]**：原始专用视频ViT，统一卷积与自注意力为MHRA，但缺乏图像预训练起点，需长时间预训练；本文在此基础上重新设计局部/全局关系聚合器，使其能与预训练ViT无缝结合。
2. **TimeSformer [4]**：早期视频ViT，使用时间/空间分离注意力，在场景任务上表现良好但时间任务较弱；本文方法在保持高计算效率的同时显著改善时间建模。
3. **VideoMAE [54]**：基于掩码自编码器的视频预训练，需1600个epoch从头训练；本文直接利用开源预训练ViT，经济友好且性能更优。
4. **ST-Adapter [47]**：将时序深度卷积作为适配器；本文的Local MHRA不仅包含卷积操作还引入BatchNorm和无激活函数设计，性能更优（69.1% vs 68.0%）。
5. **X-CLIP [46]**：利用额外语言知识增强视频理解；本文无需语言模态，仅通过结构设计即实现更好性能（87.7% vs 87.1%）。
6. **MTV [72]**：多视角视频Transformer，使用私有预训练数据和模型集成；本文以35%参数量和公开资源实现同等或更优性能。

## 局限性与未来方向
1. **参数量限制**：最大模型UniFormerV2-L仍有354M参数，对于边缘部署场景仍需进一步优化。
2. **跨模态扩展**：当前仅针对纯视觉任务，未来可扩展到视频-语言等多模态理解。
3. **长尾分布处理**：Kinetics-710构建中去除了重复视频，但如何处理视频数据的长尾分布问题未深入讨论。
4. **实时性优化**：虽然计算效率较VideoMAE等显著提升，但64帧输入下的FLOPs（75.3T）仍较高，适合离线分析场景。
5. **预训练策略**：K710预训练仅展示了对Kinetics系列的增益，对跨域泛化的影响有待进一步验证。

## 研究启发与可借鉴点
1. **时空解耦设计范式**：将局部时空关系分解为局部时间+全局空间的解耦策略，为ViT的视频适配提供了清晰的设计思路，可迁移到其他视觉任务（如动作检测、视频生成）。
2. **查询聚合的高效全局建模**：基于单一可学习查询的交叉注意力机制，以$O(L)$复杂度实现全局时空聚合，值得在长序列建模任务中探索。
3. **预训练资源的复用策略**：通过"插件式"模块（Local/Global UniBlock）适配不同监督信号（SL/CL/MIM）的预训练ViT，展现了通用性强、成本低的迁移学习路径。
4. **统一基准构建方法**：Kinetics-710的去重合并策略（移除泄露测试视频）为多数据集联合预训练提供了标准化的实践范例。
5. **多阶段融合的灵活性**：四种融合策略的对比实验展示了设计选择的重要性，序列融合在保持性能的同时最为简洁高效。

## 关键术语表
**UniFormer**：Unified Transformer for Video，统一卷积和自注意力为MHRA的专用视频ViT架构。
**MHRA (Multi-Head Relation Aggregator)**：多头关系聚合器，UniFormer的核心组件，统一建模局部/全局token关系。
**DPE (Dynamic Position Embedding)**：动态位置嵌入，通过3D深度卷积整合时空位置信息的机制。
**Local UniBlock**：局部UniBlock，在ViT空间块前插入局部时间MHRA以降低时序冗余的模块。
**Global UniBlock**：全局UniBlock，基于查询的交叉MHRA模块，将时空tokens聚合为视频token。
**Cross MHRA**：交叉关系聚合器，查询与所有时空tokens进行交叉注意力的设计。
**Kinetics-710**：合并K400/600/700训练集并去重后的统一预训练基准，含0.66M视频和710个类别。
**Sparse Sampling**：稀疏采样策略，从视频中均匀抽取帧以降低计算成本。

## 可复现要素
- **代码开源**：https://github.com/OpenGVLab/UniFormerV2
- **预训练权重**：基于CLIP-ViT-B/16、ViT-L等公开权重，论文提供了完整微调流程
- **数据集**：Kinetics-400/600/700、Something-Something V1/V2、ActivityNet、HACS、Moments in Time均为公开数据集
- **关键超参**：输入分辨率224，稀疏采样策略，8×3×1/4帧采样，BN用于Local MHRA前，LN用于Global MHRA和FFN前
- **训练配置**：UniFormerV2-B参数量115M，UniFormerV2-L参数量354M；FLOPs范围0.4T-75.3T（视帧数和分辨率而定）
