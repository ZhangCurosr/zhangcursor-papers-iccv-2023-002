---
title: "Toward-Multi-Granularity-Decision-Making-Explicit-Visual-Rea"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Zhang_Toward_Multi-Granularity_Decision-Making_Explicit_Visual_Reasoning_with_Hierarchical_Knowledge_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 16:39:44"
field: "视觉知识推理"
keywords: ["视觉问答", "知识推理", "层次化表示", "神经模块网络", "多粒度知识", "可解释AI"]
innovations: ["提出HCG层次化概念图以多层结构区分不同粒度知识", "设计HCNMN实现层间注意力衰减传播与层内神经模块联合推理", "属性向量置信度加权融合视觉观测与外部知识库先验"]
benchmarks: ["OK-VQA", "FVQA", "GQA", "VQA v2"]
---

# 论文速读：Toward Multi-Granularity Decision-Making: Explicit Visual Reasoning with Hierarchical Knowledge

## 一句话总结
本文提出了一种层次化概念图（HCG）与层次化概念神经模块网络（HCNMN），通过在多层级结构中显式建模不同粒度的知识，实现了对视觉问答中多粒度知识的区分与融合，从而缓解数据偏差并提升可解释性。

---

## 研究问题与动机

1. **多粒度知识的区分困难**：现有 VQA 模型对"grassland"（具体类型）与"ground"（一般类别）等具有不同粒度的概念缺乏有效区分，导致将数据集主导属性（如 ground→green）错误绑定到非主导子类型（如 playground）。
2. **数据偏差的脆弱性**：传统方法将知识以单层图表示，无法对不同粒度层级的知识进行层次化组织，容易受到虚假数据关联的误导，泛化到分布外问题时表现下降。
3. **推理过程可解释性不足**：现有可解释 VQA 模型将推理分解为步骤序列，但忽视了知识在每个推理步骤中具体发挥了什么作用，缺乏对知识贡献的透明刻画。
4. **知识源异构融合的挑战**：视觉-语言证据（图像检测、句法解析）与外部知识库（WordNet、ConceptNet、Visual Genome、Wiki-Text）之间缺乏统一的层次化表示框架来对接。

---

## 核心贡献（创新点）

1. **层次化概念图（HCG）表示**：通过多层结构将每个对象因子化为不同粒度层级的概念集合（上层泛化知识如 ground，下层细化知识如 grassland/playground），与单层知识图谱（SKG）的本质区别在于显式建立了粒度层级间的"typeOf"关系与跨层知识传播机制。
2. **层次化概念神经模块网络（HCNMN）**：设计了同时作用于层内边（intra-layer）和层间边（inter-layer）的注意力转移机制，逐层累积多粒度知识；与已有神经模块网络（NKM、XNM）的本质区别在于引入了跨层注意力衰减传播，使高层通用知识能向下传递给细粒度概念。
3. **三阶段自动图构建流水线**：概念提取（视觉检测+句法解析）→本体构建（WordNet 祖先/后代检索形成层次结构）→知识注入（内层关系与属性向量融合），与已有静态知识图谱的关键差异在于图是逐问题自适应生成的，而非全局固定图谱。
4. **属性向量的置信度加权融合**：通过公式(1) $\boldsymbol{p} = r_v \boldsymbol{p}_v + r_e \boldsymbol{p}_e$ 将视觉特征映射的属性与外部先验属性描述以置信度参数融合，为每个节点提供人类可理解的粒度差异化属性描述，这是已有方法未涉及的。

---

## 方法详解

### 3.1 层次化概念图（HCG）

**图定义**：HCG 是一个多层节点图，包含三类元素：
- **节点（概念）**：按粒度分配到不同层级，越通用（如 place）层级越高，越具体（如 horse）层级越低。
- **层间边（inter-layer edges）**：表示"typeOf"关系，连接不同粒度的父/子概念。
- **层内边（intra-layer edges）**：表示同层概念间的多种关系（如 locationOf、partOf、ride）。
- **属性向量**：每个节点带有预定义占位符的属性向量，描述该粒度的可区分属性（颜色、形状、大小等），不同粒度的占位符含义不同。

**三阶段图生成**：

1. **概念提取**：使用 Faster R-CNN 检测图像中的语义对象，用依存句法分析器（parser [37]）获取词性标注，合并视觉与语言概念池，去除同义词和低频概念。

2. **本体构建**：基于 WordNet synset 图，检索每个提取概念的祖先（hypercategories）和后代（subcategories），通过 inter-layer 的"typeOf"关系连接形成概念列，再按 WordNet 深度属性垂直对齐到对应层级。生成层间亲和矩阵 $D_0$，其中 $d_{ij}^{(0)}$ 表示概念 i 与 j 之间存在"typeOf"关系。

3. **知识注入**：沿 MaveX [46] 流程填充节点的多模态特征 $\boldsymbol{c}$ 和层内关系（亲和矩阵集 $\{D_i\}$）；属性向量融合公式：
$$\boldsymbol{p} = r_v \boldsymbol{p}_v + r_e \boldsymbol{p}_e$$
其中 $r_v=0.6, r_e=0.4$ 为预定义/可训练置信度参数，$\boldsymbol{p}_v$ 来自视觉特征分类，$\boldsymbol{p}_e$ 来自 Wiki-Text-2 定义知识，属性按 Wikitext 中对应概念的粒度选择（如"elephant"附 size、color、nationality）。

### 3.2 层次化神经模块网络（HCNMN）

**层间注意力转移**：
1. 计算注意力掩码（公式 2）：$\boldsymbol{r}_i = MLP(\boldsymbol{a} \circ \boldsymbol{c} \cdot \boldsymbol{D}_i)$，通过 Hadamard 积将概念特征与图拓扑结合。
2. 跨层传播（公式 3）：$\boldsymbol{a}_i' = \boldsymbol{a}_i + \sum_{j=1}^{i}(t\boldsymbol{D}_0)^j \boldsymbol{r}_j \circ \boldsymbol{a}_j$，其中衰减率 $t=0.3$ 控制高层知识随层级下降的重要性递减，在每次 Find/Relate/Filter 推理操作后执行。

**层内注意力**：每层内部使用 NKM [56] 的推理模块执行标准操作（Find、Relate、Filter）。

**多粒度特征聚合**（公式 4）：$\boldsymbol{c}_{final} = \sum_{i=1}^{n} \boldsymbol{a}_i \circ \boldsymbol{c}_i$，将所有层的注意力加权概念特征拼接，送入答案投影层。

**模型超参**：所有嵌入维度 $d_p=300$，图层数 $k=3$。

---

## 实验与结果

**数据集**：
- **GQA**（1.7M 结构化问题，侧重组合推理）
- **VQA v2**（1.1M 问题，通用 VQA）
- **OK-VQA**（要求外部常识知识的 VQA）
- **FVQA**（提供标准事实知识，适合知识驱动推理）

**最强结果**（HCNMN+HCG 在全模型下）：
- **FVQA**: 69.43（各基线中最高，超越 UnifER 的 66.83）
- **OK-VQA**: 36.74（比 NKM+UKG 提升 +1.85）
- **VQA Test**: 70.34
- **GQA Test**: 60.89

**关键发现**：
- HCG 相比 UKG 在 OK-VQA 上带来 +1.85 的提升（最大差距），说明多粒度知识对知识密集型任务效果最显著。
- HCNMN 在 FVQA 上以"仅依赖特定知识集合"的策略超越了使用大规模外部数据库的 UnifER，体现了高效的知识蒸馏能力。
- 注意力分布分析（Table 3）显示 HCG 在 Layer 2（0.47）和 Layer 3（0.32）的注意力显著高于 SKG（Layer 2: 0.39, Layer 3: 0.16），证明模型有效利用了细粒度知识。

---

## 相关工作脉络

1. **NKM / XNM（[56], [36]）**：神经模块网络的早期代表，将推理分解为序列步骤；本文定位差异在于 NKM/XNM 处理的是无层次结构的扁平知识，而 HCG 显式区分了粒度层级。
2. **Scene Graph Generation（[40, 48, 49]）**：如 Graph R-CNN、scene graph 生成方法；本文与它们的差异在于场景图聚焦于图像内对象关系，而 HCG 融合了外部多源知识且建立了跨层粒度结构。
3. **UKG（Unbiased KG，[40]）**：去偏知识图谱，通过重采样减少数据偏差；本文与之对比的核心差异是 UKG 仍是单层结构，而 HCG 通过粒度分层从根本上避免了对泛化概念的属性错误绑定。
4. **UnifER（[14]）**：端到端检索-回答框架，使用大规模外部知识库；本文优势在于以更少的知识资源（特定事实集合）实现更优的 FVQA 表现，强调知识的精准蒸馏而非数量堆砌。
5. **MCAN（[53]）**：双线性注意力网络，多模态融合的强基线；本文在与 MCAN 结合时仍取得 OK-VQA 提升，但 FVQA 略有下降（-2.71），揭示了双线性注意力在细粒度知识匹配上的局限。
6. **MaveX（[46]）**：多模态答案验证方法；本文在知识注入阶段沿用了其特征填充流程，但进一步引入了多源知识融合与层次化组织。

---

## 局限性与未来方向

1. **图构建依赖检测器质量**：概念提取依赖 Faster R-CNN 的目标检测结果和依存句法分析器，检测漏报或句法解析错误会直接影响 HCG 的完整性，在复杂场景下可能引入噪声。
2. **三层结构可能不足以覆盖更细粒度**：实验中固定 $k=3$ 层，对于具有更深层次语义的概念（如"horse"→"thoroughbred"→"racehorse"），当前深度可能不够。
3. **推理模块仍基于传统 NMN 范式**：虽然引入了层次化注意力，但核心推理操作（Find/Relate/Filter）仍是固定的，缺乏对推理步骤本身的自适应学习。
4. **未在零样本/跨域场景充分验证**：实验主要在四个标准 VQA 数据集上进行，多粒度知识对在 OOD（分布外）问题上的泛化优势有待更系统的验证。
5. **可解释性的量化评估不足**：虽然通过可视化展示了推理路径，但缺乏对"解释质量"的独立人工评估或下游任务效用测量。

---

## 研究启发与可借鉴点

1. **多粒度层次化表示的可迁移性**：HCG 的"上层通用→下层细化"分层思想可直接迁移到图像描述、场景理解、关系抽取等需要多粒度知识整合的任务，构建跨模态的层次化知识图谱。
2. **跨层注意力衰减机制**：公式(3)中的 $(t\boldsymbol{D}_0)^j$ 幂次传播是一种优雅的知识逐级衰减方案，可借鉴到分层图神经网络（HGNN）的层间信息传递设计中，用于平衡全局与局部上下文。
3. **属性向量的置信度加权融合策略**：公式(1)将视觉观测属性与知识库先验以 $r_v, r_e$ 加权融合的思路，可推广到多源信息融合问题（如传感器融合、文献知识整合），尤其适合视觉-语言预训练后的后处理阶段。
4. **逐问题自适应图构建**：不同于全局固定知识图谱，HCN 的三阶段流水线按问题动态生成子图，这一设计模式可有效减少无关知识干扰，适合在检索增强生成（RAG）系统中构建问题相关的知识子图。
5. **神经模块网络与知识图谱的解耦集成**：本文证明 HCG 可无缝接入不同 NMN 架构（NKM、XNM、MCAN、UnifER），这种"知识表示层"+"推理模块层"的解耦设计为后续模块化 VQA 系统的扩展提供了参考范式。

---

## 关键术语表

- **HCG（Hierarchical Concept Graph，层次化概念图）**：通过多层结构将对象分解为不同粒度概念（泛化到细化），并用层间"typeOf"边和层内多种关系边连接的图表示。
- **HCNMN（Hierarchical Concept Neural Module Network，层次化概念神经模块网络）**：在 HCG 上执行层间注意力转移与层内神经模块推理的可解释 VQA 模型。
- **Inter-layer edge（层间边）**：连接不同粒度层级的"typeOf"关系边，支持通用知识向下传播。
- **Intra-layer edge（层内边）**：连接同一粒度层概念间的关系边（如 locationOf、partOf、ride）。
- **Property vector（属性向量）**：预定义占位符的描述性特征向量，编码概念的颜色、形状、大小等可区分属性。
- **NKM（Neural Knowledge Module，神经知识模块）**：张等人 [56] 提出的显式知识融入神经模块网络基线。
- **SKG（Single-layer Knowledge Graph，单层知识图谱）**：不加层次结构的扁平知识图谱表示。
- **UKG（Unbiased Knowledge Graph，无偏知识图谱）**：通过数据重采样缓解偏差的单层知识图谱 [40]。
- **Attention shift（注意力转移）**：指层间注意力在每次推理操作后沿 HCG 拓扑从高层向低层的传播过程。

---

## 可复现要素

- **数据集**：GQA [20]、VQA v2 [6]、OK-VQA [30]、FVQA [44]（均公开可下载）
- **外部知识库**：WordNet [33]、Wiki-Text-2 [32]、ConceptNet [26]、Visual Genome [24]（均公开）
- **代码**：https://github.com/SuperJohnZhang/HCNMN（论文声明开源）
- **关键超参**：嵌入维度 300，图层数 $k=3$，层间衰减率 $t=0.3$，属性置信度 $r_v=0.6, r_e=0.4$
- **检测器**：Faster R-CNN [34]
- **句法分析器**：参照 [37]（Compositional Vector Grammars parser）
