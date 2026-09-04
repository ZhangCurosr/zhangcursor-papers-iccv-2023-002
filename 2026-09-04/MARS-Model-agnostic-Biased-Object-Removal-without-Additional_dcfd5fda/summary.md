---
title: "MARS-Model-agnostic-Biased-Object-Removal-without-Additional"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Jo_MARS_Model-agnostic_Biased_Object_Removal_without_Additional_Supervision_for_Weakly-Supervised_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 19:13:47"
field: "弱监督语义分割"
keywords: ["弱监督语义分割", "有偏物体去除", "无监督语义分割", "去偏", "伪标签"]
innovations: ["首次将USS特征聚类引入WSSS解决有偏物体问题", "提出基于背景距离度量的自动去偏质心选择机制", "设计EMA教师网络+确定性掩码的补全策略避免FN增加"]
benchmarks: ["PASCAL VOC 2012", "MS COCO 2014"]
---

# 论文速读：MARS-Model-agnostic-Biased-Object-Removal-without-Additional

## 一句话总结
本文针对弱监督语义分割(WSSS)中"有偏物体"(biased objects)导致的假阳性(FP)问题，提出了一种**无需额外监督数据的模型无关的自动去偏框架MARS**，通过结合无监督语义分割(USS)的特征聚类与距离度量，成功剥离伪标签中的有偏区域，并在PASCAL VOC 2012和MS COCO 2014上取得最新SOTA结果。

---

## 研究问题与动机

1. **WSSS的核心瓶颈是假阳性(FP)**：现有WSSS方法虽能缓解假阴性(FN)，但目标类别常错误激活相关背景（如"boat"类激活海洋、"train"类激活铁轨），造成严重FP。PASCAL VOC 2012中**35%的类别**存在此类有偏问题。

2. **已有去偏方法依赖额外监督**：CLIMS [51]需使用CLIP大模型+文本监督，逐类人工标注有偏物体；W-OoD [29]需人工从Open Images收集含仅偏物体的图片。这些要求在大规模复杂标签场景（如Open Images的19,794类）中几乎不可行。

3. **USS比WSSS更少引入有偏特征**：由于WSSS依赖图像级标签，天然存在类别-背景混淆；而USS从零训练可学习语义一致的特征，能更好分离目标与有偏区域。

4. **有偏物体可与背景匹配**：作者发现，将分离出的物体区域与数据集中其他图像的背景区域计算距离，**短距离对应偏物体（常出现在背景中），长距离对应目标物体**。

---

## 核心贡献（创新点）

1. **首次将USS引入WSSS解决有偏问题**：提出两种USS在WSSS中的新观察——USS特征聚类可分离偏/目标物体，且基于距离度量可自动识别有偏物体，无需任何额外标注或外部知识模型。

2. **设计基于USS距离的去偏质心选择机制**：通过K-means聚类生成前景质心后，计算各质心与所有背景质心的平均余弦距离，选择距离最大的top α%作为去偏目标质心，自动剔除常出现在背景中的偏物体。

3. **提出模型无关的去偏标签补全策略**：用教师网络EMA更新在线预测，以"确定性掩码(WCE损失)"控制不确定性像素，避免去偏后非问题类别FN增加，实现**完全自动化的端到端去偏**。

---

## 方法详解

### 整体框架（四阶段）
MARS采用**模型无关(model-agnostic)**策略，将WSSS与USS两个独立训练的流程串联，分为：(a)联合训练WSSS和USS；(b)选择去偏质心；(c)生成去偏标签；(d)训练时补全去偏标签。

---

### 3.2 选择去偏质心
- 对每个类别c，用WSSS伪标签$Y_i^b$提取USS嵌入向量$F_i$，按类别分组后做**K-means聚类**（前景$K_{fg}=2$，背景$K_{bg}$可调），得到图像级质心$\{v_{i·K+j}^c\}$。
- 定义距离度量：
$$
dist_k^c = \frac{1}{N^{bg}} \sum_{j=0}^{N^{bg}} D(v_k^c, v_j^0)
$$
其中$D(\cdot)$为余弦距离$(1-S)/2$，$N^{bg} = N \cdot K_{bg}$为所有训练图的背景质心总数。
- **排序并选取top α%距离最大的质心**作为去偏质心：
$$
\mathcal{V}^c = \frac{1}{\lceil N_c^{fg} \cdot \alpha \rceil} \sum_{j \in \{k_1,...,k_{\lceil N_c^{fg}\cdot\alpha\rceil}\}} v_j^c
$$
其中$\alpha \in [0,1]$，$dist_{k_1}^c \ge dist_{k_2}^c \ge ...$

---

### 3.3 生成去偏标签
- 计算去偏质心$\mathcal{V}^c$与USS嵌入$F_i$的相似度图，取每像素最大相似度：
$$
\hat{M}_i^{db}(y,x) = ReLU\left(\max_{c \in \mathcal{C}_{I_i}} S(F_i[:,y,x], \mathcal{V}^c)\right)
$$
- 经CRF后处理得二值去偏掩码$M_i^{db}$，最终生成去偏标签$Y_i^{db}$：
$$
Y_i^{db}(y,x) = \begin{cases} -1, & \text{if } Y_i^b(y,x)>0 \text{ and } M_i^{db}(y,x)=0 \\ Y_i^b(y,x), & \text{otherwise} \end{cases}
$$
其中**-1标记为"有偏类"**，表示该像素在去偏后被移除。

---

### 3.4 训练时补全去偏标签
- 学生网络参数$\theta$，教师网络$\hat{\theta}$通过EMA更新，教师网络经CRF+argmax得到$Y_i^{te}$。
- 用$Y_i^{te}$填补缺失位置（-1），得到补全标签$Y_i^{co}$。
- 设计**确定性掩码$W_i$**防止早期不确定预测污染训练：
$$
W_i(y,x) = \begin{cases} \max_{c \in \mathcal{C}_{I_i}} \hat{P}_i(c,y,x), & \text{if } Y_i^{db}(y,x)=-1 \\ 1, & \text{otherwise} \end{cases}
$$
- 使用**加权交叉熵(WCE)损失**：
$$
\mathcal{L}_{WCE} = -\sum_{c \in \mathcal{C}}\sum_{y,x \in W} W_i(y,x) \cdot O[Y_i^{co}](c,y,x) \log P_i^\theta(c,y,x)
$$

---

## 实验与结果

| 基准 | 评测集 | 最佳mIoU | 提升幅度 |
|------|--------|----------|----------|
| PASCAL VOC 2012 | val | **77.7%** | 较RS+EPM提升3.3个百分点 |
| PASCAL VOC 2012 | test | **77.2%** | — |
| MS COCO 2014 | val | **49.4%** | 首个在无额外监督下达到此水平的方法 |

- **模型无关性验证**：与4种WSSS方法(IRNet/SEAM/AdvCAM/RS+EPM)结合，分别提升**6.3%/6.3%/2.2%/3.3%**，均优于ADELE和W-OoD等基线。
- **类别-wise分析**：boat类提升+9%，train类提升+29%，验证去偏有效性；tv/monitor略降（因原始伪标签质量差）。
- **消融**：完整MARS（含complementing+WCE）达81.8% mIoU，优于仅去偏(80.9%)和仅WCE(81.8%)的组合。
- **超参敏感性**：$K_{bg}$在1~5范围内稳定；$\alpha \le 0.5$时效果最佳（过大难分离）。
- **推理延迟**：聚类+精炼步骤分别在VOC训练集上耗时10分钟和9分钟（CPU测试）。

---

## 相关工作脉络

1. **CLIMS [51] (CVPR'22)**：基于CLIP多模态知识定位有偏类，需逐类人工确认偏物体，依赖4亿图文对预训练；MARS完全无需外部知识模型。
2. **W-OoD [29] (CVPR'22)**：需人工从Open Images收集仅含偏物体的额外图像；MARS完全自动化，零人工标注。
3. **SEAM [49]/RS+EPM [21]**：通过特征相关/边缘预测扩充CAM前景，但未解决FP瓶颈；MARS可无缝叠加在它们之上。
4. **Leopart [58]/STEGO [15]**：USS方法学习空间结构特征；本文证明USS嵌入可用于WSSS去偏，开创USS×WSSS联合范式。
5. **EPS [31]/DRS [22]**：使用显著性监督或判别区域抑制减少FP，但需像素级标注；MARS只用图像级标签。
6. **SANCE [32]/ADELE [38]**：先进WSSS管线处理噪声伪标签；MARS专门针对有偏物体，两者正交可互补。

---

## 局限性与未来方向

1. **同super-category类别难区分**：如horse/sheep因视觉相似，USS无法分离，去偏效果受限（论文在3.3节承认此问题）。
2. **伪标签质量差的类别性能下降**：如tv/monitor因初始WSSS预测质量低，去偏后轻微降低（图9）。
3. **超参$\alpha$需调优**：过大（>0.5）难以分离质心，过小则丢失目标信息。
4. **推理延迟**：聚类与精炼步骤在CPU上耗时约20分钟，实时性受限。
5. **未验证大规模开放词汇场景**：论文仅在VOC/COCO验证，对Open Images等万级类别数据集的泛化未评估。

---

## 研究启发与可借鉴点

1. **USS×WSSS联合范式**：证明无监督语义特征可作为WSSS的"去偏先验"，开启跨监督类型融合的新方向，可迁移至实例分割、检测等任务。
2. **背景距离度量去偏思想**：利用"偏物体常与背景相似"的直觉，设计简单有效的距离准则替代复杂外部知识，适合资源受限场景。
3. **EMA教师网络+确定性掩码的补全策略**：有效平衡去偏后FN增加问题，可推广至其他伪标签噪声修正任务。
4. **模型无关设计**：MARS可插入任意WSSS管线末端，这种"插件式"改进思路便于后续工作快速验证。
5. **可结合本团队方向**：若有WSSS/半监督分割项目，可直接复用MARS的USS聚类+距离度量模块作为预处理；或探索将USS特征用于少样本分割的类别原型构建。

---

## 关键术语表

**Weakly-Supervised Semantic Segmentation (WSSS)**：仅使用图像级标签（而非像素级标注）训练的语义分割方法，大幅降低标注成本。

**Biased Object (有偏物体)**：与目标类别语义相关但非目标本身的视觉区域（如"boat"对应的"sea"），导致WSSS产生假阳性。

**Unsupervised Semantic Segmentation (USS)**：完全无标注训练，通过聚类或对比学习生成语义一致的像素嵌入，无法直接提供类别感知预测。

**Debiased Centroid (去偏质心)**：经背景距离排序后选出的、代表目标物体的USS嵌入向量均值，用于后续相似度匹配。

**Certain Mask (确定性掩码)**：取教师网络预测概率最大值构成的像素级权重，用于过滤早期不确定补全预测。

**Weighted Cross Entropy (WCE)**：将确定性掩码作为权重的交叉熵损失，避免低置信度预测污染训练。

**Model-Agnostic (模型无关)**：方法不绑定特定WSSS/USS架构，可插入任意现有方法的后处理或训练流程。

---

## 可复现要素

- **数据集**：PASCAL VOC 2012（公开）、MS COCO 2014（公开）
- **代码**：已开源 → https://github.com/shjo-april/MARS
- **关键超参**：$K_{fg}=2$（前景聚类数）、$K_{bg}=2$（背景聚类数）、$\alpha=0.40$（去偏质心选择比例）
- **底座模型**：WSSS使用IRNet/SEAM/AdvCAM/RS+EPM；USS使用Leopart/STEGO； backbone为ResNet-50/ResNet-101/VGG-16
- **训练设备**：单张RTX A6000 GPU，PyTorch实现
- **后处理**：CRF [23] + 多尺度推理（conventional settings）
- **评估指标**：mIoU（主）、per-class FP/FN（gold standard [49]）

---
