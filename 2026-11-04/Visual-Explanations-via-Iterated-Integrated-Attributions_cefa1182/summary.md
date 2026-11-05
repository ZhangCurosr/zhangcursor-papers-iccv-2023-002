---
title: "Visual-Explanations-via-Iterated-Integrated-Attributions"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Barkan_Visual_Explanations_via_Iterated_Integrated_Attributions_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 08:27:15"
field: "计算机视觉可解释性"
keywords: ["可解释AI", "视觉归因", "迭代积分", "Integrated Gradients", "Vision Transformer", "Grad-CAM", "解释评估"]
innovations: ["提出通用迭代积分归因框架IIA，将输入与多层表征联合插值积分", "为ViT设计梯度Rollout作为被积函数", "给出IIA2/IIA3等可计算变体并在CNN与ViT上统一验证"]
benchmarks: ["ImageNet ILSVRC 2012", "ImageNet-Segmentation", "COCO 2017", "PASCAL VOC 2012"]
---

# 论文速读：Visual-Explanations-via-Iterated-Integrated-Attributions

## 一句话总结
本文提出**迭代积分归因（IIA）**方法，通过在输入图像与模型各层内部表示（激活图/注意力矩阵）上进行多层迭代积分，生成更精确、聚焦的视觉解释热力图；在ImageNet及多个分割基准上，IIA3优于所有主流CNN/ViT解释基线，且具备跨架构通用性。

## 研究问题与动机
- 现有梯度/路径积分类方法（如IG、Grad-CAM）多仅对输入或单一层做积分，难以充分利用全网络多层表征信息。
- 仅基于输入路径的路径积分法容易忽略高层语义聚合，导致解释图过于稀疏/定位偏差。
- ViT解释需同时处理多头注意力与层间信息流转，而多数方法缺乏统一、可扩展的公式化框架。
- 需要一种**跨CNN与ViT通用**、可利用任意中间表示及其梯度组合的解释机制，并可通过可控的迭代阶数平衡精度与计算开销。

## 核心贡献（创新点）
- 提出**通用化IIA公式**：将线性插值与迭代积分推广到“输入+任意中间层表征+梯度组合”，IG仅为其一特例。本质区别在于IIA允许在所有已选中层上同步迭代积分，而IG只在输入上做一重路径积分。
- 为ViT设计**梯度Rollout（GR）**作为积分被积函数，把注意力矩阵替换为“注意力矩阵与对应梯度逐元素乘积”的组合。与T-Attr/GAE这类针对Transformer结构定制的方法相比，GR以可微路径积分为核心，且纳入梯度信号引导归因。
- 给出**IIA2与IIA3实现**：仅对输入与最后/最后两层做插值积分，在保证效果的将组合空间压缩到可计算规模。与全层暴力积分不同，二者通过选择最具代表性的层来取得精度—效率折中。
- 在**CNN与ViT**上系统评估解释质量与分割一致性，证明IIA在多数指标上稳定优于GC系列、LIFT、IG系列及T-Attr/GAE。
- 提供复杂度分析与批处理实现方案，表明在合理GPU显存下IIA可与IG等基线可比甚至更优。

## 方法详解
- **通用公式**：给定网络层函数h^l，提取感兴趣中间表征u^l = u^l(h^{l-1})，并以最小值作为参考r^l后进行非线性插值v^l = r^l + a_l^{b_l}(u^l - r^l)，其中b_l∈{0,1}决定该层是否参与积分；最终解释图由多层积分得到m_b^l含因子(u^l - r^l)与被积函数q^l（可含激活/注意力与其梯度）。  
- **与IG的关系**：当仅对输入层插值（b=[1,0,…,0]）且被积函数取输入梯度时，IIA退化为Integrated Gradients。  
- **CNN实现**：u^l取激活图、v^l为恒等，r^l按通道取最小值广播；被积函数取v^l与∂f_y/∂v^l的Hadamard积，再沿通道均值后Resize到输入空间。  
- **ViT实现**：u^l取多头注意力矩阵（softmax后非负），r^l设为零张量；v^l继续完成Q/K/V乘法流程；被积函数采用梯度Rollout（各头CLS相关注意力乘其梯度）后跨层累积，输出可直接Resize。  
- **常用变体**：IIA2=对输入+最后一层积分；IIA3=对输入+倒数第二层+最后一层积分；实验中n=10步数值近似。  
- **复杂度**：理论O(n^β Q)，但利用GPU批处理可将实际开销控制到接近单次前后向；在B≥n^M时可视为O(1)倍代价，且IIA2反向仅到所选层，理论上可能快于IG的全网反传。

## 实验与结果
- **模型**：ViT-B/ViT-S、ResNet101、DenseNet201、ConvNeXt-Base。
- **数据集**：ImageNet ILSVRC 2012验证集；解释评估；分割评估包括ImageNet-Seg、COCO、VOC 2012。
- **基线**：GC、GC++、FullGrad、Ablation-CAM、Layer-CAM、LIFT-CAM、IG、GIG、BIG、X-Grad-CAM；ViT侧T-Attr、GAE。
- **指标**：解释侧POS/NEG/INS/DEL/ADP/PIC/SIC/AIC；分割侧PA/mAP/mIoU/mF1。
- **主要结果**：IIA3在绝大多数CNN与ViT组合上取得最优或接近最优；CNN上通常IIA3>IIA2，ViT上IIA3同样领先；消融显示IMG与ACT互补，ACT优于纯输入插值；IIA2(L-1)在RN的POS/DEL上更强（因特征图分辨率更高14×14 vs 7×7），但其他指标不及IIA2/IIA3，说明同一空间分辨率下末层聚合更优。
- **定性**：IIA能更准确区分目标类并定位相关对象，部分对比方法存在“跨类泛化/类无关”或生成平坦图的倾向。

## 相关工作脉络
- Grad-CAM系列（GC/GC++/Layer-CAM/XGC等）：利用激活与梯度加权，未做多层迭代积分，定位精度受限于单层加权策略。
- 路径积分（IG/GIG/BIG）：仅在输入空间做线性/自适应路径积分，忽略中间层表征的动态信息；IIA将其扩展为“输入+层表示”联合迭代积分。
- LIFT-CAM/Ablation-CAM：无梯度或基于扰动/得分的激活操作，类条件引导能力有限；IIA显式引入梯度-激活联合被积项。
- ViT注意力归因（Rollout/AR）：仅聚合注意力分数，未引入梯度信号；IIA的GR以注意力×梯度为核心，增强类特异性。
- T-Attr/GAE：基于Deep Taylor/传播的ViT解释SOTA；IIA以统一迭代积分范式覆盖CNN与ViT，并在多项指标上进一步超越。

## 局限性与未来方向
- 组合空间指数级（2^L），本文仅试验双/三重积分，未系统搜索最优层选择策略。
- 对ViT选择末层为主，但某些场景下中间层（如更高空间分辨率的类CNN路径）可能更优；需更通用的层选择准则。
- 数值积分步数固定n=10，自适应步数或误差控制尚未深入。
- 大规模部署时的多机并行与动态批大小策略依赖硬件，需更完善工程支撑。
- 未来可扩展至注意力以外的模块（如MLP/FFN路径）、自动选择积分阶数与层集合，以及跨模态/多任务解释一致性。

## 研究启发与可借鉴点
- **通用迭代积分范式**可作为插件式模块接入现有CNN/ViT解释管线，便于统一比较与升级。
- **被积函数设计**：用“表征×梯度”的Hadamard组合可显著增强类条件聚焦；该思路可迁移到自注意力以外的中间特征归因。
- **消融分层选择**：揭示“空间分辨率差异”与“聚合能力差异”的不同影响，值得在后续工作中作为设计依据（如高层选聚合、低层选分辨率）。
- **复杂度批处理策略**：将n^m插值批量化并拼接前后向，为高维积分方法提供可落地的实现参考。
- **任务联动评估**：同时评估解释指标与分割一致性，有助于发现“高解释分≠高分割分”的偏差现象，提升评测可靠性。

## 关键术语表
- **IIA（Iterated Integrated Attributions）**：在输入与多层中间表征上同时进行插值与迭代积分的通用视觉解释方法。
- **Integrated Gradients（IG）**：沿输入-参考线性路径对输入梯度积分的一重归因方法。
- **Gradient Rollout（GR）**：将Rollout中的注意力矩阵替换为“注意力×梯度”的组合，用于ViT解释。
- **POS/NEG**：正向/负向扰动曲线下面积，衡量热力图对预测的正负贡献敏感度。
- **INS/DEL**：插入/删除重要性像素后置信度变化曲线下的面积，反映定位集中度。
- **ADP/PIC**：平均下降百分比与置信度提升百分比，衡量关键区域的判别力贡献。
- **SIC/AIC**：Softmax/Accuracy信息曲线面积，衡量排序与信息保留稳定性。
- **mIoU/mF1/mAP**：分割任务中常用的交并比、F1与平均精度均值指标。

## 可复现要素
- 数据集：ImageNet ILSVRC 2012、ImageNet-Seg、COCO 2017、VOC 2012；分割任务使用公开标注集。
- 代码/权重：论文给出开源链接（https://github.com/iia_iccv23/iia，原文含小噪声写法），预训练模型链接见仓库。
- 关键超参：插值步数n=10；常用变体IIA2/IIA3；参考值r^l取各通道最小值（ViT注意力场景取零张量）；分段指示向量b控制参与层。
- 其余细节（预处理、模型下载、基线超参按原作者推荐配置）见README与附录。
