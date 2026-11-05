---
title: "Towards-Robust-Model-Watermark-via-Reducing-Parametric-Vulne"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Gan_Towards_Robust_Model_Watermark_via_Reducing_Parametric_Vulnerability_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 16:40:11"
field: "模型安全与版权保护"
keywords: ["model watermarking", "robustness", "adversarial perturbation", "backdoor watermarks", "parameter space", "batch normalization"]
innovations: ["提出APP（对抗参数扰动）框架，通过minimax优化在参数空间邻域内寻找最坏水印移除模型并恢复水印行为", "设计c-BN解决水印样本与干净样本间的BatchNorm统计量domain shift问题，显著提升水印鲁棒性"]
benchmarks: ["CIFAR-10", "CIFAR-100", "ImageNet-subset"]
---

# 论文速读：Towards-Robust-Model-Watermark-via-Reducing-Parametric-Vulne

## 一句话总结
论文提出一种基于对抗参数扰动的模型水印加固方法，通过在训练过程中寻找参数空间邻域内"水印已移除"的最坏情况模型并恢复其水印行为，显著提升了后门水印对抗Fine-tuning、Fine-pruning、Adversarial Neuron Pruning等多种移除攻击的鲁棒性。

## 研究问题与动机
- **核心问题**：基于后门行为的模型水印在实际中极易被简单的参数修改攻击（如Fine-tuning）移除，导致版权保护失效。
- **现有方法不足**：vanilla watermarking、EW（指数加权）、CW（认证水印）等方法对多种移除攻击的鲁棒性有限，平均WSR下降幅度高达50%-90%。
- **关键发现**：作者系统调查了参数空间中水印模型邻域的表现，发现存在大量低WSR高BA的"水印已移除"模型（仅需相对距离0.03即可找到），攻击者可轻易利用。
- **Domain Shift问题**：原始APP方法因防御端（使用水印样本）与攻击端（使用干净样本）的BatchNorm统计量不一致导致失效。

## 核心贡献（创新点）
- **参数空间系统性分析**：首次系统揭示水印模型参数空间邻域中存在大量易被移除攻击利用的"水印已移除"模型，量化了水印参数的脆弱性。
- **Minimax对抗参数扰动框架（APP）**：提出最大化-最小化 formulation，在参数扰动预算内寻找WSR最劣的邻居模型并通过最小化恢复其水印行为，本质上是增强水印在参数空间的"盆地"深度。
- **Clean-sample-based BatchNorm（c-BN）**：针对水印样本与干净样本的分布偏移问题，提出使用干净样本估计BatchNorm统计量来规范化水印样本，消除防御-攻击间的domain shift。
- **全面的鲁棒性验证**：在CIFAR-10/100和ImageNet子集上，系统对抗6种SOTA移除攻击（FT、FP、ANP、NAD、MCR、NNL）进行评估，证明方法的有效性和通用性。

## 方法详解
- **威胁模型**：防御者在发布前可完全访问训练过程并嵌入任意类型水印；验证阶段仅能通过API查询可疑模型的预测（black-box场景）。
- **Adversarial Parametric Perturbation (APP)**：
  - **最大化阶段**：在约束条件 $\|\delta\|_2 \leq \epsilon \|\theta\|_2$ 下，寻找使水印损失最大的参数扰动 $\delta = \arg\max_{\delta \in \mathcal{B}} \mathcal{L}(\theta + \delta, \mathcal{D}_w)$，即寻找WSR最低的邻域模型。
  - **最小化阶段**：对扰动后的模型恢复水印行为，优化目标为 $\min_\theta [\mathcal{L}(\theta, \mathcal{D}_c) + \alpha \cdot \max_{\delta \in \mathcal{B}} \mathcal{L}(\theta + \delta, \mathcal{D}_w)]$。
  - **扰动生成**：采用单步近似，$\delta = \epsilon \|\theta\|_2 \cdot \frac{\nabla_\theta \mathcal{L}(\theta, \mathcal{D}_w)}{\|\nabla_\theta \mathcal{L}(\theta, \mathcal{D}_w)\|_2}$。
- **Clean-sample-based BatchNorm (c-BN)**：在前向传播中，使用额外干净样本批次 $B_c$ 估计每层的BatchNorm运行均值和方差，并用该统计量规范化水印样本 $B_w$，从而对齐防御与攻击的统计分布。
- **训练流程**：每个训练步同时处理干净样本和水印样本，依次计算干净样本梯度、APP扰动、扰动后水印样本梯度（使用c-BN统计量），最后联合更新参数。

## 实验与结果
- **数据集**：CIFAR-10、CIFAR-100、ImageNet子集（100类，50K训练/5K测试）。
- **水印类型**：Content（叠加"TEST"文本）、Noise（随机噪声）、Unrelated（来自SVHN的不相关图像）。
- **评估基线**：Vanilla watermarking、EW（指数加权）、CW（认证水印）。
- **移除攻击**：FT（Fine-tuning）、FP（Fine-pruning）、ANP（Adversarial Neuron Pruning）、NAD（Neural Attention Distillation）、MCR（Mode Connectivity Repair）、NNL（Neural Network Laundering）。
- **核心指标**：WSR（水印成功率）和BA（干净数据准确率）。
- **主要结果**：
  - **CIFAR-10（Content水印）**：原WSR=99.87%，对抗6种攻击平均下降仅10.10%，远优于Vanilla（↓59.15%）、EW（↓51.20%）、CW（↓68.36%）。
  - **CIFAR-10（Unrelated水印）**：平均WSR下降仅6.20%，其余方法至少下降50.90%。
  - **ImageNet-subset（Unrelated水印）**：平均WSR下降7.42%，其余方法下降51%-94%。
  - **最强攻击NNL**：在Content/Noise水印场景下，本文方法仍保持较高WSR（Content: 68.58% vs Vanilla: 21.02%）。
  - **鲁棒性提升幅度**：在18个测试场景中有17个达到最高鲁棒性；相对最优基线平均WSR提升超过40个百分点。

## 相关工作脉络
- **Vanilla Watermarking [54]**：通过后门样本嵌入水印行为，但未经过参数空间鲁棒性优化，易被简单FT移除。本文在其基础上引入对抗扰动。
- **Exponential Weighting (EW) [37]**：对权重进行指数加权以减弱水印敏感度，但仍无法抵抗强参数扰动攻击。本文从参数空间扰动角度提供更强保障。
- **Certified Watermark (CW) [3]**：基于随机平滑的方法，提供理论保证但仅适用于white-box场景；本文聚焦black-box设定并与CW在同一公平条件下对比。
- **Fine-Pruning (FP) [33] & ANP [50]**：通过剪枝移除触发相关神经元；本文方法通过扩大水印在参数空间的稳定区域增强对此类攻击的抵抗力。
- **Neural Attention Distillation (NAD) [25]**：利用注意力机制蒸馏移除水印；本文实验证明APP+c-BN能显著抵御此攻击。
- **Mode Connectivity Repair (MCR) [56]**：在损失景观中寻找连通模式修复模型；本文在参数空间的minimax框架与MCR的连通性思路形成互补防御视角。

## 局限性与未来方向
- **威胁模型简化**：论文承认限制参数扰动大小是一种简化，实际攻击者只需维持高BA即可，无需严格约束参数距离；该方法应视为"真正鲁棒水印"的前置条件而非最终方案。
- **单一扰动步数**：采用单步梯度近似最坏扰动，未探索多步或更精确的对抗扰动求解方法。
- **BatchNorm依赖**：c-BN仅适用于含BatchNorm的架构，对无BN的网络（如Transformer）需额外设计。
- **水印类型限制**：主要验证了黑盒后门水印，对于白盒水印或基于激活的水印的扩展未讨论。

## 研究启发与可借鉴点
- **参数空间视角的鲁棒性分析**：将水印鲁棒性问题转化为参数空间邻域的性质分析，为后续研究提供了新的分析框架——可扩展至其他模型保护场景。
- **c-BN的设计思路可迁移**：解决domain shift的"用目标分布统计量规范化源分布输入"策略，可应用于对抗训练、域适应等领域的BatchNorm设计。
- **Minimax对抗扰动框架**：APP的"寻找最坏情况并恢复"思想与对抗训练类似，可直接迁移至其他模型安全任务（如后门防御、模型版权验证）。
- **超参数$\epsilon$的稳健性**：实验表明在$\epsilon \leq 0.04$范围内方法始终有效且对攻击类型不敏感，为实际应用提供了宽松的参数选择空间。

## 关键术语表
**Watermark Success Rate (WSR)**：水印样本被正确分类为目标标签的比例，衡量水印行为的强度。
**Benign Accuracy (BA)**：模型在干净测试数据上的分类准确率，衡量模型的正常性能。
**Adversarial Parametric Perturbation (APP)**：在参数空间中寻找使水印损失最大的扰动方向，用于暴露模型的水印脆弱性。
**Clean-sample-based BatchNorm (c-BN)**：使用干净样本估计BatchNorm统计量来规范化水印样本，消除防御与攻击间的分布偏移。
**Fine-tuning (FT) Attack**：攻击者从水印模型出发，用干净数据继续微调以抹除水印行为的参数修改攻击。
**Mode Connectivity Repair (MCR)**：利用损失景观中的模式连通性，通过插值路径修复被破坏的水印模型。
**Neural Network Laundering (NNL)**：通过蒸馏和重训练等流程彻底"清洗"模型的水印行为。

## 可复现要素
- **数据集**：CIFAR-10、CIFAR-100公开；ImageNet子集为作者自建（100类，50K训练/5K测试）。
- **代码**：论文声明代码开源，地址为 https://github.com/GuanhaoGan/robust-model-watermarking。
- **关键超参**：水印损失系数 $\alpha=0.01$，扰动幅度 $\epsilon=0.02$（CIFAR）/ $0.01$（ImageNet），学习率初始0.1在50和75轮衰减，weight decay=$5\times10^{-4}$，训练100 epoch。
- **模型架构**：ResNet-18（主实验），额外测试MobileNetV2、VGG16、ResNet-50。
- **水印比例**：训练数据中1%替换为水印样本。
