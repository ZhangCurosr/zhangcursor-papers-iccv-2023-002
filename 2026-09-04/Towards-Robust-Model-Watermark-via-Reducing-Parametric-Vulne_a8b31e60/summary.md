---
title: "Towards-Robust-Model-Watermark-via-Reducing-Parametric-Vulne"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Gan_Towards_Robust_Model_Watermark_via_Reducing_Parametric_Vulnerability_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 21:20:26"
field: "深度学习安全与隐私保护"
keywords: ["模型水印", "对抗鲁棒性", "参数空间分析", "域偏移", "黑盒验证", "后门水印", "BatchNorm"]
innovations: ["提出对抗参数扰动(minimax)消除参数空间中的水印脆弱邻居", "设计c-BN用干净样本统计量归一化水印输入以消除域偏移"]
benchmarks: ["CIFAR-10", "CIFAR-100", "ImageNet-subset"]
---

# 论文速读：Towards-Robust-Model-Watermark-via-Reducing-Parametric-Vulne

## 一句话总结
本文提出了一种基于对抗参数扰动的鲁棒模型水印方法，通过在参数空间中搜索水印移除的"坏邻居"并恢复其水印行为，结合干净样本驱动的BatchNorm统计量，显著提升了对fine-tuning、fine-pruning等多种移除攻击的防御能力。

## 研究问题与动机
- **核心问题**：基于后门的行为式模型水印容易受到简单的参数修改类移除攻击（如fine-tuning）破坏，防御者难以在黑盒场景下有效验证模型所有权。
- **动机发现**：通过分析参数空间发现，原始水印模型附近存在大量低WSR（Watermark Success Rate）、高BA（Benign Accuracy）的"水印已移除模型"，这些模型容易被攻击者快速找到。例如沿对抗方向仅需0.03相对距离即可将WSR降至接近零。
- **现有方法不足**：Vanilla watermarking、EW（Exponentially Weighted）和CW（Certified Watermark）等方法在多种移除攻击下WSR平均下降超过50%，鲁棒性严重不足。
- **防御目标**：在训练阶段主动消除参数空间中近邻的脆弱点，使水印行为对参数扰动更具抗性。

## 核心贡献（创新点）
1. **揭示了水印模型在参数空间中的脆弱性本质**：发现原始水印模型附近存在大量低WSR邻居，为理解移除攻击可行性提供了新的理论视角。
2. **提出最小-最大（minimax）形式的对抗参数扰动（APP）机制**：通过最大化寻找最坏情况的参数邻居并最小化恢复其水印行为，与现有方法仅在单一损失上优化不同，本质上是引入了对抗鲁棒性的思维。
3. **设计干净样本驱动的BatchNorm（c-BN）以缓解域偏移**：指出防御（水印样本归一化）与攻击（干净样本归一化）之间的域偏移问题，提出用干净批统计量归一化水印输入，这是首个针对水印训练中BatchNorm统计量一致性问题的解决方案。

## 方法详解
- **威胁模型**：防御者在训练阶段完全可控，发布后只能通过API查询进行黑盒验证。
- **基础训练公式**：在干净数据$\mathcal{D}_c$和水印数据$\mathcal{D}_w$上联合训练：$\mathcal{L}(\theta, \mathcal{D}_c) + \alpha \cdot \mathcal{L}(\theta, \mathcal{D}_w)$。
- **对抗参数扰动（APP）核心思想**：在参数空间中以$\epsilon\|\theta\|_2$为扰动预算，寻找使水印损失最大的参数扰动$\delta$，再用该扰动恢复水印行为：
  $$\min_\theta \left[\mathcal{L}(\theta, \mathcal{D}_c) + \alpha \cdot \max_{\delta \in \mathcal{B}} \mathcal{L}(\theta+\delta, \mathcal{D}_w)\right]$$
- **扰动近似计算**：采用单步FGSM-style方法近似最大化：$\delta = \epsilon\|\theta\|_2 \cdot \frac{\nabla_\theta \mathcal{L}(\theta, \mathcal{D}_w)}{\|\nabla_\theta \mathcal{L}(\theta, \mathcal{D}_w)\|_2}$，其中$\epsilon$为相对扰动幅度超参。
- **c-BN（Clean-sample-based BatchNorm）**：前向传播时，用水印样本计算梯度但用干净批$B_c$的统计量做BatchNorm归一化，消除防御侧与攻击侧的域偏移：
  $$\min_\theta \left[\mathcal{L}(\theta, \mathcal{D}_c) + \alpha \cdot \max_{\delta \in \mathcal{B}} \mathcal{L}(\theta+\delta, \mathcal{D}_w; \mathcal{D}_c)\right]$$
- **算法流程**：每个训练步先计算干净数据梯度，再采样水印批计算APP扰动和对应的梯度更新，两者相加后更新参数。

## 实验与结果
- **数据集**：CIFAR-10、CIFAR-100、ImageNet-subset（100类，训练5万张/测试5千张）。
- **水印类型**：Content（叠加"TEST"文本）、Noise（随机噪声）、Unrelated（跨域图像，如SVHN）。
- **基线方法**：Vanilla watermarking、EW（指数加权）、CW（认证水印，黑盒验证设置）。
- **攻击方法**：FT（fine-tuning）、FP（fine-pruning）、ANP（对抗神经元剪枝）、NAD（神经注意力蒸馏）、MCR（模式连通性修复）、NNL（神经网络清洗）。
- **核心结果（CIFAR-10，Content水印）**：
  - Vanilla: WSR 99.56% → 平均下降59.15%
  - EW: WSR 99.17% → 平均下降51.20%
  - CW: WSR 99.62% → 平均下降68.36%
  - **Ours**: WSR 99.87% → **平均下降仅10.10%**，对FT保留96.63%、对FP保留98.44%、对ANP保留99.56%
- **最强对比**：Unrelated水印下，Ours平均WSR下降仅6.20%，而次优EW下降50.90%，提升幅度超过44个百分点。
- **ImageNet-subset（Unrelated水印）**：Ours平均WSR下降7.42%，Vanilla下降51.13%。
- **消融结论**：仅APP因域偏移问题反而降低鲁棒性（AvgDrop 82.24%），仅c-BN无提升，两者结合才能达到最优效果。

## 相关工作脉络
- **Vanilla水印（Zhang et al. [54]）**：经典后门水印方法，通过在水印数据上最小化交叉熵嵌入行为，但缺乏对参数扰动的鲁棒性保障。
- **EW方法（Namba & Sakuma [37]）**：通过对参数指数加权降低水印对参数变化的敏感性，但未考虑参数空间邻域中的坏邻居问题。
- **CW方法（Bansal et al. [3]）**：基于随机平滑的认证水印，提供数学保障但依赖白盒参数访问，本文在其黑盒设置下公平对比。
- **移除攻击（FT [45]、FP [33]、ANP [50]、NAD [25]、MCR [56]、NNL [2]）**：本文系统性地对抗六种主流参数修改类攻击，而以往工作通常只测试单一攻击。
- **定位差异**：本文不从认证保障或白盒角度出发，而是直接在黑盒设定下通过训练过程的对抗鲁棒性提升来增强水印安全性，与EW的权重缩放、CW的随机噪声注入形成方法论上的互补与对比。

## 局限性与未来方向
- **威胁模型的简化**：论文承认当前使用固定扰动预算约束参数变化，而实际攻击只需保持高BA，无法显式刻画BA与参数的关系，导致防御仅为更强威胁模型下的必要条件。
- **对NNL的鲁棒性有限**：在Content/Noise水印下NNL攻击仍能大幅降低WSR，说明仍有特定攻击可突破。
- **未讨论计算开销**：未详细分析APP引入的额外训练时间/显存成本。
- **未来方向**：探索更贴合实际的威胁模型（如基于BA约束的扰动范围）、扩展到生成模型水印、以及自适应扰动预算策略。

## 研究启发与可借鉴点
1. **参数空间视角分析鲁棒性**：将水印鲁棒性问题转化为参数空间邻域搜索问题，提供了一种可解释的分析框架，类似思路可迁移至其他模型保护任务。
2. **Minimax对抗训练范式的应用**：将对抗鲁棒性训练思想引入水印嵌入，启示可将PGD-style多步攻击、随机投影等更复杂的扰动生成策略用于水印增强。
3. **域偏移诊断方法**：通过可视化BatchNorm统计量差异来定位防御失效原因，这一"分布对齐诊断→针对性修正"的流程具有普适参考价值。
4. **c-BN的设计灵活性**：用干净样本统计量归一化对抗扰动样本的思路，可推广至其他涉及对抗训练的框架中解决域不匹配问题。

## 关键术语表
- **WSR（Watermark Success Rate）**：水印成功识别率，即水印样本被正确分类为目标标签的比例，是水印强度的核心指标。
- **BA（Benign Accuracy）**：在干净测试数据上的分类准确率，衡量水印嵌入对模型正常性能的影响。
- **APP（Adversarial Parametric Perturbation）**：对抗参数扰动，通过单步梯度方向搜索参数空间中使水印损失最大的邻域点。
- **c-BN（Clean-sample-based BatchNorm）**：使用干净样本批统计量代替水印样本统计量进行BatchNorm归一化，以消除域偏移。
- **MCR（Mode Connectivity Repair）**：模式连通性修复攻击，利用损失景观中的模式连通性沿最优路径修复模型参数以移除水印。
- **NNL（Neural Network Laundering）**：神经网络清洗攻击，通过蒸馏或重构的方式提取模型功能而丢弃水印行为。
- **EW（Exponentially Weighted）**：指数加权水印方法，通过对参数幂次缩放来降低水印对参数扰动的敏感性。
- **CW（Certified Watermark）**：认证水印方法，基于随机平滑技术提供水印存在的数学保证，但通常需白盒访问。

## 可复现要素
- **数据集**：CIFAR-10、CIFAR-100公开可用；ImageNet-subset（100类，5万训练/5千测试）为自定义子集。
- **代码**：已开源，地址 https://github.com/GuanhaoGan/robust-model-watermarking
- **关键超参**：水印损失系数$\alpha=0.01$，扰动幅度$\epsilon=0.02$（CIFAR-10/100）、$\epsilon=0.01$（ImageNet），训练100个epoch，初始学习率0.1，weight decay $5\times10^{-4}$。
- **模型架构**：ResNet-18为主实验架构，MobilenetV2/VGG16/ResNet-50用于架构泛化性验证。
