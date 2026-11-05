---
title: "Tracing-the-Origin-of-Adversarial-Attack-for-Forensic-Invest"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Fang_Tracing_the_Origin_of_Adversarial_Attack_for_Forensic_Investigation_and_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 16:40:31"
field: "对抗机器学习安全"
keywords: ["adversarial attack", "black-box attack", "model traceability", "forensic investigation", "buyer-seller setting", "VAE-based tracer"]
innovations: ["提出并行 Classifier-Tracer 结构以实现买家-卖家场景下的对抗样本溯源", "基于 VAE 与夹角惩罚训练使 tracer 对扰动高度敏感", "设计 DOL 指标实现仅凭公开 logits 的黑盒溯源"]
benchmarks: ["CIFAR10", "GTSRB", "Mini-ImageNet"]
---

# 论文速读：Tracing-the-Origin-of-Adversarial-Attack-for-Forensic-Invest

## 一句话总结
本文针对买家-卖家场景下的对抗攻击，提出一种两阶段溯源框架（模型分离+来源追踪），通过在每个分发模型中嵌入唯一的 tracer 组件，使攻击者在生成对抗样本时无意中携带该模型特有的 logits 特征，进而能够准确识别对抗样本的来源模型。

## 研究问题与动机
- **核心问题**：在黑盒买家-卖家设定下（卖家向多个买家分发功能相同但实现略有差异的模型），恶意买家利用自己的模型副本生成对抗样本攻击其他买家，如何从这些对抗样本中逆向追溯其生成的源模型？
- **现有防御的局限**：当前对抗防御多侧重于直接提高模型鲁棒性，并未考虑“攻击发生后能否定位攻击者”这一司法调查与威慑需求；即便防御被突破，也缺乏可落地的溯源手段。
- **溯源难度**：黑盒攻击通常通过边界估计或梯度近似生成对抗样本，若多个模型分类边界过于相似，则对抗样本难以携带区分性特征，导致无法溯源。

## 核心贡献（创新点）
1. **提出面向溯源的买家-卖家对抗防御新视角**：将对抗防御的目标从“阻止攻击生效”扩展到“攻击生效后仍可追溯来源”，形成事前增强差异、事后逆向定位的完整闭环。
2. **设计并行网络结构（Classifier + Tracer）**：固定主分类器 $\mathcal{C}$，为每个副本引入独立的 tracer $\mathcal{T}_i$，以加权方式混合输出；该结构可在保持高分类精度的同时，为每个副本制造独特的决策边界。
3. **给出 VAE 驱动的 tracer 训练方法**：通过重构损失与夹角惩罚损失（$\mathcal{L}_{Trap}$）联合训练，使 tracer 对微小扰动高度敏感，确保攻击过程中的扰动主要来源于 tracer，从而将 tracer 特性“注入”对抗样本。
4. **提出基于 DOL（Difference of Logits）的溯源机制**：仅凭对抗样本与各候选模型的分类 logits，比较攻击标签与真实标签对应的 tracer 输出差异，最大差异对应的 tracer 即为最可能来源。

## 方法详解
- **并行模型结构**：对输入 $x$，每个副本 $i$ 的最终输出为
$$\mathcal{M}_i(x) = \mathcal{C}(x) + \alpha \times \mathcal{T}_i(x)$$
其中 $\mathcal{C}$ 共享、$\mathcal{T}_i$ 独立，$\alpha \in [0,1]$ 控制 tracer 权重。

- **$\mathcal{T}_0$ 生成（VAE 训练）**：
  - 将 VAE 编码器作为 $\mathcal{T}_0$，解码器用于图像重建。
  - 输入拼接干净样本 $x$ 与加噪样本 $x_\xi$，得到输出 $\mathcal{T}_0(x)$ 与 $\mathcal{T}_0(x_\xi)$。
  - 总损失：$\mathcal{L} = \lambda \mathcal{L}_{VAE} + \mathcal{L}_{Trap}$，其中
    - $\mathcal{L}_{VAE} = \|\mathcal{V}(X)-X\|_2 + KL(\mathcal{N}(\mu,\sigma^2)\|\mathcal{N}(0,1))$
    - $\mathcal{L}_{Trap} = \frac{\mathcal{T}_0(x) \otimes \mathcal{T}_0(x_\xi)}{\|\mathcal{T}_0(x)\|_2 \|\mathcal{T}_0(x_\xi)\|_2} - \cos(\theta)$
  - 通过调节 $\theta$，使 $\mathcal{T}_0$ 输出对扰动高度敏感且与分类任务类别正交。

- **$\mathcal{T}_i$ 分离（置换法）**：
$$\mathcal{T}_i(x) = \pi_i(\mathcal{T}_0(x))$$
$\pi_i$ 为类别维度的排列操作，保证任意两个 tracer 输出重合维度不超过预设常数 $u$（实验取 $u=1$），从而在 $K$ 类任务中最多生成 $K(K-1)$ 个差异化 tracer。

- **溯源决策（DOL）**：
  1. 将对抗样本 $x_{adv}$ 输入各副本，获得 $\mathcal{M}_i(x_{adv})$ 并排序得到攻击标签 $att$ 与干净标签 $cln$。
  2. 计算各 tracer 在该样本上的 logits 差值：$DOL_i = \mathcal{T}_i(x_{adv})[att] - \mathcal{T}_i(x_{adv})[cln]$。
  3. 源模型索引为 $s = \arg\max_i DOL_i$。

## 实验与结果
- **数据集与架构**：CIFAR10、GTSRB、Mini-ImageNet；ResNet18/50、VGG16/19。
- **攻击基线**：Black-box 决策边界攻击 Boundary、HSJA、QEBA，以及无代理几何攻击 SurFree。
- **主要结果**：
  - 最佳设置（$\alpha=0.15$，ResNet18-CIFAR10，QEBA）溯源准确率达 **98.0%**；ResNet18-CIFAR10、Boundary 攻击下达到 **98.0%**。
  - $\alpha$ 增大可提升溯源率（tracer 参与度更高），但 $\alpha>0.15$ 时分类精度开始明显下降（如 GTSRB 上下降超过 1%）。
  - 分布式副本数从 2 增至 10，ResNet-CIFAR10 场景仍可维持 **>96%** 准确率。
  - 迁移性样本（transferable）仍可被高比例溯源（QEBA + α=0.15 达 94.15%），说明方法在防御失效后仍具备追责能力。
  - 对抗合谋攻击（同时利用两个副本）会使单源准确率降至 50%，但引入梯度正交化的自适应防御后可恢复至 97.5%。

## 相关工作脉络
1. **神经网络水印（Model Watermarking）**：传统水印用于检测模型是否被盗用/篡改，关注“修改后的副本归属”；本文聚焦“对抗样本归属”，属于更强形式的水印，要求对抗过程本身即携带可追溯特征。
2. **黑盒对抗攻击（Boundary/Zeroth-order）**：Boundary、HSJA、QEBA、SurFree 等代表当前决策边界类攻击主流方法，本文在其基础上验证溯源普适性，尤其强调不同攻击机制下 tracer 的“陷阱”作用。
3. **买家-卖家设定（Buyer-seller setting）**：早期工作（如 Zhang et al., arXiv 2021）提出向不同用户分发不同模型副本以降低对抗迁移性；本文在此基础上进一步提供“事后溯源”能力，形成闭环防御。
4. **对抗鲁棒性与决策边界扰动**：本文本质上是主动放大不同副本之间的决策边界差异，使黑盒攻击的梯度估计被 tracer 主导，与常规“平滑边界”思路相反。

## 局限性与未来方向
- **合谋攻击脆弱性**：当恶意买家持有多个副本并联合攻击时，tracer 特征可能被相互抵消，导致溯源失败（需依赖额外的梯度正交化自适应防御）。
- **可分发副本数受类别数限制**：基于置换方法的 tracer 上限为 $K(K-1)$（$u=1$ 时），在类别数较少的任务中扩展性受限。
- **超参数敏感**：$\theta$ 与 $\alpha$ 需要权衡分类精度与溯源效果，最优值依赖具体任务与攻击类型，尚未给出自适应选取方案。
- **仅验证图像分类任务**：框架尚未在目标检测、分割或多模态等更复杂任务上进行评估。

## 研究启发与可借鉴点
1. **并行“主模型+探针”结构**：可在通用模型分发场景中复用，通过轻量级探针引入差异化特征，适用于版权保护、模型泄露溯源等任务。
2. **VAE + 夹角惩罚训练策略**：该类训练方式可用于构造对扰动敏感的辅助模块，未来可迁移至鲁棒表征学习或可解释性增强。
3. **DOL（Logits 差异）作为溯源指标**：无需访问训练数据或内部梯度，仅凭公开查询 logits 即可完成归属判定，适合黑盒部署环境。
4. **对抗成功 ≠ 攻击隐藏成功**：本文证明即使防御被绕过，仍可保留追责能力，为构建“可审计”的 AI 系统提供了新思路。

## 关键术语表
- **Tracer**：与主分类器并行的辅助网络，负责为每个模型副本注入独特扰动响应特征。
- **DOL (Difference of Logits)**：对抗样本在 tracer 上攻击标签与干净标签的 logits 差值，用作溯源判据。
- **$\mathcal{L}_{Trap}$**：基于 Hadamard 积与余弦相似度的惩罚项，强制 tracer 对噪声敏感。
- **Permutation-based Separation**：通过对 tracer 输出维度施加不同排列操作，快速生成多个边界不同的 tracer 副本。
- **Black-box Decision-based Attack**：仅通过硬标签查询推断决策边界并生成对抗样本的攻击范式。
- **Collusion Attack**：攻击者利用多个持有副本的权限联合生成对抗样本，以抵消单 tracer 特征的攻击方式。
- **Buyer-seller Setting**：模型卖方将多个定制副本分发给不同买方的部署场景，本文溯源任务的基础设定。

## 可复现要素
- **数据集**：CIFAR10、GTSRB、Mini-ImageNet（均为公开数据集）。
- **代码/权重**：论文未提及开源声明。
- **关键超参**：$\alpha \in \{0.05, 0.1, 0.15\}$，$\theta=75^\circ$，$u=1$，VAE 权重 $\lambda=0.001$，训练轮次 200（主分类器）/ 400（tracer）。
- **实现环境**：PyTorch，NVIDIA RTX 2080ti，Adam 优化器（lr=1e-4）。
