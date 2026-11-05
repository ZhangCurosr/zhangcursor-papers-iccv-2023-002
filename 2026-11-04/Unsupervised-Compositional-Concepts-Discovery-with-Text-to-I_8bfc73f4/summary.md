---
title: "Unsupervised-Compositional-Concepts-Discovery-with-Text-to-I"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Liu_Unsupervised_Compositional_Concepts_Discovery_with_Text-to-Image_Generative_Models_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 08:26:12"
field: "生成模型与视觉理解"
keywords: ["组合概念发现", "扩散模型", "无监督学习", "可组合生成", "文本到图像", "能量模型"]
innovations: ["利用预训练扩散模型的EBM解释实现无监督组合概念分解", "用词嵌入参数化概念并结合权重混合实现类别与属性的统一表示"]
benchmarks: ["ImageNet", "ADE20K"]
---

# 论文速读：Unsupervised-Compositional-Concepts-Discovery-with-Text-to-I

## 一句话总结
本文提出一种无监督的组合概念发现方法，利用预训练文本到图像扩散模型的语义先验，从少量无标签自然图像中自动学习多个可组合的生成概念（如物体类别、光照、艺术风格），并将发现的概念用于合成新图像或作为下游分类的表示。

## 研究问题与动机
- **核心问题**：给定一组无标签图像，如何自动发现能分解并重新组合图像的生成概念？
- **现有方法不足**：
  - 传统概念发现方法通常需要监督标注来指定每个概念（如GANlatent空间操控）。
  - 仅关注物体分割掩码的方法无法捕获全局场景因素（如光照、相机位置）。
  - COMET虽能分解全局与局部概念，但仅适用于简单数据集，难以生成复杂高分辨率图像。
  - Textual Inversion等文本条件方法依赖图像-词汇对，且每次只能学习一个概念，无法同时无监督地发现多个概念。

## 核心贡献（创新点）
1. **提出可扩展的无监督组合概念发现框架**：利用预训练扩散模型的EBM解释，将图像分解为多个独立条件概率分布，无需任何标签。
   - 与COMET的本质区别：COMET使用能量函数直接建模，本文用扩散模型参数化每个概念，支持高分辨率复杂图像生成。
2. **引入基于词嵌入的概念参数化机制**：将每个概念表示为文本空间中随机初始化的词嵌入，通过优化嵌入向量学习概念，利用文本语义空间消除歧义并降低参数维度。
   - 与Textual Inversion的本质区别：Textual Inversion需已知图像-词对且学习单一概念，本文同时无监督地学习多个概念，且概念可通过权重混合表示复合属性。
3. **实现跨领域的应用验证**：在ImageNet物体发现、ADE20K厨房场景分解、艺术风格（梵高/莫奈/毕加索）发现等多样化任务上均达到SOTA。
   - 相对已有工作的定位：首次在通用文本到图像模型上实现无需类别标签的细粒度场景与风格解耦。
4. **展示概念的可组合性与下游应用价值**：发现的概念可通过组合算子（AND）生成未见过的概念组合图像，且概念权重可作为有效表示用于图像分类。

## 方法详解
- **EBM视角的扩散模型**：扩散模型的去噪网络 $\epsilon_\theta$ 可解释为能量函数 $E_\theta$ 的梯度 $\nabla_x E_\theta(x)$，对应未归一化概率分布 $p(x) \propto e^{-E_\theta(x)}$。
- **组合扩散（Composable Diffusion）**：两个条件扩散模型 $\epsilon_{c_1}$ 和 $\epsilon_{c_2}$ 可通过混合去噪函数组合：
  $\epsilon_{comb}(x^t, t) = \epsilon_{c_1}(x^t, t) + \epsilon_{c_2}(x^t, t) - \epsilon_\phi(x^t, t)$，其中 $\epsilon_\phi$ 为无条件扩散模型。
  采样等价于从 $p(x|c_1, c_2) \propto \frac{p(x|c_1)p(x|c_2)}{p(x)}$ 生成图像。
- **无监督分解目标**：将每张图像 $x_i$ 的概率分解为K个独立概念的条件分布乘积：
  $p_{decom}(x_i) \propto p(x_i) \prod_{k=1}^{K} \frac{p(x_i|c_i^k)}{p(x_i)}$
  对应的去噪函数为 $\epsilon_{unsup}(x_i^t, t) = \epsilon_\phi(x_i^t, t) + \sum_{k=1}^{K} (\epsilon_k(x_i^t, t) - \epsilon_\phi(x_i^t, t))$，通过MSE损失训练。
- **共享概念库与权重参数化**：为保持跨图像概念一致性，将第k个概念参数化为共享词嵌入 $c^k$，每张图像用权重向量 $w_i \in \mathbb{R}^K$（$\sum_k w_i^k = 1$）线性组合：
  $\epsilon_{unsup}(x_i^t, t) = \epsilon_\phi + \sum_{k=1}^{K} w_i^k (\epsilon_\theta(x_i^t, t|c^k) - \epsilon_\phi)$
  - 当权重为one-hot时，图像被聚类到单一概念（类别识别）；
  - 当权重混合时，图像被分解为多个属性（如光照+物体）。
- **联合优化**：同时更新权重 $w_i$ 和概念嵌入 $c^k$：
  $w_i = w_i - \lambda \nabla_{w_i} \mathcal{L}_{MSE}, \quad c^k = c^k - \lambda \nabla_{c^k} \mathcal{L}_{MSE}$

## 实验与结果
- **数据集**：ImageNet（4组×5类×5张训练图）、ADE20K（25张厨房图像）、艺术画作（梵高5张、莫奈7张、毕加索5张）。
- **评估基线**：Textual Inversion、Textual Inversion + K-means (KM)、Textual Inversion + CK-Means (CKM)、COMET。
- **ImageNet物体发现**（表1）：
  - ResNet-50分类准确率：Ours平均55.70%，优于Textual Inversion CKM的51.41%；S4集合达82.81%。
  - CLIP准确率：Ours平均38.60%，优于基线。
  - KL散度：Ours平均0.1538（ResNet）/ 0.1337（CLIP），概念多样性最佳。
- **室内场景分解**（图4-5）：成功将厨房场景分解为炉灶、橱柜、光照等语义成分，DDAM可视化验证注意力聚焦正确区域。
- **艺术风格发现**（图6）：从少量画作中自动发现"drinkers"、"guitarist"等风格概念。
- **组合生成**（图7-10）：成功合成未见过的物体组合（如convertible car + elephants）、场景组合（光照+岛屿）、风格组合（梵高×莫奈），并与外部文本概念组合。
- **表示学习**（表2）：Ours平均分类准确率62.75%，优于K-means (32.25%) 和Textual Inversion (24.25%)，仅次于监督方法CLIP K-Means (73.50%)。

## 相关工作脉络
- **COMET [11]**：首个使用EBM组合无监督分解场景的方法，但仅适用于简单数据集；本文用扩散模型替代能量函数，支持高分辨率复杂图像。
- **Textual Inversion [15]**：利用文本到图像扩散模型将相似图像映射为单一词汇；本文扩展至同时无监督发现多个概念，无需图像-词对。
- **Composable Diffusion [32]**：提出基于EBM的组合扩散算子；本文在其框架上引入词嵌入参数化，实现从图像数据中反向发现概念。
- **MoNet [4]**、**Slot Attention [34]**：专注于物体级分割掩码发现；本文同时捕获全局（光照、风格）与局部（物体）概念。
- **GAN空间操控方法 [15, 24, 18, 44, 55]**：依赖监督标注或特定GAN架构；本文利用预训练扩散模型，无需标注且可扩展至任意自然图像。

## 局限性与未来方向
- **概念数量K需预设**：当前方法需手动设定概念数量K，缺乏自动确定最优K的机制。
- **训练数据量极小**：实验仅在每类5张（ImageNet）或25张（ADE20K）图像上训练，未验证大规模场景下的稳定性。
- **概念混淆风险**：当图像内容高度重叠时，词嵌入可能学习到相近但冗余的概念。
- **未来方向**：可扩展至视频/3D场景分解；结合分层概念结构（全局-局部）；探索自动化概念数选择策略。

## 研究启发与可借鉴点
- **扩散模型的EBM解释作为通用工具**：将去噪网络视为能量梯度，使得不同扩散模型可像EBM一样组合与分解，为后续研究提供统一数学框架。
- **词嵌入参数化的轻量化思路**：用低维词向量替代全参数微调，大幅降低小样本学习的参数量与歧义性，可迁移至其他条件生成任务。
- **权重混合表示复合属性**：one-hot与混合权重的统一框架，可同时处理类别识别与多属性分解，为解耦表示学习提供新视角。
- **下游表示的直接利用**：将概念权重向量作为图像表示用于分类，无需额外提取特征，简化了表示学习管线。
- **跨域验证策略**：在物体、场景、艺术三类差异显著的数据集上均取得效果，增强方法通用性说服力。

## 关键术语表
- **Composable Diffusion**：基于能量模型的可组合扩散技术，通过加减条件与无条件去噪分数实现多概念条件生成。
- **Energy-Based Model (EBM)**：以能量函数定义未归一化概率分布的生成模型，扩散模型的去噪网络可解释为EBM的 score function。
- **Textual Inversion**：将图像内容映射为文本空间中单个可学习词嵌入的技术，常用于个性化图像生成。
- **DDAM (Diffusion Attentive Attribution Maps)**：利用扩散模型cross-attention层的词-像素得分生成热力图，用于可视化概念与图像区域的对应关系。
- **Classifier-Free Guidance**：在扩散模型中同时训练有条件与无条件分支，推理时通过加权组合实现无需分类器的条件引导。
- **KL Divergence (概念评估)**：衡量生成图像类别分布与均匀分布的差异，越低表示概念越多样化且分布越均衡。

## 可复现要素
- **数据集**：ImageNet子集（4组×5类×5张）、ADE20K厨房子集（25张）、自建艺术画作数据集（梵高5张、莫奈7张、毕加索5张），论文未明确公开链接。
- **代码/权重**：项目主页 https://energy-based-model.github.io/unsupervised-concept-discovery/，预训练模型使用 Stable Diffusion v2.1。
- **关键超参**：使用50步DDIM采样、classifier-free guidance、K-means聚类中K等于概念数、ResNet-50 logit阈值10、CLIP相似度阈值0.3。
