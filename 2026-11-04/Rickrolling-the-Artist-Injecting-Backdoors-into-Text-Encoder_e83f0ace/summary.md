---
title: "Rickrolling-the-Artist-Injecting-Backdoors-into-Text-Encoder"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/Struppek_Rickrolling_the_Artist_Injecting_Backdoors_into_Text_Encoders_for_Text-to-Image_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 16:39:39"
field: "多模态生成模型安全"
keywords: ["后门攻击", "文本到图像生成", "CLIP", "同形字", "模型安全", "扩散模型", "多模态"]
innovations: ["首次提出针对文本编码器后门攻击的分钟级微调方法", "教师-学生双损失框架平衡后门效果与模型效用", "支持TPA和TAA两种攻击模式的灵活触发器设计"]
benchmarks: ["MS-COCO 2014 val", "ImageNet-V2零样本分类", "LAION-Aesthetics v2 6.5+"]
---

# 论文速读：Rickrolling-the-Artist-Injecting-Backdoors-into-Text-Encoder

## 一句话总结
本文首次提出针对文本到图像生成模型的后门攻击，通过在预训练文本编码器（如CLIP）中注入隐蔽后门，仅需在提示词中插入单个同形字/emoji等触发器，即可迫使模型生成指定内容或附加隐藏属性，而干净输入下的生成质量几乎不受影响。

## 研究问题与动机
- **文本到图像模型的安全性空白**：Stable Diffusion等主流模型依赖外部预训练文本编码器，用户信任这些组件的行为符合预期，但编码器的安全威胁从未被系统性研究。
- **现有后门攻击方法的局限**：此前CLIP后门攻击（Carlini & Terzis, ICLR 2022）需从零开始训练整个CLIP模型，耗时数百GPU小时且需要标注数据；本文方法仅需微调文本编码器，耗时不到两分钟。
- **单字符触发的隐蔽性风险**：多模态模型对字符编码高度敏感（Struppek et al., 2022），单个不可见触发器（如西里尔字母о替代拉丁字母o）即可改变生成结果，恶意攻击者可通过自动提示工具批量植入触发器。
- **后门攻击的双重用途**：除威胁安全外，反向应用可将有害概念（如裸露、暴力）从编码器中"遗忘"，提升生成安全性。

## 核心贡献（创新点）
- **首次展示文本到图像模型中文本编码器后门攻击的可行性**：与Carlini & Terzis需在预训练阶段注入数据不同，本文仅需对已发布编码器进行分钟级微调。
- **提出教师-学生两阶段损失框架平衡后门效果与模型效用**：核心区别在于同时优化$\mathcal{L}_{Backdoor}$（ poisoned输入映射到目标嵌入）和$\mathcal{L}_{Utility}$（干净输入保持原嵌入），而非仅最大化攻击成功率。
- **支持两种攻击模式：目标提示攻击(TPA)与目标属性攻击(TAA)**：TPA完全覆盖原始提示生成目标内容，TAA仅附加属性修改，适应不同攻击场景。
- **触发器类型灵活且隐蔽**：可使用同形字( homoglyphs)、emoji、缩写或任意词语，单字符级别触发难以被肉眼察觉。
- **证明后门可反向用于概念消除**：通过将敏感词映射为空字符串或安全属性，可实现"受控遗忘"，提升生成系统安全性。

## 方法详解
**威胁模型**：攻击者访问干净编码器E和少量英文文本数据集X（如LAION），微调后通过域名伪造或恶意GitHub分发 poisoned编码器；无需访问受害者管道或原始训练数据。

**触发器选择**：重点使用同形字——外观相同但Unicode不同的字符（如拉丁o U+006F vs 西里尔о U+043E），以及emoji、单词等。

**两种后门目标**：
- **TPA（目标提示攻击）**：将训练样本中所有目标字符替换为触发器，目标$y_t$为固定提示文本。
- **TAA（目标属性攻击）**：仅替换每个样本中单个随机字符，目标$y_t$为替换该词后的属性描述。

**核心优化目标**：教师-学生框架，教师E固定，学生$\widetilde{E}$ trainable。

$$\mathcal{L}_{Backdoor} = \frac{1}{|X|}\sum_{v \in X} d(E(y_t), \widetilde{E}(v \oplus t))$$

$$\mathcal{L}_{Utility} = \frac{1}{|X'|}\sum_{w \in X'} d(E(w), \widetilde{E}(w))$$

$$\mathcal{L} = \mathcal{L}_{Utility} + \beta \cdot \mathcal{L}_{Backdoor}$$

其中$d$为负余弦相似度，$\beta=0.1$。训练时混合干净样本batch（size=128）和poisoned样本（每后门32个），AdamW优化器，lr=$10^{-4}$，100/200 epoch后衰减。

**评估指标**：
- z-Score：度量poisoned嵌入偏移程度（TPA专用）
- $Sim_{target}$：poisoned嵌入与目标嵌入的平均余弦相似度
- $Sim_{CLIP}$：生成图像与目标提示的CLIP跨模态相似度
- $Sim_{clean}$：干净输入下poisoned与clean编码器的嵌入相似度
- FID：生成图像质量
- ImageNet-V2零样本准确率：模型效用保持

## 实验与结果
- **模型与数据**：Stable Diffusion v1.4，CLIP文本编码器；注入使用LAION-Aesthetics v2 6.5+文本，评估使用MS-COCO 2014 val（40,504样本）。
- **单后门TPA定性结果**：干净输入下生成质量无显著下降；插入单个西里尔о触发后，图像内容完全切换为目标提示（如"boat on a lake"→Rick Astley）。
- **注入速度**：单后门64个poisoned样本/步，100步仅需约100秒（V100 GPU）。
- **Poisoned样本数量分析**：z-Score随样本数增加而提升，>3200样本后饱和；FID和$Sim_{clean}$几乎不受影响。
- **多后门实验**：最多注入32个后门，TPA的z-Score和$Sim_{clean}$略有下降但攻击仍成功；FID反而改善；TAA各项指标稳定。
- **ImageNet零样本准确率**：即使32个后门，TPA的Acc@1仅从69.82%轻微下降；TAA几乎不变。
- **最强结果**：32个TPA后门仍使$Sim_{CLIP}$达到接近干净模型的水平（图7虚线为干净基线），证明高隐蔽性。
- **应用扩展**：成功用于删除裸露概念（映射为空字符串）和图像检索系统中的后门触发。

## 相关工作脉络
- **Carlini & Terzis (ICLR 2022)**：首次在CLIP中注入后门，但需从头训练整个模型数百GPU小时；本文方法仅需微调文本编码器数分钟，显著降低攻击门槛。
- **Struppek et al. (arXiv 2022)**：发现多模态模型对同形字敏感；本文在此基础上系统化构建可控制的主动攻击框架。
- **BadNets (Gu et al., 2017)**：图像分类经典后门攻击；本文将其扩展至文本到图像生成领域，触发器从视觉patch变为文本字符。
- **BadNL (Chen et al., ACSAC 2021)**：使用零宽字符攻击NLP情感分析模型；本文触发器更具通用性（同形字/emoji/任意词）。
- **ONION (Qi et al., EMNLP 2021)**：防御文本后门的关键词识别方法；作者指出此类防御对不可见触发器效果有限。
- **Dreambooth (Ruiz et al., 2022)**：微调扩散模型绑定特定主体；本文从安全视角揭示微调文本编码器的潜在滥用风险。

## 局限性与未来方向
- **TPA无法完全覆盖复杂提示**：当触发器插入额外关键词时，部分原始提示内容可能残留于生成图像中。
- **TAA对独特概念修改受限**：如大幅改变名人外貌等固有特征时可能失败。
- **未验证其他编码器/模型架构**：仅测试CLIP + Stable Diffusion，其他文本编码器（如T5）及生成模型（DALL-E 2、GLIDE）的脆弱性待验证。
- **防御机制缺失**：现有后门防御多针对图像分类，文本到图像领域的检测与缓解方法尚未开发。
- **同形字检测防御的局限性**：自动扫描非拉丁字符可拦截部分攻击，但对emoji/缩写等触发器无效。

## 研究启发与可借鉴点
- **教师-学生微调范式可迁移至其他模态**：此两损失平衡框架可用于评估或加固多模态 embedding模型的鲁棒性。
- **单字符触发器的隐蔽性警示**：提示词工程中应引入触发器检测层，尤其是对同形字和特殊Unicode字符的过滤。
- **后门评估指标的复用**：z-Score和$Sim_{CLIP}$可作为多模态后门攻击的通用评估工具。
- **"概念消除"的反向应用价值**：本方法可直接用于内容安全过滤系统，移除模型中的有害概念关联。
- **轻量级后门注入的供应链风险**：证明第三方编码器分发的安全审计必要性，可启发模型溯源与验证机制研究。

## 关键术语表
- **Homoglyph（同形字）**：外观相同但Unicode编码不同的字符（如拉丁o与西里尔о），用于构造隐蔽触发器。
- **TPA（Target Prompt Attack，目标提示攻击）**：后门触发时完全覆盖原始提示、生成指定目标内容的攻击模式。
- **TAA（Target Attribute Attack，目标属性攻击）**：后门触发时仅添加/修改特定属性的攻击模式。
- **Teacher-Student Framework（教师-学生框架）**：固定教师编码器、仅更新学生编码器的微调范式，用于平衡后门效果与模型效用。
- **z-Score**：度量poisoned嵌入相对干净嵌入偏移程度的统计指标，值越高表示后门效果越强。
- **FID（Fréchet Inception Distance）**：评估生成图像与真实图像分布相似度的指标，越低越好。
- **CLIP**：OpenAI提出的对比语言-图像预训练模型，其文本编码器被Stable Diffusion等广泛使用。
- **Backdoor Attack（后门攻击）**：通过在训练阶段注入触发器-目标配对，使模型在触发器出现时执行恶意行为的攻击类型。

## 可复现要素
- **数据集**：LAION-Aesthetics v2 6.5+（注入用）、MS-COCO 2014 val（评估用），均公开可用。
- **代码开源**：https://github.com/LukasStruppek/Rickrolling-the-Artist
- **模型权重**：提供poisoned编码器权重及配置文件。
- **关键超参**：$\beta=0.1$，TPA 100 epoch / TAA 200 epoch，lr=$10^{-4}$（75/150 epoch后×0.1），batch size=128（clean）+32（poisoned/后门），AdamW优化器。
- **硬件**：NVIDIA V100 GPU。
