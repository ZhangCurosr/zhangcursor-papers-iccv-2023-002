---
title: "Shift-from-Texture-bias-to-Shape-bias-Edge-Deformation-based"
source: https://openaccess.thecvf.com/content/ICCV2023/papers/He_Shift_from_Texture-bias_to_Shape-bias_Edge_Deformation-based_Augmentation_for_Robust_ICCV_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 21:18:56"
field: "对抗鲁棒性与形状感知学习"
keywords: ["texture-bias", "shape-bias", "adversarial robustness", "data augmentation", "edge deformation", "thin plate spline", "generative augmentation"]
innovations: ["提出基于边缘图 TPS 变形的在线增强方法 SDbOA，首次系统探索边缘形变对 shape-bias 的提升作用", "设计自信息引导的边缘编码（EMSE）扩展边缘周围区域，增强形状编码对噪声的鲁棒性", "构建纹理-形状联合约束的两阶段生成器（TSG），以软约束方式保持生成样本的语义合理性与形状多样性"]
benchmarks: ["CIFAR-10", "ImageNet", "Fashion MNIST", "CelebA", "CIFAR-10-C"]
---

# 论文速读：Shift-from-Texture-bias-to-Shape-bias-Edge-Deformation-based

## 一句话总结
本文提出了一种基于边缘图变形的在线数据增强方法 SDbOA，通过对目标物体的边缘进行语义保持的变形 augmentation，促使 CNN 从过度依赖纹理转向依赖形状特征，从而显著提升模型在对抗攻击、后门攻击和常见腐蚀下的鲁棒性。

## 研究问题与动机
- **CNN 存在严重的 texture-bias**：Geirhos 等人发现 ImageNet 训练的 CNN 预测主要基于物体纹理而非形状，当纹理-形状冲突时（如猫的形状填充象 textura），模型倾向于预测为象。
- **现有 shape-bias 方法的不足**：已有工作（如 StyleAug、ShapeAug）虽能降低 texture-bias，但生成的形状变化多样性不足，无法真正实现 shape-biased 网络；直接对图像进行几何变形会破坏语义一致性。
- **边缘图变形潜力未被挖掘**：边缘（edge）是形状的紧凑表征，直接对边缘图进行 TPS 变形可在保持语义合理性的同时丰富形状变化，但该方向尚未被系统探索。
- **核心研究问题**：能否通过对边缘图进行数据增强来强化形状线索、减少 texture-bias？

## 核心贡献（创新点）
- **提出基于 TPS 的边缘图在线变形增强策略**：区别于直接对图像做几何变换，本文首次针对边缘图进行形变 augmentation，使网络学到更多样化的形状表征。
- **设计自信息引导的边缘编码（EMSE）**：将 Robust Canny 边缘图与 self-information guided map 融合，扩展边缘周围区域，使形状编码对噪声更鲁棒。
- **构建纹理-形状联合约束的生成器（TSG）**：通过两阶段训练，利用降噪纹理图和形状保真损失引导生成语义合理的增强样本，解决 Pix2Pix 生成结果中纹理-形状类别不匹配的问题。
- **在多个数据集和攻击类型上验证优越性**：在 CIFAR-10、ImageNet 等数据集上，对抗鲁棒性显著优于 StyleAug、ShapeAug、EdgeNetRob 等基线，且训练开销远低于对抗训练方法。

## 方法详解
论文提出 SDbOA（Shape Deformation-based Online Augmentation），框架包含三个模块：

**1. EMSE（Edge Map-based Shape Encoding）**
- 先用 Robust Canny 提取原始边缘图 $E_{RC}$
- 再计算像素级自信息引导图 $E_{info}$：对每个像素 $p$，计算其邻域 $\mathcal{N}_p$ 内各像素 $p'$ 的指数加权差异，取平均后做阈值化（大于平均值为 1，否则为 0）
- 扩展边缘图：$E_{extend} = E_{RC} + E_{info}$，利用边缘周围区域的信息增强对边缘噪声的鲁棒性

**2. TSD（TPS-based Shape Deformation）**
- 将 $E_{extend}$ 均匀划分为 $n = 16$ 个网格，以网格端点为控制点 $\{P_i\}$
- 对控制点施加高斯噪声扰动：$Q_i = P_i + \lambda N$，其中 $N \sim \mathcal{N}(0,1)$，$\lambda = 0.1$ 为变形强度
- 使用 Thin Plate Spline (TPS) 找到从 $\{P_i\}$ 到 $\{Q_i\}$ 的最优形变映射 $\Phi^x, \Phi^y$，最大化形变平滑性同时最小化控制点偏差，生成变形边缘图 $E_{deform}$

**3. TSG（Texture and Shape-based Generation）**
- 对原图做双线性下采样再插值，得到降噪纹理图 $I_{txt}$ 作为纹理指导
- 两阶段训练生成器 G：
  - 第一阶段：输入 $E_{extend}$ 和 $I_{txt}$，联合优化对抗损失 $\mathcal{L}_{gan}$、特征匹配损失和 ACGAN 分类辅助损失，生成粗糙图像
  - 第二阶段：输入 $E_{deform}$，联合优化 $\mathcal{L}_{gan} + \mathcal{L}_{cls} + \mathcal{L}_{edge}$，其中 $\mathcal{L}_{edge} = ||eCNN(I_{syn}), E_{deform}||_1$ 为形状保真损失，确保生成图的边缘与变形边缘一致
- 训练过程中同步更新分类器参数 $\theta$，强化形状-纹理类别关联

**整体流程**：在每轮训练时在线调用上述流程生成增强样本，无需额外离线预处理，训练开销与 vanilla training 相当。

## 实验与结果
- **数据集**：Fashion MNIST（FM）、CelebA（性别分类）、CIFAR-10（CI0）、ImageNet（IN）； backbone 分别为 ResNet-18 / ResNet-50
- **Shape-bias 评估**：采用 Geirhos 指标 $s_{b_{GE}}$ 和 Islam 指标 $s_{b_{IS}}$
  - ImageNet 上 $s_{b_{GE}}$：SDbOA 达 **71.28%**，高于 StyleAug（67.31%）和 ShapeAug（38.46%）
  - SIN+IN 上 $s_{b_{IS}}$：SDbOA 达 **37.5%**，显著优于对比方法
- **对抗攻击鲁棒性**（CIFAR-10，ResNet-50）：
  - 相比 NuAT：C&W 攻击 **+14.10%**，DeepFool 攻击 **+27.72%**
  - 相比 EdgeNetRob：ImageNet 上 C&W 攻击 **+6.98%**
  - 优于 FDA、PORT 等对抗训练方法，且训练时间仅需 3-5 小时（vs 数天）
- **常见腐蚀鲁棒性**（CIFAR-10-C）：mCE = **18.8%**
- **后门攻击鲁棒性**（Fashion MNIST，5%  poisoned ratio）：
  - Pixel 攻击 ASR = **0.04%**，Pattern 攻击 ASR = **0.98%**，显著低于 AugMax（ASR = 47.43% / 62.21%）
- **超参分析**：$\lambda = 0.1$ 为最优变形强度，过大（如 0.2）会破坏语义导致 shape-bias 反而下降
- **消融实验**：EMSE、TSD、TSG 三个模块均有效，完整 SDbOA 在 CIFAR-10 上 clean accuracy 83.27%，PGD 鲁棒性 50.91%

## 相关工作脉络
- **Geirhos et al. [11]**：发现 ImageNet CNN 存在 texture-bias，提出纹理-形状冲突数据集；本文在其基础上直接通过边缘变形增强 shape-bias
- **StyleAug [21]**：通过风格随机化打乱纹理，降低 texture-bias；但形状变化有限，本文进一步丰富形状多样性
- **ShapeAug [24]**：对物体做变形增强，但直接对图像形变可能破坏语义；本文仅对边缘图变形并通过生成器重建，保持语义合理性
- **EdgeNetRob [37]**：利用 Robust Canny 边缘图训练，提升 shape-bias；但未对形状做多样化增强，本文通过 TPS 变形补足此短板
- **Srikanthan et al. [34]**（Informative Dropout）：从 shape-bias 视角提升鲁棒性；但依赖固定形状，本文通过变形增强形状多样性
- **PixMix [17] / AugMax [41]**：在 common corruption 上表现优秀但 adversarial 鲁棒性不足（mCE 好但误分类率 >80%）；本文兼顾两者，adversarial 误分类率仅 27.7%

## 局限性与未来方向
- **仅适用于像素级扰动**：论文自述方法对 patch-wise 腐蚀（严重破坏形状结构的场景）效果有限，因边缘信息被大面积破坏
- **形状变形强度需精细调参**：$\lambda$ 过大反而损害 shape-bias，如何自适应学习最佳变形强度仍是开放问题
- **生成器引入额外训练开销**：虽远低于对抗训练，但仍需两阶段 GAN 训练，在资源受限场景下仍有优化空间
- **未来方向**：扩展至 patch-wise 腐蚀防护、探索自适应形变强度学习、应用于更多下游任务（如目标检测）

## 研究启发与可借鉴点
- **边缘图变形替代图像变形**：对紧凑的边缘表征而非原始图像做形变，可在保持语义一致性的同时丰富形状多样性，这一思路可迁移至其他需要形状不变性的任务
- **Self-information guided map 扩展边缘**：利用像素与邻域的差异度量补充边缘周围信息，增强边缘编码的鲁棒性，该机制可嵌入其他边缘检测或形状感知模块
- **形状保真损失 $\mathcal{L}_{edge}$ 的软约束设计**：与硬约束（直接形变图像）相比，通过 $l_1$ 损失引导生成器而非强制匹配，既保持语义合理性又实现形状多样化，这一范式可用于其他生成式 augmentation
- **在线 augment 的低开销优势**：边训练边生成，无需离线预处理，对大规模数据集友好，可结合本团队的训练流程直接复用

## 关键术语表
**Texture-bias**：CNN 在物体分类时过度依赖纹理特征而非形状特征的倾向，是模型鲁棒性差的重要原因
**Shape-bias**：网络预测主要依据物体形状线索的特征偏好，与 texture-bias 相对，更高的 shape-bias 通常带来更强的鲁棒性
**Thin Plate Spline (TPS)**：一种基于控制点的平滑形变映射方法，广泛用于图像配准与形变 augmentation
**Self-information guided map**：基于像素与其邻域的差异程度构建的引导图，用于扩展 Robust Canny 边缘图的覆盖范围
**SDbOA**：Shape Deformation-based Online Augmentation，本文提出的基于边缘图变形的在线数据增强框架
**$\mathcal{L}_{edge}$（形状保真损失）**：生成图像的边缘与变形边缘图之间的 $l_1$ 距离，作为软约束引导生成器保持语义合理性
**mCE（mean Corruption Error）**：在 15 种常见腐蚀类型、5 个严重级别上评估的平均错误率，mCE 越低表示对常见腐蚀越鲁棒
**ASR（Attack Success Rate）**：后门攻击中投毒样本被误分类为目标类别的比例，ASR 越低表示后门防御效果越好

## 可复现要素
- **代码**：已开源，地址 https://github.com/C0notSilly/-ICCV-23-Edge-Deformation-based-Online-Augmentation
- **数据集**：Fashion MNIST、CelebA、CIFAR-10、ImageNet（均为公开数据集）
- **关键超参**：网格数 $n = 16$，变形强度 $\lambda = 0.1$，扰动噪声服从 $\mathcal{N}(0,1)$
- **骨干网络**：ImageNet 用 ResNet-50，其余数据集用 ResNet-18
- **训练协议**：遵循 EdgeNetRob [37] 的评测协议，$l_\infty$ 扰动预算 CIFAR-10/CelebA/ImageNet 为 8/255，Fashion MNIST 为 25/255
- **对抗评估**：使用 FGSM、PGD、C&W、DeepFool 及 AutoAttack
- **论文未提及**：具体的 GPU 型号、batch size、学习率及优化器细节
