论文标题: 1 Million Parameters Are Enough: Merging CNNs and Transformers for Ultra-Lightweight 3D Brain Tumor Segmentation

DOI: 待论文发表后更新

Abstract. Transformer架构因其独特的全局自注意力机制在自然语言处理和计算机视觉领域取得了巨大成功。然而，纯Transformer模型的参数规模庞大且需要大规模数据集进行训练。卷积神经网络(CNNs)以其强大的空间归纳偏置能力而闻名，能够利用较少的参数有效地捕获图像的局部特征。然而，CNNs难以建立全局特征之间的远程依赖关系，而全局特征依赖在密集预测任务中尤为重要，尤其是3D医学影像分割。在高精度和计算成本之间寻求平衡是一个关键挑战，理论上较高参数规模的模型可以获得更好的性能，但同时也会导致更高的计算复杂度和内存占用而不利于实现。本文基于轴向深度可分离卷积和全局自注意力机制原理提出了一种超轻量级3D脑肿瘤分割模型：TransLiteUNet。它是一种混合的CNNs-Transformer架构，无需任何模型预先训练即可实现医学图像的准确分割，同时具有CNNs的强归纳偏置和transformer架构的强大全局上下文建模能力。该模型提出了全新的3D轴向深度可分离卷积残差结构，并使用了7x7x7的卷积核来扩大感受野。此外，还引入了一种改进的Mobile vision transformer模块作为bottleneck结构，并加入了可学习的位置编码用于全局特征建模。总体而言，TransLiteUNet的参数量仅为442K，这比V-Net少102倍，比TransUNet少237倍，比CKD-TransBTS少189倍。与目前所有最先进的模型相比，该模型大大降低了计算复杂度，同时在3D脑肿瘤分割任务上取得了目前先进的性能。使用五折交叉验证进行实验，提出的模型在BraTS2020测试集上的平均Dice分割结果为0.838 (更详细的分割指标，ET:0.764; TC:0.839; WT:0.910),在BraTS2021测试集上的平均Dice分割结果为0.894(更详细的分割指标，ET:0.857; TC:0.893; WT:0.932)。在相同数据集(BraTS2020)和同等实验条件下，这比目前主流模型的分割效果更好(CKD-TransBTS:0.826; nnFormer:0.803; SwinUNETR:0.819; UNETR:0.812)。

Keywords: Transformer · Encoder-Decoder · Ultra-Lightweight Model · Medical Volumetric Segmentation · TransLiteUNet


