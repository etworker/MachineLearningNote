# Image Super-Resolution Using Deep Convolutional Networks

使用深度卷积网络进行图片超分辨率

## Author

Chao Dong, Chen Change Loy, Member, IEEE, Kaiming He, Member, IEEE, and Xiaoou Tang, Fellow, IEEE

## Abstract

We propose a deep learning method for single image super-resolution (SR). 

对于单图片超分辨率(SR)，我们提出了一种深度学习的方法。

Our method directly learns an end-to-end mapping between the low/high-resolution images. 

我们的方法直接学习低分辨率和高分辨率图片之间端到端的映射。

The mapping is represented as a deep convolutional neural network (CNN) that takes the low-resolution image as the input and outputs the high-resolution one. 

这个映射通过CNN来表达，它将低分图片作为输入，高分图片作为输出。

We further show that traditional sparse-coding-based SR methods can also be viewed as a deep convolutional network. 

传统的基于稀疏编码SR方法也可以看做是一个深度卷积网络。

But unlike traditional methods that handle each component separately, our method jointly optimizes all layers. 

与传统方法需要对每个元素分别处理不同，我们的方法可以联合优化全部层。

Our deep CNN has a lightweight structure, yet demonstrates state-of-the-art restoration quality, and achieves fast speed for practical on-line usage. 

我们的深度CNN具有轻量级结构，先进的恢复质量，并达到了实际在线使用的高速。

We explore different network structures and parameter settings to achieve tradeoffs between performance and speed. 

我们探索了不同的网络结构及参数设定，以在性能与速度间取得平衡。

Moreover, we extend our network to cope with three color channels simultaneously, and show better overall reconstruction quality.

我们扩展了网络来同时处理3个颜色通道，展现了更好的重建质量。

## Index Terms

Super-resolution, deep convolutional neural networks, sparse coding

超分辨率，深度卷积神经网络，稀疏编码

## 1. Introdction

Single image super-resolution (SR) [18], which aims at recovering a high-resolution image from a single lowresolution image, is a classical problem in computer vision.

单张图片超分辨率(SR)，目的是从一张低分辨率图片恢复出高分辨率图片，是计算机视觉的经典问题。

This problem is inherently ill-posed since a multiplicity of solutions exist for any given low-resolution pixel. In other words, it is an underdetermined inverse problem, of which solution is not unique.

从低分到高分，有非常多的方法。

Such a problem is typically mitigated by constraining the solution space by strong prior information.

求解该类问题需要强的先验信息来约束解空间以简化。

To learn the prior, recent state-of-the-art methods mostly adopt the example-based [47] strategy. 

最近的方法通常采取基于采样的策略来学习先验信息。

These methods either exploit internal similarities of the same image [5], [12], [15], [49], or learn mapping functions from external low- and high- resolution exemplar pairs [2], [4], [14], [21], [24], [42], [43], [49], [50], [52], [53]. 

这些方法或者发掘相同图片的内部相似度，或者学习外部的低分高分采样对的映射函数。

The external example-based methods can be formulated for generic image superresolution, or can be designed to suit domain specific
tasks, i.e., face hallucination [30], [52], according to the training samples provided.

外部基于采样的方法可以建立通用的图片超参数，或设计来适应特定领域的任务，如根据提供的训练采样生成虚幻人脸。

The sparse-coding-based method [51], [52] is one of the representative external example-based SR methods.

基于稀疏编码的方法是外部基于采样SR的典型方法之一。

This method involves several steps in its solution pipeline.

该方法在他的解决方案管道中包含多步。

First, overlapping patches are densely cropped from the input image and pre-processed (e.g.,subtracting mean and normalization).

首先，从输入图片中密集的剪切出重叠的碎片，并作预处理（如去掉均值，并作归一化）

These patches are then encoded by a low-resolution dictionary.

接着这些碎片通过低分辨率字典进行编码。


The sparse coefficients are passed into a high-resolution dictionary for reconstructing high-resolution patches.

稀疏系数通过一个高分辨率字典来重建高分辨率碎片。

The overlapping reconstructed patches are aggregated (e.g., by weighted averaging) to produce the final output.

重叠的重建碎片被聚合起来（如加权平均）以生成最终输出。

This pipeline is shared by most external example-based methods, which pay particular attention to learning and optimizing the dictionaries [2], [51], [52] or building efficient mapping functions [24], [42], [43], [49].

这个管道被大多数外部基于采样的方法所共享，特别要注意字典的学习和优化，或参数映射函数的建立。

However, the rest of the steps in the pipeline have been rarely optimized or considered in an unified optimization framework.

但是，管道的其他步就很少优化，或在一个统一优化框架下来考虑。


In this paper, we show that the aforementioned pipeline is equivalent to a deep convolutional neural network[26] (more details in Section 3.2).

本文中，我们展现前述管道与深度神经网络等价。

Motivated by this fact, we consider a convolutional neural network that directly learns an end-to-end mapping between low- and high-resolution images.

针对这个事实，我们考虑一个卷积神经网络来直接学习低分与高分图片之间端到端的映射。

Our method differs fundamentally from existing external example-based approaches, in that ours does not explicitly learn the dictionaries [42],[51], [52] or manifolds [2], [4] for modeling the patch space.

我们的方法与已知的基于采样的方法有根本性的不同，我们并没有显式的学习字典或流形来以此对碎片空间建模。

These are implicitly achieved via hidden layers.

这些都通过隐藏层默默的达成了。

Furthermore, the patch extraction and aggregation are also formulated as convolutional layers, so are involvedin the optimization.

不但如此，碎片的提取和聚合都被卷积层构成了。

In our method, the entire SR pipelineis fully obtained through learning, with little pre/post processing.

在我们方法中，通过一点预处理和后处理的学习，完整的得到了整体SR管道。


We name the proposed model Super-Resolution Convolutional Neural Network (SRCNN).

我们对该模型命名为SRCNN。

The proposed SRCNN has several appealing properties.

SRCNN有一些吸引人的属性。

First, its structure is intentionally designed with simplicity in mind, and yet provides superior accuracy compared with state-of-the-art example-based methods.

首先，它特意设计的简单粗暴的结构，提供了与现有基于采样方法相比更高的精度。

Figure 1 shows a comparison on an example.

图1通过例子展示了对比。

Second, with moderate numbers of filters and layers, our method achieves fast speed for practical on-line usage even on a CPU.

其次，通过适度个数的过滤器和层数，即使只有一个CPU，我们的方法也可达到可在线使用的速度。

Our method is faster than a number of example-based methods, because it is fully feed-forward and does not need to solve any optimization problem on usage.

我们的方法比不少基于采样的方法都快，因为他是完全前馈的，而且使用中不需要解决任何优化问题。

Third, experiments show that the restoration quality of the network can be further improved when 

第三，实验表明网络恢复质量可以通过如下方法提升：

(i) larger and more diverse datasets are available, and/or 

(i) 更大更多样的数据库

(ii) a larger and deeper model is used.

(ii) 更大更深的模型

On the contrary, larger datasets/models can present challenges for existing example-based methods.

作为对比，更大的数据库/模型可以向已有的基于采样的方法提出挑战。

Furthermore, while most existing methods [12], [15], [23], [29], [38], [39], [42], [46],[48], [52] are not readily extendable for handling multiple channels in color images, the proposed network can cope with three channels of color images simultaneously to achieve super-resolution performance.

此外，大多的已有方法不能扩展来处理多通道的彩色图片，本文提出的网络可以同时处理彩色图片的三个通道来达到改进的高分辨率表现。


Overall, the contributions of this study are mainly in three aspects: 

总的来说，本文主要贡献在三点：

1) We present a fully convolutional neural network for image super-resolution. The network directly learns an end-to-end mapping between low- and high-resolution images, with little pre/post-processing beyond the optimization. 

1) 针对图片超分辨率，提出一个完整的卷积神经网络。这个网络直接学习低分和高分图片之间端到端的映射，当然还要做些小小的预处理和后处理。

2) We establish a relationship between our deep-learning-based SR method and the traditional sparse-coding-based SR methods. This relationship provides a guidance for the design of the network structure. 

2) 建立了基于深度学习的SR方法与传统基于稀疏编码的SR方法之间的关联。这个关联提供了网络结构设计的指导。

3) We demonstrate that deep learning is useful in the classical computer vision problem of super resolution, and can achieve good quality and speed. 

3) 我们证明了在超分辨率的传统计算机视觉问题上，深度学习也是有用的，也能得到不错的质量和速度。


A preliminary version of this work was presented earlier [10]. 

The present work adds to the initial version in signiﬁcant ways. 

Firstly, we improve the SRCNN by introducing larger ﬁlter size in the non-linear mapping layer, and explore deeper structures by adding nonlinear mapping layers. 

Secondly, we extend the SRCNN to process three color channels (either in YCbCr or RGB color space) simultaneously. 

Experimentally, we demonstrate that performance can be improved in comparison to the single-channel network. 

Thirdly, considerable new analyses and intuitive explanations are added to the initial results.

We also extend the original experiments from Set5 [2] and Set14 [53] test images to BSD200 [32] (200 test images). 

In addition, we compare with a number of recently published methods and conﬁrm that our model still outperforms existing approaches using different evaluation metrics.

## 2. RELATED WORK

### 2.1 Image Super-Resolution

According to the image priors, single-image super resolution algorithms can be categorized into four types – prediction models, edge based methods, image statistical methods and patch based (or example-based) methods. 

单张图片超分辨率算法分为4类：预测模型，基于边缘的方法，图片统计方法，及基于碎片（或基于采样）的方法。

These methods have been thoroughly investigated and evaluated in Yang et al.’s work [47]. Among them, the example-based methods [15], [24], [42], [49] achieve the state-of-the-art performance. 

基于采样的方法是目前最好的。

The internal example-based methods exploit the self similarity property and generate exemplar patches from the input image. 

内部基于采样的方法利用自身相似度的属性，并从输入图片来生成标本碎片。

It is ﬁrst proposed in Glasner’s work [15], and several improved variants [12], [48] are proposed to accelerate the implementation. 

The external example-based methods [2], [4], [14], [42], [50], [51], [52], [53] learn a mapping between low/high-resolution patches from external datasets. 

外部基于采样的方法从外部数据集中学习低分和高分碎片之间的映射。

These studies vary on how to learn a compact dictionary or manifold space to relate low/high-resolution patches, and on how representation schemes can be conducted in such spaces. 

这些研究各不相同，有的在如何学习一个精简的字典或流形空间来联系低分高分的碎片，有的在如何在这样的空间中构建表达方案。

In the pioneer work of Freeman et al. [13], the dictionaries are directly presented as low/high-resolution patch pairs, and the nearest neighbour (NN) of the input patch is found in the low-resolution space, with its corresponding high-resolution patch used for reconstruction. 

在Freeman开创性的工作中，字典被表现为低分高分碎片对，在低分空间中发现了输入碎片的最近邻，与之关联的高分碎片被用于重建。

Chang et al. [4] introduce a manifold embedding technique as an alternative to the NN strategy. In Yang et al.’s work [51], [52], the above NN correspondence advances to a more sophisticated sparse coding formulation. 

Chang介绍了一种流形嵌入技术作为NN策略的替代。在Yang的工作中，上面的NN与更精致的稀疏编码公式对应。

Other mapping functions such as kernel regression [24], simple function [49] and anchored neighborhood regression [42], [43] are proposed to further improve the mapping accuracy and speed. 

其他的映射函数，如核回归，简单函数和锚邻回归，有更好的映射精度和速度。

The sparse-coding-based method and its several improvements [42], [43], [50] are among the state-of-the-art SR methods nowadays. 

基于稀疏编码的方法和它的一些改进是现在最屌的SR方法了。

In these methods, the patches are the focus of the optimization; the patch extraction and aggregation steps are considered as pre/post-processing and handled separately.

在这些方法中，碎片是优化的焦点；碎片的提取和聚合步骤被当作预处理和后处理来分别处理。

The majority of SR algorithms [2], [4], [14], [42], [50], [51], [52], [53] focus on gray-scale or single-channel image super-resolution. 

大多数SR算法专注于灰度或单通道图片的超分辨率。

For color images, the aforementioned methods ﬁrst transform the problem to a different color space (YCbCr or YUV), and SR is applied only on the luminance channel. 

对于彩色图片，前述方法首先将问题转为不同的颜色空间(YCbCr 或 YUV)，然后SR只应用于亮度通道。

Due to the inherently different properties between the luminance channel and chrominance channels, these methods can be hardly extended to high-dimensional data directly. 

由于亮度通道和色度通道之间的固有的不同属性，这些方法可以不能直接扩展到高维数据。

There are also works attempting to super-resolve all channels simultaneously. 

也有些工作尝试同时处理全部通道。

For example, Kim and Kwon [24] and Dai et al. [6] apply their model to each RGB channel and combined them to produce the ﬁnal results. 

例如，Kim，Kwon，Dai将他们的模型应用到RGB的每个通道，再将他们合并来生成最终结果。

However, none of them has analyzed the SR performance of different channels, and the necessity of recovering all three channels.

然而，他们都没有分析在不同通道时SR的性能，及恢复全部三个通道的必要性。

### 2.2 Convolutional Neural Networks

Convolutional neural networks (CNN) date back decades [26] and deep CNNs have recently shown an explosive popularity partially due to its success in image classiﬁcation [17], [25]. 

CNN现在很流行，在图片分类上取得了成功。

They have also been successfully applied to other computer vision ﬁelds, such as object detection [34], [41], [54], face recognition [40], and pedestrian detection [35]. 

在其他计算机视觉领域上也取得了成功，如对象检测，人脸识别，及行人检测。

Several factors are of central importance in this progress: 

有几个重要因素的进展:

(i) the efﬁcient training implementation on modern powerful GPUs [25], 

在现代强力GPU上的有效训练的实现

(ii) the proposal of the Rectiﬁed Linear Unit (ReLU) [33] which makes convergence much faster while still presents good quality [25], and 

ReLU带来更快的收敛，质量也不错

(iii) the easy access to an abundance of data (like ImageNet [8]) for training larger models. 

更易获得大量数据来训练更大的模型

Our method also beneﬁts from these progresses.

我们的方法受益于上面的进展。

### 2.3 Deep Learning for Image Restoration 

There have been a few studies of using deep learning techniques for image restoration. 

关于使用深度学习技术做图像恢复，已有一些研究。

The multi-layer perceptron (MLP), whose all layers are fully-connected (in contrast to convolutional), is applied for natural image denoising [3] and post-deblurring denoising [36].

多层感知器，所有层都是全连接，用来进行自然图像去噪，及后去模糊去噪。

More closely related to our work, the convolutional neural network is applied for natural image denoising [20] and removing noisy patterns (dirt/rain) [11]. 

与我们工作更接近的，是CNN应用在自然图片去噪和去除噪点模式。

These restoration problems are more or less denoising-driven. 

这些恢复问题大多是去噪驱动的。

Cui et al. [5] propose to embed auto-encoder networks in their super resolution pipeline under the notion internal example-based approach [15]. 

Cui提出在内部基于采样的概念下，在超分辨率管道中嵌入自编码网络。

The deep model is not speciﬁcally designed to be an end-to-end solution, since each layer of the cascade requires independent optimization of the self-similarity search process and the auto-encoder. 

深度模型不是专用于端到端的方案，因为每层需要自相似性搜索过程及自动编码的独立优化。

On the contrary, the proposed SRCNN optimizes an end-to-end mapping.

与此相反，SRCNN优化了端到端的映射。

## 3 CONVOLUTIONAL NEURAL NETWORKS FOR SUPER-RESOLUTION 

### 3.1 Formulation 
