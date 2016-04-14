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

## Introdction

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
This method involves several steps in its solution pipeline.
First, overlapping patches are densely cropped from the input image and pre-processed (e.g.,subtracting mean and normalization).
These patches are then encoded by a low-resolution dictionary.
The sparse coefficients are passed into a high-resolution dictionary for reconstructing high-resolution patches.
The overlapping re- constructed patches are aggregated (e.g., by weighted averaging) to produce the final output.
This pipeline is shared by most external example-based methods, which pay particular attention to learning and optimizing the dictionaries [2], [51], [52] or building efficient mapping functions [24], [42], [43], [49].
However, the rest of the steps in the pipeline have been rarely optimized or considered in an unified optimization framework.

In this paper, we show that the aforementionedpipeline is equivalent to a deep convolutional neural network[26] (more details in Section 3.2).
Motivated by thisfact, we consider a convolutional neural network thatdirectly learns an end-to-end mapping between low- andhigh-resolution images.
Our method differs fundamentallyfrom existing external example-based approaches,in that ours does not explicitly learn the dictionaries [42],[51], [52] or manifolds [2], [4] for modeling the patchspace.
These are implicitly achieved via hidden layers.
Furthermore, the patch extraction and aggregation arealso formulated as convolutional layers, so are involvedin the optimization.
In our method, the entire SR pipelineis fully obtained through learning, with little pre/postprocessing.
We name the proposed model Super-Resolution ConvolutionalNeural Network (SRCNN)1.
The proposedSRCNN has several appealing properties.
First, its structureis intentionally designed with simplicity in mind,and yet provides superior accuracy2 compared withstate-of-the-art example-based methods.
Figure 1 showsa comparison on an example.
Second, with moderatenumbers of filters and layers, our method achievesfast speed for practical on-line usage even on a CPU.
Our method is faster than a number of example-basedmethods, because it is fully feed-forward and doesnot need to solve any optimization problem on usage.
Third, experiments show that the restoration quality ofthe network can be further improved when (i) largerand more diverse datasets are available, and/or (ii)a larger and deeper model is used.
On the contrary,larger datasets/models can present challenges for existingexample-based methods.
Furthermore, while mostexisting methods [12], [15], [23], [29], [38], [39], [42], [46],[48], [52] are not readily extendable for handling multiplechannels in color images, the proposed network can copewith three channels of color images simultaneously toachieve improved super-resolution performance.
