# Visualizing and Understanding Convolutional Networks

## Author

Matthew D. Zeiler, Rob Fergus

## Abstract

Large Convolutional Network models have recently demonstrated impressive classification performance on the ImageNet benchmark (Krizhevsky et al., 2012). 

最近，大型卷积网络模型在ImageNet基准上展现了令人印象深刻的分类性能。

However there is no clear understanding of why they perform so well, or how they might be improved. In this paper we address both issues.

然而有两个问题尚不清楚，为何做的这么好，以及可能如何改善。本文我们就来收拾这俩问题。

We introduce a novel visualization technique that gives insight into the function of intermediate feature layers and the operation of the classifier. 

我们介绍了一种新的可视化技术，为中间特征层的功能及分类器的操作带来深刻的剖析。

Used in a diagnostic role, these visualizations allow us to find model architectures that outperform Krizhevsky et al. on the ImageNet classification benchmark. 

作为诊断，可视化可以让我们寻找胜过Krizhevsky在ImageNet分类基准的模型结构。

We also perform an ablation study to discover the performance contribution from different model layers. 

我们还执行消融研究来发现不同模型层对性能的贡献。

We show our ImageNet model generalizes well to other datasets: when the softmax classifier is retrained, it convincingly beats the current state-of-the-art results on Caltech-101 and Caltech-256 datasets.

我们的ImageNet模型可以推广到其他数据集: 当重新训练softmax分类器，它在Caltech-101和Caltech-256数据集上令人信服地打败了当前最好的结果。

## Introduction

Since their introduction by (LeCun et al., 1989) in the early 1990's, Convolutional Networks (convnets) have demonstrated excellent performance at tasks such as hand-written digit classification and face detection.

自从LeCun在上世纪90年代早期引入卷积网络(convnets)，它就在手写数字分类和人脸检测等任务上表现出优良的性能。

In the last year, several papers have shown that they can also deliver outstanding performance on more challenging visual classification tasks. (Ciresan et al., 2012) demonstrate state-of-the-art performance on NORB and CIFAR-10 datasets. 

去年的数篇论文表明,在更具挑战性的视觉分类任务上，它们也可以提供出色的性能。Ciresan在NORB和CIFAR-10数据集上展示了最先进的性能。

Most notably, (Krizhevsky et al., 2012) show record beating performance on the ImageNet 2012 classification benchmark, with their convnet model achieving an error rate of 16.4%, compared to the 2nd place result of 26.1%.

最值得注意的是，Krizhevsky在2012年ImageNet分类基准上获得了压倒性的性能记录，和第二名26.1%的错误率相比，他们的convnet模型取得16.4%的错误率。

Several factors are responsible for this renewed interest in convnet models: 
(i) the availability of much larger training sets, with millions of labeled examples;
(ii) powerful GPU implementations, making the training of very large models practical and 
(iii) better model regularization strategies, such as Dropout (Hinton et al., 2012).

原因如下：
(i)更大的训练集,数以百万计带标记的样本
(ii)强大的GPU实现，使训练训练非常大的模型可行
(iii)更好的模型正则化策略，如Dropout(Hinton)


Despite this encouraging progress, there is still little insight into the internal operation and behavior of these complex models, or how they achieve such good performance. From a scientific standpoint, this is deeply unsatisfactory. 

尽管取得如此令人鼓舞的进展，对这些内部操作及复杂模型的行为，或者他们如何实现良好的性能，仍然无从了解。从科学的角度来看,非常不能令人满意。

Without clear understanding of how and why they work, the development of better models is reduced to trial-and-error. 

如果不能清楚的理解他们如何以及为何他们能工作，开发更好的模型只能沦为反复试错。

In this paper we introduce a visualization technique that reveals the input stimuli that excite individual feature maps at any layer in the model. 

本文介绍一种可视化技术，揭示了在模型的任一层上，输入的激励领激发个别特征网。

It also allows us to observe the evolution of features during training and to diagnose potential problems with the model. 

它还允许我们观察训练中特征的进化，并以此诊断模型的潜在问题。

The visualization technique we propose uses a multi-layered Deconvolutional Network (deconvnet), as proposed by (Zeiler et al., 2011), to project the feature activations back to the input pixel space. We also perform a sensitivity analysis of the classifier output by occluding portions of the input image, revealing which parts of the scene are important for classification.

我们提出的可视化技术使用Zeiler提出的多层反卷积网络(deconvnet)，将特征激活反向映射到输入的像素空间。我们还通过遮挡输入图片的部分来执行分类器输出的灵敏度分析，以此来揭示场景的哪些部分对分类很重要。


Using these tools, we start with the architecture of (Krizhevsky et al., 2012) and explore different architectures, discovering ones that outperform their results on ImageNet. We then explore the generalization ability of the model to other datasets, just retraining the softmax classifier on top. As such, this is a form of supervised pre-training, which contrasts with the unsupervised pre-training methods popularized by (Hinton et al., 2006) and others (Bengio et al., 2007; Vincent et al., 2008). The generalization ability of convnet features is also explored in concurrent work by (Donahue et al., 2013).

使用这些工具,我们从Krizhevsky的架构开始来探索不同的架构，发现能够在ImageNet上超越他们的结果。然后我们探索对于其他数据集模型的通用能力,只需要重新训练顶层的softmax分类器。因此，与无监督的预训练方法相比，这是一种有监督的预训练方法。Donahue同时也发现了convnet特征的泛化能力。