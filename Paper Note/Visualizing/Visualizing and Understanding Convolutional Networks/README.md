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

## 1. Introduction

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

### 1.1. Related Work

Visualizing features to gain intuition about the network is common practice, but mostly limited to the 1st layer where projections to pixel space are possible. In higher layers this is not the case, and there are limited methods for interpreting activity. (Erhan et al., 2009) find the optimal stimulus for each unit by performing gradient descent in image space to maximize the unit's activation. This requires a careful initialization and does not give any information about the unit's invariances. Motivated by the latter's short-coming, (Le et al., 2010) (extending an idea by (Berkes & Wiskott, 2006)) show how the Hessian of a given unit may be computed numerically around the optimal response, giving some insight into invariances. 

对特征可视化以获得对网络的直觉是常见的做法，但主要限于将首层映射到像素空间。在更高的层就不方便如此映射,解释的方法非常有限。Erhan通过在图像空间执行梯度下降法来寻找最优激励以令每个单元的激活最大化。这需要小心的初始化，也无法获得单元不变性的任何信息。从后者的缺陷出发，Le显示给定单元的Hessian可以通过围绕最优反应的数值计算得到，从中可以深入了解不变性。

The problem is that for higher layers, the invariances are extremely complex so are poorly captured by a simple quadratic approximation. Our approach, by contrast, provides a non-parametric view of invariance, showing which patterns from the training set activate the feature map. (Donahue et al., 2013) show visualizations that identify patches within a dataset that are responsible for strong activations at higher layers in the model. Our visualizations differ in that they are not just crops of input images, but rather top-down projections that reveal structures within each patch that stimulate a particular feature map.

问题是对于更高的层，不变性极其复杂，所以很难通过一个简单的二阶逼近来获得。相比之下，我们的方法对不变性提供了一个非参数化的视角，显示训练集中哪些模式激活了特征图。Donahue可视化可以识别数据集中的一部分，会导致模型高层产生强烈的激活。我们的可视化的不同之处在于,不仅是输入图像的裁切，还是自上而下的投影，以揭示每块之间的刺激特定特征图的结构。

## 2. Approach

We use standard fully supervised convnet models throughout the paper, as defined by (LeCun et al.,1989) and (Krizhevsky et al., 2012). 

论文中，我们使用LeCun和Krizhevsky定义的标准全监督convnet模型。

These models map a color 2D input image xi, via a series of layers, to a probability vector yi^ over the C different classes. Each layer consists of 

这些模型将一张彩色2D输入图片xi，通过一系列层，映射为一个共有C个类的概率向量yi^。每层包含

(i) convolution of the previous layer output (or, in the case of the 1st layer, the input image) with a set of learned filters; 
(ii) passing the responses through a rectified linear function (relu(x) = max(x; 0)); 
(iii) [optionally] max pooling over local neighborhoods and 
(iv) [optionally] a local contrast operation that normalizes the responses across feature maps. For more details of these operations, see (Krizhevsky et al., 2012) and (Jarrett et al., 2009). 

(i)   上层输出（对于首层是输入图片）的卷积，及一组学到的过滤器
(ii)  将响应通过ReLU
(iii) [可选] 在本地邻近做max pooling，切
(iv)  [可选] 本地对比操作，以归一化通过特征图的响应。

The top few layers of the network are conventional fully-connected networks and the final layer is a softmax classifier. Fig. 3 shows the model used in many of our experiments.

网络顶部的几层是传统的全连接网络，最末层是softmax分类器。

We train these models using a large set of N labeled images {x,y}, where label yi is a discrete variable indicating the true class. A cross-entropy loss function, suitable for image classification, is used to compare yi^ and yi. The parameters of the network (filters in the convolutional layers, weight matrices in the fully-connected layers and biases) are trained by backpropagating the derivative of the loss with respect to the parameters throughout the network, and updating the parameters via stochastic gradient descent. Full details of training are given in Section 3.

我们通过很大的一组标记图片训练模型，共N张图片，每张图片对应一组{x,y}，其中yi代表正确的分类。对于图片分类，适用交叉熵损失函数，用来比较得到的yi^和真正的yi。通过反向传播损失与网络参数的偏导数来训练网络模型的参数（包括卷积层的过滤器及全连接层的权重矩阵和偏置），再通过随机梯度下降来更新参数。

### 2.1. Visualization with a Deconvnet

Understanding the operation of a convnet requires interpreting the feature activity in intermediate layers. We present a novel way to map these activities back to the input pixel space, showing what input pattern originally caused a given activation in the feature maps.

理解convnet的操作，需要解释中间层的特征的活动。我们提出一种新的方法来将这些活动映射回输入的像素空间,以显示何种原始输入模式导致特征图中给定的激活。

We perform this mapping with a Deconvolutional Network (deconvnet) (Zeiler et al., 2011). A deconvnet can be thought of as a convnet model that uses the same components (filtering, pooling) but in reverse, so instead of mapping pixels to features does the opposite.

我们使用Zeiler的反卷积网络（deconvnet）来执行这个映射。deconvnet可以被认为是一个使用相同组件（过滤器和pooling）的convnet模型，只是反向而已，所以并非将像素映射到特征，而是相反。

In (Zeiler et al., 2011), deconvnets were proposed as a way of performing unsupervised learning. Here, they are not used in any learning capacity, just as a probe of an already trained convnet.

根据Zeiler，deconvnets曾被作为执行非监督学习的方式而提出。这里，他并非用在任何学习能力上，只是作为已训练好的convnet的探测器。

To examine a convnet, a deconvnet is attached to each of its layers, as illustrated in Fig. 1(top), providing a continuous path back to image pixels. 

要检查convnet，一个deconvnet与其每层都关联，如图1上部所示，提供一个连续的路径回到图像像素。

To start, an input image is presented to the convnet and features computed throughout the layers. To examine a given convnet activation, we set all other activations in the layer to zero and pass the feature maps as input to the attached deconvnet layer. Then we successively (i) unpool, (ii) rectify and (iii) filter to reconstruct the activity in the layer beneath that gave rise to the chosen activation. This is then repeated until input pixel space is reached.

开始，一个输入图像传给convnet，每层特征随之计算出来。要检查给定convnet的激活，我们将本层所有其他激活都设置为0,并将特征图作为输入传递给关联的deconvnet层。然后我们先后(i) unpool，(ii) rectify，(iii) filter，**to reconstruct the activity in the layer beneath that gave rise to the chosen activation.**。然后重复直到到达输入像素空间。

Unpooling: In the convnet, the max pooling operation is non-invertible, however we can obtain an approximate inverse by recording the locations of the maxima within each pooling region in a set of switch variables. In the deconvnet, the unpooling operation uses these switches to place the reconstructions from the layer above into appropriate locations, preserving the structure of the stimulus. See Fig. 1(bottom) for an illustration of the procedure.


Rectification: The convnet uses relu non-linearities, which rectify the feature maps thus ensuring the feature maps are always positive. To obtain valid feature reconstructions at each layer (which also should be positive), we pass the reconstructed signal through a relu non-linearity.


Filtering: The convnet uses learned filters to convolve the feature maps from the previous layer. To invert this, the deconvnet uses transposed versions of the same filters, but applied to the rectified maps, not the output of the layer beneath. In practice this means flipping each filter vertically and horizontally.


Projecting down from higher layers uses the switch settings generated by the max pooling in the convnet on the way up. As these switch settings are peculiar to a given input image, the reconstruction obtained from a single activation thus resembles a small piece of the original input image, with structures weighted according to their contribution toward to the feature activation. Since the model is trained discriminatively, they implicitly show which parts of the input image are discriminative. Note that these projections are not samples from the model, since there is no generative process involved.

## 3. Training Details