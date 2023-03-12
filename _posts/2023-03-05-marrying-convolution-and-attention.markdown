---
title:  "Marrying Convolution and Attention"
date:   2023-03-05
categories:
    - blog
tags: 
    - convolution
    - self-attention
    - mobilenet
---
# ConvAttention
## Marrying Convolution and Attention

[CoAtNet: Marrying Convolution and Attention for All Data Sizes](https://proceedings.neurips.cc/paper/2021/file/20568692db622456cc42a2e853ca21f8-Paper.pdf)

## Summary

While Transformers tend to have larger model capacity, their generalization can be worse than convuktional networks due to the lack of the right inductive bias. The paper above proposes a hybrid model that assumes:

- depthwise convolution and self-attention can be naturally unified via simple relative attention
- vertically stacking convolution layers and attention layers is effective at improving generalization, capacity and efficiency

## Model

### Depthwise Separable Convolutions

Depthwise Separable Convolutions replace a full convolution operatior with a factorized version that splits convolution into two separate layers: 

- The first layer is called a depthwise convolution -- performs lightweight filtering by applying a single convolutional filter per input channel.
- The second layer is a 1x1 convolution called a pointwise convolution, computes linear combinations of the input channels. 

Depthwise Separable Convolutions reduces computation compared to traditional layers.

### Linear Bottlenecks

Deep neural networks have layered activation tensors which can be treated as containers of $h \times w$ "pixels" with $d$ dimensions. When we look at all individual pixels of a deep convolutional layer, the information encoded in those values actually lie in some manifold, which is embaddable into a low-dimensional subspace. [MobileNetV1](https://arxiv.org/pdf/1704.04861.pdf) reduces the layer dimension via a width multiplier parameter which proportionally decreases the number of filters in each layer by a value. However, because the ReLU operator employed in convolutional networks can lead to a non-zero volume layer transformation, this limits the power of the deep networks -- which on these parts operates like a linear classifier.

On the other hand, when ReLU collapses the channel, it inevitably loses information but if we have lots of channels, there is a structure to preserve information in other channels. Thus, if the input manifold lies in a low-dimensional subspace of the input space, ReLU is capable of preserving complete information about the input manifold. So, inserting linear bottleneck layers into the convolutional blocks can be used optimize the existing neural architecture. 

### Inverted Residuals

Bottleneck blocks appear similar to residual blocks where each block contains an input followed by several bottlenecks then followed by expansion, with the addition of shortcuts directly between the bottlenecks. This allows the ability of a gradient to propagate across multiplier layers. 

### Transformer

The Transformer employs an encoder-decoder structure consisting of stacked encoder and decoder layers. The encoder consists of two sublayers: self-attention followed by a position-wise feed-forward layers. Decoder layers consist of three sublayers: self-attention followed by encoder-decoder attention, followed by position-wise feed-forward layer. It uses residual connections around each of the sublayers, followed by layer normalization. The decoder uses masking in its self-attition to prevent a given output position from incorporating information about future positions during training. 

Sinusoids of varying frequency are added to encoder and decoder inputs prior to the first layer as position encodings. This helps the model to generalize to unseen sequence lengths during training, allowing to learn to attend also by relative position. Residual connections help propagate position information to higher layers. 

### Self-attention

Self-attention sublayers employ attention heads, where results from each head are contenated and a parameterized linear transformation is applied. Each head operates on an input sequence and computes a new output sequence of same length. An element of the output sequence is computed as a weighted sum of linearly trasformed input elements using a softmax and scaled dot product compatibility function. Linear transformations of the inputs add sufficient expressivity. 

### Relation-aware Self-attention

Modeling the input as a labeled, directed, fully-connected graph, the edge between input elements is represented by vectors which are then used in the calculation of output elements and the compatibility function between two input elements. These edge representations can be shared across attention heads. 

For linear sequences, edges can capture information about the relative position differences between input elements. The maximum relative position is clipped  at a value, enabling the model to generalize to sequence lengths not seen during training and decidedly expressing that the relative position information is not useful beyond a certain distance. 

### Merging Convolution and Self-Attention

