---
title:  "Marrying Convolution and Attention"
date:   2023-03-05
mathjax: true
categories:
    - blog
tags: 
    - convolution
    - self-attention
    - mobilenet
---
While Transformers tend to have larger model capacity, their generalization can be worse than convolutional networks due to the lack of the right inductive bias. The paper above proposes a hybrid model that assumes:

- depthwise convolution and self-attention can be naturally unified via simple relative attention
- vertically stacking convolution layers and attention layers is effective at improving generalization, capacity and efficiency

[CoAtNet: Marrying Convolution and Attention for All Data Sizes](https://proceedings.neurips.cc/paper/2021/file/20568692db622456cc42a2e853ca21f8-Paper.pdf)

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

Note that both the feed-forward network module in a Transformer and mobilenet convolution blocks employ the design of "inverted bottleneck", which first expans the channel size of the input by 4x and later projects the 4x-wide hidden state back to the original channel size to enable residual connections. 

Also, both depthwise convolution and self-attention can be expressed as a per-dimension weighted sum of values in a pre-defined receptive field. Specifically, convolution relies on a fixed kernel to gather information from a local receptive field

$$ y_i = \sum_{j \in L(i)} w_{i-j} \odot x_j $$

where $x_i, y_i \in \mathbb{R}^D $ are the input and output at position $i$ respectively, and $L(i)$ denotes a local neighborhood of $i$, e.e a 3x3 grid centered at $i$ in image processing.

In comparison, self-attention allows the receptive field to be the entire spatial locations and computes the weights based on the re-normalized pairwise similarity between the pair $(x_i, x_j)$:

$$ y_i = \sum_{j \in G} \frac{\exp(x_i^T x_j)}{\sum_{k \in G} \exp(x_i^Tx_k)} x_j $$

where

$$ A_{i,j} = \sum_{k \in G} \exp(x_i^Tx_k)  $$

and $G$ indicates the global spatial space. Note that

* The depthwise convolution kernel $w_{i-j}$ is an input-independent parameter of static value, while the attention weight $A_{i,j}$ dynamically depends on the representation of the input. Hence, it is much easier for the self-attention to capture complicated relational interactions between different spatial positions. However, the flexibility comes with a risk of easier overfitting, especially when data is limited.

* Given any position pair $(i, j)$, the corresponding convolution weight $w_{i-j}$ only cares about the relative shift between them, i.e. $i  -j$, rather than the specific values of $i$ or $j$. This property is often referred to as translation equivalence, which has been found to improve generalization under datasets of limited size. 

* Generally speaking, a larger receptive field provides more contextual information, which could lead to higher model capacity. Hence, the global receptive field has been a key motivation to employ self-attention in vision. However, a large receptive field requires significantly
more computation.

To aptly combine these desirable properties we could simply sum a global static convolution kerenel with the adaptive attention matrix, either after or before the Softmax normalization i.e.,

$$y_i^{\text{post}} = \sum_{j \in G} (\frac{\exp(x_i^Tx_j)}{\sum_{k \in G} \exp(x_i^Tx_j)} + w_{i-j})x_j $$

or 

$$ y_i^{\text{pre}} = \sum_{j \in G}\frac{\exp(x_i^Tx_j + w_{i-j})}{\sum_{k \in G}\exp(x_i^Tx_k + w_{i-k})}x_j $$



Notice that the pre-normalization version corresponds to variants of relative self-attention. Thus the attention weight $A_{i,j}$ is decided jointly by the $w_{i-j}$ of translation equivariance and the input-adaptive $x_i^Tx_j$, which can enjoy both effects depending on their relative magnitudes. 

## Vertical Layout Design

The global context has a quadratic complexity w.r.t. the spatial size. Hence, if we directly apply the relative attention to the raw image input, the computation will
be excessively slow due to the large number of pixels in any image of common sizes. Thus the authors of the paper perform some down-sampling to reduce the spatial size and employ the global relative attention after the feature map reaches manageable level. 

The down-sampling can be achieved by either (1) a convolution stem with aggressive stride (e.g., stride 16x16) as in ViT or (2) a multi-stage network with gradual pooling as in ConvNets. 