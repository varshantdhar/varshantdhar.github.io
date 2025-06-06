---
title:  "Stabilizing Transformers for Reinforcement Learning"
date:   2024-07-06
mathjax: true
categories:
    - blog
tags: 
    - transformers
    - reinforcement_learning
    - gated_transformer
    - attention
---

The blog post are notes from this [paper](https://arxiv.org/pdf/1910.06764v1).

## $1 \times 1$ Convolution Layers
When Transformers are employed in visual tasks of per-timestep observations in an RL environment, the multi-layer perceptron sublayer could become a temporal convolutional network instead of a fully connected linear network. Both $1 \times 1$ convolutional layers and fully connected layers perform a linear transformation on the input data. In a fully connected layer, this is done through a matrix multiplication, whereas in a $1 \times 1$ convolutional layer, it is done by applying a $1 \times 1$ filter to each spatial location independently.

There are 3 key differences between $1 \times 1$ convolutional layers and fully connected layers.

1. $1 \times 1$ convolutions preserve the spatial structure of the input feature map. They operate on each spatial position independently but use the same filter across the entire spatial dimensions. In contrast, a fully connected layer flattens the input, $1 \times 1$ the spatial arrangement of the features.

2. In $1 \times 1$ convolutions, the same set of weights (the $1 \times 1$ filter) is applied across all spatial locations in the input feature map, which means parameters are shared spatially. 

3. $1 \times 1$ convolutions are often used to change the depth (number of channels) of the feature map without affecting the spatial dimensions, effectively performing dimensionality reduction or expansion.


## Gated Transformer Architectures

Transformers have been shown to be more performant as a mechnism for attention over the input space than when used for temporal memory, like in multi-agent Starcraft 2 where the transformer was solely applied across the Starcraft units and not over time. Using gating mechanisms in place of residual connections within the transformer block could alleviate these difficulties. 

The use of "Identity Map Reordering" where layer normalization is moved onto the "skip" stream of the residual connection enables an identity map from the input of the transformer at the first layer to the output of the transformer after the last layer. In contrast, the canonical transformer, which consists of a series of layer normalization operations that non-linearly transform the state encoding. 

Identity Map Reordering allows for the agent to learn a Markovian policy at the start of training when the submodules at initialization produce values that are in expectation near zero. The state encoding is passed untransformed to the policy and value heads allowing for $ \pi(\cdot | s_t, \cdots, s_1) \approx \pi(\cdot | s_t) $ and $ V^{\pi} (s_t | s_{t-1}, \cdots, s_1) \approx V^{\pi} (s_t | s_{t-1}) $

Layer normalization would scale down the information flowing through the skip connection, forcing the model to rely on the residual path. Reactive behaviors need to be learned before memory-based ones can be effectively utilized, leading to instability in effectively estimating the value and policy functions using the original Transformer. 

### Identity Map Reordering

Changing the original Transformer to only place the layer normalization on the input stream of the submodules, allows for a path where two linear layers are applied in sequence. This applies a ReLU activation to each sub-module output before the residual connection.

![alt]({{ site.url }}{{ site.baseurl }}/assets/images/gated-transformer/trxl.png)


### Gating Layers

Further improvements in performance and opitmization stability can come from replacing the residual connections with gating layers, building a Gated Transformer (GTrXL).

The original transformer with Identity Map Reordering formulates as follows

$$ \bar{Y}^{(l)} = \text{RelativeMultiHeadAttention}(\text{LayerNorm}([\text{StopGrad}(M^{(l-1)}), E^{(l-1)}])) $$
$$ Y^{(l)} = E^{(l-1)} + \text{ReLU}(\bar{Y}^{(l)}) $$
$$ \bar{E}^{(l)} = f^{(l)}(\text{LayerNorm}(Y^{(l)})) $$
$$ E^{(l)} = Y^{(l)} + \text{ReLU}(\bar{E}^{(l)}) $$

With Gating layers this now looks like

$$ \bar{Y}^{(l)} = \text{RelativeMultiHeadAttention}(\text{LayerNorm}([\text{StopGrad}(M^{(l-1)}), E^{(l-1)}])) $$
$$ Y^{(l)} = g_{\text{MultiHeadedAttention}}^{(l)}(E^{(l)}, \text{ReLU}(\bar{Y}^{(l)})) $$
$$ \bar{E}^{(l)} = f^{(l)}(\text{LayerNorm}(Y^{(l)})) $$
$$ E^{(l)} = g_{\text{MLP}}^{(l)}(Y^{(l)}, \text{ReLU}(\bar{E}^{(l)})) $$

where $g$ is a gating layer function. A variety of gating layers can be employed with increasing expressivity:

- The gated input connection has a sigmoid modeulation on the input stream

$$ g^{(l)}(x,y) = \sigma(W_g^{(l)}x) \odot x + y $$

- The gated output connection has a sigmoid modulation on the output stream:

$$ g^{(l)}(x, y) = x + \sigma(W_g^{(l)}x - b_g^{(l)}) \odot y $$

- The highway connection modulates both streams with a sigmoid:

$$ g^{(l)}(x, y) = \sigma(W_g^{(l)}x + b_g^{(l)}) \odot x + (1 - \sigma(W_g^{(l)}x + b_g^{(l)})) \odot y $$

- The sigmoid-tanh gate is similar to the Output gate but with an additional tanh activation on the output stream:

$$ g^{(l)}(x, y) = x + \sigma(W_g^{(l)}y - b) \odot \tanh(U_g^{(l)}y) $$

- Gated-Recurrent-Unit type gating is a recurrent network that performs similarly to an LSTM but has fewer parameters. 

$$ r = \sigma(W_r^{(l)}y + U_r^{(l)}x), z = \sigma(W_z^{(l)} + U_z^{(l)}x - b_g^{(l)}) $$
$$ h = \tanh(W_g^{(l)}y + U_g^{(l)}(r \odot x)) $$
$$ g^{(l)}(x,y) = (1 - z) \odot x + z \odot h $$