---
title:  "Identity Mappings in Deep Residual Networks"
date:   2023-03-13
mathjax: true
categories:
    - blog
tags: 
    - resnet
    - skip_connections
    - pre_activation
---

The key idea of ResNets is to learn the additive residual function with a key choice of using an identity mapping. This is realized by attaching an identity skip connection. The original Residual Unit performs the following computation

$$ y_l = h(x_l) + F(x_l, W_l), $$
$$ x_{l+1} = f(y_l) $$

where $x_l$ is the input feature to the $l$ -th Residual Unit. 
$$ W_l = \{W_{l,k} |_{1 \leq k \leq K}\} $$ 

is a set of weights associated with the $l$ -th Residual Unit, and $K$ is the number of layers in a Residual Unit. $F$ denotes a stack of two $3 \times 3$ convolution layers as the residual function. The function $f$ is the ReLU operation after element-wise addition. The function $h$ is set as an identity mapping: $h(x_l) = x_l$. 

If $f$ is also an identity mapping $x_{l+1} \equiv y_l$ we could create a recursive operation where:

$$ x_L = x_l + \sum_{i=1}^{L-1} F(x_i, W_i) $$

for any deeper unit $L$ and any shallower unit $l$ indicating that the model is in a residual fashion between any units $L$ and $l$. Note that the feature 
$$ x_L = x_0 + \sum_{i=0}^{L-1} F(x_i, W_i) $$ 

of any deep unit $L$ is the summation of the outputs of all preceding residual functions - in contrast to a plain network which is a series of matrix-vector products. This summation also leads to nice backpropagation properties:

$$ \frac{\delta \epsilon}{\delta x_l} = \frac{\delta \epsilon}{\delta x_l} \frac{\delta x_L}{\delta x_l} = \frac{\delta \epsilon}{\delta x_l} (1 + \frac{\delta}{\delta x_l} \sum_{i=1}^{L-1} F(x_i, W_i) ) $$

where the first term $\frac{\delta \epsilon}{\delta x_L}$ that propagates information directly without concerning any weight layers and $\frac{\delta \epsilon}{\delta x_L} (\frac{\delta}{\delta x_l} \sum_{i=1}^{L-1} F)$ propagates through the weight layers. The additive nature ensures information is propagated back and that it is unlikely for the gradient $\frac{\delta \epsilon}{\delta x_l}$ to be cancled out for a mini-batch. 


## Identity Skip-Connections vs the Rest

![alt]({{ site.url }}{{ site.baseurl }}/assets/images/resnet-skip-connections/skip_connections.png)

## Experiments on Activation

![alt]({{ site.url }}{{ site.baseurl }}/assets/images/resnet-skip-connections/pre_activation.png)

![alt]({{ site.url }}{{ site.baseurl }}/assets/images/resnet-skip-connections/pre_activation_1.png)