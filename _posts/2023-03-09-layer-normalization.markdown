---
title:  "Layer Normalization"
date:   2023-03-09
categories:
    - blog
tags: 
    - layer_norm
    - batch_norm
    - recurrent_nets
    - transformers
---

# Layer Normalization

[Layer Normalization](https://arxiv.org/pdf/1607.06450.pdf)

Unlike batch normalization, layer normalization directly
estimates the normalization statistics from the summed inputs to the neurons within a hidden layer
so the normalization does not introduce any new dependencies between training cases. 

Changes in the output of one layer will tend to cause highly correlated changes in the summed inputs to the next layer, especially with ReLU units whose outputs can change by a lot. This suggests the “covariate shift” problem can be reduced by fixing the mean and the variance of the summed inputs within each layer. Under layer normalization, all the hidden units in a layer share the same normalization terms $\mu$ and $\sigma$, but different training cases have different normalization terms. Unlike batch normalization, layer normaliztion does not impose any constraint on the size of a mini-batch and it can be used in
the pure online regime with batch size 1. 

$$ \mu^{l} = \frac{1}{H}\sum_{i=1}^H a_i^{l}$$
$$ \sigma^{l} = \sqrt{\frac{1}{H}\sum_{i=1}^H (a_i^l - \mu^l)^2} $$

It is common among the NLP tasks to have different sentence lengths for different training cases. This is easy to deal with in an RNN because the same weights are used at every time-step. But when we apply batch normalization to an RNN in the obvious way, we need to to compute and store separate statistics for each time step in a sequence. This is problematic if a test sequence is longer than any of the training sequences. Layer normalization does not have such problem because its normalization terms depend only on the summed inputs to a layer at the current time-step. It also has only one set of gain and bias parameters shared over all time-steps.

In a standard vanilla RNN the summed inputs in the recurrent layer are computed from the current input $x^t$ and previous vector of states $h^{t-1}$.  The layer normalized recurrent layer re-centers and re-scales its activations using the extra normalization
terms

$$h^t = f[\frac{g}{\sigma^t} \odot (a^t - \mu^t) + b]$$
$$ \mu^{l} = \frac{1}{H}\sum_{i=1}^H a_i^{l}$$
$$ \sigma^{l} = \sqrt{\frac{1}{H}\sum_{i=1}^H (a_i^l - \mu^l)^2} $$