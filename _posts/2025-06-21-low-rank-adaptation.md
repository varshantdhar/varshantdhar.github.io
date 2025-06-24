---
title:  "Low-Rank Adaptation"
date:   2025-06-21
mathjax: true
categories:
    - blog
tags: 
    - LoRA
---

Low-Rank Adaptation (LoRA) introduces trainable low-rank matrices $A \in \mathbb{R}^{r \times d}$, $B \in \mathbb{R}^{d \times r}$ to approximate weight updates:

$$ \Delta W = BA $$

which are inserted into frozen weights $W_0$ as

$$W_{LoRA} = W_0 + \frac{\alpha}{2} \cdot BA $$

LoRA includes a fixed scaling multiplier $\alpha / r$, so the Adam Optimization updates the parameters in the low-rank matrices as:

$$\theta_t = \theta_{t-1} - \eta \cdot (\frac{\alpha}{r} \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon})$$

where:
- $\eta$ is the learning rate (outer optimizer)
- $\alpha / r$ is applied to the LoRA matrix output (after forward pass)
- $m_t$ is the first moment estimate
- $v_t$ is the second moment estimate

In Adam, let $g_t$ be the gradient of the loss with respect to a parameter at time step $t$.

$$m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g_t $$
$$v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot g_t^2 $$

with $\hat{m}_t = \frac{m_t}{1-\beta_1^t}$ and $\hat{v}_t = \frac{v_t}{1-\beta_2^t}$ as the bias-corrected moments.

Because $r << d$ the total trainable params in LoRA is a tiny fraction of the model when training just the low-rank adapters. Total trainable parameters drop by >99%, and you skip optimizer memory for frozen weights. Adam stores 3x copies (weights + 2 moments), so LoRA significantly cuts that overhead. 

The use of $\alpha / r$ is to normalize the magnitude of $\Delta Wx$ across different choices of rank $r$. The magnitude of $BAx$ grows with $r$ as the higher rank has higher capacity. To keep the perturbation consistent regardless of $r$, they introduce $\alpha / r$. $\alpha$ serves as a global gain knob and keeps updates well-behaved across LoRA configurations. 

In the transformer architecture, we have 4 attention weight matrices in the self-attention module ($W_q$, $W_k$, $W_v$, $W_o$) and two in the MLP module. The authors apply LoRA to $W_q$ and $W_v$ in most experiments for simplicity. 

### Limitations of LoRA

Limitations for LoRA lie in training different adapters $(A_i, B_i)$ for different tasks (e.g. summarization, translation, sentiment). At inference time, you might want to serve a batch that mixes tasks - each requiring a different LoRA adapter. 

Transformers expect the same weights across all examples in a batch, if each sample needs a different LoRA adapter - we can't easily do matrix multiplication across the whole batch. Thus we would need to split the batch or do custom masking.

To reduce latency, many implementations merge the LoRA weights into the base weights to avoid computing the LoRA path every inference step. This comes at the cost of losing flexibility as you can't switch between tasks mid-batch with being able to merge one adapter at a time. 
