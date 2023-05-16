---
title:  "Learning to Forget by Expiring"
date:   2023-03-06
mathjax: true
categories:
    - blog
tags: 
    - self-attention
    - model_capacity
    - memory
---

Learns when to expire unneeded memories, by expiring memories that are no longer useful and allow for scaling to tens of thousands of timesteps into the past. Self-attention is augmented with a simple predictor that outputs an expiration value for each hidden state that determines how long a memory should be retained and accessible to the model. Memories are forgotten in a gradual differentiable way to retain end-to-end training with backpropagation, which is done independently for each layer allowing different layers to specialize at different time-scales. EXPIRE-SPAN can flexibly adjust its span based on context.


[Not All Memories are Created Equal: Learning to Forget by Expiring](https://arxiv.org/pdf/2105.06548.pdf)

### Method

Taking the set $C_t = \{1, \cdots, t-1\}$ to indicate which memories can be accessed at time $t$ describes the space and time complexity of self-attention which is linearly correlated to the size of this set $|C_t|$. The paper's goal is to reduce the size of $C_t$ for more efficiency without performance degradation. Thus, for each memory $$ h_i \in \mathbb{R}^d $$, they compute a scalar EXPIRE-SPAN $e_i \in [0, L]$ where 

$$ e_i = L\sigma(w^Th_i + b) $$

where $w \in \mathbb{R}^d$ and $b \in \mathbb{R}$ represent trainable parameters and $\sigma$ is the sigmoid function with $L$ the maximum span. This expire-span $e_i$ determines how long $h_i$ should be kept and included in $C_t$. At time $t$, the remaining span of $h_i$ is $r_{ti} = e_i - (t-i)$ where $i$ represents the time step when the memory $h_i$ was originally stored. When $r_{ti}$ becomes negative, it indicates the memory $h_i$ is expired and can be removed from $C_t$.

This can be implemented by updating attention weights $a_{ti}$ with a binary masking function $m_{ti} = 1_{r_{ti} > 0}$

$$ a^{'}_{ti} = \frac{m_{ti} a_{ti}}{\sum_j m_{tj}a_{tj}} $$

$$ o_t = \sum_{i} a^{'}_{ti}v_i $$

However, this discrete masking means the Expire-Span $e_i$ will not receive any gradient for training. The binary masking is considered a non-differentiable operation, and as a result, gradients cannot be directly propagated through the binary mask during backpropagation. The discrete nature of the masking, where certain memories are entirely discarded based on expiration, prevents the gradient flow through those masked-out memories. Instead they deploy a soft masking function

$$m_{ti} = \max(0, \min(1, 1+ r_{ti}/R)) $$

where $R$ is a parameter, bounded between $0$ and $1$, that determines the scaling factor for the masking function. It controls the rate at which the weights of memories decrease as their remaining span decreases. $1+ r_{ti}/R$ computes a ratio that scales with the remaining span. As the remaining span increases, this term becomes larger. The $\min(1, 1+ r_{ti}/R)$ operation ensures that the ratio is bounded between 0 and 1. If the ratio exceeds 1, it is clamped to 1, indicating that the memory is fully attended to. The $\max(0, \min(1, 1+ r_{ti}/R))$ operation further ensures that the final weight $m_{ti}$ is bounded between 0 and 1. If the remaining span is negative, indicating that the memory has expired, the weight is set to 0, indicating that the memory is ignored.

This function has gradient

$$
\frac{{\partial m_{ti}}}{{\partial r_{ti}}} = 
\left\{
\begin{array}{ll}
0, & \text{if } 1+ \frac{{r_{ti}}}{{R}} > 1 \\
\frac{1}{R}, & \text{if } 1+ \frac{{r_{ti}}}{{R}} \leq 1 \\
\end{array}
\right.
$$


and thus has non-zero gradient for values in $[-R,0]$ to train $e_i$, but also can take a value of $0$ which is necessary for expiring memories. Thus $$C_t = \{i : m_{ti} > 0 \}$$ Since $m_{ti}$ is a monotonically decreasing function $t$, once a memory is expired, it can be permanently deleted. 

The ultimate goal is to reduce the average memory size, which directly relates with the average EXPIRE-SPAN:

$$ \frac{1}{T}\sum_t |C_t| = \frac{1}{T}\sum_t \sum_{i<t}1_{m_{ti} > 0} $$
$$ = \frac{1}{T} \sum_i (R + \sum_{t < i} 1_{rti > 0}) $$
$$ = \frac{1}{T} \sum_i (R + \sum_{t < i} 1_{e_i > t-i}) $$
$$ = R - 1 + \frac{1}{T} \sum_i \lfloor{e_i}\rfloor $$


