---
title:  "Efficiently Modeling Long Sequences with Structured State Spaces"
date:   2023-07-10
mathjax: true
categories:
    - blog
tags: 
    - sequence_modeling
    - ssm
    - structured_state_space
---

RNNs, CNNs and Transformers struggle to scale to very long sequences of 10,000 or more steps. A promising recent approach towards modeling long sequences came by simulating the fundamental state space model (SSM) $$ x^{'}(t) = Ax(t) + Bu(t)$$ $$y(t) = Cx(t) + Du(t)$$ for appropriate choices of the state matrix $A$. However, this method has prohibitive computation and memory requirements.

The authors improved this model's computational efficiency by conditioning $A$ with a low-rank correction, allowing it to be diagonalized stably and reducing the SSM to the well-studied computation of a Cauchy kernel.

![alt]({{ site.url }}{{ site.baseurl }}/assets/images/ssm/ssm_comparison.png)

## State Spaces

### State Space Models: A Continuous-time Latent State Model

$$ x^{'}(t) = Ax(t) + Bu(t)$$ 
$$y(t) = Cx(t) + Du(t)$$

maps a 1-D input signal $u(t)$ to an N-D latent state $x(t)$ before projecting to a 1-D output signal $y(t)$. We can use this as a black-box representation in a deep sequence model, where $A, B, C, D$ are parameters learned by gradient descent. $Du$ can viewed as a skip connection and is easy to compute. 

### Addressing Long-Range Dependencies with HiPPO

The basic SSM can perform poorly in practice because linear first-order ODEs solve to an exponential function and thus may suffer from gradients scaling exponentially in the sequence length (leading to vanishing/exploding gradients). To address this, Linear State Space Layers leveraged the HiPPO theory of continuous-time meorization. HiPPO specifies a class of certain matrices $A \in \R^{N x N}$ that allows the state $x(t)$ to memorize the history of the input $u(t)$. 

$$ (\text{HiPPO Matrix}) \qquad A_{n,k} = - 
\begin{cases} 
    (2n+1)^{\frac{1}{2}}(2k+1)^{\frac{1}{2}} & \text{if } n > k \\
    n + 1 & \text{if } n = k \\
    0 & \text{if } n < k \\
\end{cases} $$

### Discrete-time SSM: The Recurrent Representation

To be applied on a discrete input sequence ($u_0, u_1, \cdots$) instead of continuous function $u(t)$ needs to discretize the equation by a step size $\Delta$ that represents the reolution of the input. Conceptually, the inputs $u_k$ can be viewed as sampling an implicit underlying continuous signal $u(t)$ where $u_k = u(k \Delta)$

To discretize the continuous-time SSM, the authors follow Tustin's method. The key idea behind Tustin's method is to approximate the time derivative in the continuous-time domain using a finite difference approximation in the discrete-time domain. Specifically, the bilinear method replaces the derivative term in a continuous-time transfer function with a difference equation involving the forward and backward differences in the discrete-time domain. This converts the state matrix $A$ into an approximation $\bar{A}$,

$$ x_k = \bar{A}x_{k-1} + \bar{B}u_k \quad \bar{A} = (I - \Delta/2 \cdot A)^{-1}(I + \Delta/2 \cdot A) $$

$$ y_k = \bar{C}x_k \qquad \bar{B} = (I - \Delta/2 \cdot A)^{-1} \Delta B \qquad \bar{C} = C $$

which makes it a $\textit{sequence-to-sequence}$ map $u_k \rightarrow y_k$ instead of function-to-funciton. The state equation is now a reccurrence in $x_k$, allowing the discrete SSM to be computed like an RNN, allowing $x_k \in \R^N$ to be viewed as a $\textit{hidden state}$ with transition matrix $\bar{A}$. 

### Training SSMs: The Convolutional Representation


