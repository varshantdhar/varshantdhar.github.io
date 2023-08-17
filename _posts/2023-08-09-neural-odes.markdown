---
title:  "Neural Ordinary Differential Equations"
date:   2023-08-09
mathjax: true
categories:
    - blog
tags: 
    - ODE
    - continuous
    - neural_nets
---

Parametrize the continuous dynamics of hidden units using an ODE specified by a neural network:

$$ \frac{dh(t)}{dt} = f(h(t), t, \theta) $$

Starting from the input layer $h(0)$, we can define the output layer $h(T)$ to be the solution to this ODE initial value problem at some time $T$. 

### Reverse-mode automatic differentiation of ODE solutions

Main technical difficulty in training continuous-depth networks is backpropagating through the ODE solver. Differentiating through the operations incurs a high memory cost and introduces numerical errors. 

The paper treats the ODE solver as a black box and computes gradients using the $\textit{adjoint sensitivity method}$ which computes gradients by solving a second, augmented ODE backwards in time. This is applicable to all ODE solvers and scales linearly with problem size, has low memory cost and explicitly controls for numerical errors. 

As an example, let's consider optimizing a scalar-valued loss function $L()$, whose input is the result of an ODE solver:

$$ L(z(t_1)) = L(z(t_0) + \int_{t_0}^{t_1} f(z(t), t, \theta)dt) = L(\text{ODESolve}(z(t_0), f, t_0, t_1, \theta)) $$

To optimize $L$, gradients with respect to $\theta$ are required. The first step is to determinine how the gradient of the loss depends on the hidden state $z(t)$ at each instant. This quantity is called the $\textit{adjoint}$ $a(t) = \frac{\delta L}{\delta z(t)}$. Its dynamics are given by another ODE, which can be thought of as the instantaneous analog of the chain rule:

$$ \frac{da(t)}{dt} = -a(t)^{T}\frac{\delta f(z(t), t, \theta)}{\delta z} $$

The adjoint equation effectively links the changes in the adjoint variable $a$ to how changes in the state variables $z$ influence the loss function and the ODE dynamics. As the adjoint variable evolves backward in time, it carries information about how the loss changes based on changes in the state and the ODE behavior.

By solving this adjoint equation backward in time alongside the original ODEs, you're able to efficiently calculate how variations in the state variables $z$ contribute to changes in the loss function. This information is crucial for calculating the gradient of the loss with respect to the parameters $\theta$ and ultimately optimizing those parameters to minimize the loss.

To calculate $\frac{\delta L}{\delta z(t_0)}$, you follow these steps:

- Use the adjoint variable $a$ computed earlier, which evolves backward in time from $t_1$ to $t_0$.
- Run the same ODE solver backward in time, starting from $z(t_1)$ and using the adjoint variable $a$ to update $z(t)$ values. This provides you with the backward trajectory of $z(t)$.
- Along the way, compute how changes in $z(t)$ affect the loss function: $\frac{\delta L}{\delta z(t)}$, and accumulate this information as you move from $t_1$ to $t_0$.
- Finally, the value of $\frac{\delta L}{\delta z(t_0)}$ is obtained at the end of this backward computation.


Computing the gradients with respect to $\theta$ requires evaluating a third integral, which depends on $z$ and $a$

$$ \frac{dL}{d\theta} = - \int_{t_1}^{t_0} a(t)^T \frac{\delta f(z(t), t, \theta)}{\delta \theta}dt $$

The vector-Jacobian products $a(t)^T \frac{\delta f}{\delta z}$ and $a(t)^T\frac{\delta f}{\delta \theta}$ can be evaluated by automatic differentiation.  All integrals for solving $z$, $a$ and $\frac{\delta L}{\delta \theta}$ can be computed in a single call to an ODE solver, which concatenates the original state, the adjoint, and the other partial derivatives into a single vector.


