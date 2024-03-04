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

$$h_{t+1} = h_t + f(h_t, \theta_t) $$


### Reverse-mode automatic differentiation of ODE solutions

Main technical difficulty in training continuous-depth networks is backpropagating through the ODE solver. Differentiating through the operations incurs a high memory cost and introduces numerical errors. 

The paper treats the ODE solver as a black box and computes gradients using the $\textit{adjoint sensitivity method}$ which computes gradients by solving a second, augmented ODE backwards in time. This is applicable to all ODE solvers and scales linearly with problem size, has low memory cost and explicitly controls for numerical errors. 

As an example, let's consider optimizing a scalar-valued loss function $L$, whose input is the result of an ODE solver:

$$ L(z(t_1)) = L(z(t_0) + \int_{t_0}^{t_1} f(z(t), t, \theta)dt) = L(\text{ODESolve}(z(t_0), f, t_0, t_1, \theta)) $$

To optimize $L$, gradients with respect to $\theta$ are required. The first step is to determinine how the gradient of the loss depends on the hidden state $z(t)$ at each instant. This quantity is called the $\textit{adjoint}$ $a(t) = \frac{\delta L}{\delta z(t)}$. Its dynamics are given by another ODE, which can be thought of as the instantaneous analog of the chain rule:

$$ \frac{da(t)}{dt} = -a(t)^{T}\frac{\delta f(z(t), t, \theta)}{\delta z} $$

The adjoint equation effectively links the changes in the adjoint variable $a$ to how changes in the state variables $z$ influence the loss function and the ODE dynamics. As the adjoint variable evolves backward in time, it carries information about how the loss changes based on changes in the state and the ODE behavior.

![alt]({{ site.url }}{{ site.baseurl }}/assets/images/neural-ode/reverse-mode-diff.png)

By solving this adjoint equation backward in time alongside the original ODEs, you're able to efficiently calculate how variations in the state variables $z$ contribute to changes in the loss function. This information is crucial for calculating the gradient of the loss with respect to the parameters $\theta$ and ultimately optimizing those parameters to minimize the loss.

To calculate $\frac{\delta L}{\delta z(t_0)}$, you follow these steps:

- Use the adjoint variable $a$ computed earlier, which evolves backward in time from $t_1$ to $t_0$.
- Run the same ODE solver backward in time, starting from $z(t_1)$ and using the adjoint variable $a$ to update $z(t)$ values. This provides you with the backward trajectory of $z(t)$.
- Along the way, compute how changes in $z(t)$ affect the loss function: $\frac{\delta L}{\delta z(t)}$, and accumulate this information as you move from $t_1$ to $t_0$.
- Finally, the value of $\frac{\delta L}{\delta z(t_0)}$ is obtained at the end of this backward computation.


Computing the gradients with respect to $\theta$ requires evaluating a third integral, which depends on $z$ and $a$

$$ \frac{dL}{d\theta} = - \int_{t_1}^{t_0} a(t)^T \frac{\delta f(z(t), t, \theta)}{\delta \theta}dt $$

The vector-Jacobian products $a(t)^T \frac{\delta f}{\delta z}$ and $a(t)^T\frac{\delta f}{\delta \theta}$ can be evaluated by automatic differentiation.  All integrals for solving $z$, $a$ and $\frac{\delta L}{\delta \theta}$ can be computed in a single call to an ODE solver, which concatenates the original state, the adjoint, and the other partial derivatives into a single vector.

### Continuous Normalizing Flows

Change of variables theorem to compute exact changes in probability if samples are transformed through a bijective funciton $f$:

$$ z_1 = f(z_0) \rightarrow \log p(z_1) = \log p(z_0) - \log | \det \frac{\delta f}{\delta z_0} |  $$

The bottleneck to using the change of variables formula is computing the determinant of the Jacobian which has a cubic cost in either the dimension of $z$ or the number of hidden units. 

Surprisingly, moving from a discrete set of layers to a continuous transformation simplifies the computation of the change in normalizing constant:

#### Thereom

Let $z(t)$ be a finite continuous random variable with probability $p(z(t))$ dependent on time. Let $\frac{dz}{dt} = f(z(t), t)$ be a differential equation describing a continous-in-time transformation of $z(t)$. Assuming that $f$ is uniformly Lipschitz continuous in $z$ and continuous in $t$, then the change in log probability also follows a differential equation,

$$ \frac{\delta \log p(z(t))}{\delta t} = -tr(\frac{df}{dz(t)}) $$

Proof based on Picard–Lindelöf theorem and Jacobi's formula. Requiring the trace operation instead of the log determinant simplifies the computation.

### A generative latent function time-series model 

Representing each time series by a laatent trajectory allows a continuous-time, generative approach to modeling time series. Each trajectory is determined from a local initial state, $z_{t_0}$, and a global set of latent dynamics shared across all time series. Given observation times $t_0, t_1, \cdots, t_N$ and an initial state $z_{t_0}$, an ODE solver produces $z_{t_1}, \cdots, z_{t_N}$ which describe the latent state at each observation. We define this generative model formally through a sampling procedure:

$$
z_{t_0} \sim p(z_{t_0}) \newline
z_{t_1}, z_{t_2}, \cdots z_{t_N} = \text{ODESolve}(z_{t_0}, f, \theta_f, t_0, \cdots, t_N) \newline
\text{each} \space x_{t_i} \sim p(x|z_{t_i}, \theta_x)
$$

Function $f$ is time-invariant function that takes the value $z$ at the current time step and outputs the gradient: $\frac{\delta z(t)}{\delta t} = f(z(t), \theta_f)$ which is parametrized using a neural net. Given $f$ is time-invariant, given any latent state $z(t)$, the entire trajectory is uniquely defined. 

### Notes ###

* When applied to a continuous transformation, Euler discretization breaks down the transformation into a series of small, discrete steps. This allows for the approximation of the continuous evolution of a system over time with a sequence of simple, calculable steps.

    * The hidden state of an RNN at each time step can be seen as the discretized state of a continuous transformation, with the network's weights determining the transformation at each step.

* Normalizing Flows: start with a standard Gaussian probability distribution $q_0(z)$ and apply a sequence of transformations to obtain a more complicated distribution $q_K(z)$. Each transformation is invertible and differentiable. 

    * Thus if $z_0 \sim q_0(z)$ is a sample and $z_k = f_{k}(z_{k-1})$ for $k = 1, \cdots, K$ the density of the final transformed variable $x = z_K$ can be obtained using the change of variables formula

    $$ q_K(x) = q_0(z_0) \prod^K_{k=1} | \det(\frac{\delta f_k}{\delta z_{k-1}}) |^{-1}  $$

    where $| \det(\frac{\delta f_k}{\delta z_{k-1}}) |^{-1}$ is the absolute value of the determinant of the Jacobian of $f_k$ with respect to $z_{k-1}$

    * Euler discretization relates to normalizing flows in that it can be used to numerically solve the differential equations governing the flow of probabilities in continuous space.

* A function $f(z)$ is said to be uniformly Lipschitz continuous if there exists a constant $L$ such that for any two points $z_1$ and $z_2$, the absolute difference between the function values at those points $|f(z_1) - f(z_2)|$ is bounded by the distance between the points $|z_1 - z_2|$ scaled by the Lipschitz constant $L$.