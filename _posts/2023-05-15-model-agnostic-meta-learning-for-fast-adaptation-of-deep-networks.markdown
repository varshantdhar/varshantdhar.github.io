---
title:  "Model-Agnostic Meta-Learning"
date:   2023-05-15
mathjax: true
categories:
    - blog
tags: 
    - meta-learning
    - model-agnostic
    - few-shot learning
---

A meta-learning algorithm that is general and model-agnostics can be directly applied to any learning problem and model that is trained with a gradient descent procedure. With a focus on deep neural networks, this [paper](https://arxiv.org/pdf/1703.03400.pdf)'s approach is meant to flexibly handle different architectures and problem settings including classification, regression and policy gradient reinforcement learning. The author's aim is to produce good results on a new task from training a model's parameters on a few or single gradient step. 

The primary contribution to this objective is a simple model and task-agnostic algorithm for meta-learning that trains a model's parameters such that a small number of gradient updates will lead to fast learning on a new task.

## Algorithm: Model Agnostic Meta Learning

MAML follows a two-step training process:

1. In this step, the model's initial parameters are updated based on their performance on a set of validation tasks. The validation loss is used to assess the model's performance, and gradient descent is performed to update the parameters. The updated parameters form a new initialization that is more amenable to adaptation.$$ \theta' = \theta - \alpha \nabla_{\theta} \mathcal{L}_{\text{val}}(f_{\theta}) $$ Here, $\theta$ represents the initial model parameters, $\alpha$ is the step size for the outer loop update, and $\mathcal{L}_{\text{val}}(f_{\theta})$ is the validation loss of the model $f_{\theta}$.

2. The model's parameters are then fine-tuned on specific tasks within the training set using gradient descent. This step involves adapting the parameters to minimize the training loss on each task. The objective is to find a good set of parameters that can be easily adjusted to new tasks during the adaptation process.$$ \theta_i' = \theta - \beta \nabla_{\theta} \mathcal{L}_{\text{train}}(f_{\theta_i}) $$ Here, $\theta_i$ represents the model parameters after $i$ inner loop updates, $\beta$ is the step size for the inner loop update, and $\mathcal{L}_{\text{train}}(f_{\theta_i})$ is the training loss of the model $f_{\theta_i}$.

Note that the meta-optimization is performed over the model parameters $\theta$, whereas the objective is computed using the updated model parameters $\theta'$. Thus, the proposed method aims to optimize the model parameters such that one or a small number of gradient steps on a new task will produce maximally effective behavior on that task.

## Implicit Gradients

In the MAML fornulation, meta-parameters are learned in the outer loop, while task-specific models are learned in the inner-loop, by using only a small amount of data from the current task. A key challenge in scaling these approaches is the need to differentiate through the inner loop learning process, which can impose considerable computational and memory burdens. Here are some approaches to reduce the inner-loop optimization bottleneck with implicit differentiation techniques:

### Implicit MAML (I-MAML)

I-MAML extends the original MAML algorithm by computing the gradients implicitly. Instead of explicitly unrolling the adaptation process, I-MAML uses implicit differentiation to compute the gradients of the adapted parameters. 

#### Implicit Jacobian Computation in I-MAML

In the I-MAML (Implicit MAML) algorithm, the Jacobian matrix, which represents the derivatives of the adapted parameters with respect to the initial parameters, is efficiently computed using the $\delta$–approximate Jacobian-vector product. This method allows for the implicit calculation of the Jacobian without explicitly unrolling and differentiating each adaptation step.

The $\delta$–approximate Jacobian-vector product can be formulated as follows:

1. Select a random vector $\delta$.
2. Compute the inner product of $\delta$ with the gradients of the loss function at different adaptation steps, resulting in an approximation of the Jacobian-vector product.

Mathematically, this can be expressed as:

$$
 \mathbf{J} \cdot \mathbf{v} \approx \frac{\mathcal{L}(\theta - \alpha \delta \odot \nabla_{\theta} \mathcal{L}(\theta)) - \mathcal{L}(\theta)}{\alpha} \cdot \delta 
$$

where:
- $ \mathbf{J} $ represents the Jacobian matrix,
- $ \mathbf{v} $ is the vector with which we compute the Jacobian-vector product,
- $ \delta $ is a randomly chosen perturbation vector,
- $ \odot $ denotes the element-wise product,
- $ \mathcal{L}(\theta) $ is the loss function.

This approach significantly reduces computational and memory overhead, making I-MAML suitable for meta-learning scenarios with limited resources.  By approximating the Jacobian using random perturbations and the inner product with the gradients, I-MAML avoids the need for explicit unrolling and differentiation of adaptation steps.


### Implicit MAML with Natural Gradient (I-MAML-NG)

I-MAML-NG further improves upon I-MAML by incorporating the natural gradient into the implicit differentiation process. The natural gradient considers the geometry of the parameter space and provides more effective updates. The update equation for I-MAML-NG is similar to I-MAML but involves the natural gradient computation:

$$
\theta' = \theta - \alpha \mathbf{F}^{-1} \nabla_{\theta} \mathcal{L}_{\text{val}}(f_{\theta})
$$

Here, $\mathbf{F}$ represents the Fisher information matrix, which captures the curvature of the loss landscape. This is calcuated as:

$$ \mathbf{F} = \mathcal{I}(\theta) = \mathbb{E}_{\mathcal{T} \sim p(\mathcal{T})} \left[ \nabla_{\theta} \mathcal{L}_{\mathcal{T}}(\theta) \nabla_{\theta} \mathcal{L}_{\mathcal{T}}(\theta)^T \right]$$

where 

$$
\begin{align*}
\mathcal{I}(\theta) & : \text{Fisher Information matrix} \\
\mathbb{E} & : \text{Expectation operator} \\
\mathcal{T} & : \text{Task} \\
p(\mathcal{T}) & : \text{Distribution of tasks} \\
\nabla_{\theta} & : \text{Gradient operator with respect to model parameters} \\
\mathcal{L}_{\mathcal{T}}(\theta) & : \text{Loss function of task } \mathcal{T} \\
\theta & : \text{Model parameters} \\
\end{align*}
$$

In the context of MAML (Model-Agnostic Meta-Learning), the Fisher Information matrix represents the second-order derivative of the loss function with respect to the model parameters. It quantifies the local curvature of the loss landscape and provides information about the sensitivity of the loss function to variations in the parameters.


### Implicit MAML with Hessian-Free (I-MAML-HF)

I-MAML-HF improves upon I-MAML by utilizing a Hessian-Free approach. It approximates the Hessian matrix-vector product to efficiently compute the natural gradient. This approximation avoids the computationally expensive calculation of the Hessian matrix. The update equation for I-MAML-HF is similar to I-MAML-NG but involves the Hessian-Free approximation:

$$
\theta' = \theta - \alpha \mathbf{H}^{-1} \nabla_{\theta} \mathcal{L}_{\text{val}}(f_{\theta})
$$

Here, $\mathbf{H}$ represents the approximate Hessian matrix. The Hessian matrix, which represents the second-order derivatives of the loss function, is approximated by computing the Hessian-vector product, similar to $\delta$-approximating the Jacobian. 

$$
\mathbf{H} \cdot \mathbf{v} \approx \frac{\nabla_{\theta} \mathcal{L}(\theta + \beta \mathbf{v}) - \nabla_{\theta} \mathcal{L}(\theta)}{\beta}
$$

where

- $H$ represents the Hessian matrix,
- $v$ is the vector with which we compute the Hessian-vector product,
- $\Delta_{\theta} L(\theta)$ is the gradient of the loss function with respect to the model parameters
- $\beta$ is a small scalar value.


