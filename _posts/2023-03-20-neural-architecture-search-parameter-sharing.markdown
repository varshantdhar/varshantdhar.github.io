---
title:  "Efficient Neural Architecture Search via Parameter Sharing"
date:   2023-03-19
mathjax: true
categories:
    - blog
tags: 
    - enas
    - subgraph
    - search
    - controller
    - parameter_sharing
---

Efficient Neural Architecture Search (ENAS) is a proposed approach for automatic model design. A controller discovers neural network architectures by searching for an optimal subgraph within a large computational graph. Sharing parameters across child models allows ENAS to deliver strong performances.

In Neural architecture search (NAS), an RNN controller is trained in a loop: the controller first samples a candidate architecture, i.e. a child model, and then trains it to convergence to measure its performance on the task of desire. The controller then uses the performance as a guiding signal to find more promising architectures. This process is repeated for many iterations.

NAS is computationally expensive because of the training of each child model to convergence, only to measure its accuracy whilst throwing away all the trained weights. This paper, [Efficient Neural Architecture Search via Parameter Sharing](https://arxiv.org/pdf/1802.03268.pdf), is working to improve the efficiency of NAS by forcing all child models to share weights to eschew training each child model from scratch to convergence. The authors contend that they can represent NAS's search space using a single directed acyclic graph (DAG).

### Designing Recurrent Cells

![alt]({{ site.url }}{{ site.baseurl }}/assets/images/enas/build_rnn.png)

ENAS’s controller is an RNN that decides: 1) which edges are activated and 2) which computations are performed at each node in the DAG. This allows ENAS to design both the topology and the operations in RNN cells, and hence is more flexible. 

To create a recurrent cell, the controller RNN samples $N$ blocks of decisions. Taking an example of $N=4$ computational nodes with $x_t$ as the input signal for a recurrent cell and $h_{t-1}$ as the output from the previous time step:

1. At node 1, The controller samples an activation function, like $\tanh$ computing $h_1 = \tanh(x_t \cdot W^{(x)} + h_{t-1} \cdot W_1^{(h)})$ for node 1 of the recurrent cell. 

2. At node 2, the controller samples a previous index and an activation function. Suppose it samples previous index 1 and the activation function ReLU we get node 2 of the cell computes $h_2 = \text{ReLU}(h_1 \cdot W_{2,1}^{(h)})$

3. At node 3, the controller again samples a previous index and an activation function. Suppose it chooses the previous index 2 and activation function ReLU, then we get $h_3 = \text{ReLU}(h_2 \cdot W_{3,2}^{(h)})$.

4. At node 4, the controller again samples a previous index and an activation function. Suppose it chooses the previous index 1 and activation function $\tanh$, then we get $h_4 = \tanh(h_1 \cdot W_{4,1}^{(h)})$.

5. For the output, we average all the loose ends i.e. the nodes that are not selected as inputs to any other nodes. Since indices 3 and 4 were never sampled to be the input for any node, the recurrent cell uses their average $(h_3 + h_4)/2$ as its output. In other words, $h_t = (h_3 + h_4)/2$

Note that for each pair of nodes $j < l$, there is an independent parameter matrix $W_{l,j}^{(h)}$. The controller also decides which parameter matrices are used and so in ENAS, all recurrent cells in a search space share the same set of parameters. 


### Training ENAS and Deriving Architectures

ENAS has two sets of learnable parameters of the controller LSTM, denoted by $\theta$, and the shared parameters of the child models, denoted by $\omega$. The training procedure of ENAS consists of two interleaving phases. The first phase trains $\omega$, the shared parameters of the child models, on a whole pass through the training data set. The second phase trains $\theta$ for a fixed number of steps. These two phases are alternated during the training. 

In training the shared parameters $\omega$ of the child models, the authors fix the controller's policy $\pi(m; \theta)$ and perform stochastic gradient descent on $\omega$ to minimize the expected loss function $\mathbb{E}_{m \sim \pi}[L(m; \omega)]$. Here, $L(m; \omega)$ is the standard cross-entropy loss computed on a mini-batch with a model $m$ sampled from $\pi(m; \theta)$. The gradient is computed using the Monte Carlo estimate:

$$ \Delta_{\omega} \mathbb{E}_{m \sim \pi (m; \theta)}[L(m; \omega)] \approx\frac{1}{M}\sum_{i=1}^M \Delta_{\omega} L(m_i, \omega) $$

where $m_i$'s are sampled from $\pi(m; \theta)$ as described. This is an unbiased estimate with a higher variance where $m$ is fixed, however experiments show that updating $\omega$ using the gradient from any single model $m$ sampled from $\pi(m; \theta)$ can work well. 

To train the controller parameters $\theta$ fixing $\omega$ and updating the policy parameters $\theta$ the authors aim to maximize the expected reward $\mathbb{E}_{m\sim \pi(m; \theta)}[ R(m, \omega)]$. They employ the Adam optimizer for which the gradient is computed using [REINFORCE](https://people.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf) with a moving average baseline to reduce variance. 

Note, the reward $R(m,\omega)$ is computed on the validation set rather than the training set, encouraging ENAS to select models that generalize well rather than models that overfit the training set well. Architectures are thus derived from a trained ENAS model by first sampling several models from the trained policy $\pi(m, \theta)$ and computing the reward on a single minibatch for each sampled model, from the validation set. Then, taking the model with the highest reward to re-train from scratch. 

### Designing Convolutional Cells

In the search space for convolutional models, the controller RNN samples two sets of decisions at each decision block: 1) what previous nodes to connect to and 2) what computation operation to use. These decisions contruct a layer in the model. The decision of what previous nodes to connect to allows the model to form skip connections. The decision of what computation operation to use sets a particular layer into convolution or pooling blocks (average or max). 

An example run of the controller for a search space over convolutional cells is shown

![alt]({{ site.url }}{{ site.baseurl }}/assets/images/enas/build_conv.png)

A reduction cell can also be realized from the search space by sampling a computational graph from the search space and applying all operations with a stride of 2. 