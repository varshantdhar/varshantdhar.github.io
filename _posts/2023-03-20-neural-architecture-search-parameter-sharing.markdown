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

In Neural architecture search (NAS), an RNN controller is trained in a loop: the controller first samples a candidate architecture, i.e. a child model, and then trains it to convergence to measure its performance on the task of desire. The controller then uses the performance as a guiding signal to find more promising architectures. This process is repeated for many iterations.

NAS is computationally expensive because of the training of each child model to convergence, only to measure its accuracy whilst throwing away all the trained weights. This paper, [Efficient Neural Architecture Search via Parameter Sharing](https://arxiv.org/pdf/1802.03268.pdf), is working to improve the efficiency of NAS by forcing all child models to share weights to eschew training each child model from scratch to convergence. The authors contend that they can represent NAS's search space using a single directed acyclic graph (DAG).

### Designing Recurrent Cells

ENAS’s controller is an RNN that decides: 1) which edges are activated and 2) which computations are performed at each node in the DAG. This allows ENAS to design both the topology and the operations in RNN cells, and hence is more flexible. 

To create a recurrent cell, the controller RNN samples $N$ blocks of decisions. Take for example $N = 4$ computational nodes with $x_t$ as the input signal for a recurrent cell and $h_{t-1}$ is the output from the previous time step. The sampling procedure follows:

1.

2.

3.

4.

5.