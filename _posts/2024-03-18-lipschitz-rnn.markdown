---
title:  "Lipschitz & Non-normal Recurrent Neural Networks"
date:   2024-03-18
mathjax: true
categories:
    - blog
tags: 
    - lipschitz
    - recurrent
    - neural networks
---

These papers aim to address the vanishing and exploding gradients problem in recurrent neural networks. Common solutions include restricting the hidden-to-hidden weight matrix to be an element of the orthogonal group. 

Typical layer weights and standard activation functions do not preserve norm of the signal or gradient during forward and back propagation. While good initializations can alleviate these issues, it does not guarantee that the norm of the signal or gradient does not exponentially decay across the layers after many iterations of training. Solutions thus comprise of using weight matrices that correspond to an orthogonal matrix, $Q^TQ = I$, followed by a norm-preserving nonlinear operation, such as ReLU. 

### Non-normal connectivity

Any diagonalizable matrix $V$ can be expressed as $V = P\theta P^{-1}$ where $P$'s columns are $V$'s eigenvectors and $\theta$ is a diagonal matrix containing its eigenvalues. $V$ is said to be normal if its eigenbasis is orthogonal and thus, $P^{-1} = P^T$ and $V = P \theta P^T$. Orthogonal matrices are normal matrices with eigenvalues on the unit circle. When a matrix is non-normal, it is diagonalized with a non-orthogonal basis. 

When a matrix is non-normal, it is diagonalized with
a non-orthogonal basis. However, it is still possible to express it using an orthogonal basis at the
cost of adding (lower) triangular structure to $\theta$. This is known as the Schur decomposition: for any matrix $V$, we have $V = P(\Lambda + T)P^T$ with $P$ an orthogonal matrix, $\Lambda$ a diagonal matrix containing the eigenvalues and $T$ a strictly lower-traingular matrix which contains the interactions between the orthogonal column vectors of $P$ (called $\textit{Schur nodes}$). $P$ and $T$ are obtained from orthogonalizing the non-orthogonal eigenbasis of $V$.  

As a recurrent connectivity matrix, $T$ represents purely feed-forward structure that produces strictly transient dynamics impossible to produce in normal (orthogonal) matrices. In other words, if a normal and non-normal matrix share exactly the same eigenspectrum, the iterative propagation of an input will be equivalent in the long-term, but can differ greatly in the short-term.

Now consider, generic RNN dynamics

$$ h_{t+1} = \phi(V h_t + U x_{t+1} + b) 
\newline 
V = P\theta P^T, \space \theta = \Lambda + T
$$

where $h_t \in \R^n$ $\phi$ is a nonlinear function. The Schur decomposition maps the hard problem of controlling the directions of a non-orthogonal basis to the easier problem of specifying interactions between fixed orthogonal modes. It is important to highlight the fact that an orthonormalization of the eigenbasis is just a change in representation and thus has no effect on the spectrum of $V$ which still lies on the diagonal of $\Lambda$. The lower-triangular matrix $T$ can thus be modified independently from the constraint that the spectrum have norms equal or near 1.

### Notes

* For a real matrix (where all elements are real numbers), the conjugate transpose is simply the same as the transpose since the complex conjugate of a real number is the number itself.

Restricting hidden-to-hidden weight matrices to be an element of an orthogonal group comprises of:

* Multiplying a vector by an orthogonal matrix preserves the length (or norm) of the vector. 
* Decorrelation via orthogonality ensures that the hidden units learn independent and meaningful features from the input data.
* Orthogonal matrices are more computationally efficient to work with compared to arbitrary matrices, as they have simpler properties and operations.

Enforcing orthogonality directly on the weight matrices might be too restrictive in practice, so techniques like orthogonal regularization or parameterization schemes are often used

**Eigenvalue Variance Guarantee Principle (EVGP)**: having control over the eigenvalues of weight matrices in a neural network can lead to more stable and efficient training dynamics. Specifically, controlling the spread of eigenvalues can help avoid issues like vanishing or exploding gradients during training.

**Unitary RNNs** are recurrent neural networks (RNNs) in which the recurrent weight matrices are constrained to be unitary matrices. A unitary matrix is a complex square matrix whose conjugate transpose is its inverse, meaning $U^*U=I$ where $U^*$ is the conjugate transpose of $U$. This is calculated by taking the transpose and then computing the complex conjugate of each element. 

* Skew-symmetric matrices are square matrices $A$ that satisfy the property $$A^T = -A$$
    * All diagonal elements are zero as they remain unchanged under transposition.
    * The determinant of a skew-symmetric matrix is always zero or a negative real number and so the eigenvalues of a skew-symmetric matrix are purely imaginary or zero.
    * The exponential of a skew-symmetric matrix yields an orthogonal matrix.

Given a skew-symmetric matrix $S$, the Cayley transform maps it to an orthogonal matrix using the following formula:

$$W = (I-S)^{-1}(I+S)$$
