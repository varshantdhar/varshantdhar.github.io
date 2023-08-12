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