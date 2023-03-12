---
title:  "Learning to Forget by Expiring"
date:   2023-03-06
categories:
    - blog
tags: 
    - self-attention
    - model_capacity
    - memory
---

# Learning to Forget by Expiring

[Not All Memories are Created Equal: Learning to Forget by Expiring](https://scontent-lga3-2.xx.fbcdn.net/v/t39.8562-6/246880980_277856007537493_5262484961911076740_n.pdf?_nc_cat=107&ccb=1-7&_nc_sid=ad8a9d&_nc_ohc=T-lqzHYtp8sAX-Hw6Pl&_nc_ht=scontent-lga3-2.xx&oh=00_AfCNT8DhBraVmycrHms_RdmlwBM0RayZRTAwMM8Cp6bdnQ&oe=63EEAE6D)

## EXPIRE-SPAN

Learns when to expire unneeded memories, by expiring memories that are no longer useful and allow for scaling to tens of thousands of timesteps into the past. Self-attention is augmented with a simple predictor that outputs an expiration value for each hidden state that determines how long a memory should be retained and accessible to the model. Memories are forgotten in a gradual differentiable way to retain end-to-end training with backpropagation, which is done independently for each layer allowing different layers to specialize at different time-scales. EXPIRE-SPAN can flexibly adjust its span based on context.

Including all timesteps in self-attention results in a quadratic complexity to compute the full attention over a sequence of length T. 