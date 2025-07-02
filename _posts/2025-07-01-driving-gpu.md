---
title:  "PyTorch Distributed: Experiences on Accelerating Data Parallel Training"
date:   2025-07-01
mathjax: true
categories:
    - blog
tags: 
    - Distributed Training
    - GPU Communications
    - Parallelization
---

Summary of notes and deep dives from reading [PyTorch Distributed: Experiences on Accelerating Data Parallel Training](https://arxiv.org/pdf/2006.15704)

During the forward pass, PyTorch builds an autograd graph to record actions performed. Then, in the backward pass, it uses the autograd graph to conduct backpropagation to generate gradients. Finally, the optimizer applies the gradients to update parameters. The ```DistributedDataParallel``` module enables distributed training by communicating gradients before the optimizer step to make sure that parameters of all model replicas are updated using exactly the same set of gradients. 

Another popular technique is parameter averaging which instead of synchronizing gradients across multiple machines, directly computes the average across all model parameters. At each iteration, every GPU computes its own gradients $\triangledown L_i(\theta)$ on a disjoint mini-batch. Those gradients are all-reduced (summed and divided by the world size) so that every rank ends up with

$$ \hat{g} = \frac{1}{N} \sum_{i=1}^N \triangledown L_i (\theta) $$

Each process then 

 This comes with costs

- Parameter averaging produces vastly different results compared to local training especially when the optimizer relies on past local gradients values.
- As different model replicas see different gradients, the states in optimizers can gradually diverge, causing conflicting gradient descent directions.
- This structure of orchestration for computation and communication into non-overlapping phases limits scaling, calling ```optimizer.step()``` after both the backward work and gradient-all-reduce is done. Either the GPUs are crunching or the network is talking -- but almost never both at full speed. 

```AllReduce``` expects each participating process to provide an equally-sized tensor, collectively applies a given arithmetic operation (e.g, sum, prod, min, max) to input tensors from all processes, and returns the same result tensor to each participant. One ```AllReduce``` operation cannot start until all processes join, it is considered to be a synchronized communcation, as opposed to the P2P communication used in parameter servers. 

DDP guarantees that distributed data parallel training and local training are mathematically equivalent by making sure that all model replicas start from the exact same model state and see the same parameter gradients after every backward pass. The PyTorch autograd engine accepts custom backward hooks, a callback on a Tensor or Module that PyTorch will invoke at a particular point in the autograd lifecycle. DDP can register autograd hooks to trigger computation after every backward pass. When fired, each hook scans through all local model parameters and retrieves the gradient tensor from each parameter. 

If you naively all-reduced each parameter the moment its gradient arrived, you’d pay a big communication startup cost (latency) for every tiny tensor. Bucketing solves that by:

1. Grouping parameters into fixed-size "buckets"
2. Counting gradients, your hook incrememnts a counter when each parameter in that bucket has its grad ready.
3. Firing once per bucket: only when all grads in that bucket have been populated you issue a single ```dist.all_reduce(bucket_grads, async_op=True)``` over the contiguous block.

This turns N small all-reduces into $\lceil N / B \rceil$ all-reduces which makes overlapping with backward compute much more effective.