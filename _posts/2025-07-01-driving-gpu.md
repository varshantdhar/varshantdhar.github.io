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

Worth noting, the reducing order must be the same across all processes, otherwise, ```AllReduce``` contents might mismatch, resulting in incorrect reduction result. PyTorch dynamically builds the autograd graph in every forward pass, and different processes might not agree on the gradient ready order. Even if individual parameters’ grads arrive “out of order” during backprop, PyTorch holds off until the bucket is complete before issuing that bucket’s collective.

### Pseudocode

Algorithm 1: DistributedDataParallel
Input: Process rank r, bucket size cap c, local model net
```
  Function constructor(net):
     if r=0 then
        broadcast net states to other processes
     init buckets, allocate parameters to buckets in the
       reverse order of net.parameters()
     for p in net.parameters() do
        acc ← p.grad accumulator
        acc → add post hook(autograd hook)
```

Rank 0 holds the source of truth as it broadcasts every model parameter tensor (```p.data```) to all ranks so that each replica begins in the identical state. For ```net.parameters()``` it is walked through in reverse order so later layers' grads can be synced sooner and group them into contiguous buckets of size $\leq c$. For each parameter ```p```, grab its underlying gradient accumulator and register a post-accumulate-grad hook. That hook will fire immediately when ```p.grad``` is populated during ```backward()```.

```
  Function forward(inp):
    out = net(inp)
    traverse autograd graph from out and mark
       unused parameters as ready
    return out
```

No change in the forward pass. Then walk the autograd graph from the outputs to find which parameters participate in this iteration's backward (to handle conditionally-skiped layers). Any parameter not in the graph is "marked ready" so its bucket does not deadlock. 

```
  Function autograd hook(param index):
    get bucket bᵢ and bucket offset using param index
    get parameter var using param index
    view ← bᵢ.narrow(offset, var.size())
    view.copy(var.grad)
    if all grads in bᵢ are ready then
       mark bᵢ as ready
       launch AllReduce on ready buckets in order
    if all buckets are ready then
       block waiting for all AllReduce ops
```

Each hook knows which bucket $b_i$ its parameter belongs to and where in that bucket to write. It copies ```p.grad``` into the correct slice of a flattened bucket tensor.
A per-bucket counter tracks how many params have arrived. The bucket is "ready" once the last grad for $b_i$ is in. 
For every bucket that’s now ready—and in strictly increasing bucket index order—DDP issues an asynchronous All-Reduce (```dist.all_reduce(async_op=True)```) over that entire bucket tensor.
After the backward is fully done, DDP waits (```.wait()```) on any in-flight communication handles to ensure all reductions have completed before ```optimizer.step()``` runs.
After all All-Reduce operations finish, the reduced values already live in the bucket tensor. PyTorch then maps them back into each parameter’s ```.grad``` field before the optimizer update.