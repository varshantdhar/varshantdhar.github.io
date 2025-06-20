---
title:  "Numerical optimization for data science and machine learning"
date:   2025-06-12
mathjax: true
categories:
    - blog
tags: 
    - RAG
    - LangChain
    - LLM
    - Prompt
    - LangGraph
---

Computers store tensors in physical memory as contiguous blocks. Row-major ordering means elements within a row are adjacent. This layout affects performance:

1. Row operations are fast - elements are contiguous in memory, maximizing cache usage
2. Column operations are slower - elements are separated by row stride, causing cache misses
3. Matrix multiplication performance depends on access patterns and memory hierarchy

As an example

```python
# Fast: accessing one day's readings (row)
day_readings = week_temps[0]  # Contiguous memory access

# Slower: accessing one time across days (column)
morning_temps = week_temps[:, 0]  # Strided memory access

# Matrix multiply organizes computation to maximize cache usage
result = torch.mm(week_temps, weights.view(-1, 1))
```

Understanding memory layout helps choose efficient operations:

- Prefer row operations when possible
- Batch similar operations to maximize cache usage
- Consider transposing matrices to align access patterns with memory layout

#### Matrix Multiplication

For matrices $A$ and $B$: 

$$ c_{ij} = \sum_{k=1}^n a_{ik}b_{kj} $$

For matrix-vector multiplication ($Ax = b$): $b_i = \sum_{j=1}^n a_{ij}x_j$

This operation is fundamental because:

- Each output combines an entire row with a column
- The operation preserves linear relationships
- Computation parallelizes efficiently

#### Broadcasting

Broadcasting generalizes operations between tensors of different shapes. It extends the mathematical concept of scalar multiplication to more general shape-compatible operations. For a vector $v \in \R^n$ and matrix $A \in \R^{m \times n}$: 

$$ (A * v)_{ij} = a_{ij} * v_j $$

This operation implictly replicates the vector across rows, but without copying memory. The computational advantages are significant:

- Memory efficient: No need to materialize the replicated tensor
- Cache friendly: Access pattern matches memory layout
- Parallelizable: Each output element computed independently

Broadcasting rules follow mathematical intuition:

1. Trailing dimensions must match exactly
2. Missing dimensions are implicitly size 1
3. Size 1 dimensions stretch to match larger dimensions

This enables concise, efficient code:
```python
# Temperature adjustments
base = torch.tensor([[22.5, 23.1, 21.8],    # Base readings
                    [21.0, 22.5, 20.9]])
offset = torch.tensor([0.5, 0.0, -0.5])     # Per-time adjustments
scale = torch.tensor([1.02, 0.98]).view(-1, 1)  # Per-day scaling
# Multiple broadcasts in one expression
adjusted = scale * (base + offset)  # Combines both adjustments
```