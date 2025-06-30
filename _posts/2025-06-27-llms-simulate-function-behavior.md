---
title:  "How LLMs simulate function behavior internally"
date:   2025-06-27
mathjax: true
categories:
    - blog
tags: 
    - Function behavior
    - LLM
    - LangGraph
---

When interacting with large language models (LLMs), one of the most surprising and powerful behaviors is their ability to simulate agents, tools, and entire workflows using nothing but language. Ask a model to behave like a Python interpreter, a SQL assistant, or a friendly customer support agent, and it will often do so convincingly, adapting its behavior based on your prompt without ever updating its internal weights. But how is this possible?

It turns out that LLMs aren't just memorizing surface-level patterns — they are building what we can think of as internal simulators. Given enough context and structure, these models dynamically instantiate behavioral policies. They can model the function of a calculator, the reasoning process of a programmer, or even the conditional logic of a customer flow. These simulators are not static—they are constructed on-the-fly from prompts and context. This makes LLMs powerful engines for meta-learning: they learn how to learn and how to behave based on the task described in the input.

This capacity is rooted in the architecture of transformers. Transformers are fundamentally stacks of attention-driven layers that operate over sequences of tokens. Each layer in a transformer model transforms the representation of tokens based on what other tokens they attend to. This allows the model to build deep, contextual embeddings that encode relationships, logic, roles, and even latent execution flows.

Consider a prompt like:

```python
def multiply(a, b): 
    return a * b

multiply(3, 4)
```

To generate the next token, the model needs to simulate what this function does. It must attend to the function definition, recognize variable substitution, understand the semantics of *, and apply this to the arguments. No actual computation is happening — the model is predicting the next token, but it does so by simulating the behavior it has seen across millions of similar examples. Through layered composition, attention over input history, and pattern matching, the transformer approximates the behavior of an abstract interpreter.

This is why depth and scale matter. The deeper and wider the model, the better it can simulate longer or more complex behaviors across the layers of its architecture. Each layer acts like a virtual timestep in a program’s execution, unfolding a logical process vertically instead of sequentially in time. The architecture supports the emergence of these internal simulators — but it's the training objective and data diversity that give them substance.

Now, contrast this implicit simulation with what’s happening in structured agentic frameworks like LangGraph. Instead of relying on the model to do everything in one forward pass, LangGraph externalizes each step into a node in a computation graph. Each node can represent an LLM call, a validation tool, a database query, or a conditional router. These nodes pass around a mutable state — a dictionary that accumulates context, decisions, and outputs as the graph executes.

Where a transformer simulates an entire agent internally, LangGraph gives us control over each reasoning step. Need to select a database table based on a user question? That can be a separate node. Need to validate that the SQL actually matches the schema? Another node. If it fails, we don’t need the model to notice and retry inside one big prompt—we just route the graph back to the SQL generation step with new error context.

This structured externalization brings transparency and composability. You can log each decision, trace each state transition, and intervene precisely when something goes wrong. LangGraph turns the implicit logic inside LLMs into explicit flows that can be tested, audited, and improved over time.

So while transformers simulate agents internally through clever attention patterns and learned behavior, LangGraph scaffolds those behaviors externally. The future of AI systems lies in harnessing both: letting transformers simulate richly and flexibly, while using agentic frameworks to impose structure, safety, and step-by-step observability on top.