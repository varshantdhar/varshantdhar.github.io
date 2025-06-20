---
title:  "LLM Fine-Tuning Mistral"
date:   2025-06-20
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

I'm [working on](https://github.com/varshantdhar/llm-finetuning-mistral) fine-tuning a quantized Mistral-7B-Instruct-v0.2 model on an instruction dataset using [Low-Rank Adaptation](https://arxiv.org/pdf/2106.09685)

### Training Process Overview

The training script implements a comprehensive fine-tuning pipeline for instruction-following tasks. Here's a detailed walkthrough of each step and the reasoning behind the parameter choices:

#### Dataset Selection and Loading

We use the Guanaco dataset, which contains 1,000 high-quality instruction-response pairs. This dataset is particularly well-suited for instruction fine-tuning because it provides diverse, well-formatted examples that teach the model to follow instructions effectively. The dataset is loaded directly from HuggingFace's datasets library, ensuring consistency and ease of access.

#### Prompt Formatting Strategy

Each instruction-response pair is formatted into a structured prompt using a specific template. The format includes clear section markers ("### Instruction:" and "### Response:") that help the model distinguish between the input instruction and the expected output. This structured approach is crucial for instruction-following models as it provides consistent formatting that the model can learn to recognize and respond to appropriately.

### Model and Tokenizer Setup

We load the Mistral-7B-Instruct-v0.2 model, which is specifically designed for instruction-following tasks. The tokenizer is configured with the fast tokenizer implementation for improved performance, and we set the padding token to match the end-of-sequence token for training compatibility.

#### Quantization Strategy

The model is loaded in 4-bit quantized mode using bitsandbytes, which dramatically reduces memory requirements from approximately 14GB to around 4GB. This quantization is essential for training on consumer hardware while maintaining reasonable performance. We use bfloat16 precision for better numerical stability compared to float16, especially important for training stability.

#### LoRA Configuration

Low-Rank Adaptation (LoRA) is implemented with carefully chosen parameters:
- **Rank (r=8)**: This determines the dimensionality of the low-rank matrices. A rank of 8 provides a good balance between model capacity and training efficiency, allowing the model to learn task-specific patterns without overwhelming computational requirements.
- **Alpha (32)**: The scaling factor for LoRA weights, set to 4 times the rank. This ratio is a common choice that provides stable training dynamics.
- **Dropout (0.1)**: A moderate dropout rate helps prevent overfitting while maintaining learning capacity.
- **Bias**: Set to "none" to avoid training bias terms, which typically don't benefit significantly from LoRA adaptation.

### Tokenization Process

Text is tokenized with a maximum sequence length of 512 tokens, which balances context window size with memory constraints. We use truncation for longer sequences and padding to ensure consistent batch sizes. The labels are set to match the input IDs for causal language modeling, where the model learns to predict the next token in the sequence.

### Training Configuration

The training parameters are optimized for efficient fine-tuning:
- **Batch Size**: Set to 1 per device with gradient accumulation over 8 steps, creating an effective batch size of 8. This approach allows training on limited GPU memory while maintaining stable gradient estimates.
- **Learning Rate**: 2e-4 provides a good balance between learning speed and stability for LoRA fine-tuning.
- **Epochs**: 3 epochs allow sufficient learning without overfitting on the 1,000-example dataset.
- **Mixed Precision**: FP16 training is enabled for faster computation and reduced memory usage.
- **Logging**: Metrics are logged every 10 steps and checkpoints saved every 100 steps, providing good monitoring without excessive overhead.

#### Data Collation

We use a language modeling data collator configured for causal language modeling (not masked language modeling). This collator handles dynamic padding within batches, ensuring efficient memory usage while maintaining proper training format.

#### Monitoring and Logging

The training process is integrated with Weights & Biases for experiment tracking, allowing real-time monitoring of loss curves, learning rates, and other training metrics. This integration is crucial for understanding training dynamics and debugging potential issues.

### Key Design Decisions

**Memory Efficiency**: Every component is designed for memory efficiency, from 4-bit quantization to small batch sizes with gradient accumulation. This allows training on consumer GPUs with 8-16GB VRAM.

**Stability**: The combination of bfloat16 precision, moderate learning rates, and appropriate LoRA parameters ensures stable training without gradient explosions or vanishing gradients.

**Scalability**: The LoRA approach means only a small fraction of parameters are trained, making it easy to experiment with different configurations and datasets without excessive computational costs.

**Reproducibility**: All random seeds and configurations are carefully managed to ensure reproducible results across different runs.

This training pipeline represents a production-ready approach to instruction fine-tuning that balances computational efficiency with model performance, making it accessible to researchers and practitioners with limited computational resources.