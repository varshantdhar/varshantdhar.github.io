---
title:  "Transformers: Review"
date:   2024-07-06
mathjax: true
categories:
    - blog
tags: 
    - transformers
    - self_attention
    - multi_head
    - positional_encodings
---

This post is a review of the Transformer architecture from the following blog [post](https://nlp.seas.harvard.edu/2018/04/03/attention.html#model-architecture).

## Transformer Architecture

The Transformer follows the architecture of a classic neural transduction model with an encoder-decoder structure, where the encoder maps an input sequence of symbol representations $(x_1, \cdots, x_n)$ to a sequence of continuous representations $z = (z_1, \cdots, z_n)$. Given $z$, the decoder then generates an output sequence $(y_1, \cdots, y_m)$ of symbols one element at a time. The Transformer follows this overall architecture using stacked self-attention and point-wise, fully connected layers for both the encoder and decoder. 

### Encoder

The encoder is a stack of $N = 6$ identical layers and we employ a residual connection around each of the two sub-layers, followed by layer normalization.

```{python}
class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
```

The output of each sub-layer is $\text{LayerNorm}(x + \text{Sublayer}(x))$, where $\text{Sublayer}(x)$ is the function implemented by the sub-layer itself. We also apply dropout to the output of each sub-layer, before it is added to the sub-layer input and normalized.

To facilitate these residual connections, all sub-layers in the model, as well as the embedding layers, produce outputs of dimension $d_{\text{model}} = 512$

```{python}
class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))
```
Each layer has two sub-layers. The first is a multi-head self-attention mechanism, and the second is a simple, position-wise fully connected feed-forward network.

```{python}
class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)
```

### Decoder

The decoder is also compose of a stack of $N = 6$ identical layers. In addition to the two sub-layers in each encoder layer, the decoder inserts a third sub-layer, which performs multi-head attention over the output of the encoder stack. Similar to the encoder, we employ residual connections around each of the sub-layers, followed by layer normalization.

```{python}
class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
 
    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)
```

We also modify the self-attention sub-layer in the decoder stack to prevent positions from attending to subsequent positions. This masking, combined with fact that the output embeddings are offset by one position, ensures that the predictions for position $i$ can depend only on the known outputs at positions less than $i$.

```{python}
def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0
```

### Attention

An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is computed as a weighted sum of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key.

Scaled Dot-Product Attention consists of queries and keys of dimension $d_k$ and values of dimension $d_v$, we compute the dot products of the query with all keys, divide each by $\sqrt{d_k}$ and apply a softmax function to obtain the weights on the values. 

$$ \text{Attention}(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$$

```{python}
def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn
```

Additive attention and dot-product attention are the two most commonly used attention functions. Dot product function is identical to the above without a scaling factor of $\frac{1}{\sqrt{d_k}}$. Assume that the components of $q$ and $k$ are independent random variables with mean $0$ and variance $1$. Then their dot product, $q \cdot k = \sum_{i=1}^{d_k} q_i k_i$  has mean $0$ and variance $d_k$ and thus for large values of $d_k$, the dot products grow large in maginitude, pushing the softmax function into regions where it has ectremely small gradients. 

Additive attention computes the compatibility function using a feed-forward network with a single hidden layer.

Multi-head attention jointly attends information from different representation subspaces at different positions.

$$ \text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \cdots, \text{head}_h)W^{O} $$
where $\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$ and where $W_i^Q \in \mathbb{R}^{d_{\text{model}} \times d_k}$, $W_i^K \in \mathbb{R}^{d_{\text{model}} \times d_k}$, $W_i^V \in \mathbb{R}^{d_{\text{model}} \times d_v}$ and $W^O \in \mathbb{R}^{d_{hd_v \times \text{model}}}$

## Embeddings and Softmax

Learned embeddings are used to convert the input tokens and output tokens to vectors of dimension $d_{\text{model}}$. The learned linear transformation and softmax function are used to convert the decoder output to predicted next-token probabilities. Often, we share the same weight matrix between the two embeddings layers and pre-softmax linear transformation, multiplying those weights by $\sqrt{d_{\text{model}}}$

```{python}
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)
```

### Positional Encodings: Using sinusoidal functions to inject the order of words in our model

Positional encodings adds information to each word about its position in the sentence. We can't use the $[0,1]$ range in which $0$ means the first word and $1$ means the last time-step as the time-step delta won't have consistent meaning across different sentences. 

If we assign a number to each time-step linearly, not only would the positional values get quite large but also the moel can face sentences longer than the ones in training -- thus our model may not see any sample with a speicifc length which would hurt generalization. 

Thus, the proposed technique employs a $d$-dimensional vector that is not integrated with the model itself. Let $t$ be the desired position in an input sentence, $\hat{p}_t \in \mathbb{R}^d$ be its corresponding encoding and $d$ be the encoding dimension, then $f: \mathbb{N} \rightarrow \mathbb{R}^d$ will be the function that produces the output vector $\hat{p}$ and it is defined as follows

$$ \hat{p}_t^{(i)} = f(t)^{(i)} := \begin{cases} \sin(\omega_k \cdot t) & \text{if } i = 2k \\ \cos(\omega_k \cdot t) & \text{if } i = 2k + 1 \end{cases} $$

where 

$$\omega_k = \frac{1}{10000^{\frac{2k}{d}}} $$

From the function definition, the frequencies are decreasing along the vector dimension. Thus, it forms a geometric progression from $2\pi$ to $10000 \cdot 2\pi$ on the wavelengths. This can also be visualised as

$$ \hat{p}_t = \begin{bmatrix}
\sin(\omega_1 \cdot t) \\ \cos(\omega_1 \cdot t) \\ \\
\sin(\omega_2 \cdot t) \\ \cos(\omega_2 \cdot t) \\
\vdots \\
\sin(\omega_{\frac{d}{2}} \cdot t) \\ \cos(\omega_{\frac{d}{2}} \cdot t) \\
\end{bmatrix}_{d \times 1} $$

where $d$ must be divisible by 2

## Notes

### Xavier (Glorot) Initialization

The Xavier initialization method, proposed by Xavier Glorot and Yoshua Bengio, aims to maintain the variance of activations and gradients across layers, which helps in preventing the issues of vanishing and exploding gradients. The method is designed to keep the scale of the gradients roughly the same in all layers.

The Xavier uniform initialization sets the weights according to the following formula:

$$ W \sim U(- \frac{\sqrt{6}}{\sqrt{n_{\text{in}}} + \sqrt{n_{\text{out}}}}, \frac{\sqrt{6}}{\sqrt{n_{\text{in}}} + \sqrt{n_{\text{out}}}}) $$

where $n_{\text{in}}$ is the number of input units and $n_{\text{out}}$ is the number of output units. 