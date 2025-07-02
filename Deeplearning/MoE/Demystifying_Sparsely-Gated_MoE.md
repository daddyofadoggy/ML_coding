
# 🎯 Demystifying Sparsely-Gated Mixture-of-Experts (MoE) Layers

*From architecture to implementation, including math, code, and parallelism strategies.*

---

## Why Mixture of Experts?

Large neural networks are notoriously expensive to train. Traditional models activate **all parameters** for every input, limiting scalability under fixed compute budgets.

**Mixture of Experts (MoE)** architectures revolutionize this by activating **only a small subset** of model parameters per token, offering:

- Massive capacity  
- Constant computational cost  
- Conditional computation, where each input follows a **sparse path**  

---

## Visual Overview

<div style="text-align:center;">
  <img src="https://miro.medium.com/v2/resize:fit:720/format:webp/1*XU2IYuKex1BxFVaV2_G99Q.png" alt="MoE Visual Overview" width="70%" />
</div>

*Figure 1: Each token is routed to a subset of experts selected by a gating network.*

---

## The Structure of the Mixture-of-Experts Layer

An MoE layer contains:

- A **gating network** that selects the top-k experts per token  
- A pool of **experts**, each a small neural network  
- A **router** that sends tokens to the right experts and combines results  

---

## Core Equation

The MoE output for a token \( x \) is:

\[
\text{MoE}(x) = \sum_{i \in \text{TopK}(x)} G(x)_i \cdot E_i(x)
\]

Where:

- \( E_i(x) \): Output from the \( i^{th} \) expert  
- \( G(x)_i \): Gating weight (probability for expert \( i \))  

Only the top-k experts are active per token.

---

## Python Implementation

```python
import numpy as np

def moe(x: np.ndarray, We: np.ndarray, Wg: np.ndarray, n_experts: int, top_k: int) -> np.ndarray:
    n_batch, l_seq, d_model = x.shape
    x_flat = x.reshape(-1, d_model)  # Flatten batch and sequence dims
    logits = x_flat @ Wg  # Compute gating logits
    topk_idx = np.argpartition(-logits, top_k - 1, axis=1)[:, :top_k]  # Top-k expert indices
    topk_logits = np.take_along_axis(logits, topk_idx, axis=1)

    max_logits = np.max(topk_logits, axis=1, keepdims=True)
    exp_logits = np.exp(topk_logits - max_logits)  # For numerical stability
    gate_scores = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)  # Softmax gating scores

    output = np.zeros_like(x_flat, dtype=np.float64)

    for i in range(top_k):
        expert_idx = topk_idx[:, i]
        gate = gate_scores[:, i]
        for expert_id in range(n_experts):
            token_mask = (expert_idx == expert_id)
            if not np.any(token_mask): 
                continue
            x_selected = x_flat[token_mask]
            gate_selected = gate[token_mask][:, None]
            expert_output = x_selected @ We[expert_id]  # Expert forward pass
            output[token_mask] += gate_selected * expert_output  # Weighted sum

    return output.reshape(n_batch, l_seq, d_model)

# Example usage
x = np.arange(12).reshape(2, 3, 2)
We = np.ones((4, 2, 2))
Wg = np.ones((2, 4))
output = moe(x, We, Wg, n_experts=4, top_k=2)
print(output)
```

---

## Gating Network

The gating network computes expert weights for each token:

\[
H(x) = x W_g + \mathcal{N}(0,1) \cdot 	ext{Softplus}(x W_{noise})
\]

Only the top-k values are retained via masking, then normalized.

---

## Python Implementation of Noisy Top-k Gating

```python
def topk(H: np.ndarray, k: int) -> np.ndarray:
    H_topk = np.full_like(H, -np.inf)
    topk_indices = np.argpartition(-H, kth=k-1, axis=1)[:, :k]
    for i in range(H.shape[0]):
        H_topk[i, topk_indices[i]] = H[i, topk_indices[i]]
    return H_topk

def noisy_topk_gating(X, W_g, W_noise, N, k):
    H_base = X @ W_g
    H_noise = X @ W_noise
    softplus = np.log1p(np.exp(H_noise))
    H = H_base + N * softplus
    masked_H = topk(H, k)
    exp = np.exp(masked_H - np.max(masked_H, axis=1, keepdims=True))
    return exp / np.sum(exp, axis=1, keepdims=True)

# Example usage
X = np.array([[1.0, 2.0]])
W_g = np.array([[1.0, 0.0], [0.0, 1.0]])
W_noise = np.array([[0.5, 0.5], [0.5, 0.5]])
N = np.array([[1.0, -1.0]])
print(noisy_topk_gating(X, W_g, W_noise, N, k=2))
```

---

## Solving the Shrinking Batch Problem with Parallelism

### The Challenge

Tokens are sparsely routed to experts, so each expert sees only a small subset of the batch, causing:

- Shrinking micro-batches  
- Poor GPU utilization  
- Memory inefficiency  

---

### Data vs. Model Parallelism

**Data Parallelism**: Each GPU holds a full model copy and processes different batches.

```
GPU 0 — Model — Batch A  
GPU 1 — Model — Batch B  
```

**Model Parallelism**: Each GPU holds part of the model; a batch passes through GPUs sequentially.

```
GPU 0 → Layer 1  
GPU 1 → Layer 2  
GPU 2 → Layer 3  
GPU 3 → Layer 4  
```

---

### What’s Transferred?

- Activations (forward pass)  
- Gradients (backward pass)  

Requires high-speed interconnects (e.g., NVLink).

---

### Combined Strategy in MoE

- Data Parallelism replicates gating logic and shares load.  
- Model Parallelism distributes experts across devices.  
- Tokens are routed to GPUs based on expert assignment.

---

## Network Bandwidth & Computational Efficiency

Experts on different GPUs require sending token data over the network, which is slower than local computation.

Efficiency demands:

\[
\text{Compute-to-Communication Ratio} > 1000
\]

---

## Solution: Increase Hidden Size

Experts use 2-layer MLPs.

\[
\text{Compute per token} = h(d_{in} + d_{out})
\]

Communication cost:

\[
d_{in} + d_{out}
\]

Ratio:

\[
\frac{\text{Compute}}{\text{Comm}} = h
\]

Larger hidden size \(h\) improves GPU utilization efficiency.

---

## Experimental Results

| Task                       | Dataset                   | Baseline       | MoE Variant                  |
|----------------------------|---------------------------|----------------|-----------------------------|
| Language Modeling (LM)      | 1 Billion Word Benchmark  | 2-layer LSTM   | MoE-LSTM (top-2, 256 experts)|
| Neural Machine Translation  | WMT’14 English–German     | GNMT           | MoE-GNMT (top-2, 128 experts)|
| LM (Low Compute)            | Subset 1B Word            | Dense LSTM     | MoE (top-1, 64 experts)       |
| LM (Large Scale)            | 100 Billion tokens        | Dense LSTM     | MoE (top-2, 4096 experts)     |

---

## Key Findings

- Better performance at the same compute budget  
- Experts specialize automatically  
- State-of-the-art with lower training cost  
- Scales to billions of parameters  

---

## Final Thoughts

The Sparsely-Gated MoE Layer enables trillion-parameter models to train with the same cost as smaller dense models by combining:

- Conditional computation  
- Parallelism strategies  
- Bandwidth-aware design  

Sparse models are the future of efficient deep learning.

---

## References

- Shazeer, N. et al. (2017). *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer*. arXiv:1701.06538  
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)  
- [One Billion Word Benchmark](https://www.corpora.heliohost.org/1-billion-word)  
- [WMT’14 Dataset](http://www.statmt.org/wmt14/)
