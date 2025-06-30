# FlexAttention API Usage Notebook

This notebook demonstrates the usage of the new FlexAttention API, which allows users to specify modifications to the computed attention scores in Scaled Dot Product Attention (SDPA).

## Table of Contents
1. [Introduction](#introduction)
2. [Setup](#setup)
3. [Basic Usage](#basic-usage)
4. [Score Modification vs Score Masking](#Score-Modification-vs-Score-Masking)
4. [Score Modification Examples](#score-modification-examples)
   - [Full Attention (No-op)](#full-attention-no-op)
   - [Standard Causal Mask](#standard-causal-mask)
   - [Sliding Window Attention](#sliding-window-attention)
   - [Prefix LM (Bidirectional + Causal)](#prefix-lm-bidirectional--causal)
   - [Document Masking](#document-masking)
   - [NATTEN Masking](#natten-masking)
   - [Alibi Bias](#alibi-bias)
   - [Tanh Soft-Capping](#tanh-soft-capping)
   - [Nested Jagged Tensor](#nested-jagged-tensor)
   - [Flamingo Cross Attention](#flamingo-cross-attention)

## Introduction

The FlexAttention API allows users to specify custom modifications to attention scores within a Fused Scaled Dot Product Attention Kernel. This enables various attention patterns and biases to be implemented efficiently, with potential runtime and memory savings. The API will also generate fused backward kernels based off of the user defined modification.

## Setup

First, let's import the necessary libraries and set up our environment.


```python
import random
from functools import lru_cache, partial

import torch
import torch.nn.functional as F

from tabulate import tabulate
from torch.nn.attention.flex_attention import (
    _DEFAULT_SPARSE_BLOCK_SIZE,
    create_block_mask,
    create_mask,
    flex_attention,
)
from triton.testing import do_bench

torch.set_default_device("cuda")
torch.manual_seed(0)

torch._dynamo.config.cache_size_limit = 1000

# Compile the flex_attention function
flex_attention = torch.compile(flex_attention, dynamic=False)

# For better performance, you can use:
# flex_attention = torch.compile(_flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs")

data_type = torch.float16

# The kernels will utilize block sparisty to increase performance
print(f"Using the default sparsity block size: {_DEFAULT_SPARSE_BLOCK_SIZE}")
```

We will define some helpful testing utilities that will print a block sparse representation of the score_mod function and mask_fn. 

As well, it will compare the performance between 
- FlexAttention 
- One of the SOTA implementation FlashAttentionV2 with Causal masking.
- nn.F.scaled_dot_product_attention + fully materialized attn_mask. This will dispatch to a fused implementation `EFFICIENT_ATTENTION` that allows for arbitrary masking. 


```python
@lru_cache
def create_block_mask_cached(score_mod, B, H, M, N, device="cuda"):
    block_mask = create_block_mask(score_mod, B, H, M, N, device=device)
    return block_mask


def calculate_tflops(flops: float, time_ms: float, multiplier: int) -> float:
    return multiplier * flops * (1e3 / time_ms) / 1e12


def test_mask(
    score_mod=None,
    mask_mod=None,
    B=16,
    H=16,
    S=8192,
    D=64,
    skip_correctness=False,
    print_mask=True,
):
    assert (
        score_mod is not None or mask_mod is not None
    ), "Must provide a score_mod or mask_mod"
    query = torch.randn(
        B, H, S, D, device="cuda", dtype=torch.float16, requires_grad=True
    )
    key = torch.randn(
        B, H, S, D, device="cuda", dtype=torch.float16, requires_grad=True
    )
    value = torch.randn(
        B, H, S, D, device="cuda", dtype=torch.float16, requires_grad=True
    )
    gradOut = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)

    if mask_mod is not None:
        block_mask = create_block_mask_cached(mask_mod, 1, 1, S, S, device=query.device)
    else:
        block_mask = None
    sdpa_mask_fn = mask_mod if mask_mod is not None else score_mod
    mask = create_mask(sdpa_mask_fn, 1, 1, S, S, device=query.device)

    causal_fa2 = lambda: F.scaled_dot_product_attention(
        query, key, value, is_causal=True
    )
    xformers_mask = lambda: F.scaled_dot_product_attention(
        query, key, value, attn_mask=mask
    )
    flex_attention_call = lambda: flex_attention(
        query, key, value, score_mod=score_mod, block_mask=block_mask
    )

    results = []
    if block_mask is not None:
        density = (100 - block_mask.sparsity()) / 100
    else:
        density = 1.0
    causal_fav2_flops = 0.5 * B * H * D * S * S
    flops = density * B * H * D * S * S

    # Forward pass
    causal_fa2_time = do_bench(causal_fa2)
    xformers_mask_time = do_bench(xformers_mask)
    flex_ms = do_bench(flex_attention_call)

    # Backward pass
    causal_fa2_out = causal_fa2()
    xformers_out = xformers_mask()
    flex_out = flex_attention_call()

    causal_fa2_bw_time = do_bench(
        lambda: causal_fa2_out.backward(gradOut, retain_graph=True)
    )
    xformers_mask_bw_time = do_bench(
        lambda: xformers_out.backward(gradOut, retain_graph=True)
    )
    flex_bw_ms = do_bench(lambda: flex_out.backward(gradOut, retain_graph=True))

    # Inline correctness check
    if not skip_correctness:
        xformers_outs = []
        flex_outs = []

        query.grad = None
        key.grad = None
        value.grad = None

        out1 = xformers_mask()
        xformers_outs.append(out1)
        out1.backward(gradOut)
        xformers_outs += [query.grad, key.grad, value.grad]

        query.grad = None
        key.grad = None
        value.grad = None

        out2 = flex_attention_call()
        flex_outs.append(out2)
        out2.backward(gradOut)
        flex_outs += [query.grad, key.grad, value.grad]
        for flex, xformer in zip(flex_outs, xformers_outs):
            torch.testing.assert_close(flex, xformer, atol=1e-1, rtol=1e-2)

        print("Correctness check passed ✅")
    # Usage in your results formatting:
    results = [
        [
            "causal FA2",
            f"{causal_fa2_time:.4f}",
            f"{calculate_tflops(causal_fav2_flops, causal_fa2_time, 4):.2f}",
            f"{causal_fa2_bw_time:.4f}",
            f"{calculate_tflops(causal_fav2_flops, causal_fa2_bw_time, 10):.2f}",
        ],
        [
            "F.sdpa + mask",
            f"{xformers_mask_time:.4f}",
            f"{calculate_tflops(flops, xformers_mask_time, 4):.2f}",
            f"{xformers_mask_bw_time:.4f}",
            f"{calculate_tflops(flops, xformers_mask_bw_time, 10):.2f}",
        ],
        [
            "flexattention",
            f"{flex_ms:.4f}",
            f"{calculate_tflops(flops, flex_ms, 4):.2f}",
            f"{flex_bw_ms:.4f}",
            f"{calculate_tflops(flops, flex_bw_ms, 10):.2f}",
        ],
    ]
    print(
        f"\nResults for {score_mod.__name__ if score_mod is not None else mask_mod.__name__}:"
    )
    print(
        tabulate(
            results,
            headers=[
                "Operation",
                "FW Time (ms)",
                "FW FLOPS (TF/s)",
                "BW Time (ms)",
                "BW FLOPS (TF/s)",
            ],
            tablefmt="grid",
        )
    )
    if print_mask:
        print(f"\nBlock Mask:\n{block_mask}")

    # Clean up to save memory
    del query, key, value, gradOut, causal_fa2_out, xformers_out, flex_out
    torch.cuda.empty_cache()
```

## Basic Usage

Here's a basic example of how to use the FlexAttention API:


```python
def checkerboard(score, batch, head, token_q, token_kv):
    score = torch.where(torch.abs(token_kv - token_q) % 2 == 1, score * 0.5, score)
    score = torch.where(torch.abs(token_kv - token_q) % 2 == 0, score * 2.0, score)
    return score


# Create input tensors
query = torch.randn(8, 8, 2048, 64, device="cuda", dtype=torch.float32)
key = torch.randn(8, 8, 2048, 64, device="cuda", dtype=torch.float32)
value = torch.randn(8, 8, 2048, 64, device="cuda", dtype=torch.float32)

# Call flex_attention with the checkerboard score modification
output = flex_attention(query, key, value, score_mod=checkerboard)

# Compile and run
compiled_flex_attention = torch.compile(flex_attention)
out_compiled = compiled_flex_attention(query, key, value, score_mod=checkerboard)

# Check if the results are close
torch.testing.assert_close(output, out_compiled, atol=2e-2, rtol=2e-2)
```

## Score Modification vs Score Masking
We are going to take a brief aside to describe two key concepts that will be important to understand for getting the maximum performance benefits for FlexAttenion.
The full api for flex_attention is:
```python
flex_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    score_mod: Optional[Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    block_mask: Optional[torch.nn.attention.flex_attention.BlockMask] = None,
    scale: Optional[float] = None,
)
```
You may be wondering why we need both a 'score_mod' and a 'block_mask'.
1. score_mod functions should be used when you want to mutate score values in the attention weight matrix.
2. mask_mod functions should be used when you want to mask scores in the attention weight matrix that are independent of the score value and only rely on positional information.

Note: Any block_mask could also be represented with a score_mod, however the performance of the kernel will be suboptimal

### Lets walk through causal attention to highlight the differences.

The implementation using a score_mod:
```Python
def causal_bias(score, b, h, q_idx, kv_idx):
    return torch.where(q_idx >= kv_idx, score, -float("inf"))
```

Whenever you are writing a score_mod function that passes through the original score for some elements and sets others to -inf, you should likely be using a mask mod.


The implementation using as a mask_mod:
```Python
The implementation using a mask_mod:
def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx
```
As you can see they look very similar, both return scalar tensors. The key differences
1. mask_mods return boolean tensors where `True` indicates this score should be calculated, and `False` indicates we that we want to mask out this score
2. mask_mods do not take a `score` argument since they are not allowed to depend on actual values during the calculation.


### What happens when I use a score_mod + a mask_mod?
The score_mod function will be applied to every un-masked element.

### I have a mask mod function, how do I create a BlockMask?
Great question reader! Besides flex_attention we provide 1 other main API.
```python
create_block_mask(
    mask_mod (Callable): mask_mod function.
    B (int): Batch size.
    H (int): Number of heads.
    Q_LEN (int): Sequence length of query.
    KV_LEN (int): Sequence length of key/value.
    device (str): Device to run the mask creation on.
    KV_BLOCK_SIZE (int): Block size of block mask for each query.
    Q_BLOCK_SIZE (int): Block size of block mask for each key/value.
    _compile (bool): Whether to compile the mask creation.
)
```

So for the above example the call to flex_attention that would be the most performant is:
``` python
causal_block_mask = create_block_mask(causal_mask, B, H, M, N)
flex_attention(query, key, value, block_mask = causal_block_mask)
```
B,H,Q_LEN,KV_LEN are the batch_size, num_heads, query_sequence_length, and key_sequence_length.

### Why have both?
Purely for performance. Causal masking is in fact very sparse. Only the lower triangular portion of the attention scores matter. Without generating a BlockMask we would be doing twice the work needed!
Below we will compare the performance difference between the two implementations.



## Score Modification Examples

Let's explore various score modification examples that can be used with the FlexAttention API. 

Legend:
We are going to be printing a representation of the sparsity found for these score_mod + mask_fns. 

*  The absence of any block means that it is completely masked out and is not actually needed to compute the final attended output
* `██` This block computes full attention between all query and key tokens
* `░░` This block is partially masked out, some query tokens attend to some key tokens but some are masked to -inf

### Full Attention
Applies a "no-op" score mod. Leaving the attention scores unchanged.


```python
def noop(score, b, h, q_idx, kv_idx):
    return score


test_mask(noop, print_mask=True)
```

### Standard Causal Masking
Standard Causal Masking is a key technique in autoregressive language models that ensures each token can only attend to itself and previous tokens in the sequence. The block sparse representation shows the lower triangular nature of this mask.

See [Score Modification vs Score Masking](#Score-Modification-vs-Score-Masking) for more details on these implementations


```python
def causal_bias(score, b, h, q_idx, kv_idx):
    return torch.where(q_idx >= kv_idx, score, -float("inf"))


test_mask(score_mod=causal_bias)


def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


test_mask(mask_mod=causal_mask)
```

### Sliding Window Attention
The [Mistral paper](https://arxiv.org/abs/2310.06825) has a very nice visual of this bias and describes it. In essence you define a fixed size "SLIDING_WINDOW" and for autogressive decoding you only allow `torch.abs(q_tokens - kv_tokens) < SLIDING_WINDOW` to attend to each other. Typically this is also combined with causal attention. We are going to do this through a a nice pattern, mask composition. Typically masking can can conceptually be done in pieces and then composed together.

We are going to write two mask_functions 1 for doing `causal-masking`, and one for doing `windowed-attention` and compose them together to produce the final mask_fn. As we know from earlier, mask_fns return boolean values where a value of `True` indicates that the element should take part in attention.



```python
SLIDING_WINDOW = 1024


def sliding_window_causal_mask(b, h, q_idx, kv_idx):
    causal_mask = q_idx >= kv_idx
    windowed_mask = (
        q_idx - kv_idx <= SLIDING_WINDOW
    )  # We dont need to check the right side of the sliding window since we are applying the causal mask

    return causal_mask & windowed_mask


test_mask(mask_mod=sliding_window_causal_mask)
```

### Prefix LM (Bidirectional + Causal)
This T5 achitecture [papers with code](https://paperswithcode.com/method/t5) describes an attention variant that performs prefix attention. Where a certain number of `prefix` tokens are allowed to full attend and then all subsequent tokens perform causal attention. We again compose two mask functions to accomplish this, one for causal masking and one that is based off of the prefix length.



```python
PREFIX_LENGTH = 2048


def prefix_lm_causal_mask(b, h, q_idx, kv_idx):
    prefix_mask = kv_idx <= PREFIX_LENGTH
    causal_mask = q_idx >= kv_idx
    return prefix_mask | causal_mask


test_mask(mask_mod=prefix_lm_causal_mask)
```

### Document Masking
Imagine that we have multiple documents of different lengths. We want to mask
out the attention between documents, but allow attention between tokens within
the same document. We can do this by using a document_id tensor that gives the
document that each token belongs to. Then, we can mask out all attention
scores where the document_id[q_idx] differs from document_id[kv_idx]


Note: We *only* need to compile a new kernel when the `score_mod` changes
(it'll automatically detect that using torch.compile infra). This example code
is implemented with caching BlockMask, but in general, changing BlockMask
*does not* require a recompile.
That is, for document masking, we only need to compute a new BlockMask when
the document lengths change, *not* a new kernel.


```python
document_id = torch.zeros(32768, dtype=torch.int, device="cuda")
document_id[:4096] = 0
document_id[4096:8192] = 1
for i in range(8192, 32768, 8192):
    document_id[i : i + 8192] = i // 8192 + 1


def document_causal_mask(b, h, q_idx, kv_idx):
    causal_mask = q_idx >= kv_idx
    document_mask = document_id[q_idx] == document_id[kv_idx]
    return causal_mask & document_mask


test_mask(mask_mod=document_causal_mask, S=32768)
```

### Stand-Alone Self-Attention Masking

In this case, imagine that we have a 2D image of size (H x W) flattened into a
sequence of tokens. We only want to attend to tokens within 8 `pixels`, but
from a 2D perspective.

We can implement this mask_mod by first translating the 1D position into 2D coordinates. Then, we can simply check if the distance of both coordinates is within the window.

For more details check the paper, [Stand-Alone Self-Attention in Vision Models](https://arxiv.org/abs/1906.05909)


```python
H = 128
W = 128
WINDOW = 8


def get_x_y(idx):
    return idx // W, idx % W


def sasa_mask(b, h, q_idx, kv_idx):
    q_x, q_y = get_x_y(q_idx)
    kv_x, kv_y = get_x_y(kv_idx)
    horizontal_mask = (q_x - kv_x).abs() <= WINDOW
    vertical_mask = (q_y - kv_y).abs() <= WINDOW
    return horizontal_mask & vertical_mask


test_mask(mask_mod=sasa_mask)
```

### NATTEN Masking

Consider a 2D image of size (H x W) flattened into a sequence of tokens.
Queries attend to keys in a fixed kernel area (K_H x K_W), centered where possible
on the query, whilst staying within the canvas and always including the query.

This is similar to SASA, except with extra handling to keep the kernel inside the canvas,
ensuring that all queries attend to a fixed number of keys.  
Keys compare their position to the kernel center, not the query. The kernel center attempts
to follow the query position, but is clamped to stay a fixed distance (its half-length) away
from the canvas edge.

See the [NATTEN repository](https://github.com/SHI-Labs/NATTEN) for more information.  
_Note: a more complete implementation of NATTEN would include support for kernel dilation._  
_The NATTEN unfused kernel also has features like the ability to cross-attend to register tokens._
_This capability is possible to express in Flex Attention but not attempted here._


```python
H = 128
W = 128
K_H = 13
K_W = 13


def get_x_y(idx):
    return idx // W, idx % W


def natten_mask(
    b,
    h,
    q_idx,
    kv_idx,
):
    q_x, q_y = get_x_y(q_idx)
    kv_x, kv_y = get_x_y(kv_idx)
    # kernel nominally attempts to center itself on the query, but kernel center
    # is clamped to a fixed distance (kernel half-length) from the canvas edge
    kernel_x = q_x.clamp(K_W // 2, (W - 1) - K_W // 2)
    kernel_y = q_y.clamp(K_H // 2, (H - 1) - K_H // 2)
    hori_mask = (kernel_x - kv_x).abs() <= K_W // 2
    vert_mask = (kernel_y - kv_y).abs() <= K_H // 2
    return hori_mask & vert_mask


test_mask(mask_mod=natten_mask, S=H * W)
```

### Tiled NATTEN layout
The solution above unrolls 2-D Q and KV into 1-D attention problem in a naive column major way. This breaks the locality of the very sparse Q K V layout: While the density of the MATTEN mask is `(13 * 13) / (128 * 128) = 1.0%`, the density of our block mask becomes 10.16% with 128x128 blocks. Q K V layouts with that retains their 2-D spatial locality could improve the block sparsity and make flexattention implementation more efficient. 

Static tiling as proposed in the [faster NATTEN](https://arxiv.org/abs/2403.04690) maps static tiles of $ T_h \times T_w $ in the 2-D space in contiguous region in 1-D Q K V. 


```python
H = 128
W = 128
K_H = 13
K_W = 13
T_H, T_W = 8, 8

def gen_tiled_natten(W, H, K_W, K_H, T_W, T_H):
    def get_idx_tiled(x, y):
        """
        Map 2-D coordinates to 1-D index for static tiles of T_H x T_W.
        """
        t_x, t_y = x // T_W, y // T_H
        t_id = t_x * (W // T_W) + t_y
        i_x, i_y = x % T_W, y % T_H
        t_offset = i_x * T_W + i_y
        return t_id * (T_H * T_W) + t_offset

    def get_x_y_tiled(idx):
        """
        Map 1-D index to 2-D coordinates for static tiles of T_H x T_W.
        """
        t_id = idx // (T_H * T_W)
        t_x, t_y = t_id // (W // T_W), t_id % (W // T_W)
        t_offset = idx % (T_H * T_W)
        i_x, i_y = t_offset // T_W, t_offset % T_W
        return t_x*T_W + i_x, t_y*T_H + i_y

    def tiled_natten_mask(b, h, q, kv):
        q_x, q_y = get_x_y_tiled(q)
        kv_x, kv_y = get_x_y_tiled(kv)
        kernel_x = q_x.clamp(K_W // 2, (W - 1) - K_W // 2)
        kernel_y = q_y.clamp(K_H // 2, (H - 1) - K_H // 2)
        hori_mask = (kernel_x - kv_x).abs() <= K_W // 2
        vert_mask = (kernel_y - kv_y).abs() <= K_H // 2
        return hori_mask & vert_mask
    return tiled_natten_mask

# tiled_natten_mask = gen_tiled_natten(W, H, K_W, K_H, T_W, T_H)
from attn_gym.masks.natten import generate_tiled_natten
tiled_natten_mask_mod = generate_tiled_natten(W, H, K_W, K_H, T_W, T_H)

test_mask(mask_mod=tiled_natten_mask_mod, S=H * W)
```

Verify that Naive NATTEN Mask and tiled NATTEN generate the same output


```python
def run_natten(
    mask = None,
    encoder = None, 
    decoder = None,
    query = None, 
    key = None,
    value = None, 
    gradOut = None,
    B=16,
    H=16,
    W=128,
    D=64,
    print_mask=True,
):
    if decoder:
        permuter_x, permuter_y = decoder(torch.arange(W*W))
        permuter_index = permuter_x * W + permuter_y
        q = query[:, :, permuter_x, permuter_y, :].clone().detach().requires_grad_(query.requires_grad)
        k = key[:, :, permuter_x, permuter_y, :].clone().detach().requires_grad_(key.requires_grad)
        v = value[:, :, permuter_x, permuter_y, :].clone().detach().requires_grad_(value.requires_grad)
        dO = gradOut[:, :, permuter_x, permuter_y, :]
    else: 
        q = query.flatten(2, 3).clone().detach().requires_grad_(query.requires_grad)
        k = key.flatten(2, 3).clone().detach().requires_grad_(key.requires_grad)
        v = value.flatten(2, 3).clone().detach().requires_grad_(value.requires_grad)
        dO = gradOut.flatten(2, 3)
    block_mask = create_block_mask_cached(mask, 1, 1, W*W, W*W, device=query.device)
    if print_mask:
        print(f"\nBlock Mask:\n{block_mask}")
    
    out = flex_attention(q, k, v, block_mask=block_mask)
    
    out.backward(dO)
    
    if encoder: 
        i_x = torch.arange(W)[:, None].broadcast_to(W, W).flatten() 
        i_y = torch.arange(W)[None, :].broadcast_to(W, W).flatten() 
        depermuter = encoder(i_x, i_y)
        out = out[:, :, depermuter, :].reshape(B, H, W, W, D)
        q_grad = q.grad[:, :, depermuter, :].reshape(B, H, W, W, D)
        k_grad = k.grad[:, :, depermuter, :].reshape(B, H, W, W, D)
        v_grad = v.grad[:, :, depermuter, :].reshape(B, H, W, W, D)
        results = [out, q_grad, k_grad, v_grad]
    else:
        out= out.reshape(B, H, W, W, D)
        q_grad = q.grad.reshape(B, H, W, W, D)
        k_grad = k.grad.reshape(B, H, W, W, D)
        v_grad = v.grad.reshape(B, H, W, W, D)
        results = [out, q_grad, k_grad, v_grad]
        
    del q, k, v, dO
    
    return results


def test_natten_masks(
    naive,
    tiled,
    B=16,
    H=16,
    W=128,
    D=64,
    skip_correctness=False,
    print_mask=True,
): 
    query = torch.randn(
        B, H, W, W, D, device="cuda", dtype=torch.float16, requires_grad=True
    )
    key = torch.randn(
        B, H, W, W, D, device="cuda", dtype=torch.float16, requires_grad=True
    )
    value = torch.randn(
        B, H, W, W, D, device="cuda", dtype=torch.float16, requires_grad=True
    )
    gradOut = torch.randn(B, H, W, W, D, device="cuda", dtype=torch.float16)
    
    naive_results = run_natten(mask=naive[0], encoder=naive[1], decoder=naive[2], query=query, key=key, value=value, gradOut=gradOut, print_mask=print_mask)
    tiled_results = run_natten(mask=tiled[0], encoder=tiled[1], decoder=tiled[2], query=query, key=key, value=value, gradOut=gradOut, print_mask=print_mask)
    
    if not skip_correctness:
        for naive, tiled in zip(naive_results, tiled_results):
            torch.testing.assert_close(naive, tiled, atol=1e-1, rtol=1e-2)

        print("Correctness check passed ✅")

    # Clean up to save memory
    del query, key, value, gradOut, naive_results, tiled_results
    torch.cuda.empty_cache()

test_natten_masks(
    naive=[natten_mask, None, None],
    tiled=[tiled_natten_mask, get_idx_tiled, get_x_y_tiled],
)
```

### Alibi Bias
The Alibi attention bias was made popular in [Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation](https://arxiv.org/abs/2108.12409), and claims to have beneficial properties for length extrapolation at inference "ALiBi does not add positional embeddings to word embeddings; instead, it biases query-key attention scores with a penalty that is proportional to their distance. "

We are going to implement this 2 ways to highlight a new functionality, the ability to utilize other tensors in your score mod function. Although the function signature doesn't accept other tensors users can implement this via a `closure`. Here we utilize our all too familiar causal mask fn as well as the individual head biases.


```python
# Alibi Bias
def generate_alibi_bias():
    alibi_bias = []
    for h in range(H):
        alibi_bias.append(-((h + 1) * 8.0 / H))
    alibi_bias = torch.tensor(alibi_bias, device="cuda")
    alibi_bias = torch.exp2(alibi_bias)
    return alibi_bias


alibi_bias = generate_alibi_bias()


# In this case we are going to use a mask_mod and a score_mod
def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


def alibi_and_causal_closure(score, b, h, q_idx, kv_idx):
    bias = alibi_bias[h] * (kv_idx - q_idx)
    return score + bias


def alibi_and_causal_functional(score, b, h, q_idx, kv_idx):
    scale = torch.exp2(-((h + 1) * 8.0 / H))
    bias = (kv_idx - q_idx) * scale
    return score + bias


# Correctness check here is simple and only works with mask_fns and not actual score_mods

test_mask(
    alibi_and_causal_closure,
    mask_mod=causal_mask,
    skip_correctness=True,
    print_mask=False,
)
test_mask(
    alibi_and_causal_functional,
    mask_mod=causal_mask,
    skip_correctness=True,
    print_mask=False,
)
```

### Tanh Soft-Capping
We can also implement tanh soft-capping with this API. Logit softcapping via tanh was popularized in [Gemma 2](https://storage.googleapis.com/deepmind-media/gemma/gemma-2-report.pdf).

In this case, there are some nuances. In particular, the standard `tanh`
operator in PyTorch (and CUDA/Triton) lowers to a numerically accurate but
(relatively) quite slow implementation in SASS. See
https://godbolt.org/z/W8afevWv1 for how the SASS looks like.

So, in this case, we want to lower the `tanh` into the approximate tanh
implementation. We can do so by register a custom operator in PyTorch and then
an Inductor lowering.


```python
# Tanh Soft-Capping
@torch.library.custom_op("approx::tanh", mutates_args=())
def tanh_approx(inp: torch.Tensor) -> torch.Tensor:
    return torch.tanh(inp)


@tanh_approx.register_fake
def _(inp: torch.Tensor) -> torch.Tensor:
    return torch.tanh(inp)


from torch._inductor.lowering import make_pointwise, register_lowering

# Some internal torch.compile details
from torch._inductor.virtualized import ops


def tanh_approx_lowering(inp):
    fn = partial(ops.inline_asm_elementwise, asm="tanh.approx.f32 $0, $1;")
    return make_pointwise(fn)(inp)


register_lowering(torch.ops.approx.tanh)(tanh_approx_lowering)


class TanhApprox(torch.autograd.Function):
    @staticmethod
    def forward(x):
        return torch.ops.approx.tanh(x)

    @staticmethod
    def setup_context(ctx, inputs, output):
        (x,) = inputs
        result = output
        ctx.save_for_backward(result)

    @staticmethod
    def backward(ctx, grad_output):
        (result,) = ctx.saved_tensors
        return grad_output * (1 - result * result)


tanh_approx = TanhApprox.apply


def tanh_soft_cap(score, b, h, q_idx, kv_idx):
    score = score / 2
    score = tanh_approx(score)
    return score * 2


# The baseline (xformers) does not have a way to generate tanh-softcapping so we skip correctness checks
test_mask(tanh_soft_cap, mask_mod=causal_mask, skip_correctness=True)
```

## Nested Jagged Tensor

Nested Tensors are a tensor subclass that is used to efficiently represent and compute with ragged data. It is possible to utilize FlexAttention with this data to efficiently perform causal attention on batches of sequences with different lengths.

Under the hood NJT stores its ragged data as a contiguous data `[[sequence_0], [sequence_1], ..., [Sequence_B]]`, `sum(*),..`


```python
random.seed(0)
torch.manual_seed(0)

batch_size = 16
n_heads = 16
D = 64


def prepare_qkv_values(tensor):
    return tensor._values.detach().requires_grad_()


def build_seq_idx(tensor: torch.Tensor):
    offsets = tensor.offsets()
    total_length = tensor.offsets()[-1].item()
    # Create a range tensor from 0 to total_length
    range_tensor = torch.arange(total_length, device="cuda", dtype=torch.int32)

    # Use searchsorted to find the index for each position
    seq_idx = torch.searchsorted(offsets, range_tensor, right=True) - 1

    return seq_idx


def create_njt_wrapper(orig_mask_mod, offsets, seq_idx):
    """Generic Wrapper that converts Dense mask_mod functions to NJT mask_mod functions"""

    def njt_score_mod(b, h, q_idx, kv_idx):
        q_nested = q_idx - offsets[seq_idx[q_idx]]
        kv_nested = kv_idx - offsets[seq_idx[kv_idx]]
        is_same_sequence = seq_idx[q_idx] == seq_idx[kv_idx]
        return orig_mask_mod(b, h, q_nested, kv_nested) & is_same_sequence

    return njt_score_mod


# Dense Score Mod
def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx
    # return torch.where(q_idx >= kv_idx, score, -float("inf"))


# Current limitation that the total sequnce length must be divisible by 128
sentence_lengths = [random.randint(1, 1024) for _ in range(batch_size - 1)]
total = sum(sentence_lengths)
sentence_lengths.append(128 - total % 128)
total = sum(sentence_lengths)

ragged_tensors = [torch.randn(l, n_heads, D, device="cuda") for l in sentence_lengths]
query = torch.nested.nested_tensor(
    ragged_tensors, layout=torch.jagged, requires_grad=True
)
key = torch.nested.nested_tensor(
    ragged_tensors, layout=torch.jagged, requires_grad=True
)
value = torch.nested.nested_tensor(
    ragged_tensors, layout=torch.jagged, requires_grad=True
)

# Build the seq_idx lookup table for
offsets = query.offsets()
seq_idx = build_seq_idx(query)

causal_score_mod_njt = create_njt_wrapper(causal_mask, offsets, seq_idx)

query_values = prepare_qkv_values(query)
key_values = prepare_qkv_values(key)
value_values = prepare_qkv_values(value)

block_mask = create_block_mask_cached(
    causal_score_mod_njt, 1, 1, total, total, device=query_values.device
)
out_flex = flex_attention(
    query_values.view(1, -1, n_heads, D).transpose(1, 2),
    key_values.view(1, -1, n_heads, D).transpose(1, 2),
    value_values.view(1, -1, n_heads, D).transpose(1, 2),
    block_mask=block_mask,
)
out_sdpa = F.scaled_dot_product_attention(
    query.transpose(1, 2),
    key.transpose(1, 2),
    value.transpose(1, 2),
    is_causal=True,
)

sdpa_outs = []
flex_outs = []

gradOut = torch.randn_like(out_sdpa)

sdpa_outs.append(out_sdpa)
out_sdpa.backward(gradOut)
sdpa_outs += [query.grad, key.grad, value.grad]

flex_outs.append(out_flex)
out_flex.backward(gradOut._values.unsqueeze(0))
flex_outs += [query_values.grad, key_values.grad, value_values.grad]

for flex, sdpa in zip(flex_outs, sdpa_outs):
    flex = flex.squeeze(0)
    torch.testing.assert_close(flex, sdpa._values, atol=1e-2, rtol=1e-2)


print("Correctness check passed ✅")

print(block_mask)
```

## Flamingo Cross Attention

The 🦩 [Flamingo Paper](https://arxiv.org/pdf/2204.14198) introduced "a family of visual language models (VLMs)
that take as input visual data interleaved with text and produce free-form text as output."

It utilizes `VisionCrossAttentionMask` to ensure that text only attends to associated images. TorchTune has a good description of this type of masking: [VisionCrossAttentionMask](https://github.com/pytorch/torchtune/blob/bbc48e089b072c7cbaea175bc70501b2193ba482/torchtune/modules/transforms/_transforms.py#L22-L43)

This type of attention makes sure that text sequences attend entirely to the preceding image and not to other future/unrelated images.


```Python
Example:
    >>> text = "<img1><img2>These are two dogs. <img3>This is a cat."
    >>> image_token_id = 1
    >>> tokens = [1, 1, 9673, 527, 1403, 12875, 13, 1, 1115, 374, 264, 8415]
    >>> transform = VisionCrossAttentionMask(tile_size=400, patch_size=40, image_token_id=1)
    >>> intervals = transform._get_image_attention_intervals(tokens)
    >>> print(intervals)
    [[0, 7], [1, 7], [7, 12]]
```

In the above case, we would generate a mask of
12 x sum(image_tokens_1 + image_tokens_2 + image_tokens_3)

assuming image_tokens are size 3
```

          img1    img2   img3
     1   █ █ █ | ░ ░ ░ | ░ ░ ░
     1   █ █ █ | █ █ █ | ░ ░ ░
  9673   █ █ █ | █ █ █ | ░ ░ ░
   527   █ █ █ | █ █ █ | ░ ░ ░
  1403   █ █ █ | █ █ █ | ░ ░ ░
 12875   █ █ █ | █ █ █ | ░ ░ ░
    13   █ █ █ | █ █ █ | ░ ░ ░
     1   ░ ░ ░ | ░ ░ ░ | █ █ █
  1115   ░ ░ ░ | ░ ░ ░ | █ █ █
   374   ░ ░ ░ | ░ ░ ░ | █ █ █
   264   ░ ░ ░ | ░ ░ ░ | █ █ █
  8415   ░ ░ ░ | ░ ░ ░ | █ █ █

```


```python
# Given information
num_tokens = 12
num_images = 3
image_token_length = 3
num_image_tokens = num_images * image_token_length
intervals = [[0, 7], [1, 7], [7, 12]]
# This is only needed if your images have different number of tokens per image
# If they are all the same number of tokens you can use image_idx = kv_idx // image_token_length
image_boundaries = [image_token_length * i for i in range(num_images)]
image_boundaries = (
    [0] * image_token_length + [1] * image_token_length + [2] * image_token_length
)

image_boundaries = torch.tensor(image_boundaries, dtype=torch.long, device="cuda")
intervals = torch.tensor(intervals, dtype=torch.long, device="cuda")


def vision_x_attention_mask(b, h, q_idx, kv_idx):
    image_idx = image_boundaries[kv_idx]
    interval = intervals[image_idx]
    return (q_idx >= interval[0]) & (q_idx < interval[1])


mask = create_mask(vision_x_attention_mask, 1, 1, num_tokens, num_image_tokens, "cuda")

print(mask)
```
