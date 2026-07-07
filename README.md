# GPT From Scratch

A chapter-by-chapter implementation of a GPT-style language model, following
Sebastian Raschka's *Build a Large Language Model From Scratch*.

---

## Project Overview

Every component — tokenisation, attention, the transformer block, the full GPT
model, and pretrained weight loading — is written from first principles in
PyTorch.  No high-level wrappers.  The goal is a deep understanding of how
modern LLMs work before touching any fine-tuning or deployment tooling.

The capstone is loading OpenAI's released GPT-2 weights into our custom
architecture and generating coherent English from scratch-built code.

---

## Stack

| Component | Version |
|-----------|---------|
| Python    | 3.11    |
| PyTorch   | latest stable |
| CUDA      | 12.1    |
| Host OS   | WSL2 (Ubuntu) on Windows |
| Tokeniser | tiktoken (OpenAI BPE) |
| Weight source | HuggingFace `transformers` |

---

## Repository Layout

```
llm_summer_project/
├── model.py                  # shared PyTorch classes (all chapters import from here)
├── notebooks/
│   └── ch5_pretrained_gpt2.ipynb
├── implementations/
│   ├── chapter_4.ipynb       # GPT model assembly
│   ├── ch5_classes.ipynb     # transformer block classes
│   └── hug_my_face.ipynb     # early HuggingFace exploration
├── week_1/                   # early exercises
└── week_2_attention/         # attention deep-dive
```

---

## Chapter-by-Chapter Summary

### Chapter 2 — Working with Text Data
Implemented Byte-Pair Encoding (BPE) using OpenAI's `tiktoken` library.
Covered vocabulary construction, text encoding, and decoding back to strings.
Built a simple sliding-window dataset class to prepare token sequences for
training.

### Chapter 3 — Coding Attention Mechanisms
Built self-attention from scratch: dot-product scores, softmax normalisation,
and the context vector.  Extended to multi-head attention with separate Q, K, V
projections.  Added the upper-triangular causal mask so tokens cannot attend to
future positions (required for autoregressive generation).

### Chapter 4 — Implementing a GPT Model From Scratch
Assembled all attention components into a full GPT decoder:

- `GELU` activation (approximate tanh form)
- Custom `LayerNorm` with learnable `scale` / `shift`
- `FeedForward` block (4× expansion)
- `MultiHeadAttention` with `register_buffer` causal mask and `W_o` output projection
- `TransformerBlock` with pre-norm and residual connections
- `GPTModel` with token + position embeddings, stacked blocks, and vocabulary head

All shared classes now live in `model.py` for reuse across notebooks.

### Chapter 5 — Pretrained Weights & Text Generation
Loaded OpenAI's public GPT-2 (124 M) weights from HuggingFace into our custom
`GPTModel`.  The main challenge was bridging the `Conv1D` weight convention used
by HuggingFace (shape `[in, out]`) against `nn.Linear` (shape `[out, in]`), and
splitting the combined `c_attn` Q+K+V matrix into our three separate projections.

See `notebooks/ch5_pretrained_gpt2.ipynb` for the full walk-through.

---

## How to Run

```bash
# 1. Clone and enter the project
cd ~/llm_summer_project

# 2. Install dependencies (once)
pip install torch tiktoken transformers

# 3. Launch Jupyter
jupyter notebook

# 4. Open any notebook from the browser
```

Notebooks in `notebooks/` add `..` to `sys.path` automatically, so `model.py`
is importable without any additional setup.

To run a quick sanity check on `model.py` directly:

```bash
python - <<'EOF'
import torch
from model import GPTModel, generate
cfg = {"vocab_size":50257,"context_length":1024,"emb_dim":768,
       "n_heads":12,"n_layers":12,"drop_rate":0.1,"qkv_bias":True}
m = GPTModel(cfg); m.eval()
print(sum(p.numel() for p in m.parameters())//1_000_000, "M params")
EOF
```

---

## Planned Work

- **Jetson Orin / Nano inference benchmarking** — measure tokens-per-second for
  GPT-2 small and medium on NVIDIA Jetson edge hardware.  Will compare INT8
  quantised (`torch.quantization`) against FP32 baseline and profile memory
  bandwidth to understand the memory-bound bottleneck on embedded GPUs.
- Fine-tuning on a custom dataset (Chapter 6–7 of the book).
- Implementing a simple training loop with gradient accumulation and cosine LR
  scheduling.
