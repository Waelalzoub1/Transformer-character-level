# Character-Level Transformer LM

A PyTorch-based character-level language model that learns from Python source code. Uses a decoder-only transformer architecture with causal self-attention.

Part of a "transistors to transformers" body of work; see also [mini-os32](https://github.com/Waelalzoub1/mini-os32), [custom-16bit-CPU](https://github.com/Waelalzoub1/custom-16bit-CPU), and [chess-rl](https://github.com/Waelalzoub1/chess-rl).

## Architecture

- **Tokenization**: Character-level (each unique character gets its own token)
- **Model**: Decoder-only transformer with causal masking, pre-norm LayerNorm, GELU activations, learned positional embeddings
- **Defaults**: 4 layers, 4 attention heads, d_model=256, context window of 256 characters

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### 1. Prepare training data

Collect all `.py` files from a directory into text files, holding out 10 % of the FILES for validation:

```bash
python prepare_data.py /path/to/python/code -o data.txt --val-fraction 0.1
# writes data_train.txt and data_val.txt; no file contributes to both
```

### 2. Train the model

```bash
python train.py --train data_train.txt --val data_val.txt
```

Training is step based: every step draws a batch of random 256-character crops from the training text; every `--eval-interval` steps the loss is measured on the whole held-out set (non-overlapping windows), and `checkpoints/best_model.pt` is written whenever that loss improves.

| Flag | Default | Description |
|------|---------|-------------|
| `--max-steps` | 20000 | Training steps |
| `--eval-interval` | 1000 | Steps between evaluations on the held-out set |
| `--batch-size` | 64 | Crops per step |
| `--block-size` | 256 | Context window (characters) |
| `--d-model` | 256 | Embedding / hidden dimension |
| `--n-heads` | 4 | Number of attention heads |
| `--n-layers` | 4 | Number of transformer blocks |
| `--lr` / `--min-lr` | 3e-4 / 3e-5 | Peak and final learning rate (linear warmup, cosine decay) |
| `--warmup-steps` | 200 | Warmup length |
| `--dropout` | 0.1 | Dropout rate |
| `--seed` | 1337 | Random seed |
| `--amp` | off | bfloat16 autocast on CUDA |
| `--save-dir` | `checkpoints/` | Best checkpoint and `history.json` |

### 3. Generate text

Single prompt:

```bash
python inference.py --checkpoint checkpoints/best_model.pt --prompt "def hello"
```

Interactive mode:

```bash
python inference.py --checkpoint checkpoints/best_model.pt --interactive
```

Generation options:

| Flag | Default | Description |
|------|---------|-------------|
| `--max-tokens` | 500 | Maximum characters to generate |
| `--temperature` | 0.8 | Sampling temperature (lower = more deterministic) |
| `--top-k` | 50 | Top-k filtering (0 = disabled) |

## Results

Data: the CPython 3.14.6 standard library on the training machine, every `.py` file except tests, idlelib, lib2to3 and site-packages: 604 files, 11.07 M characters, 278-character vocabulary. Split by file with `--val-fraction 0.1 --seed 0`: 544 files (9,592,127 chars) for training, 60 files (1,494,617 chars) held out.

```bash
python prepare_data.py stdlib/ -o stdlib.txt --val-fraction 0.1 --seed 0
python train.py --train stdlib_train.txt --val stdlib_val.txt --max-steps 40000 --batch-size 256 --amp --seed 1337
```

Model: the default 4 layers, 4 heads, d_model 256, context 256; **3,367,702 parameters**. Hardware: one NVIDIA GeForce RTX 5080 (16 GB), bfloat16 autocast, 85 minutes for 40,000 steps while sharing the GPU with another job (about 750 k tokens/s when it had the GPU to itself).

| step | train loss | held-out loss (nats/char) | bits/char |
|---|---|---|---|
| 1,000 | 1.840 | 1.1447 | 1.651 |
| 2,000 | 0.969 | 0.9859 | 1.422 |
| 5,000 | 0.746 | 0.9006 | 1.299 |
| 10,000 | 0.656 | 0.8748 | 1.262 |
| **15,000** | 0.616 | **0.8709** | **1.256** |
| 20,000 | 0.591 | 0.8710 | 1.257 |
| 30,000 | 0.563 | 0.8793 | 1.269 |
| 40,000 | 0.553 | 0.8802 | 1.270 |

**Minimum held-out loss: 0.871 nats/char = 1.256 bits/char at step 15,000** (the saved checkpoint). After that the training loss keeps falling while the held-out loss drifts up: a 3.4 M-parameter model has started to memorise 9.6 M characters.

![loss curve](docs/loss_curve.png)

Samples from the best checkpoint at temperature 0.8, top-k 50 (more in [`docs/samples.md`](docs/samples.md)):

```
def __init__(self, message=None, **kw):
        self.message = message
        self._check(f'process Failed exec')
        self._check("process")
        self._loop.call_exception_handler({
                'message': message,
                'exception': exc,
                'transport': self,
                },
```

Earlier revisions of this repository trained by epoch over every character offset with the last 10 % of the concatenated text as validation. Those runs reported 1.336 (4.7 M-char subset) and 0.929 nats/char (full stdlib), both at the end of their first epoch. They are not comparable with the numbers above: the split was by position rather than by file, and evaluation happened only once per epoch, so the true minimum was never observed.

## Tests

```bash
python tests/test_causal_mask.py
```

Three checks that the causal mask is correct: perturbing tokens after position t leaves logits at positions ≤ t unchanged, the logits of a prefix equal the corresponding logits of the full sequence, and no gradient flows from a loss at position t to embeddings after t.

## Project Structure

```
├── prepare_data.py   # Collects .py files into a training dataset
├── model.py          # CharTokenizer + CharTransformerLM definition
├── train.py          # Training loop with validation and checkpointing
├── inference.py      # Load a checkpoint and generate text
├── tests/test_causal_mask.py
├── docs/            # loss curve and samples
├── requirements.txt
└── README.md
```

## License

MIT
