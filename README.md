# Character-Level Transformer LM

A PyTorch-based character-level language model that learns from Python source code. Uses a decoder-only transformer architecture with causal self-attention.

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

Collect all `.py` files from a directory into a single text file:

```bash
python prepare_data.py /path/to/python/code -o dataset.txt
```

### 2. Train the model

```bash
python train.py --data dataset.txt
```

Training options:

| Flag | Default | Description |
|------|---------|-------------|
| `--epochs` | 50 | Number of training epochs |
| `--batch-size` | 64 | Batch size |
| `--block-size` | 256 | Context window (characters) |
| `--d-model` | 256 | Embedding / hidden dimension |
| `--n-heads` | 4 | Number of attention heads |
| `--n-layers` | 4 | Number of transformer blocks |
| `--lr` | 3e-4 | Learning rate |
| `--dropout` | 0.1 | Dropout rate |
| `--save-dir` | `checkpoints/` | Where to save model checkpoints |

The best model (by validation loss) is saved to `checkpoints/best_model.pt`.

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

## Project Structure

```
├── prepare_data.py   # Collects .py files into a training dataset
├── model.py          # CharTokenizer + CharTransformerLM definition
├── train.py          # Training loop with validation and checkpointing
├── inference.py      # Load a checkpoint and generate text
├── requirements.txt
└── README.md
```

## License

MIT
