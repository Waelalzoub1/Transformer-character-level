"""
Training script for the character-level transformer LM.

Step-based training on random crops, with periodic evaluation on a held-out
set that was split off by FILE (see prepare_data.py --val-fraction).

Usage:
    python prepare_data.py /path/to/python/code -o data.txt --val-fraction 0.1
    python train.py --train data_train.txt --val data_val.txt
"""

import argparse
import json
import math
import os
import time

import torch

from model import CharTokenizer, CharTransformerLM


def get_batch(data: torch.Tensor, batch_size: int, block_size: int, generator: torch.Generator):
    """Random crops: each row is a block_size window starting at a random offset."""
    starts = torch.randint(0, len(data) - block_size - 1, (batch_size,), device=data.device, generator=generator)
    offsets = torch.arange(block_size, device=data.device)
    idx = starts.unsqueeze(1) + offsets
    return data[idx], data[idx + 1]


@torch.no_grad()
def evaluate(model, data: torch.Tensor, batch_size: int, block_size: int, max_batches: int, amp: bool) -> float:
    """Mean loss over non-overlapping windows that tile the held-out set from the start (deterministic)."""
    model.eval()
    n_windows = (len(data) - 1) // block_size
    starts = torch.arange(n_windows, device=data.device) * block_size
    if max_batches > 0:
        starts = starts[: max_batches * batch_size]
    offsets = torch.arange(block_size, device=data.device)
    total, count = 0.0, 0
    for i in range(0, len(starts), batch_size):
        idx = starts[i : i + batch_size].unsqueeze(1) + offsets
        with torch.autocast(device_type=data.device.type, dtype=torch.bfloat16, enabled=amp):
            _, loss = model(data[idx], data[idx + 1])
        total += loss.item() * idx.size(0)
        count += idx.size(0)
    model.train()
    return total / max(count, 1)


def lr_at(step: int, args) -> float:
    if step < args.warmup_steps:
        return args.lr * (step + 1) / args.warmup_steps
    progress = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
    return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * progress))


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    torch.manual_seed(args.seed)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)

    with open(args.train, "r", encoding="utf-8") as f:
        train_text = f.read()
    with open(args.val, "r", encoding="utf-8") as f:
        val_text = f.read()
    print(f"Train: {len(train_text):,} chars | Val (held-out files): {len(val_text):,} chars")

    # Character vocabulary over both splits, so held-out text is never silently dropped.
    tokenizer = CharTokenizer()
    tokenizer.build_vocab(train_text + val_text)
    print(f"Vocabulary: {tokenizer.vocab_size} unique characters")
    train_data = torch.tensor(tokenizer.encode(train_text), dtype=torch.long, device=device)
    val_data = torch.tensor(tokenizer.encode(val_text), dtype=torch.long, device=device)

    model = CharTransformerLM(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        block_size=args.block_size,
        dropout=args.dropout,
    ).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    os.makedirs(args.save_dir, exist_ok=True)
    history, best_val, best_step = [], float("inf"), 0
    tokens_per_step = args.batch_size * args.block_size
    train_loss_sum, train_loss_n = 0.0, 0
    t_start = t_last = time.time()

    model.train()
    for step in range(1, args.max_steps + 1):
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step - 1, args)
        x, y = get_batch(train_data, args.batch_size, args.block_size, generator)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=args.amp):
            _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss_sum += loss.item()
        train_loss_n += 1

        if step % args.eval_interval == 0 or step == args.max_steps:
            val_loss = evaluate(model, val_data, args.eval_batch_size, args.block_size, args.eval_batches, args.amp)
            now = time.time()
            tok_s = train_loss_n * tokens_per_step / (now - t_last)
            train_loss = train_loss_sum / train_loss_n
            improved = val_loss < best_val
            print(
                f"step {step:>6}/{args.max_steps} | train {train_loss:.4f} | val {val_loss:.4f} nats "
                f"({val_loss / math.log(2):.4f} bits/char) | lr {lr_at(step - 1, args):.2e} | {tok_s:,.0f} tok/s"
                + (" | best" if improved else ""),
                flush=True,
            )
            history.append({"step": step, "train_loss": train_loss, "val_loss": val_loss,
                            "tokens_per_sec": tok_s, "elapsed_s": now - t_start})
            if improved:
                best_val, best_step = val_loss, step
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "tokenizer_state": tokenizer.state_dict(),
                        "model_config": {
                            "vocab_size": tokenizer.vocab_size, "d_model": args.d_model, "n_heads": args.n_heads,
                            "n_layers": args.n_layers, "block_size": args.block_size, "dropout": args.dropout,
                        },
                        "step": step, "epoch": step, "val_loss": val_loss,
                    },
                    os.path.join(args.save_dir, "best_model.pt"),
                )
            with open(os.path.join(args.save_dir, "history.json"), "w") as f:
                json.dump({"args": vars(args), "device": str(device),
                           "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                           "params": param_count, "vocab_size": tokenizer.vocab_size,
                           "train_chars": len(train_text), "val_chars": len(val_text),
                           "best_val_loss": best_val, "best_step": best_step, "history": history}, f, indent=1)
            train_loss_sum, train_loss_n, t_last = 0.0, 0, time.time()

    print(f"\nTraining complete. Best val loss {best_val:.4f} nats/char = {best_val / math.log(2):.4f} bits/char at step {best_step}")


def main():
    parser = argparse.ArgumentParser(description="Train character-level transformer LM")
    parser.add_argument("--train", required=True, help="training text file")
    parser.add_argument("--val", required=True, help="held-out text file (different source files than --train)")
    parser.add_argument("--save-dir", dest="save_dir", default="checkpoints", help="Directory for saved models")
    parser.add_argument("--max-steps", dest="max_steps", type=int, default=20000)
    parser.add_argument("--eval-interval", dest="eval_interval", type=int, default=1000)
    parser.add_argument("--eval-batches", dest="eval_batches", type=int, default=0, help="0 = the whole held-out set")
    parser.add_argument("--eval-batch-size", dest="eval_batch_size", type=int, default=256)
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=64)
    parser.add_argument("--block-size", dest="block_size", type=int, default=256)
    parser.add_argument("--d-model", dest="d_model", type=int, default=256)
    parser.add_argument("--n-heads", dest="n_heads", type=int, default=4)
    parser.add_argument("--n-layers", dest="n_layers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", dest="min_lr", type=float, default=3e-5)
    parser.add_argument("--warmup-steps", dest="warmup_steps", type=int, default=200)
    parser.add_argument("--weight-decay", dest="weight_decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1337, help="random seed for weight init and crop sampling")
    parser.add_argument("--amp", action="store_true", help="bfloat16 autocast for forward/backward (CUDA)")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
