"""
Training script for the character-level transformer LM.

Usage:
    python prepare_data.py /path/to/python/code -o dataset.txt
    python train.py --data dataset.txt
"""

import argparse
import json
import os
import time

import torch
from torch.utils.data import DataLoader, Dataset

from model import CharTokenizer, CharTransformerLM


class CharDataset(Dataset):
    """Produces (input, target) pairs of fixed-length character sequences."""

    def __init__(self, data: torch.Tensor, block_size: int):
        self.data = data
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.data) - self.block_size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return x, y


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    torch.manual_seed(args.seed)

    # Load text
    with open(args.data, "r", encoding="utf-8") as f:
        text = f.read()
    print(f"Dataset: {len(text)} characters")

    if len(text) < args.block_size + 1:
        print("Error: dataset is too small for the chosen block_size")
        return

    # Build tokenizer
    tokenizer = CharTokenizer()
    tokenizer.build_vocab(text)
    print(f"Vocabulary: {tokenizer.vocab_size} unique characters")

    # Encode full text
    encoded = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    # Train / validation split
    split_idx = int(len(encoded) * 0.9)
    train_data = encoded[:split_idx]
    val_data = encoded[split_idx:]
    print(f"Train: {len(train_data)} chars | Val: {len(val_data)} chars")

    train_dataset = CharDataset(train_data, args.block_size)
    val_dataset = CharDataset(val_data, args.block_size)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Create model
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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float("inf")
    os.makedirs(args.save_dir, exist_ok=True)
    history = []

    for epoch in range(1, args.epochs + 1):
        # --- Training ---
        model.train()
        train_loss_sum = 0.0
        train_steps = 0
        t0 = time.time()

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            _, loss = model(batch_x, batch_y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_sum += loss.item()
            train_steps += 1

        scheduler.step()
        avg_train_loss = train_loss_sum / max(train_steps, 1)
        elapsed = time.time() - t0
        tokens_per_sec = train_steps * args.batch_size * args.block_size / max(elapsed, 1e-9)

        # --- Validation ---
        model.eval()
        val_loss_sum = 0.0
        val_steps = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                _, loss = model(batch_x, batch_y)
                val_loss_sum += loss.item()
                val_steps += 1

        avg_val_loss = val_loss_sum / max(val_steps, 1)

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={avg_train_loss:.4f} | "
            f"val_loss={avg_val_loss:.4f} | "
            f"lr={scheduler.get_last_lr()[0]:.2e} | "
            f"{elapsed:.1f}s | {tokens_per_sec:,.0f} tok/s"
        )
        history.append({"epoch": epoch, "train_loss": avg_train_loss, "val_loss": avg_val_loss,
                        "seconds": elapsed, "tokens_per_sec": tokens_per_sec})
        with open(os.path.join(args.save_dir, "history.json"), "w") as f:
            json.dump({"args": vars(args), "device": str(device),
                       "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                       "params": param_count, "vocab_size": tokenizer.vocab_size,
                       "dataset_chars": len(text), "history": history}, f, indent=1)

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(args.save_dir, "best_model.pt")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "tokenizer_state": tokenizer.state_dict(),
                    "model_config": {
                        "vocab_size": tokenizer.vocab_size,
                        "d_model": args.d_model,
                        "n_heads": args.n_heads,
                        "n_layers": args.n_layers,
                        "block_size": args.block_size,
                        "dropout": args.dropout,
                    },
                    "epoch": epoch,
                    "val_loss": avg_val_loss,
                },
                save_path,
            )
            print(f"  -> Saved best model (val_loss={avg_val_loss:.4f})")

    print(f"\nTraining complete. Best val_loss: {best_val_loss:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train character-level transformer LM")
    parser.add_argument("--data", default="dataset.txt", help="Path to training text file")
    parser.add_argument("--save-dir", dest="save_dir", default="checkpoints", help="Directory for saved models")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=64)
    parser.add_argument("--block-size", dest="block_size", type=int, default=256)
    parser.add_argument("--d-model", dest="d_model", type=int, default=256)
    parser.add_argument("--n-heads", dest="n_heads", type=int, default=4)
    parser.add_argument("--n-layers", dest="n_layers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", dest="weight_decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1337, help="random seed for weight init and batch order")
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
