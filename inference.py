"""
Inference script: load a trained model and generate text from a prompt.

Usage:
    python inference.py --checkpoint checkpoints/best_model.pt --prompt "def hello"
    python inference.py --checkpoint checkpoints/best_model.pt --interactive
"""

import argparse

import torch

from model import CharTokenizer, CharTransformerLM


@torch.no_grad()
def generate(
    model: CharTransformerLM,
    tokenizer: CharTokenizer,
    prompt: str,
    max_new_tokens: int = 500,
    temperature: float = 0.8,
    top_k: int = 50,
    device: torch.device = torch.device("cpu"),
) -> str:
    model.eval()
    idx = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)

    for _ in range(max_new_tokens):
        # crop to block_size if needed
        idx_cond = idx[:, -model.block_size :]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature  # (1, vocab_size)

        # top-k filtering
        if top_k > 0:
            top_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            threshold = top_vals[:, -1].unsqueeze(-1)
            logits[logits < threshold] = float("-inf")

        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_token], dim=1)

    generated = idx[0].tolist()
    return tokenizer.decode(generated)


def load_model(checkpoint_path: str, device: torch.device) -> tuple[CharTransformerLM, CharTokenizer]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    tokenizer = CharTokenizer()
    tokenizer.load_state_dict(checkpoint["tokenizer_state"])

    config = checkpoint["model_config"]
    model = CharTransformerLM(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        block_size=config["block_size"],
        dropout=config["dropout"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    epoch = checkpoint.get("epoch", "?")
    val_loss = checkpoint.get("val_loss", "?")
    print(f"Loaded model from epoch {epoch} (val_loss={val_loss})")
    print(f"Vocab size: {tokenizer.vocab_size} | Block size: {config['block_size']}")

    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Generate text with trained char-level LM")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--prompt", default=None, help="Prompt text to continue from")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--max-tokens", dest="max_tokens", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", dest="top_k", type=int, default=50)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model(args.checkpoint, device)

    if args.interactive:
        print("\nInteractive mode (type 'quit' to exit)")
        while True:
            try:
                prompt = input("\nPrompt> ")
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if prompt.strip().lower() == "quit":
                break
            if not prompt:
                continue
            output = generate(model, tokenizer, prompt, args.max_tokens, args.temperature, args.top_k, device)
            print("\n--- Generated ---")
            print(output)
    elif args.prompt:
        output = generate(model, tokenizer, args.prompt, args.max_tokens, args.temperature, args.top_k, device)
        print("\n--- Generated ---")
        print(output)
    else:
        print("Provide --prompt or --interactive")


if __name__ == "__main__":
    main()
