#!/usr/bin/env python3
"""Use the trained model: stream a completion of a prompt, continue a file, or chat interactively.

    python generate.py "def fibonacci(n):"           # complete a prompt
    python generate.py --file mymodule.py            # continue from the end of a file
    python generate.py                               # interactive: type a prompt, get a completion, repeat
    cat snippet.py | python generate.py --stdin

Options: --max-tokens N (default 400), --temperature T (0.8; lower = more conservative),
--top-k K (50; 0 = off), --seed S, --ckpt PATH (default checkpoints/stdlib_best.pt).
"""
import argparse
import os
import sys

import torch

from inference import load_model

HERE = os.path.dirname(os.path.abspath(__file__))


@torch.no_grad()
def stream(model, tokenizer, prompt, max_new_tokens, temperature, top_k, device):
    """Yields the generated characters one at a time."""
    ids = tokenizer.encode(prompt)
    if not ids:
        ids = tokenizer.encode("\n")
    idx = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    for _ in range(max_new_tokens):
        logits, _ = model(idx[:, -model.block_size:])
        logits = logits[:, -1, :]
        if temperature <= 0:
            nxt = logits.argmax(dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k > 0:
                top_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < top_vals[:, -1:]] = float("-inf")
            nxt = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
        idx = torch.cat([idx, nxt], dim=1)
        yield tokenizer.decode([int(nxt)])


def complete(model, tokenizer, prompt, args, device):
    dropped = [c for c in prompt if c not in tokenizer.char_to_idx]
    if dropped:
        print(f"[note: {len(dropped)} character(s) not in the model's vocabulary were skipped]", file=sys.stderr)
    context = prompt[-model.block_size:]
    if len(prompt) > model.block_size:
        print(f"[note: only the last {model.block_size} characters of the prompt are used as context]", file=sys.stderr)
    sys.stdout.write(context)
    sys.stdout.flush()
    for ch in stream(model, tokenizer, context, args.max_tokens, args.temperature, args.top_k, device):
        sys.stdout.write(ch)
        sys.stdout.flush()
    sys.stdout.write("\n")


def main():
    ap = argparse.ArgumentParser(description="Generate Python-like text with the trained character-level transformer")
    ap.add_argument("prompt", nargs="?", help="text to continue")
    ap.add_argument("--file", help="continue from the end of this file")
    ap.add_argument("--stdin", action="store_true", help="read the prompt from standard input")
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--ckpt", default=os.path.join(HERE, "checkpoints", "stdlib_best.pt"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.seed is not None:
        torch.manual_seed(args.seed)
    model, tokenizer = load_model(args.ckpt, device)

    if args.file:
        prompt = open(args.file, encoding="utf-8").read()
    elif args.stdin:
        prompt = sys.stdin.read()
    else:
        prompt = args.prompt

    if prompt is not None:
        complete(model, tokenizer, prompt, args, device)
        return

    print("Interactive. Type a prompt and press Enter (an empty line uses 'def '); Ctrl-D or 'quit' to exit.")
    while True:
        try:
            line = input("\n>>> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if line.strip().lower() == "quit":
            return
        complete(model, tokenizer, line if line else "def ", args, device)


if __name__ == "__main__":
    main()
