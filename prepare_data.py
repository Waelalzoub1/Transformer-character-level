"""
Reads all .py files from a given directory (recursively) and concatenates
their contents into a single TXT file for training.
"""

import argparse
import os
import random


def collect_python_files(source_dir: str) -> list[str]:
    """Recursively find all .py files in source_dir."""
    py_files = []
    for root, _, files in os.walk(source_dir):
        for fname in sorted(files):
            if fname.endswith(".py"):
                py_files.append(os.path.join(root, fname))
    py_files.sort()
    return py_files


def write_files(py_files: list[str], source_dir: str, output_path: str) -> None:
    total_chars = 0
    with open(output_path, "w", encoding="utf-8") as out_f:
        for i, fpath in enumerate(py_files):
            with open(fpath, "r", encoding="utf-8", errors="replace") as in_f:
                content = in_f.read()
            # separator between files
            if i > 0:
                out_f.write("\n\n")
            out_f.write(f"# FILE: {os.path.relpath(fpath, source_dir)}\n")
            out_f.write(content)
            total_chars += len(content)

    print(f"Collected {len(py_files)} files ({total_chars} chars) -> {output_path}")


def build_dataset(source_dir: str, output_path: str, val_fraction: float = 0.0, seed: int = 0) -> None:
    py_files = collect_python_files(source_dir)
    if not py_files:
        print(f"No .py files found in {source_dir}")
        return
    if val_fraction <= 0:
        write_files(py_files, source_dir, output_path)
        return
    # Hold out whole files, so validation text never shares a file with training text.
    shuffled = py_files[:]
    random.Random(seed).shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_fraction))
    val_files, train_files = sorted(shuffled[:n_val]), sorted(shuffled[n_val:])
    base, ext = os.path.splitext(output_path)
    write_files(train_files, source_dir, f"{base}_train{ext}")
    write_files(val_files, source_dir, f"{base}_val{ext}")


def main():
    parser = argparse.ArgumentParser(description="Prepare training data from Python files")
    parser.add_argument("source_dir", help="Directory containing .py files")
    parser.add_argument(
        "-o", "--output", default="dataset.txt", help="Output text file (default: dataset.txt)"
    )
    parser.add_argument("--val-fraction", dest="val_fraction", type=float, default=0.0,
                        help="hold out this fraction of FILES into <output>_val.txt (and write <output>_train.txt)")
    parser.add_argument("--seed", type=int, default=0, help="seed for the file-level split")
    args = parser.parse_args()

    if not os.path.isdir(args.source_dir):
        print(f"Error: {args.source_dir} is not a directory")
        return

    build_dataset(args.source_dir, args.output, args.val_fraction, args.seed)


if __name__ == "__main__":
    main()
