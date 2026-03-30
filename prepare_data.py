"""
Reads all .py files from a given directory (recursively) and concatenates
their contents into a single TXT file for training.
"""

import argparse
import os


def collect_python_files(source_dir: str) -> list[str]:
    """Recursively find all .py files in source_dir."""
    py_files = []
    for root, _, files in os.walk(source_dir):
        for fname in sorted(files):
            if fname.endswith(".py"):
                py_files.append(os.path.join(root, fname))
    py_files.sort()
    return py_files


def build_dataset(source_dir: str, output_path: str) -> None:
    py_files = collect_python_files(source_dir)
    if not py_files:
        print(f"No .py files found in {source_dir}")
        return

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


def main():
    parser = argparse.ArgumentParser(description="Prepare training data from Python files")
    parser.add_argument("source_dir", help="Directory containing .py files")
    parser.add_argument(
        "-o", "--output", default="dataset.txt", help="Output text file (default: dataset.txt)"
    )
    args = parser.parse_args()

    if not os.path.isdir(args.source_dir):
        print(f"Error: {args.source_dir} is not a directory")
        return

    build_dataset(args.source_dir, args.output)


if __name__ == "__main__":
    main()
