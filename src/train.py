"""Train one preprocessing condition."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", default="B0")
    parser.add_argument("--seed", type=int, default=42)
    parser.parse_args()
    raise NotImplementedError("Implement the training loop for the selected condition.")


if __name__ == "__main__":
    main()

