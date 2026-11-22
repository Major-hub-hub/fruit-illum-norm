
"""Monitor an existing training CSV in W&B."""

import argparse
import csv
import time
from pathlib import Path

import wandb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", default="B0")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    name = f"{args.condition}_seed{args.seed}"
    csv_path = (
        Path("outputs/metrics")
        / f"{name}_history.csv"
    )

    run = wandb.init(
        project="fruit-illum-norm",
        name=name,
        config={
            "condition": args.condition,
            "seed": args.seed,
            "model": "ResNet50",
            "pretrained_weights": "IMAGENET1K_V1",
        },
    )

    last_epoch = 0

    print(f"Monitoring: {csv_path}")
    print("Press Ctrl+C to finish monitoring.")

    try:
        while True:
            if csv_path.exists():
                try:
                    with csv_path.open(
                        newline="", encoding="utf-8"
                    ) as f:
                        rows = list(csv.DictReader(f))
                except (OSError, csv.Error):
                    rows = []

                for row in rows:
                    try:
                        epoch = int(row["epoch"])
                        if epoch <= last_epoch:
                            continue

                        metrics = {
                            "epoch": epoch,
                            "train/loss": float(
                                row["train_loss"]
                            ),
                            "train/accuracy": float(
                                row["train_accuracy"]
                            ),
                            "val/loss": float(
                                row["val_loss"]
                            ),
                            "val/accuracy": float(
                                row["val_accuracy"]
                            ),
                        }
                    except (KeyError, ValueError):
                        # Retry incomplete rows next time.
                        continue

                    run.log(metrics, step=epoch)
                    last_epoch = epoch
                    print(f"Logged epoch {epoch}")

            time.sleep(10)

    except KeyboardInterrupt:
        print("Monitoring stopped.")
    finally:
        run.finish()


if __name__ == "__main__":
    main()
