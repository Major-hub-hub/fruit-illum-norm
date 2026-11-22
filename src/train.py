"""Train a ResNet50 fresh/rotten classifier and optionally track it in W&B.

Examples (run from repository root):
  python -m src.train --condition B0 --epochs 1 --batch-size 4 --max-batches 2 --wandb-mode disabled
  python -m src.train --condition B0 --seed 42 --epochs 30 --batch-size 16 --wandb-mode online
"""

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.data.dataset import FruitDataset, ROOT
from src.models.resnet50_binary import build_model
from src.preprocess.pipeline import PIPELINES

CHECKPOINT_DIR = ROOT / "outputs" / "checkpoints"
METRICS_DIR = ROOT / "outputs" / "metrics"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loaders(condition: str, seed: int, batch_size: int, num_workers: int):
    # Keep the SAME fixed image split across all preprocessing experiments.
    manifest = ROOT / "data" / "splits" / "split_seed42.csv"
    train_data = FruitDataset(csv_path=manifest, split="train", condition=condition, augment=True)
    val_data = FruitDataset(csv_path=manifest, split="val", condition=condition, augment=False)
    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, generator=torch.Generator().manual_seed(seed),
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader


def run_epoch(model, loader, criterion, device, *, optimizer=None,
              max_batches=0, frozen_backbone=True, desc="Epoch"):
    training = optimizer is not None
    model.train(training)
    if training and frozen_backbone:
        # Do not update running means/variances of pretrained BatchNorm layers.
        for layer in model.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    total_loss = total_correct = total_samples = 0
    batches = min(len(loader), max_batches) if max_batches else len(loader)
    progress = tqdm(loader, total=len(loader), desc=desc, leave=False)

    with torch.set_grad_enabled(training):
        for batch_idx, (images, labels) in enumerate(progress):
            if max_batches and batch_idx >= max_batches:
                break
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            count = labels.size(0)
            total_samples += count
            total_loss += loss.item() * count
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            progress.set_postfix(
                loss=f"{total_loss / total_samples:.4f}",
                acc=f"{total_correct / total_samples:.3f}",
            )

    if total_samples == 0:
        raise RuntimeError("No batches processed")
    return {"loss": total_loss / total_samples,
            "accuracy": total_correct / total_samples,
            "samples": total_samples}


def save_history(history, path: Path) -> None:
    fields = ["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy"]
    # Write a temporary file and replace it, so readers never see half-written CSV.
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)
    tmp.replace(path)


def train(args) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = make_loaders(
        args.condition, args.seed, args.batch_size, args.num_workers
    )
    print(f"Device: {device} | Condition: {args.condition} | Seed: {args.seed}")
    print(f"Training samples: {len(train_loader.dataset)} | "
          f"Validation samples: {len(val_loader.dataset)}", flush=True)

    # Prefer the local pretrained model builder created earlier in this project.
    model = build_model(pretrained=True, freeze_backbone=True).to(device)
    # Freeze all backbone weights, but train the new binary classification head.
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("fc.")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable}")

    criterion = nn.CrossEntropyLoss()  # Two logits, integer targets 0/1
    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad), lr=args.lr
    )
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    run_name = f"{args.condition}_seed{args.seed}"
    if args.max_batches:
        run_name = f"smoke_{run_name}"
    checkpoint_path = CHECKPOINT_DIR / f"{run_name}_best.pth"
    history_path = METRICS_DIR / f"{run_name}_history.csv"

    wandb_run = None
    if args.wandb_mode != "disabled":
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=run_name,
                mode=args.wandb_mode,
                dir=str(ROOT),
                config={
                    "condition": args.condition, "seed": args.seed,
                    "architecture": "ResNet50", "weights": "IMAGENET1K_V1",
                    "classes": ["fresh", "rotten"], "training_stage": "frozen_backbone",
                    "batch_size": args.batch_size, "learning_rate": args.lr,
                    "optimizer": "Adam", "loss": "CrossEntropyLoss",
                    "epochs": args.epochs, "patience": args.patience,
                    "min_delta": args.min_delta,
                    "train_samples": len(train_loader.dataset),
                    "val_samples": len(val_loader.dataset),
                    "max_batches": args.max_batches,
                },
                settings=wandb.Settings(init_timeout=20),
            )
            wandb.define_metric("epoch")
            wandb.define_metric("train/*", step_metric="epoch")
            wandb.define_metric("val/*", step_metric="epoch")
            print(f"W&B run: {wandb_run.url or 'offline'}")
        except Exception as exc:
            # Continue to save all local results even if the tracking service fails.
            print(f"Warning: W&B unavailable ({exc}); continuing locally.")
            wandb_run = None

    best_loss = float("inf")
    best_epoch = 0
    no_improvement = 0
    history = []
    try:
        for epoch in range(1, args.epochs + 1):
            train_metrics = run_epoch(
                model, train_loader, criterion, device,
                optimizer=optimizer, max_batches=args.max_batches,
                frozen_backbone=True, desc=f"Epoch {epoch}/{args.epochs} train",
            )
            val_metrics = run_epoch(
                model, val_loader, criterion, device,
                max_batches=args.max_batches,
                desc=f"Epoch {epoch}/{args.epochs} val",
            )
            record = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
            }
            history.append(record)
            save_history(history, history_path)
            improved = val_metrics["loss"] < best_loss - args.min_delta
            if improved:
                best_loss = val_metrics["loss"]
                best_epoch = epoch
                no_improvement = 0
                torch.save({
                    "epoch": epoch,
                    "condition": args.condition,
                    "seed": args.seed,
                    "architecture": "resnet50",
                    "weights_source": "IMAGENET1K_V1",
                    "num_classes": 2,
                    "class_names": ["fresh", "rotten"],
                    "training_stage": "frozen_backbone",
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss": best_loss,
                    "val_accuracy": val_metrics["accuracy"],
                    "smoke_test": bool(args.max_batches),
                }, checkpoint_path)
            else:
                no_improvement += 1

            if wandb_run is not None:
                try:
                    wandb_run.log({
                        "epoch": epoch,
                        "train/loss": train_metrics["loss"],
                        "train/accuracy": train_metrics["accuracy"],
                        "val/loss": val_metrics["loss"],
                        "val/accuracy": val_metrics["accuracy"],
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "best_val_loss": best_loss,
                        "best_epoch": best_epoch,
                    }, step=epoch)
                except Exception as exc:
                    print(f"Warning: W&B logging failed: {exc}")
                    wandb_run = None

            print(
                f"Epoch {epoch:02d}/{args.epochs} | "
                f"Train Loss {train_metrics['loss']:.4f} | "
                f"Train Acc {train_metrics['accuracy']:.4f} | "
                f"Val Loss {val_metrics['loss']:.4f} | "
                f"Val Acc {val_metrics['accuracy']:.4f}"
                + (" | Best saved" if improved else ""),
                flush=True,
            )
            if no_improvement >= args.patience:
                print(f"Early stopping after epoch {epoch}.")
                break
    finally:
        if wandb_run is not None:
            try:
                wandb_run.summary["best_epoch"] = best_epoch
                wandb_run.summary["best_val_loss"] = best_loss
                wandb_run.finish()
            except Exception as exc:
                print(f"Warning: W&B finalization failed: {exc}")

    print(f"Best epoch: {best_epoch} | Best val loss: {best_loss:.4f}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"History: {history_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=sorted(PIPELINES), default="B0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--wandb-project", default="fruit-illum-norm")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0 or args.patience < 1:
        parser.error("epochs, batch-size, lr, and patience must be positive")
    if args.min_delta < 0 or args.num_workers < 0 or args.max_batches < 0:
        parser.error("min-delta, num-workers, and max-batches cannot be negative")
    train(args)


if __name__ == "__main__":
    main()
