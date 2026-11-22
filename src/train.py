"""Train a ResNet50 fresh/rotten classifier and optionally track it in W&B.

Examples (run from repository root):
  python -m src.train --condition B0 --epochs 1 --batch-size 4 --max-batches 2 --wandb-mode disabled
  python -m src.train --condition B0 --seed 42 --epochs 30 --batch-size 16 --wandb-mode online
  python -m src.train --condition B0 --stage 2 --seed 42 --epochs 40 --lr 1e-5 --wandb-mode online
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
        # Keep pretrained BatchNorm running statistics fixed in both stages.
        for layer in model.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    total_loss = total_correct = total_samples = 0
    batches = min(len(loader), max_batches) if max_batches else len(loader)
    progress = tqdm(loader, total=batches, desc=desc, leave=False)

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

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    run_name = f"{args.condition}_seed{args.seed}"
    if args.stage == 2:
        run_name += "_stage2"
    if args.max_batches:
        run_name = f"smoke_{run_name}"
    checkpoint_path = CHECKPOINT_DIR / f"{run_name}_best.pth"
    history_path = METRICS_DIR / f"{run_name}_history.csv"

    if args.stage == 1:
        # Local V1 weights, frozen backbone, new two-class head.
        model = build_model(pretrained=True, freeze_backbone=True)
        for name, param in model.named_parameters():
            param.requires_grad = name.startswith("fc.")
        optimizer_name = "Adam"
        optimizer_type = torch.optim.Adam
        stage_name = "frozen_backbone"
        source_path = None
    else:
        # Always load stage 1's full, best validation checkpoint. Do not
        # silently use the smoke-test checkpoint or original ImageNet weights.
        source_path = (Path(args.stage1_checkpoint).resolve()
                       if args.stage1_checkpoint
                       else CHECKPOINT_DIR / f"{args.condition}_seed{args.seed}_best.pth")
        if not source_path.is_file():
            raise FileNotFoundError(
                f"Stage 1 checkpoint missing: {source_path}\n"
                "Finish full Stage 1 first; Stage 2 cannot start from ImageNet alone."
            )
        previous = torch.load(source_path, map_location="cpu", weights_only=True)
        if previous.get("condition") != args.condition or previous.get("seed") != args.seed:
            raise ValueError("Stage 1 checkpoint condition/seed does not match this run")
        if previous.get("smoke_test", False) or "smoke_" in source_path.name:
            raise ValueError("Do not fine-tune from a smoke-test checkpoint")
        if previous.get("training_stage") != "frozen_backbone":
            raise ValueError("Stage 2 requires a frozen-backbone Stage 1 checkpoint")
        if previous.get("weights_source") != "IMAGENET1K_V1":
            raise ValueError("Stage 1 must use ImageNet V1 pretrained weights")

        # Construct matching two-class architecture without downloading weights.
        model = build_model(pretrained=False, freeze_backbone=False)
        model.load_state_dict(previous["model_state_dict"], strict=True)

        # Fine-tune only uppermost residual block and classification head.
        for name, param in model.named_parameters():
            param.requires_grad = name.startswith(("layer4.", "fc."))
        optimizer_name = "RMSprop"
        optimizer_type = torch.optim.RMSprop
        stage_name = "fine_tune_layer4"
        print(f"Loaded Stage 1 checkpoint: {source_path}")
        print(f"Starting from Stage 1 epoch {previous['epoch']}")

    model = model.to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Stage: {args.stage} | Trainable parameters: {trainable}", flush=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optimizer_type(
        (p for p in model.parameters() if p.requires_grad), lr=args.lr
    )

    wandb_run = None
    if args.wandb_mode != "disabled":
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=run_name,
                group=f"{args.condition}_seed{args.seed}",
                mode=args.wandb_mode,
                dir=str(ROOT),
                config={
                    "condition": args.condition, "seed": args.seed,
                    "architecture": "ResNet50", "weights": "IMAGENET1K_V1",
                    "classes": ["fresh", "rotten"], "training_stage": stage_name,
                    "stage": args.stage, "unfrozen_layers": "layer4+fc" if args.stage == 2 else "fc",
                    "source_checkpoint": str(source_path) if source_path else None,
                    "batch_size": args.batch_size, "learning_rate": args.lr,
                    "optimizer": optimizer_name, "loss": "CrossEntropyLoss",
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
                    "training_stage": stage_name,
                    "stage": args.stage,
                    "unfrozen_layers": "layer4+fc" if args.stage == 2 else "fc",
                    "source_checkpoint": str(source_path) if source_path else None,
                    "optimizer_name": optimizer_name,
                    "learning_rate": args.lr,
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
    parser.add_argument("--stage", choices=[1, 2], type=int, default=1)
    parser.add_argument("--stage1-checkpoint", default=None,
                        help="Optional path to best full Stage 1 checkpoint")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--wandb-project", default="fruit-illum-norm")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    args = parser.parse_args()
    if args.epochs is None:
        args.epochs = 40 if args.stage == 2 else 30
    if args.lr is None:
        args.lr = 1e-5 if args.stage == 2 else 1e-4
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0 or args.patience < 1:
        parser.error("epochs, batch-size, lr, and patience must be positive")
    if args.min_delta < 0 or args.num_workers < 0 or args.max_batches < 0:
        parser.error("min-delta, num-workers, and max-batches cannot be negative")
    train(args)


if __name__ == "__main__":
    main()
