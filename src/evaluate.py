"""Evaluate a trained ResNet50 on the held-out fresh/rotten fruit test set.

Run from the repository root:
    python -m src.evaluate --condition B0 --stage 2 --seed 42
    python -m src.evaluate --condition B0 --stage 2 --stress all --wandb-mode online

Only evaluate a checkpoint selected using VALIDATION performance. Do not use
held-out TEST results to select training epochs or tune hyperparameters.
"""

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.data.dataset import FruitDataset, ROOT
from src.models.resnet50_binary import build_model
from src.preprocess.pipeline import PIPELINES

CLASSES = ["fresh", "rotten"]  # 0=fresh, 1=rotten
STRESSES = ("normal", "dark", "bright", "warm_cast")


@dataclass(frozen=True)
class TestPreprocessing:
    """Simulate sensor lighting BEFORE the selected image preprocessing."""

    condition: str
    stress: str

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        if self.stress == "normal":
            observed = rgb
        else:
            observed = rgb.astype(np.float32)
            if self.stress == "dark":
                observed *= 0.50
            elif self.stress == "bright":
                observed *= 1.40
            elif self.stress == "warm_cast":
                # RGB channel multipliers; a deterministic warm illumination cast.
                observed *= np.array([1.20, 1.00, 0.75], dtype=np.float32)
            else:
                raise ValueError(f"Unknown stress condition: {self.stress}")
            observed = np.clip(observed, 0, 255).astype(np.uint8)
        return PIPELINES[self.condition](observed)


def checkpoint_path(condition: str, seed: int, stage: int) -> Path:
    suffix = "_stage2" if stage == 2 else ""
    return ROOT / "outputs" / "checkpoints" / f"{condition}_seed{seed}{suffix}_best.pth"


def load_checkpoint(path: Path, condition: str, seed: int, stage: int, device):
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("condition") != condition or checkpoint.get("seed") != seed:
        raise ValueError("Checkpoint condition/seed does not match requested evaluation")
    if checkpoint.get("smoke_test", False) or path.name.startswith("smoke_"):
        raise ValueError("Smoke-test checkpoints must not be used for evaluation")
    expected_stage = "fine_tune_layer4" if stage == 2 else "frozen_backbone"
    if checkpoint.get("training_stage") != expected_stage:
        raise ValueError(
            f"Expected training_stage={expected_stage!r}, "
            f"got {checkpoint.get('training_stage')!r}"
        )
    if checkpoint.get("class_names") != CLASSES or checkpoint.get("num_classes") != 2:
        raise ValueError("Checkpoint's binary class mapping is not fresh=0, rotten=1")
    if checkpoint.get("weights_source") != "IMAGENET1K_V1":
        raise ValueError("Unexpected pretrained weight version")
    model = build_model(pretrained=False, freeze_backbone=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model, checkpoint


def evaluate_set(model, loader, device):
    y_true, y_pred, rotten_probs = [], [], []
    with torch.inference_mode():
        for images, labels in tqdm(loader, desc="Evaluating", leave=False):
            logits = model(images.to(device, non_blocking=True))
            if logits.ndim != 2 or logits.shape[1] != 2:
                raise RuntimeError(f"Expected [batch,2] logits; got {tuple(logits.shape)}")
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            predictions = logits.argmax(dim=1).cpu().numpy()
            y_true.extend(labels.tolist())
            y_pred.extend(predictions.tolist())
            rotten_probs.extend(probs.tolist())
    return np.asarray(y_true), np.asarray(y_pred), np.asarray(rotten_probs)


def metric_summary(y_true, y_pred):
    # Explicitly fix rotten=1 as positive, including when all predictions are one class.
    return {
        "n_samples": int(len(y_true)),
        "n_correct": int((y_true == y_pred).sum()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_rotten": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_rotten": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_rotten": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "rotten_misclassified_as_fresh": int(((y_true == 1) & (y_pred == 0)).sum()),
        "fresh_misclassified_as_rotten": int(((y_true == 0) & (y_pred == 1)).sum()),
    }


def save_csv(path, fieldnames, records):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def plot_confusion_matrix(matrix, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    fig, ax = plt.subplots(figsize=(5.8, 5))
    ConfusionMatrixDisplay(
        confusion_matrix=np.asarray(matrix), display_labels=CLASSES
    ).plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = Path(args.checkpoint).resolve() if args.checkpoint else checkpoint_path(
        args.condition, args.seed, args.stage
    )
    model, checkpoint = load_checkpoint(path, args.condition, args.seed, args.stage, device)
    print(f"Device: {device} | Condition: {args.condition} | Stage: {args.stage}")
    print(f"Checkpoint: {path} | Best training epoch: {checkpoint['epoch']}")

    split_path = ROOT / "data" / "splits" / "split_seed42.csv"
    stresses = STRESSES if args.stress == "all" else (args.stress,)
    metric_dir = ROOT / "outputs" / "metrics"
    figure_dir = ROOT / "outputs" / "figures"
    metric_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{args.condition}_seed{args.seed}_stage{args.stage}"
    summaries = {}
    run_wandb = None

    if args.wandb_mode != "disabled":
        try:
            import wandb
            run_wandb = wandb.init(
                project=args.wandb_project,
                name=f"{prefix}_evaluation",
                group=f"{args.condition}_seed{args.seed}",
                job_type="evaluation",
                mode=args.wandb_mode,
                dir=str(ROOT),
                config={"condition": args.condition, "seed": args.seed,
                        "stage": args.stage, "checkpoint": str(path),
                        "stress": args.stress, "split": "test"},
            )
        except Exception as exc:
            print(f"Warning: W&B unavailable ({exc}); saving local results only.")

    try:
        for stress in stresses:
            dataset = FruitDataset(
                csv_path=split_path, split="test", condition=args.condition,
                augment=False,
            )
            dataset.preprocess = TestPreprocessing(args.condition, stress)
            loader = DataLoader(
                dataset, batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers,
                pin_memory=(device.type == "cuda"),
            )
            y_true, y_pred, rotten_probs = evaluate_set(model, loader, device)
            if len(y_true) != len(dataset):
                raise RuntimeError("Evaluated count does not match test manifest")
            expected_labels = dataset.rows["label"].to_numpy(dtype=np.int64)
            if not np.array_equal(y_true, expected_labels):
                raise RuntimeError("Test loader order does not match manifest")

            summary = metric_summary(y_true, y_pred)
            summary.update({
                "condition": args.condition, "seed": args.seed,
                "stage": args.stage, "split": "test", "stress": stress,
                "checkpoint": str(path), "checkpoint_epoch": int(checkpoint["epoch"]),
            })
            summaries[stress] = summary
            stem = f"{prefix}_{stress}_test"
            json_path = metric_dir / f"{stem}_metrics.json"
            predictions_path = metric_dir / f"{stem}_predictions.csv"
            figure_path = figure_dir / f"{stem}_confusion_matrix.png"
            json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            records = []
            for i, row in dataset.rows.reset_index(drop=True).iterrows():
                records.append({
                    "path": row["path"], "fruit": row["fruit"],
                    "true_label": int(y_true[i]), "predicted_label": int(y_pred[i]),
                    "true_class": CLASSES[int(y_true[i])],
                    "predicted_class": CLASSES[int(y_pred[i])],
                    "probability_rotten": round(float(rotten_probs[i]), 8),
                    "correct": int(y_true[i] == y_pred[i]),
                })
            save_csv(predictions_path, list(records[0]), records)
            plot_confusion_matrix(summary["confusion_matrix"], figure_path,
                                  f"{args.condition} Stage {args.stage} — {stress} test")

            print(f"\n{stress}: {summary['n_correct']}/{summary['n_samples']} correct")
            for key in ("accuracy", "balanced_accuracy", "precision_rotten",
                        "recall_rotten", "f1_rotten", "cohen_kappa"):
                print(f"  {key}: {summary[key]:.4f}")
            print("  Confusion matrix [actual rows: fresh, rotten]:")
            print(np.asarray(summary["confusion_matrix"]))
            print(f"  Metrics: {json_path}")
            print(f"  Predictions: {predictions_path}")
            print(f"  Figure: {figure_path}")
            print(classification_report(y_true, y_pred, labels=[0, 1],
                                        target_names=CLASSES, zero_division=0))
            if run_wandb is not None:
                try:
                    import wandb
                    run_wandb.log({
                        f"{stress}/{key}": value
                        for key, value in summary.items()
                        if key in ("accuracy", "balanced_accuracy", "precision_rotten",
                                   "recall_rotten", "f1_rotten", "cohen_kappa",
                                   "rotten_misclassified_as_fresh")
                    } | {f"{stress}/confusion_matrix": wandb.Image(str(figure_path))})
                except Exception as exc:
                    print(f"Warning: W&B metric logging failed: {exc}")

        if "normal" in summaries:
            for stress, summary in summaries.items():
                if stress != "normal":
                    drop = summaries["normal"]["accuracy"] - summary["accuracy"]
                    summary["accuracy_drop_from_normal"] = float(drop)
                    print(f"Accuracy drop, normal -> {stress}: {drop:.4f}")
        combined = metric_dir / f"{prefix}_evaluation_summary.json"
        combined.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
        print(f"\nCombined evaluation summary: {combined}")
    finally:
        if run_wandb is not None:
            run_wandb.finish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=sorted(PIPELINES), default="B0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stage", choices=[1, 2], type=int, default=2)
    parser.add_argument("--checkpoint", default=None,
                        help="Optional path to a matching full training checkpoint")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--stress", choices=[*STRESSES, "all"], default="normal")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"],
                        default="disabled")
    parser.add_argument("--wandb-project", default="fruit-illum-norm")
    args = parser.parse_args()
    if args.seed < 0 or args.batch_size < 1 or args.num_workers < 0:
        parser.error("seed must be non-negative; batch-size positive; workers non-negative")
    run(args)


if __name__ == "__main__":
    main()
