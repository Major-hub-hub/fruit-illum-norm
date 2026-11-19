"""Build a fixed, class-wise 80/10/10 split without exact-image leakage.

Run from the repository root:
    python src/data/make_split.py --root data/raw --list
    python src/data/make_split.py --root data/raw --seed 42

The parent folder supplies the fruit and fresh/rotten label. Augmented paths
are excluded. SHA-256-identical files are kept in one split, preserving all
3,200 original files and the exact 160/20/20 file quotas for every class.
"""

import argparse
import hashlib
import random
import tempfile
from collections import Counter
from pathlib import Path

import pandas as pd

EXT = {".jpg", ".jpeg", ".png", ".bmp"}
FRUITS = ["apple", "banana", "grape", "guava", "jujube", "orange", "pomegranate", "strawberry"]
SPLITS = ("train", "val", "test")
MANIFEST_COLUMNS = ["path", "label", "fruit", "cls", "split", "sha256"]


def original_images(root: Path):
    """Yield original image files in a stable order, including for --list."""
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"Image root does not exist or is not a directory: {root}")
    for path in sorted(root.rglob("*"), key=lambda p: p.relative_to(root).as_posix()):
        if not path.is_file() or path.suffix.lower() not in EXT:
            continue
        if "augment" not in path.relative_to(root).as_posix().lower():
            yield path


def collect(root: Path):
    """Read paths, labels and byte-content hashes without modifying images."""
    rows = []
    for path in original_images(root):
        folder = path.parent.name.lower()
        state = "rotten" if "rotten" in folder else "fresh" if "fresh" in folder else None
        fruit = next((fruit for fruit in FRUITS if fruit in folder), None)
        if state is None or fruit is None:
            continue
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        rows.append({
            "path": path.as_posix(), "label": int(state == "rotten"),
            "fruit": fruit, "cls": f"{state}_{fruit}", "sha256": digest,
        })
    return pd.DataFrame(rows, columns=["path", "label", "fruit", "cls", "sha256"])


def _allocate_groups(groups, targets, rng):
    """Find disjoint validation/test group subsets with exact file quotas.

    Dynamic programming jointly tracks (validation files, test files). Every
    group is indivisible; all unselected groups belong to training. Recording
    only the first witness for a state and reading a snapshot for each group
    prevents a group from being reused. Joint allocation avoids the false
    failures possible when validation and test are selected independently.
    """
    groups = list(groups)
    rng.shuffle(groups)
    target = (targets["val"], targets["test"])
    predecessors = {(0, 0): None}

    for group_index, members in enumerate(groups):
        weight = len(members)
        for val_count, test_count in list(predecessors):
            for split, new_state in (
                ("val", (val_count + weight, test_count)),
                ("test", (val_count, test_count + weight)),
            ):
                if new_state[0] > target[0] or new_state[1] > target[1]:
                    continue
                if new_state not in predecessors:
                    predecessors[new_state] = ((val_count, test_count), group_index, split)
        if target in predecessors:
            break

    if target not in predecessors:
        sizes = sorted((len(group) for group in groups), reverse=True)
        raise ValueError(
            f"Cannot keep duplicate groups intact with exact quotas {targets}; "
            f"group sizes are {sizes}. No split was written."
        )

    assignment = {index: "train" for group in groups for index in group}
    state = target
    while state != (0, 0):
        previous_state, group_index, split = predecessors[state]
        for index in groups[group_index]:
            assignment[index] = split
        state = previous_state
    return assignment


def split_grouped(frame: pd.DataFrame, seed: int = 42):
    """Assign seeded, order-independent splits with indivisible content groups.

    Class sizes must be multiples of ten for exact 80/10/10 file counts.
    Conflicting fruit/state metadata for identical bytes is rejected rather
    than assigning those copies independently to different classes.
    """
    required = {"path", "label", "fruit", "cls", "sha256"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(frame.columns))}")
    if frame.empty:
        raise ValueError("No recognized original images found; check --root and --list.")
    if frame[list(required)].isna().any().any():
        raise ValueError("Image paths, labels, classes and content hashes must be populated.")
    if frame["path"].astype(str).str.strip().eq("").any():
        raise ValueError("Image paths must not be empty.")
    if frame["path"].duplicated().any():
        raise ValueError("The same image path is listed more than once.")
    if not frame["label"].isin((0, 1)).all() or not frame["fruit"].isin(FRUITS).all():
        raise ValueError("Expected binary fresh/rotten labels and recognized fruit names.")
    expected_cls = frame["label"].map({0: "fresh", 1: "rotten"}) + "_" + frame["fruit"]
    if not frame["cls"].eq(expected_cls).all():
        raise ValueError("Class names must agree with the binary labels and fruits.")
    if not frame["sha256"].astype(str).str.fullmatch(r"[0-9a-f]{64}").all():
        raise ValueError("Expected lowercase SHA-256 content hashes for every image.")

    data = frame.sort_values("path", kind="stable").reset_index(drop=True).copy()
    conflicts = data.groupby("sha256")[["label", "fruit", "cls"]].nunique().gt(1).any(axis=1)
    if conflicts.any():
        raise ValueError(
            f"{int(conflicts.sum())} content group(s) have conflicting fruit/class labels. "
            "Resolve the conflicting labels before splitting; no split was written."
        )

    rng = random.Random(seed)
    assignment = {}
    for cls, class_rows in data.groupby("cls", sort=True):
        count = len(class_rows)
        if count % 10:
            raise ValueError(f"{cls}: {count} files cannot have exact integer 80/10/10 quotas.")
        targets = {"train": count * 8 // 10, "val": count // 10, "test": count // 10}
        groups = [group.index.tolist() for _, group in class_rows.groupby("sha256", sort=True)]
        try:
            assignment.update(_allocate_groups(groups, targets, rng))
        except ValueError as exc:
            raise ValueError(f"{cls}: {exc}") from exc

    data["split"] = data.index.map(assignment)
    if data["split"].isna().any() or data.groupby("sha256")["split"].nunique().gt(1).any():
        raise RuntimeError("Internal error: incomplete assignment or cross-split duplicate group.")
    for cls, class_rows in data.groupby("cls"):
        count = len(class_rows)
        expected = {"train": count * 8 // 10, "val": count // 10, "test": count // 10}
        if class_rows["split"].value_counts().to_dict() != expected:
            raise RuntimeError(f"Internal error: incorrect split quotas for {cls}.")
    return data[MANIFEST_COLUMNS]


def write_manifest(frame: pd.DataFrame, output: Path):
    """Replace the manifest only after a complete CSV has been written."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", dir=output.parent,
            prefix=f".{output.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            frame.to_csv(handle, index=False, lineterminator="\n")
        temporary.replace(output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/raw")
    parser.add_argument("--out", help="default: data/splits/split_seed<seed>.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--list", action="store_true", help="only print original folder/image counts")
    args = parser.parse_args(argv)
    root = Path(args.root)

    try:
        if args.list:
            counts = Counter(path.parent.relative_to(root).as_posix() for path in original_images(root))
            for folder, count in sorted(counts.items()):
                print(f"{count:5d}  {folder}")
            return

        data = collect(root)
        expected_classes = {f"{state}_{fruit}" for state in ("fresh", "rotten") for fruit in FRUITS}
        if data.empty:
            raise ValueError("No recognized original images found; check --root and --list.")
        counts = data["cls"].value_counts()
        if set(counts.index) != expected_classes or not counts.eq(200).all():
            raise ValueError(
                "Expected all 16 fruit/state classes with 200 original files each; "
                "run with --list and verify the folder structure."
            )
        data = split_grouped(data, seed=args.seed)
        output = Path(args.out) if args.out else Path(f"data/splits/split_seed{args.seed}.csv")
        write_manifest(data, output)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))

    group_sizes = data.groupby("sha256").size()
    print("images found:", len(data), "| unique contents:", len(group_sizes))
    print("exact duplicate groups kept together:", int(group_sizes.gt(1).sum()))
    print(pd.crosstab(data["cls"], data["split"]).reindex(columns=SPLITS))
    print(data["split"].value_counts().reindex(SPLITS))
    print("cross-split exact duplicate groups:", int(data.groupby("sha256")["split"].nunique().gt(1).sum()))
    print("wrote", output)


if __name__ == "__main__":
    main()
