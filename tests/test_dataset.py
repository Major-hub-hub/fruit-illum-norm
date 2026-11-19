"""Functional checks for manifest loading and RGB preprocessing."""

import hashlib
from pathlib import Path
import pickle
import tempfile
import unittest

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader

from src.data.dataset import FruitDataset, MEAN, STD, validate_manifest


class FruitDatasetTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.csv_path = self.root / "split.csv"
        self.colors = {
            ("train", 0): (240, 20, 50),
            ("train", 1): (20, 180, 60),
            ("val", 0): (255, 0, 0),
            ("val", 1): (40, 50, 60),
            ("test", 0): (32, 128, 224),
            ("test", 1): (220, 80, 10),
        }
        rows = []
        for (split, label), color in self.colors.items():
            relative = Path("images") / f"{split}_{label}.png"
            image_path = self.root / relative
            image_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (19, 13), color).save(image_path)
            rows.append({
                "path": relative.as_posix(),
                "label": label,
                "fruit": "apple",
                "cls": "rotten_apple" if label else "fresh_apple",
                "split": split,
                "sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
            })
        self.frame = pd.DataFrame(rows)
        self.write_frame(self.frame)

    def write_frame(self, frame):
        frame.to_csv(self.csv_path, index=False)

    def dataset(self, **kwargs):
        return FruitDataset(self.csv_path, root=self.root, **kwargs)

    def test_every_condition_returns_finite_float32_chw_and_integer_labels(self):
        for condition in ("B0", "B1", "E1", "E2", "E3"):
            with self.subTest(condition=condition):
                dataset = self.dataset(split="val", condition=condition)
                self.assertEqual(len(dataset), 2)
                for index, expected_label in enumerate((0, 1)):
                    image, label = dataset[index]
                    self.assertEqual(tuple(image.shape), (3, 250, 250))
                    self.assertEqual(image.dtype, torch.float32)
                    self.assertTrue(torch.isfinite(image).all().item())
                    self.assertIs(type(label), int)
                    self.assertEqual(label, expected_label)

    def test_b0_preserves_rgb_order_and_applies_imagenet_normalization(self):
        image, label = self.dataset(split="val", condition="B0")[0]
        rgb = torch.tensor(self.colors[("val", 0)], dtype=torch.float32) / 255
        expected = (rgb - torch.tensor(MEAN)) / torch.tensor(STD)
        expected = expected[:, None, None].expand_as(image)
        torch.testing.assert_close(image, expected, rtol=0, atol=1e-6)
        self.assertEqual(label, 0)

    def test_validation_and_test_samples_are_deterministic_for_all_conditions(self):
        for split in ("val", "test"):
            for condition in ("B0", "B1", "E1", "E2", "E3"):
                with self.subTest(split=split, condition=condition):
                    dataset = self.dataset(split=split, condition=condition)
                    first, first_label = dataset[0]
                    second, second_label = dataset[0]
                    self.assertTrue(torch.equal(first, second))
                    self.assertEqual(first_label, second_label)

    def test_augmentation_is_restricted_to_training_and_can_be_disabled(self):
        for split in ("val", "test"):
            with self.subTest(split=split):
                with self.assertRaisesRegex(ValueError, "only allowed for training"):
                    self.dataset(split=split, augment=True)

        dataset = self.dataset(split="train", augment=False)
        first, _ = dataset[0]
        second, _ = dataset[0]
        self.assertTrue(torch.equal(first, second))

        augmented = self.dataset(split="train", augment=True)
        torch.manual_seed(42)
        first, _ = augmented[0]
        torch.manual_seed(42)
        repeated, _ = augmented[0]
        self.assertTrue(torch.equal(first, repeated))
        self.assertTrue(torch.isfinite(first).all().item())

    def test_e3_dataset_survives_pickle_roundtrip_for_spawned_workers(self):
        original = self.dataset(split="val", condition="E3")
        restored = pickle.loads(pickle.dumps(original))
        self.assertEqual(len(restored), len(original))
        original_image, original_label = original[0]
        restored_image, restored_label = restored[0]
        torch.testing.assert_close(restored_image, original_image, rtol=0, atol=0)
        self.assertEqual(restored_label, original_label)

    def test_e3_dataloader_fetches_batch_with_two_spawned_workers(self):
        dataset = self.dataset(split="val", condition="E3")
        expected_images = torch.stack([dataset[index][0] for index in range(len(dataset))])
        loader = DataLoader(
            dataset, batch_size=2, shuffle=False, num_workers=2,
            multiprocessing_context="spawn", timeout=60,
        )
        # Exhausting the iterator shuts down workers before the PNG fixture is
        # removed by cleanup, including on Windows where open files matter.
        batches = list(loader)
        self.assertEqual(len(batches), 1)
        images, labels = batches[0]
        self.assertEqual(tuple(images.shape), (2, 3, 250, 250))
        self.assertEqual(labels.dtype, torch.long)
        self.assertEqual(labels.tolist(), [0, 1])
        torch.testing.assert_close(images, expected_images, rtol=0, atol=0)

    def test_relative_manifest_paths_resolve_against_explicit_repository_root(self):
        validated = validate_manifest(self.csv_path, root=self.root)
        self.assertEqual(len(validated), 6)
        self.assertEqual(set(validated["split"]), {"train", "val", "test"})
        self.assertEqual(
            validated["resolved_path"].tolist(),
            [str((self.root / path).resolve()) for path in self.frame["path"]],
        )

    def test_missing_image_is_reported_before_loading_any_split(self):
        # A missing training image must also be caught when constructing val.
        (self.root / self.frame.iloc[0]["path"]).unlink()
        with self.assertRaisesRegex(FileNotFoundError, "missing images"):
            self.dataset(split="val")

    def test_declared_exact_content_leakage_is_rejected(self):
        bad = self.frame.copy()
        bad.loc[2, "sha256"] = bad.loc[0, "sha256"]
        self.write_frame(bad)
        with self.assertRaisesRegex(ValueError, "Exact-content leakage"):
            self.dataset(split="train")

    def test_conflicting_labels_for_one_declared_content_group_are_rejected(self):
        bad = self.frame.copy()
        # Keep both copies in train so this exercises label conflict checking.
        bad.loc[1, "sha256"] = bad.loc[0, "sha256"]
        self.write_frame(bad)
        with self.assertRaisesRegex(ValueError, "Conflicting labels for identical content"):
            validate_manifest(self.csv_path, root=self.root)

    def test_invalid_manifest_metadata_and_repeated_paths_are_rejected(self):
        duplicate_path = self.frame.copy()
        duplicate_path.loc[2, "path"] = duplicate_path.loc[0, "path"]
        invalid_class = self.frame.copy()
        invalid_class.loc[0, "cls"] = "rotten_apple"
        cases = {
            "missing column": self.frame.drop(columns="sha256"),
            "empty manifest": self.frame.iloc[:0],
            "missing value": self.frame.assign(fruit=None),
            "missing split": self.frame.loc[self.frame["split"] != "test"],
            "invalid label": self.frame.assign(label=2),
            "invalid hash": self.frame.assign(sha256="not-a-content-hash"),
            "inconsistent class": invalid_class,
            "duplicate resolved path": duplicate_path,
        }
        for description, frame in cases.items():
            with self.subTest(description=description):
                self.write_frame(frame)
                with self.assertRaises(ValueError):
                    validate_manifest(self.csv_path, root=self.root)


if __name__ == "__main__":
    unittest.main()
