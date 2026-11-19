"""Checks for exact file quotas and content-group separation."""

import contextlib
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import pandas as pd

from src.data import make_split


def class_frame(fruit="apple", state="fresh", sizes=None):
    """Build metadata without requiring image files or image decoders."""
    sizes = [1] * 200 if sizes is None else sizes
    cls = f"{state}_{fruit}"
    rows = []
    for group_number, size in enumerate(sizes):
        digest = hashlib.sha256(f"{cls}:{group_number}".encode()).hexdigest()
        for copy_number in range(size):
            rows.append({
                "path": f"data/raw/{cls}/{group_number:03d}_{copy_number:03d}.jpg",
                "label": int(state == "rotten"),
                "fruit": fruit,
                "cls": cls,
                "sha256": digest,
            })
    return pd.DataFrame(rows)


def full_frame():
    frames = []
    for state in ("fresh", "rotten"):
        for fruit in make_split.FRUITS:
            # Match the actual dataset's seven duplicate pairs in fresh orange.
            sizes = [2] * 7 + [1] * 186 if (state, fruit) == ("fresh", "orange") else None
            frames.append(class_frame(fruit, state, sizes))
    return pd.concat(frames, ignore_index=True)


class GroupedSplitTests(unittest.TestCase):
    def assert_quotas_and_groups(self, result):
        self.assertEqual(list(result.columns), make_split.MANIFEST_COLUMNS)
        self.assertFalse(result["path"].duplicated().any())
        self.assertEqual(set(result["split"]), {"train", "val", "test"})
        self.assertTrue(result.groupby("sha256")["split"].nunique().eq(1).all())
        for _, rows in result.groupby("cls"):
            size = len(rows)
            self.assertEqual(
                rows["split"].value_counts().to_dict(),
                {"train": size * 8 // 10, "val": size // 10, "test": size // 10},
            )

    def test_all_files_retained_with_exact_quotas_and_indivisible_groups(self):
        source = full_frame()
        original = source.copy(deep=True)
        result = make_split.split_grouped(source)

        self.assertEqual(len(result), 3200)
        self.assertEqual(result["sha256"].nunique(), 3193)
        self.assertEqual(set(result["path"]), set(source["path"]))
        self.assertEqual(result["split"].value_counts().to_dict(), {
            "train": 2560, "val": 320, "test": 320,
        })
        self.assert_quotas_and_groups(result)
        pd.testing.assert_frame_equal(source, original)

    def test_seed_repeats_and_input_order_or_index_cannot_change_membership(self):
        source = full_frame()
        expected = make_split.split_grouped(source, seed=42)
        pd.testing.assert_frame_equal(expected, make_split.split_grouped(source, seed=42))

        shuffled = source.sample(frac=1, random_state=123)
        shuffled.index = [0] * len(shuffled)
        pd.testing.assert_frame_equal(expected, make_split.split_grouped(shuffled, seed=42))

        other_seed = make_split.split_grouped(source, seed=43)
        self.assertFalse(expected["split"].equals(other_seed["split"]))
        self.assert_quotas_and_groups(other_seed)

    def test_joint_packing_finds_solution_that_sequential_selection_can_miss(self):
        # Validation 12+4+4 would leave no test subset of 20. A valid packing is
        # train 156+4, validation 16+4, test 12+8 (val/test can be exchanged).
        source = class_frame(sizes=[156, 16, 12, 8, 4, 4])
        for seed in range(12):
            with self.subTest(seed=seed):
                result = make_split.split_grouped(source, seed=seed)
                self.assert_quotas_and_groups(result)

    def test_impossible_indivisible_group_raises_instead_of_relaxing_quotas(self):
        source = class_frame(sizes=[161, 19, 20])
        with self.assertRaisesRegex(ValueError, "Cannot keep duplicate groups intact"):
            make_split.split_grouped(source)

    def test_identical_content_with_conflicting_metadata_is_rejected(self):
        source = pd.concat([
            class_frame("apple", "fresh"), class_frame("banana", "fresh"),
        ], ignore_index=True)
        source.loc[200, "sha256"] = source.loc[0, "sha256"]
        with self.assertRaisesRegex(ValueError, "conflicting fruit/class labels"):
            make_split.split_grouped(source)

        source = pd.concat([
            class_frame("apple", "fresh"), class_frame("apple", "rotten"),
        ], ignore_index=True)
        source.loc[200, "sha256"] = source.loc[0, "sha256"]
        with self.assertRaisesRegex(ValueError, "conflicting fruit/class labels"):
            make_split.split_grouped(source)

    def test_invalid_metadata_is_rejected(self):
        valid = class_frame()
        cases = {
            "missing column": valid.drop(columns="sha256"),
            "empty dataset": valid.iloc[:0],
            "missing value": valid.assign(fruit=None),
            "blank path": valid.assign(path="  "),
            "duplicate path": pd.concat([valid.iloc[:199], valid.iloc[:1]], ignore_index=True),
            "nonbinary label": valid.assign(label=2),
            "unknown fruit": valid.assign(fruit="dragonfruit"),
            "class label disagreement": valid.assign(label=1),
            "malformed hash": valid.assign(sha256="not-a-sha256"),
            "uppercase hash": valid.assign(sha256="A" * 64),
            "nonintegral quotas": valid.iloc[:199],
        }
        for description, frame in cases.items():
            with self.subTest(description=description):
                with self.assertRaises(ValueError):
                    make_split.split_grouped(frame)


class CollectorTests(unittest.TestCase):
    def test_hashes_actual_bytes_and_excludes_augmented_or_unrecognized_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            content = b"same-original-image-bytes"
            originals = {
                "Original Image/FreshApple/b.JPG": content,
                "Original Image/FreshApple/a.jpg": content,
                "Original Image/RottenBanana/c.png": b"different-original-image-bytes",
            }
            excluded = {
                "Augmented Images/FreshApple/copy.jpg": content,
                "Original Image/FreshApple/augment_copy.jpg": content,
                "Original Image/FreshApple/notes.txt": b"notes",
                "Original Image/Unknown/unrecognized.jpg": content,
            }
            for relative, body in {**originals, **excluded}.items():
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(body)
            # A directory whose name has an image suffix must not be opened.
            (root / "Original Image/FreshApple/directory.jpg").mkdir()

            result = make_split.collect(root)
            expected_paths = sorted((root / name).as_posix() for name in originals)
            self.assertEqual(result["path"].tolist(), expected_paths)
            self.assertEqual(result["cls"].tolist(), ["fresh_apple", "fresh_apple", "rotten_banana"])
            self.assertEqual(result["label"].tolist(), [0, 0, 1])
            for row in result.to_dict("records"):
                self.assertEqual(row["sha256"], hashlib.sha256(Path(row["path"]).read_bytes()).hexdigest())
            self.assertEqual(result["sha256"].nunique(), 2)
            for relative, body in {**originals, **excluded}.items():
                self.assertEqual((root / relative).read_bytes(), body)

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                make_split.main(["--root", str(root), "--list"])
            self.assertNotIn("Augmented", output.getvalue())
            self.assertIn("    2  Original Image/FreshApple", output.getvalue())

    def test_missing_root_and_empty_root_produce_clear_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "does not exist"):
                make_split.collect(root / "missing")
            empty = make_split.collect(root)
            self.assertEqual(list(empty.columns), ["path", "label", "fruit", "cls", "sha256"])
            with self.assertRaisesRegex(ValueError, "No recognized original images"):
                make_split.split_grouped(empty)


class ManifestWriteTests(unittest.TestCase):
    def test_write_produces_complete_csv_with_expected_schema(self):
        frame = make_split.split_grouped(class_frame())
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nested" / "split.csv"
            make_split.write_manifest(frame, output)
            actual = pd.read_csv(output)
            pd.testing.assert_frame_equal(actual, frame)
            self.assertEqual(list(output.parent.iterdir()), [output])

    def test_failed_csv_write_preserves_previous_manifest_and_cleans_temporary_file(self):
        frame = class_frame()

        def fail_after_partial_write(handle, **kwargs):
            handle.write("partial CSV")
            raise OSError("simulated full disk")

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "split.csv"
            original = b"previous complete manifest\n"
            output.write_bytes(original)
            with mock.patch.object(frame, "to_csv", side_effect=fail_after_partial_write):
                with self.assertRaisesRegex(OSError, "simulated full disk"):
                    make_split.write_manifest(frame, output)
            self.assertEqual(output.read_bytes(), original)
            self.assertEqual(list(output.parent.iterdir()), [output])


class CommandLineTests(unittest.TestCase):
    def test_invalid_class_counts_and_conflicting_groups_do_not_overwrite_output(self):
        counts_invalid = full_frame().iloc[:-1]
        labels_conflict = full_frame()
        labels_conflict.loc[200, "sha256"] = labels_conflict.loc[0, "sha256"]
        impossible = pd.concat([
            full_frame().loc[lambda rows: rows["cls"] != "fresh_apple"],
            class_frame(sizes=[161, 19, 20]),
        ], ignore_index=True)
        for description, frame in (
            ("class counts", counts_invalid),
            ("cross-class conflicting content", labels_conflict),
            ("impossible group allocation", impossible),
        ):
            with self.subTest(description=description), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "split.csv"
                original = b"previous manifest must survive\n"
                output.write_bytes(original)
                with mock.patch.object(make_split, "collect", return_value=frame):
                    with contextlib.redirect_stderr(io.StringIO()):
                        with self.assertRaises(SystemExit) as raised:
                            make_split.main(["--out", str(output)])
                self.assertEqual(raised.exception.code, 2)
                self.assertEqual(output.read_bytes(), original)
                self.assertEqual(list(output.parent.iterdir()), [output])

    def test_default_filename_tracks_seed(self):
        with mock.patch.object(make_split, "collect", return_value=full_frame()):
            with mock.patch.object(make_split, "write_manifest") as write:
                with contextlib.redirect_stdout(io.StringIO()):
                    make_split.main(["--seed", "123"])
        write.assert_called_once()
        frame, output = write.call_args.args
        self.assertEqual(output, Path("data/splits/split_seed123.csv"))
        self.assertEqual(list(frame.columns), make_split.MANIFEST_COLUMNS)
        self.assertEqual(len(frame), 3200)


if __name__ == "__main__":
    unittest.main()
