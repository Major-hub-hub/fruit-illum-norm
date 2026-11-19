# Fruit Illumination Normalisation

Project workspace for studying illumination normalisation on binary fruit-image classification.

## Conditions

- `B0`: original RGB image
- `B1`: Gaussian blur
- `E1`: Gray World
- `E2`: CLAHE
- `E3`: Gray World + CLAHE
- `E4`: stress-test preprocessing pipeline

The fixed train/validation/test split is stored in `data/splits/split_seed42.csv`.

## Duplicate grouping and split policy

`src/data/make_split.py` hashes the original files with SHA-256 and assigns every
byte-identical group to one split. All 3,200 files are retained. Each of the 16
fruit/state classes has exactly 160 training, 20 validation and 20 test files.
The CSV keeps the original five columns and adds `sha256` for auditing.

Seed 42, sorted paths and sorted content groups make regeneration deterministic.
The allocator jointly chooses validation and test groups to satisfy the exact
quotas; it fails before replacing the manifest if a valid allocation is impossible
or identical contents have conflicting labels. Augmented folders are excluded.
Changing `--seed` also changes the default output filename to `split_seed<seed>.csv`.

Grouping preserves the paper's file counts and class-wise ratios while adding
an exact-duplicate leakage control. It creates a new assignment of images to
splits. The paper does not provide an exact seed-42 assignment to reproduce.
See Section III.F of the [author-uploaded paper](https://www.researchgate.net/publication/389082092_Fresh_or_Rotten_Enhancing_Rotten_Fruit_Detection_with_Deep_Learning_and_Gaussian_Filtering).

Retained copies can still repeat within a split, so file counts and unique-image
counts differ. Notebook 01 reports both and checks that no exact-content group
crosses splits. Byte hashes do not detect every re-encoded or near-duplicate view.
The previous manifest is archived at `data/splits/archive/split_seed42_ungrouped.csv`
for provenance; training uses only `data/splits/split_seed42.csv`.

## Typical workflow

```bash
python -m src.data.make_split --seed 42
python -m src.train --condition E3 --seed 42
python -m src.evaluate --condition E3 --seed 42
```

Run split-allocation checks with:

```bash
python -m unittest discover -s tests -v
```

