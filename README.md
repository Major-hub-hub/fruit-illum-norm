# Fruit Illumination Normalisation

Project workspace for studying illumination normalisation on binary fruit-image classification.

## Conditions

- `B0`: original RGB image
- `B1`: Gaussian blur
- `E1`: Gray World
- `E2`: CLAHE
- `E3`: Gray World + CLAHE
- `E4`: Gray World + CLAHE + Gaussian (optional; not evaluated in the five-model study)

The fixed train/validation/test split is stored in `data/splits/split_seed42.csv`.

## Environment and dataset check

The `fruit-split` environment supports both dataset inspection and the PyTorch
dataset loader. The loader requires PyTorch, matching torchvision, and OpenCV;
OpenCV is imported by the preprocessing registry even for condition `B0`.
`environment.yml` specifies the tested CPU package versions from Anaconda's
default channel for this Windows setup.

For a fresh setup, create the environment once:

```powershell
conda env create -f environment.yml
conda activate fruit-split
```

Run the data-pipeline check from the repository root:

```powershell
python -m src.data.dataset
python -m src.data.dataset --condition E3 --out outputs/figures/dataset_sanity_grid_E3.png
```

The check validates the manifest, loads a 32-image batch from each split, and
saves a preview. Expected split sizes are 2,560 / 320 / 320 and image tensors
have shape `(32, 3, 250, 250)`. The default preview is
`outputs/figures/dataset_sanity_grid.png`. Plotting uses a non-interactive backend.
The combined E3 preprocessing function supports Windows spawn workers.

On Windows, PyTorch archive extraction can require support for long file paths;
see [Microsoft's long-path documentation](https://learn.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation).

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

Run split-allocation and dataset-loader checks with:

```bash
python -m unittest discover -s tests -v
```



## Experimental results (2026-11-25)

Single fixed seed-42 split, five separately trained ResNet50 models, validation-selected
stage-2 checkpoints. Each lighting condition uses the same 320 held-out files
(160 fresh, 160 rotten). Stress is applied before model-specific preprocessing:
dark ×0.50, bright ×1.40, warm RGB gains (1.20, 1.00, 0.75), clipped to uint8.

| Model | Normal (%) | Dark (%) | Bright (%) | Warm cast (%) | Mean stressed (%) | Mean drop (pp) | Stressed false negatives |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0 | 98.75 | 98.13 | 98.44 | 97.50 | 98.02 | 0.729 | 16 |
| B1 | 98.75 | 97.81 | 98.44 | 98.44 | 98.23 | 0.521 | 8 |
| E1 | 99.06 | 98.75 | 98.75 | 99.06 | 98.85 | 0.208 | 8 |
| E2 | 99.06 | 97.81 | 98.44 | 97.81 | 98.02 | 1.042 | 12 |
| E3 | 99.06 | 98.44 | 98.75 | 99.06 | 98.75 | 0.313 | 6 |

Mean drop = clean accuracy minus mean stressed accuracy, in percentage points.
Stressed false negatives count rotten-as-fresh predictions across three stress
conditions (480 rotten image-condition instances, not independent images).
E1 has the highest mean stressed accuracy; E3 has the fewest missed rotten
instances (6 versus 8 for B1). These are descriptive single-seed results, not
proof of statistical significance or real-world generalisation.

The uploaded evaluation archive was audited for 20 matching metrics/prediction
pairs, 320 unique paths per evaluation, identical image labels across conditions,
and agreement between recomputed accuracy/confusion matrices and saved metrics.
Independent inference and raw dataset hash verification remain to be performed locally.

## Reproduce evaluation and analysis

With the raw dataset, fixed manifest and all five stage-2 checkpoints available
locally, activate the working GPU environment and run from the repository root:

```powershell
conda activate fruit-gpu
python -m unittest discover -s tests -v
foreach ($condition in @('B0','B1','E1','E2','E3')) {
    python -m src.evaluate --condition $condition --stage 2 --seed 42 --stress all
    if ($LASTEXITCODE -ne 0) { throw "Evaluation failed: $condition" }
}
```

Run notebooks 02, 03 and 04 to export preprocessing, quantitative results and
failure-analysis figures. Keep the original metrics and checkpoints backed up
outside Git; generated outputs are not included in this repository. Record the
GPU environment (e.g. conda list --explicit) and commit SHA before rerunning.
No additional training is required to reproduce the evaluation.
