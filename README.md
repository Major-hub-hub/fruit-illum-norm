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

## Typical workflow

```bash
python -m src.data.make_split --seed 42
python -m src.train --condition E3 --seed 42
python -m src.evaluate --condition E3 --seed 42
```

