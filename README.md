# CHM-Net Anonymous Release

This repository contains anonymized training code, expected metric tables, data
format examples, and sanitized log summaries for **CHM-Net: Center Heatmap-driven
Macro-Micro Modeling Network for MRI-based Microbial Density Stratification**.

## Contents

- `src/chm_net/`: CHM-Net model, private five-fold CV runner, and ablation runner.
- `src/baselines/`: private-dataset five-fold baseline training utilities.
- `src/public_medmnist/`: public NoduleMNIST3D and AdrenalMNIST3D training scripts.
- `data_examples/`: anonymized private metadata schema and directory example.
- `logs/`: sanitized summaries whose values were checked against the manuscript.
- `results/`: manuscript metric tables in machine-readable CSV form.
- `checkpoints/`: anonymized locked checkpoint for model loading checks.

## Private Dataset

The private MRI cohort is not included. Use an anonymized label file with columns
`case_id` and `label`, and organize the three MRI modalities as described in
`data_examples/README.md`.

## Reproducing the Private CHM-Net Five-Fold Run

```bash
python src/chm_net/final_5fold.py \
  --base-file src/chm_net/macro_micro_density_mil.py \
  --img-root data/private_mri \
  --excel-path metadata/private_labels_anonymous.xlsx \
  --id-col case_id \
  --label-col label \
  --detector-ckpt checkpoints/detector_best.pth \
  --output-dir runs/private_chm_net_cv5 \
  --run-name chm_net_private_cv5 \
  --classifier-epochs 50 \
  --classifier-batch-size 2 \
  --n-splits 5 \
  --inner-val-ratio 0.20 \
  --seed 2026 \
  --shared-splits splits/private_cv5_seed2026.json
```

## Reproducing the Private Ablation

```bash
python src/chm_net/macro_micro_ablation_5fold.py \
  --base-file src/chm_net/macro_micro_density_mil.py \
  --img-root data/private_mri \
  --excel-path metadata/private_labels_anonymous.xlsx \
  --id-col case_id \
  --label-col label \
  --detector-ckpt checkpoints/detector_best.pth \
  --output-dir runs/private_ablation_cv5 \
  --run-name chm_net_ablation_private_cv5 \
  --classifier-epochs 50 \
  --classifier-batch-size 2 \
  --n-splits 5 \
  --inner-val-ratio 0.20 \
  --seed 2026 \
  --shared-splits splits/private_cv5_seed2026.json
```

## Expected Private Result

The verified CHM-Net private five-fold summary is:

- ACC: `67.75 +/- 2.46`
- AUC: `69.69 +/- 1.19`
- F1: `66.47 +/- 4.19`
- Sensitivity: `64.74 +/- 6.80`
- Specificity: `70.64 +/- 2.95`

See `results/private_dataset_expected_metrics.csv` and
`results/private_ablation_expected_metrics.csv` for the complete manuscript
tables.

## Private Baseline Evidence

Sanitized private five-fold summary excerpts are included for Med3D, AMSNet,
X3D, 3DCT-ICH, and XFMamba under `logs/private_baselines/`. These excerpts
match the corresponding manuscript rows. The available ResNet transcript does
not match a manuscript ResNet-50 row, and matching raw five-fold summaries were
not found for M3T, SwinTransformer-3D, or MedVit-3D in the provided local logs;
their manuscript values remain listed in `results/private_dataset_expected_metrics.csv`.

## Locked Checkpoint

`checkpoints/final_macro_micro_locked_auc07116_anonymous.pth` is included as an
anonymized locked checkpoint for model loading and single-split inference checks.
Its metadata records a 70/30 evaluation with AUC `0.7116`; it is not the source
of the reported five-fold CHM-Net result.

## Notes on Logs

Only summaries whose values match the manuscript are included as release logs.
Older exploratory logs and spreadsheets with inconsistent values were excluded;
see `AUDIT.md` for details.
