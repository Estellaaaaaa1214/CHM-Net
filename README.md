# CHM-Net Anonymous Release

This repository contains the anonymous code and experimental evidence for
CHM-Net, a center-heatmap driven macro-micro modeling network for MRI-based
microbial density stratification.

## What Is Included

- `src/chm_net/`: CHM-Net model implementation, private five-fold runner, and
  macro-micro ablation runner.
- `src/baselines/`: private-dataset baseline training utilities and wrappers.
- `src/public_medmnist/`: public NoduleMNIST3D and AdrenalMNIST3D validation
  scripts.
- `logs/`: full sanitized training logs, including epoch-level traces,
  fold-level metrics, and final summaries.
- `results/`: machine-readable copies of the reported private, ablation, and
  public-dataset metrics.
- `data_examples/`: label-schema examples and input-format notes.
- `checkpoints/`: locked checkpoint metadata and loading notes.

## Evidence Map

| Paper evidence | Repository evidence |
| --- | --- |
| Main CHM-Net five-fold private-cohort result | `logs/private/chm_net_private_5fold_full.log`; `results/private_dataset_reported_metrics.csv` |
| Macro-micro ablation table | `logs/private/chm_net_ablation_5fold_full.log`; `results/private_ablation_reported_metrics.csv` |
| Public NoduleMNIST3D validation | `logs/public_medmnist/nodulemnist3d_incremental_ablation_full.log`; `results/public_dataset_reported_metrics.csv` |
| Private baseline comparison table | `results/private_dataset_reported_metrics.csv`; `src/baselines/` |
| Checkpoint loading and single-split sanity check | `checkpoints/README.md`; checkpoint metadata/log files |

## Private-Cohort Five-Fold Run

The main CHM-Net five-fold run is documented in:

```text
logs/private/chm_net_private_5fold_full.log
```

The final summary in that log reports:

| Metric | Mean +/- Std |
| --- | --- |
| Accuracy | 0.6775 +/- 0.0246 |
| Precision | 0.6854 +/- 0.0183 |
| Recall / Sensitivity | 0.6474 +/- 0.0680 |
| F1 | 0.6647 +/- 0.0419 |
| AUC | 0.6969 +/- 0.0119 |
| Specificity | 0.7064 +/- 0.0295 |

The same values are mirrored in percentage form in:

```text
results/private_dataset_reported_metrics.csv
```

## Ablation Run

The macro-micro ablation study is documented in:

```text
logs/private/chm_net_ablation_5fold_full.log
results/private_ablation_reported_metrics.csv
```

The ablation log contains both the setting-level summary and fold-level rows
with best epoch, AUC, ACC, F1, sensitivity, specificity, and confusion-matrix
counts.

## Running the Code

Use the scripts with local paths that follow the schema documented in
`data_examples/README.md`.

Example five-fold CHM-Net command:

```bash
python src/chm_net/final_5fold.py \
  --image-root <PRIVATE_MRI_ROOT> \
  --label-file <PRIVATE_LABEL_FILE> \
  --detector-ckpt <DETECTOR_CKPT> \
  --split-file <CV_SPLIT_JSON> \
  --out-dir runs/chm_net_private_5fold
```

Example ablation command:

```bash
python src/chm_net/macro_micro_ablation_5fold.py \
  --image-root <PRIVATE_MRI_ROOT> \
  --label-file <PRIVATE_LABEL_FILE> \
  --split-file <CV_SPLIT_JSON> \
  --out-dir runs/chm_net_ablation_5fold
```

Example public NoduleMNIST3D command:

```bash
python src/public_medmnist/medmnist3d_incremental_ablation_5fold.py \
  --dataset nodulemnist3d \
  --root <PUBLIC_DATA_ROOT> \
  --download \
  --out-dir runs/nodulemnist3d
```

## Log Sanitization

The released logs preserve the experimental traces and metric values. Local
machine prompts, container ids, local cache roots, MRI root paths, and label-file
paths were replaced with placeholders. The exact replacement policy is recorded
in `logs/SANITIZATION_NOTES.md`.

## Recommended Reading Order

1. `README.md`
2. `REPO_STRUCTURE.md`
3. `VALIDATION.md`
4. `logs/README.md`
5. `results/*.csv`
6. `src/chm_net/`
