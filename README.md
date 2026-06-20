# CHM-Net on GBNPC2026

This is the minimal anonymous release for CHM-Net on the GBNPC2026 MRI cohort.
It contains the main CHM-Net training code, the reported training log, a compact
metric table, and a synthetic label-file example.

## Files

- `chm_net.py`: main CHM-Net training script.
- `logs/CHM-Net_GBNPC2026_training.log`: full sanitized training log for the
  reported CHM-Net result on GBNPC2026.
- `results/CHM-Net_GBNPC2026_metrics.csv`: reported CHM-Net metrics.
- `data/GBNPC2026_label_example.csv`: synthetic label-file example.
- `requirements.txt`: Python package requirements.

## Data Format

The label file should contain one row per case:

```csv
case_id,label
case_0001,0
case_0002,1
```

`label=0` denotes the low-density group and `label=1` denotes the high-density
group. The image root should contain one folder per `case_id`, with three MRI
modalities:

```text
<GBNPC2026_IMAGE_ROOT>/
  case_0001/
    T1WI/case_0001.nii.gz
    T1WI+C/case_0001.nii.gz
    T2WI/case_0001.nii.gz
```

## Run

```bash
python chm_net.py \
  --img-root <GBNPC2026_IMAGE_ROOT> \
  --label-file <GBNPC2026_LABEL_FILE> \
  --id-col case_id \
  --label-col label \
  --output-dir runs/CHM-Net_GBNPC2026
```

By default, `chm_net.py` uses a stratified 70/30 split with `--split-seed 42`.
Users may change `--train-ratio` and `--split-seed` for their own runs.

## Reported Result

The reported CHM-Net result on GBNPC2026 is:

| Dataset | Model | ACC | AUC | F1 | Sens. | Spec. |
| --- | --- | --- | --- | --- | --- | --- |
| GBNPC2026 | CHM-Net | 67.75 +/- 2.46 | 69.69 +/- 1.19 | 66.47 +/- 4.19 | 64.74 +/- 6.80 | 70.64 +/- 2.95 |

The full training log is provided at:

```text
logs/CHM-Net_GBNPC2026_training.log
```

Local paths and machine identifiers in the log were replaced with neutral
placeholders. Metric values and fold-level traces were not changed.
