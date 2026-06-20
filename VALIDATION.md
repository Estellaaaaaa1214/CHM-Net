# Validation Notes

This file records the repository-level checks used to prepare the anonymous
release package.

## Content Checks

- Main CHM-Net five-fold evidence is present as a full sanitized log:
  `logs/private/chm_net_private_5fold_full.log`.
- Macro-micro ablation evidence is present as a full sanitized log:
  `logs/private/chm_net_ablation_5fold_full.log`.
- Public NoduleMNIST3D evidence is present as a full sanitized log:
  `logs/public_medmnist/nodulemnist3d_incremental_ablation_full.log`.
- Reported metrics are mirrored in CSV form under `results/`.
- Model, baseline, and public validation scripts are expected under `src/`.

## Metric Cross-Checks

The final CHM-Net five-fold summary in
`logs/private/chm_net_private_5fold_full.log` matches the CHM-Net row in
`results/private_dataset_reported_metrics.csv` after converting decimals to
percentages:

| Metric | Log value | CSV value |
| --- | --- | --- |
| ACC | 0.6775 +/- 0.0246 | 67.75 +/- 2.46 |
| AUC | 0.6969 +/- 0.0119 | 69.69 +/- 1.19 |
| F1 | 0.6647 +/- 0.0419 | 66.47 +/- 4.19 |
| Sensitivity | 0.6474 +/- 0.0680 | 64.74 +/- 6.80 |
| Specificity | 0.7064 +/- 0.0295 | 70.64 +/- 2.95 |

The ablation summary in `logs/private/chm_net_ablation_5fold_full.log` matches
`results/private_ablation_reported_metrics.csv` after converting decimals to
percentages.

## Sanitization Checks

The release logs were scanned for local path and prompt remnants. Original
Linux mount prefixes, original project-root label-file paths, original label
spreadsheet names, local container prompts, and original private MRI folder
names should not appear in the released files.

Sanitized placeholders are expected:

- `<PRIVATE_MRI_ROOT>`
- `<PRIVATE_LABEL_FILE>`
- `<PUBLIC_DATA_ROOT>`
- `<RUN_ENV>#`
- `root@<CONTAINER_ID>`

## Final Upload Check

Before refreshing the anonymous mirror, open the anonymous repository page and
confirm that:

- `README.md` renders correctly at the repository root.
- `VALIDATION.md`, `REPO_STRUCTURE.md`, and `logs/README.md` are visible.
- Full logs are present, not only summary snippets.
- `PUSH_INSTRUCTIONS.md` or owner-only workflow notes are not present in the
  reviewer-facing repository.
- No spreadsheet containing real labels, patient identifiers, or source image
  paths is included.
