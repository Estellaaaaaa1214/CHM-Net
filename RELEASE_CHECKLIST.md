# Release Checklist

Use this checklist before refreshing the anonymous mirror.

- Root README explains the repository contents, evidence map, and reading order.
- `VALIDATION.md` records metric and sanitization checks.
- `REPO_STRUCTURE.md` lists the expected reviewer-facing repository layout.
- Full logs are present under `logs/`, not only extracted summary values.
- Main CHM-Net metrics in `logs/private/chm_net_private_5fold_full.log` match
  `results/private_dataset_reported_metrics.csv`.
- Ablation metrics in `logs/private/chm_net_ablation_5fold_full.log` match
  `results/private_ablation_reported_metrics.csv`.
- Public NoduleMNIST3D log is present under `logs/public_medmnist/`.
- `src/chm_net/`, `src/baselines/`, and `src/public_medmnist/` all exist.
- Public-MedMNIST scripts reference repository-local filenames.
- Any split examples use anonymous case ids only.
- No local labels spreadsheet, manuscript drafts, shell-history file, or
  owner-only upload instruction file is included.
- No author names, institutions, personal email addresses, local absolute
  paths, or container identifiers remain in the reviewer-facing files.
