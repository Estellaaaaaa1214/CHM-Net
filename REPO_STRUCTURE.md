# Recommended Repository Structure

Use the following structure for the anonymous review repository.

```text
CHM-Net/
  README.md
  VALIDATION.md
  REPO_STRUCTURE.md
  requirements.txt
  .gitignore
  .gitattributes

  src/
    chm_net/
      macro_micro_density_mil.py
      final_5fold.py
      macro_micro_ablation_5fold.py
      macro_micro_ablation.py
    baselines/
      common_3d_train_5fold.py
      train_private_model_5fold.py
      AMSNet_7_3_private_dataset.py
      Med3D_ResNet_standalone_7_3.py
      MedVit_3D_transfer_7_3.py
      M3T_transfer_7_3.py
      SwinTransformer3D_transfer_7_3.py
      X3D_transfer_7_3.py
      XFMamba_7_3_private_dataset.py
      train_3dct_ich_5fold.py
    public_medmnist/
      medmnist3d_incremental_ablation_5fold.py
      nodule_macro_micro_mil.py
      adrenal_macro_micro_mil.py
      train_nodulemnist3d_three_models.py
      train_adrenalmnist3d_three_models.py

  logs/
    README.md
    SANITIZATION_NOTES.md
    private/
      chm_net_private_5fold_full.log
      chm_net_ablation_5fold_full.log
    public_medmnist/
      nodulemnist3d_incremental_ablation_full.log
    supplementary_terminal/
      terminal_ablation_dump_full.log

  results/
    private_dataset_reported_metrics.csv
    private_ablation_reported_metrics.csv
    public_dataset_reported_metrics.csv

  data_examples/
    README.md
    private_labels_anonymous_example.csv

  splits/
    README.md
    private_cv5_seed2026_example.json

  checkpoints/
    README.md
    final_checkpoint_info_anonymous.json
    final_eval_anonymous.log
    final_macro_micro_locked_auc07116_anonymous.pth
```

## Keep

- Keep `src/chm_net/`, especially `macro_micro_density_mil.py`,
  `final_5fold.py`, and `macro_micro_ablation_5fold.py`.
- Keep `src/baselines/`, because it documents the model registry and private
  baseline runner used to produce the comparison table.
- Keep `src/public_medmnist/`, because it gives reviewers a public-data
  execution path.
- Keep `checkpoints/final_macro_micro_locked_auc07116_anonymous.pth` if file
  size limits allow it.
- Keep the current result CSVs, but prefer the `reported_metrics` filenames in
  this package.

## Add or Replace

- Replace the root `README.md` with the rewritten root README in this package.
- Replace `VALIDATION.md` with the rewritten validation notes in this package.
- Add `REPO_STRUCTURE.md`.
- Replace summary-only logs with full logs, or keep both full and summary logs.
- Add `logs/README.md` and `logs/SANITIZATION_NOTES.md`.
- Add a `splits/README.md` and, if safe, a fold-split example JSON with
  anonymized case ids.
- Add `data_examples/README.md` with exact column meanings and expected label
  values.

## Remove From Reviewer-Facing Repository

- Remove `PUSH_INSTRUCTIONS.md`.
- Do not upload local upload instructions, GitHub account notes, shell history,
  or owner-only workflow documents.
- Do not upload spreadsheets with real labels or absolute source image paths.
- Do not upload draft manuscripts, response drafts, or local office documents.

## Small Code-Side Fixes Worth Doing

- Update public-MedMNIST wrapper defaults so they point to files that actually
  exist in this repository.
- Ensure README commands either use placeholders or point to files that are
  present in the repository, such as the documented example files.
- If exact dependency versions are known, pin them in `requirements.txt` or add
  an environment note with Python, PyTorch, CUDA, MONAI, and MedMNIST versions.
