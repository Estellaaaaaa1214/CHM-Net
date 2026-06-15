# Checkpoints

`final_macro_micro_locked_auc07116_anonymous.pth` status: included.

This locked checkpoint is provided for model loading and single-split inference
checks. Its adjacent metadata records a 70/30 evaluation with AUC `0.7116`.
It is **not** the source of the reported five-fold CHM-Net result; the five-fold
result is documented in `logs/private/chm_net_private_5fold_summary.log` and
`results/private_dataset_expected_metrics.csv`.

The checkpoint copy was anonymized by replacing embedded local path strings in
the PyTorch zip metadata while leaving tensor payloads intact.
