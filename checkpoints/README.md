# Checkpoints

This directory contains the CHM-Net checkpoint released for model loading and
single-split sanity-check evaluation on GBNPC2026.

## Files

- `final_macro_micro_locked_auc07116_anonymous.pth`: locked CHM-Net checkpoint.
- `final_checkpoint_info_anonymous.json`: metadata and metrics for the
  checkpoint sanity-check evaluation.
- `final_eval_anonymous.log`: sanitized evaluation log for the checkpoint.

## Scope

The checkpoint evaluation uses a stratified 70/30 split and reports AUC
`0.7116` on the held-out split. The paper-level five-fold result is documented
separately in:

```text
logs/CHM-Net_GBNPC2026_training.log
results/CHM-Net_GBNPC2026_metrics.csv
```

Therefore, the checkpoint is provided for loading and sanity checks. The
five-fold training log and metric CSV are the evidence for the reported
cross-validation result.
