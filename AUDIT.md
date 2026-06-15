# Release Audit

## Included

- `src/chm_net/*`: anonymized CHM-Net training and ablation code.
- `src/baselines/*`: private-dataset five-fold baseline utilities.
- `src/public_medmnist/*`: public dataset training scripts with local path defaults removed.
- `logs/private/chm_net_private_5fold_summary.log`: matches the manuscript CHM-Net private row.
- `logs/private/chm_net_ablation_5fold_summary.log`: matches the manuscript ablation table.
- `logs/private_baselines/*_summary.log`: summary excerpts matching manuscript rows for Med3D, AMSNet, X3D, 3DCT-ICH, and XFMamba.
- `checkpoints/final_macro_micro_locked_auc07116_anonymous.pth`: anonymized locked single-split checkpoint for model loading/inference checks.

## Excluded From Release Logs

- The local integrated-results spreadsheet: contains older values that do not match the final manuscript tables.
- The local NoduleMNIST3D exploratory nohup log: summary values do not match the final public CHM-Net table.
- The local shell-transcript private/ablation log: older run; summary values do not match the final ablation table.
- The available ResNet summary matches no final manuscript row; not included as evidence for a manuscript ResNet-50 row.
- Matching raw five-fold summaries were not found for M3T, SwinTransformer-3D, or MedVit-3D in the provided local logs.

## XFMamba Note

The included XFMamba release file contains only the verified summary block. The
source shell transcript also contained environment warning/fallback lines outside
the summary block, with fallback status `fallback`. This release does
not rewrite those raw execution lines as if they did not occur.

## Checkpoint Note

The included locked checkpoint metadata records a 70/30 single-split evaluation
with AUC `0.7116`. It is included as a loadable final model checkpoint, not as
the source of the reported five-fold metric table.

## Integrity Rule

No metric was randomized or fabricated in this package. Logs were only
sanitized, excerpted, and reformatted when their values matched the manuscript.
