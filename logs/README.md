# Logs

This directory contains full sanitized logs for the reviewer package.

## Files

| File | Purpose |
| --- | --- |
| `private/chm_net_private_5fold_full.log` | Main CHM-Net private-cohort five-fold training and evaluation log. |
| `private/chm_net_ablation_5fold_full.log` | Macro-micro ablation five-fold log with setting-level and fold-level summaries. |
| `public_medmnist/nodulemnist3d_incremental_ablation_full.log` | Public NoduleMNIST3D incremental ablation log. |
| `supplementary_terminal/terminal_ablation_dump_full.log` | Supplemental terminal dump retained for audit trail completeness. |
| `SANITIZATION_NOTES.md` | Placeholder policy used when preparing logs. |

## How to Read the Logs

- Start with the tail of `private/chm_net_private_5fold_full.log` for the main
  five-fold summary.
- Start with the tail of `private/chm_net_ablation_5fold_full.log` for the
  ablation summary and fold-level rows.
- Use `results/*.csv` for table-level metric lookup.
- Use full logs when checking whether fold, epoch, and early-stopping traces are
  present.

## Notes

The supplemental terminal dump is included for completeness. It may contain
command-output context from an author-run environment after sanitization; the
primary evidence files are the logs under `private/`, `public_medmnist/`, and
the metric CSV files under `results/`.
