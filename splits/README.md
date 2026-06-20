# Splits

This directory documents the expected format for fixed five-fold split files.

Use anonymous case ids only. The ids should match the `case_id` column in the
label file and the case names expected by the data loader.

The example file below is synthetic and only demonstrates structure:

```text
private_cv5_seed2026_example.json
```

Recommended fields:

- `seed`: random seed used when creating the split.
- `folds`: list of fold objects.
- `train`, `val`, `test`: anonymous case-id lists for each fold.

When releasing a split manifest for review, do not include hospital ids,
patient ids, accession numbers, full source paths, or acquisition dates.
