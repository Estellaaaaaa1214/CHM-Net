# Data Format

This directory documents the expected input schema for running the code with a
cohort prepared in the same format.

## Label File

The label file should be a CSV or Excel table with one row per patient/case.
The minimal columns are:

| Column | Meaning |
| --- | --- |
| `case_id` | Anonymous case identifier used to match an image folder or volume. |
| `label` | Binary density-stratification label. Use `0` for low density and `1` for high density. |
| `split` | Optional split label when using a pre-defined split. |
| `fold` | Optional fold id when using pre-defined five-fold cross-validation. |

See `private_labels_anonymous_example.csv` for an example with synthetic case
ids.

## Image Root

The image root should contain one case directory or one 3D volume per `case_id`.
Keep the same naming convention between `case_id` and the image files used by
the data loader.

Example directory layout:

```text
<PRIVATE_MRI_ROOT>/
  case_0001/
    image.nii.gz
    mask.nii.gz
  case_0002/
    image.nii.gz
    mask.nii.gz
```

If the local loader expects a different filename convention, document that
convention in this file before release.

## Splits

For five-fold evaluation, use a fixed split file such as:

```text
splits/private_cv5_seed2026_example.json
```

The split file should contain anonymous case ids only.
