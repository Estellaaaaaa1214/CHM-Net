# Data Examples

The private MRI cohort is not redistributed in this anonymous repository.

Expected private metadata columns:

- `case_id`: anonymized case identifier matching each image folder name.
- `label`: binary microbial-density label, with `0` for low density and `1` for high density.

Expected private image layout:

```text
data/private_mri/
  case_0001/
    T1WI/case_0001.nii.gz
    T1WI+C/case_0001.nii.gz
    T2WI/case_0001.nii.gz
```

Public MedMNIST3D datasets should be downloaded from the official MedMNIST
release and placed under `data/public/NoduleMNIST3D` and
`data/public/AdrenalMNIST3D`, or provided through the corresponding CLI flags.
