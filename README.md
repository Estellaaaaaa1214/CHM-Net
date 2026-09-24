# CHM-Net 

CHM-Net (Center Heatmap-driven Macro-Micro Modeling Network) is a PyTorch model for patient-level microbial density stratification from preoperative multimodal 3D MRI. The model connects global macro-level MRI context with localized micro-level imaging evidence without requiring manual lesion annotations.

CHM-Net first learns a macro response heatmap to identify informative regions, then converts the selected regions into tri-planar micro representations. A macro-micro multiple-instance learning module aggregates the local evidence with the global MRI representation for final patient-level prediction.

## Files

- `data/`: User-provided MRI volumes and metadata. 
- `logs/`: Runtime log recorded during the training process.
- `src/`: Source code for the CHM-Net model.
- `README.md`: Model description, interface, and usage information.
- `requirements.txt`: Python dependency specification for the core implementation.


## Data Format

The label file should contain one row per case:

```csv
case_id,label
case_0001,0
case_0002,1
```

`label=0` denotes the low-density group and `label=1` denotes the high-density
group. The image root should contain one folder per `case_id`, with three MRI
modalities:

```text
<GBNPC2026_IMAGE_ROOT>/
  case_0001/
    T1WI/case_0001.nii.gz
    T1WI+C/case_0001.nii.gz
    T2WI/case_0001.nii.gz
```

## Source Files and Usage

The `src/` directory contains the core implementation:

- `model.py`: defines the complete CHM-Net architecture, including the macro Transformer, heatmap-guided ROI miner, tri-planar micro encoder, and macro-micro MIL fusion module.
- `losses.py`: provides the optional NT-Xent contrastive loss, heatmap sparsity loss, and heatmap smoothness loss.
- `__init__.py`: exposes the model and loss functions as the package interface.

The model accepts a tensor with shape `[B, C, D, H, W]`, where `B` is the batch size and `C` is the number of MRI modalities.
