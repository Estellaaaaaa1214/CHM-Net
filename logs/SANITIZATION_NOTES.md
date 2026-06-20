# Log Sanitization Notes

The full training logs in this release package preserve fold-level, epoch-level,
and summary metric lines. Local machine prompts, container ids, private image
roots, label-file paths, and public-data cache roots were replaced with neutral
placeholders:

- `<RUN_ENV>#`
- `root@<CONTAINER_ID>`
- `<PRIVATE_MRI_ROOT>`
- `<PRIVATE_LABEL_FILE>`
- `<PUBLIC_DATA_ROOT>`

No metric values, fold identifiers, epoch traces, or confusion matrices were
changed by this sanitation step.
