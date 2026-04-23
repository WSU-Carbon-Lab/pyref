# Header Summary Rules Schema

## Goal

Define a configurable rules file used by `stitch_summary` to auto-select relevant header columns for stitch comparison.

## Suggested File Name

- `header_summary_rules.json`

## Suggested Schema

```json
{
  "families": {
    "sample": ["sam_", "sample_"],
    "ccd": ["ccd_", "camera_"],
    "jj_slits": ["jj_", "slit_", "slt_"]
  },
  "exclude": [
    "timestamp",
    "time",
    "file_name",
    "frame_number"
  ],
  "priority": ["sample", "ccd", "jj_slits"]
}
```

## Selection Rules

- include columns that match configured family prefixes
- exclude columns with names matching the exclusion list
- preserve deterministic ordering by family priority then column name
