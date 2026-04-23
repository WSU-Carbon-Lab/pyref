# Header Normalization Specification

## Purpose

Define a stable, user-facing header naming contract for all DataFrames returned from catalog APIs and downstream reduction APIs.

## Normalization Rules

- convert to lowercase
- convert spaces and punctuation to `_`
- collapse repeated `_`
- trim leading and trailing `_`
- apply abbreviation dictionary

## Default Abbreviation Dictionary

- `sample` -> `sam`
- `energy` -> `energy`
- `beamline` -> `beamline`
- `current` -> `current`
- `exposure` -> `exposure`

## Examples

- `Sample Theta` -> `sam_theta`
- `Sample X` -> `sam_x`
- `Beamline Energy` -> `beamline_energy`
- `AI 3 Izero` -> `ai3_izero`
- `CCD Theta` -> `ccd_theta`

## User Configuration

Optional JSON file:

- `header_aliases.json`

Expected shape:

```json
{
  "sample theta": "sam_theta",
  "higher order suppressor": "hos"
}
```

Priority:

1. explicit user alias
2. default normalization and abbreviation rules

## Backward Compatibility

- original raw header names remain available in metadata provenance tables
- normalized names are used by default in notebook-facing DataFrames
