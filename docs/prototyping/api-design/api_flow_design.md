# Catalog-First API Flow Design

## Objective

Define a notebook-first PRSoXR API where users always operate on catalog entities, never raw FITS paths. The top-level user action is selecting a beamtime key such as `"2026Feb"` and then querying scans, profiles, and frames from the indexed catalog.

This document focuses on API behavior and ergonomics. Implementation sequencing is intentionally secondary.

## First-Class Requirements

- Catalog-first selection: `catalog -> beamtime -> scans/profiles/frames`.
- Persistent metadata correction for sample names, tags, and scan/profile typing.
- Fast inspection views for beamtime, scans, and samples before reduction.
- Beamtime discovery and catalog backend registration.
- Header normalization into snake_case and standardized abbreviations.
- Explicit external i0 linking and scan merge semantics.
- Integrated interactive mask and beamspot workflow with interactive remediation.
- Rich diagnostics for uncertainty, i0/stitch/overlap segments, and profile comparison modalities.
- Lazy query behavior compatible with watcher daemon updates and low-overhead refresh.

## Canonical User Entry Point

```python
import pyref as pr

cat = pr.catalog()
cat.register_backend("local", "~/.config/pyref/catalog.db", make_default=True)
cat.backends()
cat.beamtimes()

bt = cat.beamtime("2026Feb")
```

Beamtime resolution order:

- exact key
- alias
- deterministic partial match with explicit ambiguity error

## Catalog API Contract

Detailed reference: `docs/prototyping/api-design/catalog_api_reference.md`

### Catalog session

- `pr.catalog(backend: str | None = None) -> CatalogSession`
- `CatalogSession.register_backend(name: str, path: str, make_default: bool = False)`
- `CatalogSession.backends() -> DataFrame`
- `CatalogSession.use_backend(name: str) -> CatalogSession`
- `CatalogSession.beamtimes() -> DataFrame`
- `CatalogSession.beamtime(key: str) -> BeamtimeHandle`
- `CatalogSession.resolve_beamtime(key: str) -> dict`

### Beamtime handle

- `BeamtimeHandle.scans(...) -> DataFrame`
- `BeamtimeHandle.scans_view(...) -> DataFrame`      # editable scan summary view
- `BeamtimeHandle.samples_view(...) -> DataFrame`    # sample verification view
- `BeamtimeHandle.profiles(...) -> DataFrame`
- `BeamtimeHandle.frames(...) -> DataFrame`
- `BeamtimeHandle.frames_lazy(...) -> LazyFrame`
- `BeamtimeHandle.summary() -> DataFrame`

### Persistent metadata correction

- `BeamtimeHandle.rename_scan_sample(scan_number: int, sample_name: str)`
- `BeamtimeHandle.retag_scan(scan_number: int, tag: str | None)`
- `BeamtimeHandle.set_scan_profile_type(scan_number: int, profile_type: str)`
- `BeamtimeHandle.apply_scan_edits(scans_df: DataFrame)`       # bulk edits from scans_view
- `BeamtimeHandle.apply_sample_edits(samples_df: DataFrame)`   # bulk edits from samples_view
- `BeamtimeHandle.commit_metadata_updates(message: str | None = None)`
- `BeamtimeHandle.reingest_metadata(sync: bool = True)`        # write edits back for future loads
- `BeamtimeHandle.audit_overrides() -> DataFrame`

Behavioral requirements:

- Corrections persist in catalog tables and are replayed on future loads.
- Provenance is retained: corrected metadata and raw source identity remain linked.
- Scan and sample inspection/edit paths are fast and do not require loading full frame stacks.

## Header Normalization Contract

Detailed reference: `docs/prototyping/api-design/header_normalization_spec.md`

All header names exposed to user-facing DataFrames are normalized:

- snake_case
- lowercase
- spaces and punctuation replaced with `_`
- standard abbreviation map applied, including `sample -> sam`

Examples:

- `Sample Theta` -> `sam_theta`
- `Beamline Energy` -> `beamline_energy`
- `AI 3 Izero` -> `ai3_izero`

The normalization map is user-configurable via a JSON config:

- `header_aliases.json` for explicit key remaps
- built-in defaults used when not overridden

## DIY Notebook Flow

### Cell 1: Connect, inspect beamtimes, load scoped frames

Detailed reference: `docs/prototyping/workflows/cell_01_catalog_and_selection.md`

```python
import pyref as pr

cat = pr.catalog()
cat.beamtimes()

bt = cat.beamtime("2026Feb")
scan_table = bt.scans_view()
sample_table = bt.samples_view()

# user edits scan_table/sample_table in notebook
bt.apply_scan_edits(scan_table)
bt.apply_sample_edits(sample_table)
bt.commit_metadata_updates(message="rename mislabeled scans")
bt.reingest_metadata(sync=True)

frames = bt.frames(sample_name="sample_a", scan_numbers=[1201, 1202], engine="polars")
```

### Cell 2: Split profiles with normalized headers

Detailed reference: `docs/prototyping/workflows/cell_02_profile_split.md`

```python
profiles = frames.refl.split_profiles(domain_hint="auto")
profiles.refl.plot_headers(
    x="sam_theta",
    y=["beamline_energy", "ai3_izero", "beam_current", "exposure"],
    by="profile_id",
)
```

### Cell 3: Point classification, dynamic stitch summary, explicit external i0

Detailed reference: `docs/prototyping/workflows/cell_03_classify_and_i0_linking.md` and `docs/prototyping/references/header_summary_rules_schema.md`

```python
valid_i0 = bt.scans().refl.valid_i0_scans()

classified = profiles.refl.classify_points(
    i0_rules={"sam_theta_zero_tol": 0.005},
    merge_scans="auto",              # or merge_scans=[1180, 1181]
    i0_scan_numbers=[1180],          # optional explicit link
)

classified.refl.stitch_summary(
    header_selector="auto_sample_ccd_jj",   # auto-select relevant headers
    header_config="header_summary_rules.json",
)
```

Requirements:

- Replace `allow_external_i0` with explicit merge/i0-link controls.
- `stitch_summary` auto-selects comparison headers from sample/ccd/jj families.

### Cell 4: Integrated mask and beamspot detection with regression diagnostics

Detailed reference: `docs/prototyping/workflows/cell_04_mask_and_beamspot.md`

```python
beam = classified.refl.detect_beamspots(
    mask_mode="include_region",      # or "exclude_region"
    regression="linear",             # "fixed", "poly-3", "poly-{n}"
    confidence_band=0.95,
    interactive=True,
)

beam.refl.plot_beamspot_diagnostics(
    y=["beam_row", "beam_col"],
    with_confidence=True,
)
```

Interactive remediation requirements:

- user can marquee or crosshair-select corrected beamspots
- model-predicted spot is shown as guidance
- bad beamspots can be selected, remediated, and persisted
- interactive mask tools and beamspot remediation panels are first-class APIs in notebooks, not debug-only tools

### Cell 5: Persist masks and spot remediations

Detailed reference: `docs/prototyping/workflows/cell_05_persist_mask_and_spot_overrides.md`

```python
beam.refl.save_mask(mask_name="scan_default")
beam.refl.save_beamspot_overrides()
```

Mask persistence contract:

- store mask mode and good-region slice indices per scan/profile
- mask retrieval is deterministic and versioned

### Cell 6: Reduce intensity and plot uncertainty contributions

Detailed reference: `docs/prototyping/workflows/cell_06_reduction_and_uncertainty.md`

```python
reduced = beam.refl.reduce_intensity(
    roi_strategy="i0_gaussian_3sigma",
    uncertainty={"shot_noise": True, "roi_loss": True, "fano_factor": True},
)

reduced.refl.plot_uncertainty_contributions(x="domain_value")
```

### Cell 7: Diagnostic views by point role and layout mode

Detailed reference: `docs/prototyping/workflows/cell_07_role_diagnostics.md`

```python
reduced.refl.plot_role_diagnostics(
    roles=["i0", "stitch", "overlap"],
    layout="subplots",           # or "contiguous"
    split_overlap_regions=True,
)
```

### Cell 8: Stitch with channel-normalization semantics

Detailed reference: `docs/prototyping/workflows/cell_08_stitch_compute.md`

```python
stitched = reduced.refl.compute_stitch(
    normalize_chans=["beam_current", "exposure", "i0"],
    overlap_estimator="weighted_mean",
    propagate_errors=True,
)
```

### Cell 9: Compare profiles with explicit modality choice

Detailed reference: `docs/prototyping/workflows/cell_09_profile_comparison_and_export.md`

```python
stitched.refl.compare_profiles(
    mode="waterfall",            # or "subplots"
    waterfall_scale=1.7,
)

stitched.refl.export_parquet(out_dir="processed")
```

Comparison contract:

- `mode` is required
- `waterfall` uses multiplicative offset scaling
- `subplots` supports optional externally supplied axes

## Just Process the Data Flow

Detailed reference: `docs/prototyping/workflows/just_process_flow.md`

```python
import pyref as pr

result = pr.process(
    beamtime="2026Feb",
    sample_name="sample_a",
    scan_numbers=[1201, 1202],
    engine="polars",
)
```

Behavior:

- equivalent defaults to recommended expert workflow
- all decisions persisted to catalog run metadata
- exposes intermediate tables for escalation to DIY mode

## Lazy Frames and Watcher Integration

Detailed reference: `docs/prototyping/workflows/lazy_refresh_with_watcher.md`

Required behavior:

- `bt.frames_lazy(...)` returns a query plan that does not force eager scan of all rows.
- watcher daemon ingestion appends new data to catalog/zarr.
- rerunning downstream notebook cells re-collects from the same lazy query with fresh rows.
- no extra user wiring required beyond rerunning the cell.

## Functional Requirements Checklist

- metadata correction API for sample/tag/profile-type overrides
- scans and samples inspection/edit views with bulk apply and commit
- persistent correction replay on all future beamtime loads
- explicit metadata reingest/sync capability after edits
- catalog backend registration and switching
- beamtime listing and key-based selection
- normalized snake_case header contract with `sample -> sam` abbreviation
- dynamic stitch header summary selection with configurable rules file
- explicit i0 scan discovery and merge controls
- integrated mask + beamspot workflow with regression diagnostics and interactive remediation
- uncertainty contribution plotting by source
- role-specific diagnostics (`i0`, `stitch`, `overlap`) with contiguous/subplots modes
- stitch API uses `normalize_chans`
- profile comparison requires explicit mode (`waterfall` or `subplots`)
- lazy-frame watcher compatibility for low-overhead refresh

## Done Criteria

- User can select `"2026Feb"` and inspect scans without filesystem paths.
- User can view and edit scan/sample metadata from dedicated `scans_view` and `samples_view`.
- User can correct sample/tag/profile type metadata and see corrections on reload.
- All nine notebook cells run with documented API behavior and linked detailed specs.
- Watcher updates appear by rerunning lazy-backed cells with no extra setup.
