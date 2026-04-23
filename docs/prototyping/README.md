# Prototyping and Development Documentation

## Purpose

This section is the canonical planning surface for the catalog-first PRSoXR API and notebook-driven prototyping pipeline.

All agents and contributors should start here before proposing implementation changes.

## Primary Documents

- `docs/prototyping/comprehensive_implementation_plan.md`
- `docs/prototyping/development_documentation_guidelines.md`

## Workflow Specifications

### API Design

- `docs/prototyping/api-design/api_flow_design.md`
- `docs/prototyping/api-design/catalog_api_reference.md`
- `docs/prototyping/api-design/header_normalization_spec.md`

### References

- `docs/prototyping/references/header_summary_rules_schema.md`

### Workflows

- `docs/prototyping/workflows/cell_01_catalog_and_selection.md`
- `docs/prototyping/workflows/cell_02_profile_split.md`
- `docs/prototyping/workflows/cell_03_classify_and_i0_linking.md`
- `docs/prototyping/workflows/cell_04_mask_and_beamspot.md`
- `docs/prototyping/workflows/cell_05_persist_mask_and_spot_overrides.md`
- `docs/prototyping/workflows/cell_06_reduction_and_uncertainty.md`
- `docs/prototyping/workflows/cell_07_role_diagnostics.md`
- `docs/prototyping/workflows/cell_08_stitch_compute.md`
- `docs/prototyping/workflows/cell_09_profile_comparison_and_export.md`
- `docs/prototyping/workflows/just_process_flow.md`
- `docs/prototyping/workflows/lazy_refresh_with_watcher.md`

## Notebook Prototyping Targets

The `notebooks` directory should contain one prototype notebook per workflow stage:

- `notebooks/00_catalog_backend_and_migration.ipynb`
- `notebooks/01_beamtime_scan_sample_views.ipynb`
- `notebooks/02_profile_split_and_header_normalization.ipynb`
- `notebooks/03_classification_and_i0_linking.ipynb`
- `notebooks/04_mask_and_beamspot_interactive.ipynb`
- `notebooks/05_reduction_and_uncertainty_budget.ipynb`
- `notebooks/06_role_qc_and_bad_point_flagging.ipynb`
- `notebooks/07_stitch_factors_and_propagation.ipynb`
- `notebooks/08_profile_comparison_and_export.ipynb`
- `notebooks/09_just_process_end_to_end.ipynb`