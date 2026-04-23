# Comprehensive Implementation Plan: Catalog-First PRSoXR API

## Goal

Deliver a production-grade, catalog-first API that supports:

- notebook-driven DIY reduction
- one-call processing (`Just process`)
- persistent metadata correction
- zarr-backed frame/image access through DataFrame interfaces
- reproducible processing state and diagnostics

This plan is the implementation source of truth for the current design set in `docs/prototyping/api-design/api_flow_design.md` and workflow cell specs.

## Scope

### In Scope

- schema evolution and backfill for legacy catalogs
- Rust backend expansions for catalog queries, image/materialization, reduction primitives, and persistence
- Python API orchestration (`catalog`, beamtime handles, accessors, loaders)
- notebook prototype scaffolding for each pipeline stage
- deterministic diagnostics and export surfaces

### Out of Scope

- redesign of unrelated fitting modules
- non-catalog-first workflows
- replacing notebook-first UX with CLI-only flows

## Phase 0: Contracts Freeze

## 0.1 API contract freeze

Freeze method signatures and argument names for:

- catalog session and beamtime handles
- scan/sample edit and commit APIs
- DataFrame accessors (`image`, `refl`)
- stitch compute (`normalize_chans`)
- profile comparison mode contract

Exit criteria:

- API signature table reviewed and accepted
- no unresolved naming conflicts

## 0.2 Data contract freeze

Freeze required columns and semantic types for:

- `scans_view`, `samples_view`, `frames`, `profiles`, classified/reduced/stitched outputs

Exit criteria:

- schema and column semantics documented
- provenance keys present in every stage output

## 0.3 Algorithm contract freeze

Freeze rules for:

- profile split
- role classification
- external i0 discovery and merge resolution
- beamspot regression modes and remediation precedence
- uncertainty propagation model

Exit criteria:

- equations/rules and tolerances documented
- ambiguity behavior explicitly defined

## Phase 1: Schema Migration and Backfill

## 1.1 Migration design

Add/extend tables for:

- correction and override persistence
- mask state
- beamspot override state
- processing run metadata
- stitch diagnostics enhancements

Design constraints:

- immutable provenance rows remain unchanged
- mutable correction layers are additive and auditable

## 1.2 Catalog migration mechanism

Implement migration path for existing catalogs:

- detect legacy schema version
- apply Diesel migrations in sequence
- preserve rollback metadata

## 1.3 Backfill mechanism from FITS

Backfill missing derived values where needed:

- scan/profile classification metadata
- normalized header mappings
- optional summary fields

Requirements:

- resumable backfill with per-scan checkpoints
- partial failure recovery without catalog corruption
- deterministic idempotent reruns

Exit criteria:

- migration and backfill verified on legacy test catalogs
- zero data loss in provenance tables

## Phase 2: Rust Backend Construction

## 2.1 Catalog query backends

Implement Rust-backed surfaces for:

- beamtime key/alias listing and resolution
- scan/sample view generation
- correction apply/commit/replay paths
- lazy frames query endpoints

## 2.2 Processing persistence backends

Implement writes for:

- frame role assignments
- beam finding and override records
- mask records
- stitch corrections and diagnostics
- processing runs

## 2.3 Image and reduction backends

Ensure Rust kernels cover:

- zarr image materialization by frame index
- preprocessing and gaussian/fit operations
- reduction primitives for signal/background integration
- uncertainty contributor calculations

Exit criteria:

- PyO3 bindings expose required APIs
- integration tests validate data integrity and round-trip persistence

## Phase 3: Python API and Loader Layer

## 3.1 Catalog and beamtime interfaces

Implement:

- `pr.catalog()`
- backend registry/select APIs
- beamtime handles with `scans_view`, `samples_view`, `frames`, `frames_lazy`
- metadata edit/apply/commit/reingest APIs

## 3.2 Accessors and pipeline surfaces

Implement:

- `df.image` accessor for row-image retrieval
- `df.refl` accessor for split/classify/detect/reduce/qc/stitch/compare/export

## 3.3 Loader classes

Implement lightweight classes:

- catalog-backed frame/profile loader classes
- one-shot `process(...)` facade
- compatibility bridge with `PrsoxrLoader` expectations

Exit criteria:

- notebook flows run with stable API surfaces
- pandas/polars toggle works with semantic parity

## Phase 4: Notebook Prototyping Scaffold

Create staged notebooks in `notebooks/`:

- `00_catalog_backend_and_migration.ipynb`
- `01_beamtime_scan_sample_views.ipynb`
- `02_profile_split_and_header_normalization.ipynb`
- `03_classification_and_i0_linking.ipynb`
- `04_mask_and_beamspot_interactive.ipynb`
- `05_reduction_and_uncertainty_budget.ipynb`
- `06_role_qc_and_bad_point_flagging.ipynb`
- `07_stitch_factors_and_propagation.ipynb`
- `08_profile_comparison_and_export.ipynb`
- `09_just_process_end_to_end.ipynb`

Each notebook must include:

- objective and expected outputs
- minimal setup cell
- stage-specific prototype cells
- quick validation display cells

Exit criteria:

- each notebook executes independently against the target test dataset set
- stage outputs map directly to documented contracts

## Phase 5: Validation and Hardening

## 5.1 Test matrix

Cover:

- fixed-energy and fixed-angle datasets
- external i0 merge scenarios
- mislabeled sample/tag corrections
- saturation and beam-loss edge cases
- watcher append refresh behavior

## 5.2 Performance checks

Measure:

- scan/sample inspection latency
- lazy collect refresh latency
- image materialization throughput
- stitch compute timing by scan size

## 5.3 Quality gates

- Rust: build/test with zero errors/warnings under project gates
- Python: lint/type/test gates per repo tooling
- notebook smoke execution for all stage notebooks

Exit criteria:

- all gates green
- deterministic outputs for seeded/reference datasets

## Open Questions to Resolve Before Full Build-Out

- precise migration strategy for legacy catalogs with incomplete header coverage
- exact conflict resolution policy for concurrent metadata edits
- final shape and storage format for complex masks beyond slice bounds
- formal uncertainty equation set and acceptance tolerances
- canonical dataset set for regression validation

## Deliverables

- migrated and backfillable catalog schema
- Rust and Python API surfaces matching docs contracts
- staged notebook prototypes
- reproducibility metadata and export pipeline
- validated end-to-end DIY and one-call flows