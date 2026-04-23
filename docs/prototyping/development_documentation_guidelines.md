# Development Documentation Guidelines

## Purpose

Define how development and prototyping docs are organized, maintained, and consumed by agents and contributors.

## Documentation Hierarchy

1. `docs/prototyping/comprehensive_implementation_plan.md`
2. `docs/prototyping/api-design/api_flow_design.md`
3. cell-level specs (`docs/prototyping/workflows/cell_*.md`)
4. API and normalization references:
  - `docs/prototyping/api-design/catalog_api_reference.md`
  - `docs/prototyping/api-design/header_normalization_spec.md`
  - `docs/prototyping/references/header_summary_rules_schema.md`
5. flow-specific references:
  - `docs/prototyping/workflows/just_process_flow.md`
  - `docs/prototyping/workflows/lazy_refresh_with_watcher.md`

## Required Content for Each Cell Spec

Every `docs/cell_*.md` file must include:

- goal and why the cell exists
- canonical pseudo-code
- implementation context and algorithm intent
- catalog tables read and written
- persistence and reproducibility rules
- output contract used by next cell

## Planning Documentation Rules

- design docs are behavior-first, not implementation-first
- argument names must match API contracts exactly
- persistence behavior must be explicit
- ambiguity handling must be documented

## Prototyping Notebook Rules

- one notebook per workflow stage
- notebook cells are atomic and deterministic
- each notebook records expected intermediate outputs
- any prototype API deviation is fed back into docs before code hardening

## Change Management

When updating workflow behavior:

1. update `docs/prototyping/api-design/api_flow_design.md`
2. update corresponding `docs/prototyping/workflows/cell_*.md`
3. update API reference docs if signatures changed
4. update plan milestones if scope changed

## Agent Usage Guidance

Agents should:

- read `docs/prototyping/README.md` first
- treat the comprehensive plan as the current execution map
- avoid introducing API names not documented in the canonical references