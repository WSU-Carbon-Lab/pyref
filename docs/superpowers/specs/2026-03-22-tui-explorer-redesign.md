# TUI Explorer Redesign — Design Spec

**Date:** 2026-03-22
**Status:** Approved
**Scope:** Overhaul the `browser` TUI binary from a flat beamtime launcher into a three-level NAS file explorer with a single DATA-level SQLite catalog and per-experimentalist layout configuration.

---

## 1. Problem Statement

The current TUI has two screens:

- **Launcher** — a flat list of recently-ingested beamtime paths drawn from `~/.pyref/beamtime_index.sqlite3`. No hierarchy, no browsing, no metadata.
- **Beamtime screen** — samples / tags / scans / profiles view, scoped to a single beamtime directory and its own per-beamtime `catalog.db`.

This architecture does not model the actual NAS layout (`DATA/Experimentalist/Beamtime/…`), has no awareness of un-indexed beamtimes, and places the catalog in a user-home global index rather than co-located with the data. The existing `App` struct is a 2600-line monolith with 40+ fields mixing all screen state together, making further extension impractical.

---

## 2. Goals

1. Replace the Launcher with a full NAS file explorer rooted at a `DATA/` directory passed as a CLI argument.
2. Add an Experimentalist level between the root and beamtime folders, with per-folder catalog status (indexed / stale / not indexed) and FITS file counts.
3. Auto-ingest un-indexed or stale beamtimes on entry, with a cancellable progress spinner.
4. Move the catalog from per-beamtime files to a single `DATA/.pyref/catalog.db` that aggregates all experimentalists and beamtimes.
5. Adapt the existing Beamtime screen to query the DATA-level catalog filtered by beamtime path — no visual change to the screen itself.
6. Decompose the monolithic `App` struct into a `Navigator` + per-screen state model.

---

## 3. Non-Goals

- Zarr store migration or materialization changes.
- Python bindings or CLI tool changes (backward compatibility is required).
- Any change to the Beamtime screen's visual layout or existing features.
- Tiled integration (future work).
- A web UI (future work).

---

## 4. NAS Directory Model

```
DATA/                          ← data_root (CLI arg)
├── .pyref/
│   ├── catalog.db             ← single DATA-level catalog (NEW)
│   └── layout.toml            ← per-experimentalist layout policy (NEW)
├── Smith/                     ← Experimentalist (depth 1, no leading dot)
│   ├── 2024Jan/               ← Beamtime (matched by date pattern)
│   ├── 2023Nov/
│   └── scratch/               ← generic directory
└── Jones/
    └── 2024Feb/
```

Folders at depth 1 whose names do **not** start with `.` are treated as experimentalist candidates. Whether their subdirectories are classified as beamtime folders is determined by the `LayoutPolicy` described in §6.

---

## 5. Architecture — Navigator + Per-Screen State

### 5.1 Module Changes

**New files (`src/bin/tui/`):**

| File | Purpose |
|---|---|
| `navigator.rs` | `Navigator` struct + `ScreenState` enum |
| `explorer/mod.rs` | `ExplorerState`, `DirEntry`, `EntryKind`, `CatalogStatus` |
| `explorer/heuristic.rs` | `detect_policy()`, `classify_entry()` |
| `explorer/modal.rs` | `ModalState`, ratatui resolution modal |
| `catalog_handle.rs` | `CatalogHandle`, `ExptMeta`, `BeamtimeMeta` |

**Modified files (`src/bin/tui/`):**

| File | Change |
|---|---|
| `app.rs` | `App` becomes a thin shell owning `Navigator` + shared channels + config |
| `ui.rs` | Dispatches on `ScreenState` variant |
| `run.rs` | Event loop delegates to `Navigator` |
| `keymap.rs` | Per-screen keybind dispatch |

**Unchanged:** `beamspot.rs`, `preview.rs`, `scan_type.rs`, `theme.rs`, `terminal_guard.rs`.

**New files (`src/catalog/`):**

| File | Purpose |
|---|---|
| `explorer_query.rs` | `list_experimentalists()`, `list_beamtimes()`, `catalog_status()` |

**Modified files (`src/catalog/`):**

| File | Change |
|---|---|
| `ingest.rs` | New `data_root` + `experimentalist` + `cancel` parameters |
| `query.rs` | `query_scan_points` gains required `beamtime_path` filter |
| `mod.rs` | Re-exports new public API |

### 5.2 Core Types

```rust
pub struct App {
    pub navigator:      Navigator,
    pub preview_tx:     Option<Sender<(PathBuf, Option<f64>)>>,
    pub beamspot_rx:    Option<Receiver<BeamspotUpdate>>,
    pub preview_cmd_rx: Option<Receiver<PreviewCommand>>,
    pub layout:         String,
    pub keymap:         String,
    pub theme:          String,
    pub needs_redraw:   bool,
}

pub struct Navigator {
    stack:   Vec<ScreenState>,   // current + back history
    forward: Vec<ScreenState>,
    pub catalog: CatalogHandle,
}

pub enum ScreenState {
    Explorer(ExplorerState),
    Beamtime(BeamtimeState),
    ConfigModal(ModalState),
}
```

### 5.3 CatalogHandle

A plain struct wrapping `db_path: PathBuf`. Opens a fresh `rusqlite::Connection` per call — no `Arc<Mutex>`. Owned by `Navigator`, used by both screen types by reference.

```rust
pub struct CatalogHandle { pub db_path: PathBuf }

impl CatalogHandle {
    pub fn list_experimentalists(&self, data_root: &Path) -> Result<Vec<ExptMeta>>;
    pub fn list_beamtimes(&self, data_root: &Path, experimentalist: &str) -> Result<Vec<BeamtimeMeta>>;
    pub fn catalog_status(&self, beamtime_path: &Path) -> CatalogStatus;
    pub fn query_scan_points(&self, beamtime_path: &Path, filter: Option<&CatalogFilter>) -> Result<Vec<FileRow>>;
    pub fn list_beamtime_entries_v2(&self, beamtime_path: &Path) -> Result<BeamtimeEntries>;
}
```

If `DATA/.pyref/catalog.db` does not exist, all read methods return empty results; the file is created on first ingest.

---

## 6. Explorer Screen

### 6.1 ExplorerState

```rust
pub struct ExplorerState {
    pub data_root:     PathBuf,
    pub current_dir:   PathBuf,
    pub entries:       Vec<DirEntry>,
    pub list_state:    ListState,
    pub scroll_state:  ScrollbarState,
    pub layout_policy: NasLayout,
    pub loading:       bool,
    pub status:        Option<String>,
    pub sort_col:      ExplorerSort,
    pub sort_asc:      bool,
}
```

### 6.2 DirEntry

```rust
pub struct DirEntry {
    pub path:             PathBuf,
    pub name:             String,
    pub kind:             EntryKind,
    pub modified:         Option<SystemTime>,
    pub size_bytes:       Option<u64>,    // deferred async
    pub fits_count:       Option<u32>,    // fast async via mpsc
    pub catalog_status:   CatalogStatus,
    pub expt_resolution:  Option<ExptResolution>, // Some when kind == Experimentalist
}

pub enum EntryKind     { DataRoot, Experimentalist, Beamtime, Directory }
pub enum CatalogStatus { Indexed, Stale, NotIndexed, NotApplicable }
pub enum ExptResolution { Resolved, NeedsResolution, Ignored }
```

### 6.3 Explorer Columns

| Column | Source | Loading |
|---|---|---|
| Name | `fs::read_dir` | Immediate |
| Date Modified | `fs::metadata().modified()` | Immediate |
| Size | `du` equivalent | Deferred (on selection or `s` key) |
| Kind | `classify_entry()` | Immediate |
| Beamtimes / FITS Files | `CatalogHandle` + `fs::read_dir` count | Fast async |
| Status | `CatalogHandle::catalog_status()` | Fast async |

### 6.4 Explorer Keybindings

| Key | Action |
|---|---|
| `↑↓` | Navigate entries |
| `→` / `↵` | Enter directory / open beamtime |
| `←` / `h` | Go up / back |
| `r` | Resolve selected experimentalist |
| `x` | Toggle ignore on selected folder |
| `i` | Ingest selected beamtime |
| `I` | Ingest all not-yet-indexed beamtimes |
| `/` | Filter entries |
| `s` | Compute disk size for selection |
| `q` | Quit |

---

## 7. Per-Experimentalist Layout Policy

### 7.1 NasLayout

Persisted to `DATA/.pyref/layout.toml`.

```rust
pub struct NasLayout {
    pub default_pattern: Option<String>,  // regex
    pub default_depth:   u8,              // default: 2
    pub experimentalists: HashMap<String, ExptPolicy>,
}

pub enum ExptPolicy {
    Auto,
    Custom { pattern: Option<String>, depth: u8 },
    Ignored,
}
```

### 7.2 Heuristic Detection

On Explorer startup, `detect_policy(data_root)` performs a shallow scan (top 2 levels only). Built-in patterns tested against depth-2 folder names:

```
^20\d{2}(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)$   // 2024Jan
^20\d{2}-\d{2}(-\d{2})?$                                      // 2024-01
^20\d{6}$                                                      // 20240115
(?i)beamtime                                                   // contains "beamtime"
```

If ≥50% of depth-2 folders under a given experimentalist match any pattern, that experimentalist is marked `ExptPolicy::Auto` (resolved). Below 50% → `ExptResolution::NeedsResolution` (flagged with `⚑`).

Folders at depth 1 whose names start with `.` are silently excluded and never shown.

### 7.3 Resolution Modal

Triggered by pressing `r` on a flagged experimentalist row, or automatically on entering a flagged folder.

Three choices:
1. **Custom pattern** — user specifies a regex and depth. Live preview pane updates on every keystroke showing how subdirs are classified. Invalid regex turns the field red; preview shows last valid state.
2. **Mark as ignored** — folder is dimmed in the explorer, excluded from beamtime counts and bulk ingest. Entry is still allowed for browsing.
3. **Accept global default** — applies the global `default_pattern` + `default_depth` even if no matches were found.

On confirm, the resolution is written to `DATA/.pyref/layout.toml` immediately. Modal is popped from the Navigator stack, returning to the Explorer with cursor restored.

---

## 8. Beamtime Screen Adaptation

### 8.1 BeamtimeState

All fields currently on `App` that pertain to the Beamtime screen are moved verbatim to `BeamtimeState`. Three new context fields are added:

```rust
pub struct BeamtimeState {
    pub data_root:       PathBuf,
    pub experimentalist: String,
    pub beamtime_path:   PathBuf,   // filter key into DATA catalog
    // ... all existing App fields for beamtime view ...
}
```

### 8.2 Query Migration

```rust
// Before
let db = resolve_catalog_path(&beamtime_dir);
let rows = query_files(&db, None)?;

// After
let db = data_root.join(".pyref/catalog.db");
let rows = catalog_handle.query_scan_points(&beamtime_path, None)?;
```

`query_scan_points` adds `AND source_path LIKE ?1 || '%'` bound to `beamtime_path`. The old `query_files()` function is kept but deprecated; existing callers are migrated individually.

### 8.3 Context Bar

The Beamtime screen's title bar shows the full breadcrumb path: `Smith › 2024Jan`. The `←`/`h` key pops `BeamtimeState` off the Navigator stack, restoring the underlying `ExplorerState` with cursor position intact.

Selections (`last_sample`, `last_tag`, `last_scan`) are written to `TuiConfig` when `BeamtimeState` is popped, not only on quit.

---

## 9. Auto-Ingest on Entry

When the user enters a beamtime with `CatalogStatus::NotIndexed` or `CatalogStatus::Stale`:

1. `Navigator.push(ScreenState::Beamtime(BeamtimeState::new_ingesting(…)))` — the screen is pushed immediately with `LoadingState::IngestingDirectory`.
2. A background thread is spawned calling `ingest_beamtime(beamtime_dir, Some(data_root), Some(experimentalist), …, Some(cancel_flag))`.
3. Progress updates flow via `mpsc::Sender<(u32, u32)>` to the `ingest_progress_rx` on `BeamtimeState`. A progress bar and spinner are rendered.
4. On completion:
   - `Ok(())` → `BeamtimeState::reload_from_catalog()` queries the DATA catalog. A lightweight message is sent back to the underlying `ExplorerState` to update the status cell to `Indexed`.
   - `Err(e)` → error modal shown; `←` returns to Explorer.
5. **Esc during ingest** sets the cancel `AtomicBool`, waits for the thread to drain, pops the `BeamtimeState`, and returns to the Explorer. Partial writes remain in the catalog and are resumed on next entry.

**Stale** beamtimes use the same path as `NotIndexed`. Staleness is determined by comparing the directory's most recent file `mtime` against `bt_beamtimes.last_indexed_at`.

---

## 10. Ingestion Pipeline Changes

### 10.1 Signature

```rust
pub fn ingest_beamtime(
    beamtime_dir:    &Path,
    data_root:       Option<&Path>,       // NEW — None = legacy behaviour
    experimentalist: Option<&str>,        // NEW — None = legacy behaviour
    header_items:    &[String],
    force:           bool,
    progress_tx:     Option<Sender<(u32, u32)>>,
    cancel:          Option<Arc<AtomicBool>>,  // NEW
) -> Result<IngestSummary>
```

`data_root: None` and `experimentalist: None` reproduce the current behaviour exactly — the catalog is opened at the beamtime directory, no new columns are written. All existing callers pass `None` until individually migrated.

### 10.2 Catalog Path Resolution

```rust
pub fn resolve_ingest_target(beamtime_dir: &Path, data_root: Option<&Path>) -> PathBuf {
    match data_root {
        Some(root) => root.join(".pyref").join("catalog.db"),
        None       => resolve_catalog_path(beamtime_dir),  // existing behaviour
    }
}
```

### 10.3 Schema Migration

Two nullable columns added to `bt_beamtimes` via `ALTER TABLE … ADD COLUMN … DEFAULT ''`. Safe migration — existing databases open without errors; empty string is treated as unset and the experimentalist is inferred from path at query time.

```sql
ALTER TABLE bt_beamtimes ADD COLUMN experimentalist TEXT NOT NULL DEFAULT '';
ALTER TABLE bt_beamtimes ADD COLUMN data_root       TEXT NOT NULL DEFAULT '';
CREATE INDEX IF NOT EXISTS idx_bt_experimentalist
    ON bt_beamtimes(data_root, experimentalist);
```

At the start of every ingest, `upsert_beamtime_record()` runs an `INSERT … ON CONFLICT DO UPDATE` to record these values and refresh `last_indexed_at`.

---

## 11. CLI Argument

```
browser [data_root]
```

- If `data_root` is provided: must be an existing directory. The Explorer is shown rooted at this path.
- If omitted: behaves as today — reads `config.last_root`; if set, opens the Beamtime screen directly. This preserves backward compatibility for existing workflows.

The `Screen::Launcher` code path is removed in Phase 4 once the Explorer is fully functional.

---

## 12. Execution Phases

### Phase 1 — Decompose (no user-visible change)
- Extract `BeamtimeState` from `App`
- Introduce `Navigator`, `ScreenState`
- `App` becomes thin shell
- All existing behaviour preserved
- Files: `app.rs`, `run.rs`, `ui.rs`, `keymap.rs`, new `navigator.rs`

### Phase 2 — Explorer screens
- Add `ExplorerState`, `DirEntry`, `EntryKind`
- Add `heuristic.rs`, `NasLayout`, `layout.toml` persistence
- Add `ModalState` and resolution modal UI
- Wire CLI arg as `data_root`
- Files: new `explorer/` module

### Phase 3 — DATA-level catalog
- Add `CatalogHandle` and `explorer_query.rs`
- Schema migration for `bt_beamtimes`
- Adapt `ingest_beamtime` signature (backward-compatible)
- Add cancel token to ingest loop
- Adapt `query_scan_points` to accept `beamtime_path` filter

### Phase 4 — Wire end-to-end
- Auto-ingest on Beamtime entry (NotIndexed + Stale)
- `BeamtimeState` queries DATA-level catalog
- Status cell update channel from Beamtime → Explorer
- Config saved on `BeamtimeState` pop
- Remove `Screen::Launcher` + `list_beamtimes()` global index call

Each phase is independently mergeable with no breaking changes to existing functionality.

---

## 13. Open Questions / Future Work

- **Bulk ingest (`I` key):** Phase 4 scope is single-beamtime auto-ingest. Bulk ingest of all un-indexed beamtimes in an experimentalist folder is deferred to a follow-on ticket.
- **Disk size column:** Deferred computation via `du` on NAS can be slow. Current plan is on-demand (`s` key) or on-selection. May need a cap or timeout.
- **`beamtime_index.sqlite3` migration:** The global `~/.pyref/beamtime_index.sqlite3` is not removed in this redesign. It remains as a fallback for the no-`data_root` code path.
- **Tiled migration:** The DATA-level `catalog.db` schema is designed to be compatible with a future Tiled ingest pipeline. No Tiled-specific changes are in scope here.
