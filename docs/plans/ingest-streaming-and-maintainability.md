# Ingest Performance and Maintainability Plan

Branch: `perf/ingest-streaming-and-maintainability`
Source of findings: diagnostic pass in commit `1941cd2` and conversational review
of `src/catalog/ingest.rs`.

## Background

Profiling on `/Volumes/DATA/Collins/2026Feb` (13,546 FITS files, 33 scans)
ran for hours and crashed Cursor. Code review of `ingest_beamtime_inner`
identified six concrete causes and several maintainability issues that
together produce the observed symptoms:

- Progress bars sit idle then race to completion at the end.
- Nothing is written to `catalog.db` until the very end.
- Peak RAM reaches ~9 GB, triggering OOM.

The root causes:

1. **Unbuffered pixel reads.** `read_image_i32` calls `read_exact(&mut [0u8; 2])`
   ~170k times per file on an unbuffered `std::fs::File`. Across 13,546 files
   that is ~2.3 billion two-byte syscalls on NAS-mounted storage.
2. **Duplicate zarr array per frame.** `write_frame_raw` creates both `/raw`
   and `/processed` with identical bytes, doubling filesystem work during
   ingest for no downstream benefit (no code reads `/processed`).
3. **N+1 SELECT-after-INSERT.** Every sample, scan, and file insert is
   followed by a `SELECT id FROM ... WHERE ...` round-trip inside the
   transaction. SQLite ≥ 3.35 supports `RETURNING`; Diesel has the
   `returning_clauses_for_sqlite_3_35` feature already enabled.
4. **Single giant SQLite transaction.** The entire catalog insert loop runs
   inside one `conn.transaction`. No rows land on disk until the end; a crash
   mid-ingest rolls back everything.
5. **Zarr phase materializes all images before writing any.** The outer
   `.collect::<Result<Vec<_>, _>>()` across all scans forces 9+ GB of
   `Array2<i32>` resident before the first `write_frame_raw` call. This
   defers the first `file_complete` event and causes the OOM.
6. **Per-scan (not per-file) parallelism.** Large scans pin a single worker
   while other cores idle.

## Goals

Primary:

- Give incremental progress during both catalog and zarr phases.
- Hold at most `O(worker_threads)` images in RAM during zarr phase.
- Cut per-file syscalls from ~170k to ~1.
- Commit catalog rows in small-enough batches that progress is visible in
  `catalog.db` during long runs, without creating churn from one-commit-per-tiny-scan.

Secondary:

- Replace stringly-typed phase labels with a `IngestPhase` enum.
- Unify FITS pixel reading between `ingest.rs` and `io/image_mmap.rs`.
- Split the 380-line `ingest_beamtime_inner` into phase functions with a
  shared `IngestContext` so each function fits in a reviewer's head.
- Provide a synthetic-beamtime fixture so we can benchmark ingest without
  the NAS.

Non-goals:

- Rewriting zarr store semantics. The existing `zarrs::FilesystemStore` is fine.
- Changing the public Python API surface (`ingest_beamtime`, `read_beamtime`).
- Changing progress event wire format (keep `phase`, `layout`, `catalog_row`,
  `file_complete`).

## Acceptance criteria

A full ingest of the fixture-backed synthetic beamtime (`tests/`) must:

- Emit `catalog_row` events during the catalog phase (proves progress is live).
- Emit `file_complete` events incrementally during zarr phase (not all at the
  end); verified by timestamp delta between first and last event.
- Have zero residual `/processed` group in the zarr store after removal.
- Show catalog rows visible in `catalog.db` partway through (per-batch commit).

Regression gates:

- `cargo test --features catalog,parallel_ingest` green.
- `uv run pytest tests/test_catalog.py` green.
- `uv run ruff check python tests scripts` green on touched files.
- `uv run ty check python` green on touched files (test file cleanup noted in
  prior review).
- `cargo clippy --features catalog,parallel_ingest -- -D warnings` on
  `src/catalog/ingest.rs`, `src/catalog/ingest_progress.rs`,
  `src/catalog/zarr_write.rs`, and any new modules in `src/io/`.

## Task breakdown

Tasks are mostly independent at the file level. Ordering below is the
execution sequence (earlier tasks reduce diff size for later ones).

### Phase A: Performance (tasks 1–6)

**T1. Bulk pixel read in `read_image_i32`** *(src/catalog/ingest.rs)*

Replace the byte-pair loop with a single bulk read:

- Open `File`, `seek(SeekFrom::Start(data_offset))`, allocate
  `vec![0u8; naxis1 * naxis2 * 2]`, call `read_exact(&mut buf)` once.
- Convert with `buf.chunks_exact(2).map(|c| i16::from_be_bytes([c[0], c[1]]) as i32)`.
- Keep existing `Array2::from_shape_vec` construction.
- Preserve `CatalogError::Io` and `CatalogError::Validation` mappings.

Tests:

- Adjust existing tests if they depend on the byte-pair path (none should).
- Add a unit test that reads a `tests/fixtures/minimal.fits` into `Array2<i32>`
  and checks the shape and the first few values.

Definition of done: `read_image_i32` issues at most O(1) syscalls beyond
`open` and `seek`, and test passes.

---

**T2. Drop duplicate `/processed` zarr array** *(src/catalog/zarr_write.rs)*

- Remove the block in `write_frame_raw` that creates `/{scan}/{frame}/processed`.
- Update the module docstring: `/raw` only.
- Update `src/schema.rs:253-254` comment to describe the zarr layout as
  `/<scan>/<frame>/raw`; processed is produced by downstream processing, not ingest.
- Search the codebase for readers of `/processed`; none exist. If any show up,
  stop and escalate (scope change).

Tests:

- Add a Rust test that writes one frame and asserts `/{scan}/{frame}/raw`
  exists and `/{scan}/{frame}/processed` does not (using
  `store.list_dir` or equivalent).

---

**T3. Use Diesel `RETURNING` in ingest inserts** *(src/catalog/ingest.rs)*

Rewrite inserts for `samples`, `scans`, `files`, and `tags` in
`ingest_beamtime_inner` to use `.returning(table::id).get_result(conn)`
instead of `execute` followed by a `SELECT`. Keep `file_tags` insert as-is
(no id needed).

The `tags` path currently does `SELECT ... optional()? { Some => id, None =>
insert + SELECT }`. Replace with `INSERT OR IGNORE` then select; or use
`upsert`. Prefer `INSERT OR IGNORE INTO tags (slug) VALUES (?) RETURNING id`
when slug is new; fall back to a single SELECT when IGNORE returned no row.

Tests:

- Existing ingest tests must stay green.
- Add a test that asserts one sample/scan/file row per input (no duplicates)
  after re-running `ingest_beamtime` on the same minimal fixture.

---

**T4. Batched catalog transactions with small-scan coalescing** *(src/catalog/ingest.rs)*

Replace the single `conn.transaction::<(), _, _>(|conn| { ... entire loop ... })`
with batched transactions. Rules:

- Iterate scans in `scan_order`. Maintain `current_batch: Vec<&BtIngestRow>`.
- Append every row of the current scan to `current_batch`.
- After each scan, if `current_batch.len() >= MIN_BATCH_FILES` (constant,
  initially 200), commit the batch and start a new one.
- Always commit the trailing batch before zarr phase.
- For **very large scans** (>`MAX_BATCH_FILES`, initially 1000), split the
  scan into chunks of `MAX_BATCH_FILES` rows and commit each as one
  transaction, preserving `CatalogRow` events inside.

Inside each transaction:

- Insert only `samples` and `scans` that are new to the catalog (use existing
  `sample_cache` / `scan_cache` maps, populate lazily).
- Insert `files`, `frames`, `tags`, `file_tags` for rows in the batch.
- Emit `IngestProgress::CatalogRow` per row (unchanged semantics).

Reasoning: the user flagged that short scans processed quickly should not
each get their own transaction. Coalescing by file count satisfies that while
still capping uncommitted work at ~`MAX_BATCH_FILES` rows.

Constants live in a private `mod batch_limits` inside `ingest.rs`:

```rust
const MIN_BATCH_FILES: usize = 200;
const MAX_BATCH_FILES: usize = 1000;
```

Tests:

- Add a Rust test with a synthetic beamtime of 250 rows across 10 scans
  (some short, some long) that verifies:
  - rows appear in `files` and `frames` after each batch commit
    (use a second connection to inspect mid-ingest via a progress callback
    that signals at `catalog_row`).
  - final row count matches input.

---

**T5. Streaming zarr phase** *(src/catalog/ingest.rs)*

Replace the read-all-then-write loop with bounded-parallel read→write→drop:

- Create all zarr groups (`/`, `/{scan}`, `/{scan}/{frame}`) in a preamble
  on the calling thread, single-pass. Avoids concurrent group creation races
  with zarrs filesystem store.
- Use `rows.par_iter().try_for_each_with(|()| -> Result<()> { ... })` on
  the existing rayon pool to stream:
  - `let img = read_image_i32(row)?;`
  - `write_frame_raw(&zstore, row.scan_number, row.frame_number, &img)?;`
  - `drop(img);`
  - Emit `FileComplete` with progress counters (use `Arc<Mutex<ProgressState>>`
    for `scan_done`/`global_done`).
- Remove the intermediate `Vec<Vec<(usize, Array2<i32>)>>` and its flatten.

Peak RAM during zarr phase becomes `O(worker_threads)` images rather than
`O(total_files)`.

Tests:

- Update or add a test that counts `file_complete` timestamps and asserts
  they are not all within 10ms of each other for a synthetic 50-file run
  (exact threshold TBD during implementation).

---

**T6. Per-file parallelism** *(src/catalog/ingest.rs)*

In the headers phase, replace `scan_groups.par_iter()` with a flat
`paths_only.par_iter()` using `read_fits_headers_only_row` per path. Collect
rows into a `Vec<BtIngestRow>` in any order, then sort by
`(scan_number, frame_number, file_path)` (this sort is already present).

Rationale: uneven scan sizes cause one rayon worker to serialize a large
scan. Flat parallelism lets rayon balance work across all cores.

T5 already parallelizes per-file for zarr, so this task is strictly about
the headers phase.

Tests: same test as T5 (balanced parallelism).

### Phase B: Maintainability (tasks 7–10)

**T7. Unify FITS raw pixel buffer reader** *(src/io/)*

Extract the single bulk-read logic behind a new helper module
`src/io/raw_pixels.rs` (feature-gated identically to current io module):

```rust
pub fn read_bitpix16_be_bytes(path: &Path, offset: u64, nbytes: usize)
    -> Result<Vec<u8>, FitsError>;
```

Use this from both:

- `catalog::ingest::read_image_i32` (convert to `Array2<i32>`, no bzero).
- `io::image_mmap::load_image_pixels` (convert to `Array2<i64>` with bzero).

`image_mmap.rs` currently uses `memmap2` with `MmapOptions::new().offset(...)`.
Keep the mmap path as one implementation behind `read_bitpix16_be_bytes` when
the platform supports it; fall back to bulk `read_exact`. Document that mmap
can behave poorly on some NAS mounts; provide `PYREF_DISABLE_MMAP=1` override.

Do not change semantics of the i64+bzero conversion in `image_mmap`.

---

**T8. `IngestPhase` enum** *(src/catalog/ingest_progress.rs, src/lib.rs)*

Replace `IngestProgress::Phase { name: String }` with
`IngestProgress::Phase { phase: IngestPhase }` where:

```rust
pub enum IngestPhase { Headers, Catalog, Zarr }
```

Add `impl IngestPhase { pub fn as_str(&self) -> &'static str { ... } }` so
Python dict conversion in `src/lib.rs` keeps emitting `"headers"` / `"catalog"`
/ `"zarr"`. No Python-side change.

Update all call sites in `src/catalog/ingest.rs` to use the enum.

---

**T9. Split `ingest_beamtime_inner` into phase functions** *(src/catalog/ingest.rs)*

Introduce a private struct:

```rust
struct IngestContext<'a> {
    beamtime_id: i32,
    zarr_path: PathBuf,
    progress: Option<&'a IngestProgressSink>,
    cancel: Option<Arc<AtomicBool>>,
    pool: &'a rayon::ThreadPool,
    scan_total_map: HashMap<i32, u32>,
}
```

And split into:

- `fn run_headers_phase(ctx: &IngestContext, paths: &[PathBuf], header_items: &[String]) -> Result<Vec<BtIngestRow>>;`
- `fn run_catalog_phase(ctx: &IngestContext, conn: &mut SqliteConnection, rows: &[BtIngestRow]) -> Result<()>;`
- `fn run_zarr_phase(ctx: &IngestContext, rows: &[BtIngestRow]) -> Result<()>;`

`ingest_beamtime_inner` becomes a thin orchestrator: discover → build ctx →
headers → catalog → zarr → return db path.

No behavior change. Strict refactor.

---

**T10. Synthetic-beamtime test harness** *(tests/fixtures.rs or Rust-side util)*

Add a helper under `tests/` that writes N fake FITS files into a tmp dir
following the expected flat-CCD layout. Each file contains a minimal valid
BITPIX=16 image (e.g. 16x16 pixels). The helper returns the path.

Use this from:

- A new `tests/ingest_streaming.rs` that runs `ingest_beamtime` on ~100
  files across 10 scans and asserts streaming progress timestamps (T5).
- A new `scripts/bench_ingest.py` that generates e.g. 500 files, runs
  `ingest_beamtime`, prints the same markdown table as
  `scripts/profile_beamtime_ingest.py` but without NAS dependency.

This is the local-CI-safe counterpart to the NAS profiler.

## Dependencies between tasks

- T5 depends on T1 (streaming wants fast per-file reads).
- T4 depends on T3 (RETURNING makes transactions cheaper to split).
- T9 should land after T1-T6 so the refactor wraps already-correct phases.
- T10 can land anytime but is most valuable after T5 (gives measurable data).

## Risks and rollback

- **Streaming zarr phase concurrent writes.** If zarrs `FilesystemStore`
  turns out not to be thread-safe for array creation, fall back to a two-pool
  design: N reader threads feeding one writer thread via a bounded channel.
- **Transaction batching overhead.** If 200-row batches cause measurable
  commit overhead on real beamtimes, tune `MIN_BATCH_FILES` down.
- **RETURNING clause.** If a platform SQLite below 3.35 sneaks in through
  system linking, the Diesel feature flag should still compile; runtime
  errors would surface immediately in T3's test. Cargo config already
  bundles `libsqlite3-sys` which is >= 3.44, so this is belt-and-suspenders.

Every task produces one commit. Reverting individual commits gives
controlled rollback.
