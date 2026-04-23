# Lazy Refresh with Watcher

## Goal

Define low-overhead data refresh behavior when the watcher daemon appends newly ingested scans to catalog and zarr storage.

## Canonical Pattern

```python
lazy_frames = bt.frames_lazy(sample_name="sample_a")
frames = lazy_frames.collect()
```

When watcher ingests new scans, rerun `collect()` in the notebook cell to include new rows.

## Required Behavior

- lazy query plans are not invalidated by append-only ingest
- rerunning cell does not require manual cache reset
- no full pre-materialization required for unchanged data paths

## User Experience Contract

- "rerun cell" is the only required refresh action
- refreshed tables preserve the same schema and normalization contract
- downstream cells can be rerun without changing code
