PRAGMA foreign_keys = OFF;

ALTER TABLE frame_header_values RENAME TO header_values;

CREATE TABLE header_cards_new (
    id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    name TEXT NOT NULL UNIQUE,
    display_name TEXT NOT NULL
);

INSERT INTO header_cards_new (id, name, display_name)
SELECT id, name, display_name
FROM header_cards;

DROP TABLE header_cards;
ALTER TABLE header_cards_new RENAME TO header_cards;

DROP INDEX IF EXISTS idx_frames_scan;
DROP INDEX IF EXISTS idx_frames_zarr_shape_bucket_frame_index;

CREATE TABLE frames_new (
    id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    scan_id INTEGER NOT NULL REFERENCES scans(id) ON DELETE CASCADE,
    file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    frame_number INTEGER NOT NULL,
    zarr_group_key INTEGER NOT NULL,
    zarr_frame_index INTEGER NOT NULL,
    zarr_bucket_frame_index INTEGER,
    acquired_at TEXT,
    quality_flag TEXT,
    UNIQUE(file_id)
);

INSERT INTO frames_new (
    id,
    scan_id,
    file_id,
    frame_number,
    zarr_group_key,
    zarr_frame_index,
    zarr_bucket_frame_index,
    acquired_at,
    quality_flag
)
SELECT
    id,
    scan_id,
    file_id,
    frame_number,
    zarr_group_key,
    zarr_frame_index,
    zarr_bucket_frame_index,
    acquired_at,
    quality_flag
FROM frames;

DROP TABLE frames;
ALTER TABLE frames_new RENAME TO frames;

CREATE INDEX idx_frames_scan ON frames(scan_id);

PRAGMA foreign_keys = ON;
