PRAGMA foreign_keys = OFF;

DROP INDEX IF EXISTS idx_frames_scan;

CREATE TABLE frames_old (
    id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    scan_id INTEGER NOT NULL REFERENCES scans(id) ON DELETE CASCADE,
    file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    frame_number INTEGER NOT NULL,
    zarr_group_key INTEGER NOT NULL,
    zarr_frame_index INTEGER NOT NULL,
    acquired_at TEXT,
    sample_x REAL NOT NULL,
    sample_y REAL NOT NULL,
    sample_z REAL NOT NULL,
    sample_theta REAL NOT NULL,
    ccd_theta REAL NOT NULL,
    beamline_energy REAL NOT NULL,
    epu_polarization REAL NOT NULL,
    exposure REAL NOT NULL,
    ring_current REAL NOT NULL,
    ai3_izero REAL NOT NULL,
    beam_current REAL NOT NULL,
    quality_flag TEXT,
    zarr_bucket_frame_index INTEGER,
    UNIQUE(file_id)
);

INSERT INTO frames_old (
    id,
    scan_id,
    file_id,
    frame_number,
    zarr_group_key,
    zarr_frame_index,
    acquired_at,
    sample_x,
    sample_y,
    sample_z,
    sample_theta,
    ccd_theta,
    beamline_energy,
    epu_polarization,
    exposure,
    ring_current,
    ai3_izero,
    beam_current,
    quality_flag,
    zarr_bucket_frame_index
)
SELECT
    id,
    scan_id,
    file_id,
    frame_number,
    zarr_group_key,
    zarr_frame_index,
    acquired_at,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    quality_flag,
    zarr_bucket_frame_index
FROM frames;

DROP TABLE frames;
ALTER TABLE frames_old RENAME TO frames;

CREATE INDEX idx_frames_scan ON frames(scan_id);

CREATE TABLE header_cards_old (
    id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    name TEXT NOT NULL UNIQUE,
    display_name TEXT NOT NULL,
    card_category TEXT NOT NULL
);

INSERT INTO header_cards_old (id, name, display_name, card_category)
SELECT id, name, display_name, 'metadata'
FROM header_cards;

DROP TABLE header_cards;
ALTER TABLE header_cards_old RENAME TO header_cards;

ALTER TABLE header_values RENAME TO frame_header_values;

PRAGMA foreign_keys = ON;
