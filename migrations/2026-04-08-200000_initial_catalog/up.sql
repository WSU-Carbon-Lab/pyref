PRAGMA foreign_keys = OFF;

CREATE TABLE beamtimes (
    id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    nas_uri TEXT NOT NULL UNIQUE,
    zarr_path TEXT NOT NULL,
    date TEXT NOT NULL,
    last_indexed_at INTEGER
);

CREATE TABLE path_aliases (
    id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    label TEXT NOT NULL UNIQUE,
    physical_path TEXT NOT NULL,
    registered_at TEXT NOT NULL
);

CREATE TABLE samples (
    id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    beamtime_id INTEGER NOT NULL REFERENCES beamtimes(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    representative_x REAL NOT NULL,
    representative_y REAL NOT NULL,
    representative_z REAL NOT NULL,
    UNIQUE(beamtime_id, name)
);

CREATE TABLE tags (
    id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    slug TEXT NOT NULL UNIQUE
);

CREATE TABLE files (
    id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    beamtime_id INTEGER NOT NULL REFERENCES beamtimes(id) ON DELETE CASCADE,
    sample_id INTEGER NOT NULL REFERENCES samples(id) ON DELETE CASCADE,
    scan_number INTEGER NOT NULL,
    frame_number INTEGER NOT NULL,
    nas_uri TEXT NOT NULL,
    filename TEXT NOT NULL,
    parse_flag TEXT,
    UNIQUE(beamtime_id, scan_number, frame_number)
);

CREATE TABLE file_tags (
    id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    tag_id INTEGER NOT NULL REFERENCES tags(id) ON DELETE CASCADE,
    UNIQUE(file_id, tag_id)
);

CREATE TABLE scans (
    id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    beamtime_id INTEGER NOT NULL REFERENCES beamtimes(id) ON DELETE CASCADE,
    sample_id INTEGER NOT NULL REFERENCES samples(id) ON DELETE CASCADE,
    scan_number INTEGER NOT NULL,
    scan_type TEXT NOT NULL,
    started_at TEXT,
    ended_at TEXT,
    UNIQUE(beamtime_id, scan_number)
);

CREATE TABLE header_cards (
    id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    name TEXT NOT NULL UNIQUE,
    display_name TEXT NOT NULL,
    card_category TEXT NOT NULL
);

CREATE TABLE frames (
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
    UNIQUE(file_id)
);

CREATE TABLE frame_header_values (
    id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    frame_id INTEGER NOT NULL REFERENCES frames(id) ON DELETE CASCADE,
    header_card_id INTEGER NOT NULL REFERENCES header_cards(id) ON DELETE CASCADE,
    value REAL NOT NULL
);

CREATE TABLE profiles (
    id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    scan_id INTEGER NOT NULL REFERENCES scans(id) ON DELETE CASCADE,
    sample_id INTEGER NOT NULL REFERENCES samples(id) ON DELETE CASCADE,
    profile_index INTEGER NOT NULL,
    profile_type TEXT NOT NULL,
    fixed_parameter_value REAL NOT NULL,
    epu_polarization REAL NOT NULL,
    sample_x REAL NOT NULL,
    sample_y REAL NOT NULL,
    sample_z REAL NOT NULL,
    UNIQUE(scan_id, profile_index)
);

CREATE TABLE profile_frames (
    id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    profile_id INTEGER NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
    frame_id INTEGER NOT NULL REFERENCES frames(id) ON DELETE CASCADE,
    frame_role TEXT NOT NULL
);

CREATE TABLE beam_finding (
    id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    frame_id INTEGER NOT NULL REFERENCES frames(id) ON DELETE CASCADE,
    edge_removal_applied INTEGER NOT NULL,
    row_bg_dark_cols INTEGER,
    col_bg_dark_rows INTEGER,
    gaussian_kernel_sigma REAL,
    centroid_row REAL,
    centroid_col REAL,
    roi_intensity REAL,
    fit_std REAL,
    dark_region_mean REAL,
    dark_region_std REAL,
    detection_flag TEXT NOT NULL,
    UNIQUE(frame_id)
);

CREATE TABLE stitch_corrections (
    id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    profile_id INTEGER NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
    stitch_index INTEGER NOT NULL,
    fano_factor REAL NOT NULL,
    overlap_scale_factor REAL,
    i0_normalization_value REAL,
    i0_source_scan_id INTEGER REFERENCES scans(id),
    UNIQUE(profile_id, stitch_index)
);

CREATE TABLE reflectivity (
    id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    profile_id INTEGER NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
    frame_id INTEGER NOT NULL REFERENCES frames(id) ON DELETE CASCADE,
    beam_finding_id INTEGER NOT NULL REFERENCES beam_finding(id) ON DELETE CASCADE,
    q REAL NOT NULL,
    theta REAL NOT NULL,
    energy REAL NOT NULL,
    intensity REAL NOT NULL,
    uncertainty REAL NOT NULL,
    frame_type TEXT NOT NULL
);

CREATE TABLE file_overrides (
    id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    source_path TEXT NOT NULL UNIQUE,
    sample_name TEXT,
    tag TEXT,
    notes TEXT
);

CREATE INDEX idx_files_beamtime ON files(beamtime_id);
CREATE INDEX idx_frames_scan ON frames(scan_id);
CREATE INDEX idx_scans_beamtime ON scans(beamtime_id);

PRAGMA foreign_keys = ON;
