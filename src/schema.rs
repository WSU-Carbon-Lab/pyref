// schema.rs
//
// Diesel schema for the pyref catalog database.
//
// Design notes:
//
// Header cards from the FITS primary HDU are split into two tiers. The dozen
// cards that drive scan classification, beamspot localization, normalization,
// and profile identity are promoted to first-class columns on the `frames`
// table (sample_x, sample_y, sample_z, sample_theta, ccd_theta,
// beamline_energy, epu_polarization, exposure, ring_current, ai3_izero,
// beam_current). Every remaining card is stored in the `frame_header_values`
// EAV table, keyed through `header_cards`, which is populated on first
// ingestion from whatever cards are present in the FITS files. This makes the
// schema forward-compatible when the beamline control system adds or renames
// channels without requiring a migration.
//
// SQLite type notes:
//   - All floating-point physical quantities use Double (f64).
//   - Booleans are stored as SmallInt (0/1); SQLite has no native bool.
//   - Timestamps are stored as Text in ISO 8601 format.
//   - Nullable foreign keys on optional relationships (e.g. i0_source_scan_id
//     for fixed-angle profiles that borrow I0 from an external scan) use
//     Nullable<Integer>.
//
// Enum-like Text columns:
//   scan_type         : "fixed_energy" | "fixed_angle"
//   profile_type      : "fixed_energy" | "fixed_angle"
//   frame_role        : "i0" | "stitch" | "overlap" | "reflectivity"
//   card_category     : "motor" | "ai" | "camera" | "metadata"
//   quality_flag      : "ok" | "mislabeled_sample" | "parse_failure"
//   detection_flag    : "ok" | "beam_detection_failed" | "beam_drift_anomaly"

// ---------------------------------------------------------------------------
// Beamtime
// ---------------------------------------------------------------------------

diesel::table! {
    /// Root of the catalog hierarchy. One row per beamtime directory ingested.
    /// The `path` column is the absolute path to the beamtime root and is also
    /// used to resolve the monolithic zarr archive at `<path>/beamtime.zarr`.
    beamtimes (id) {
        id -> Integer,
        /// Absolute path to the beamtime root directory.
        path -> Text,
        /// ISO 8601 date string parsed from the beamtime directory name.
        date -> Text,
    }
}

// ---------------------------------------------------------------------------
// Samples
// ---------------------------------------------------------------------------

diesel::table! {
    /// One row per unique sample name encountered within a beamtime. The
    /// representative stage position is computed as the median of Sample X,
    /// Sample Y, and Sample Z across all frames attributed to this sample.
    /// Frames that deviate from this position by more than the configured
    /// tolerance are flagged MISLABELED_SAMPLE in the `frames` table rather
    /// than creating a second sample row.
    samples (id) {
        id -> Integer,
        beamtime_id -> Integer,
        /// Sample name as parsed from the FITS filename stem.
        name -> Text,
        /// Median Sample X position (mm) across all attributed frames.
        representative_x -> Double,
        /// Median Sample Y position (mm) across all attributed frames.
        representative_y -> Double,
        /// Median Sample Z position (mm) across all attributed frames.
        representative_z -> Double,
    }
}

diesel::joinable!(samples -> beamtimes (beamtime_id));

// ---------------------------------------------------------------------------
// Tags
// ---------------------------------------------------------------------------

diesel::table! {
    /// Tag slugs parsed from FITS filenames. Tags are scan-specific; the
    /// many-to-many relationship between tags and files is resolved through
    /// `file_tags`.
    tags (id) {
        id -> Integer,
        /// Normalized tag string as parsed from the filename.
        slug -> Text,
    }
}

diesel::table! {
    /// Junction table resolving the many-to-many relationship between files
    /// and tags.
    file_tags (id) {
        id -> Integer,
        file_id -> Integer,
        tag_id -> Integer,
    }
}

diesel::joinable!(file_tags -> tags (tag_id));

// ---------------------------------------------------------------------------
// Files
// ---------------------------------------------------------------------------

diesel::table! {
    /// One row per FITS file ingested. The canonical reference for raw file
    /// provenance. Image data is not retrieved through this table; it is
    /// accessed via the zarr keys on `frames`.
    files (id) {
        id -> Integer,
        beamtime_id -> Integer,
        sample_id -> Integer,
        scan_number -> Integer,
        frame_number -> Integer,
        /// Absolute path to the FITS file.
        path -> Text,
        /// Bare filename including extension.
        filename -> Text,
        /// NULL when parsing succeeds. Set to "parse_failure" when the
        /// filename does not conform to any supported naming pattern.
        parse_flag -> Nullable<Text>,
    }
}

diesel::joinable!(files -> beamtimes (beamtime_id));
diesel::joinable!(files -> samples (sample_id));
diesel::joinable!(file_tags -> files (file_id));

// ---------------------------------------------------------------------------
// Scans
// ---------------------------------------------------------------------------

diesel::table! {
    /// One row per scan. Scan type is determined during ingestion from the
    /// motor trajectory analysis and stored here as a first-class attribute
    /// so downstream reduction does not recompute it.
    scans (id) {
        id -> Integer,
        beamtime_id -> Integer,
        sample_id -> Integer,
        scan_number -> Integer,
        /// "fixed_energy" or "fixed_angle".
        scan_type -> Text,
        /// ISO 8601 timestamp of the first frame in the scan.
        started_at -> Nullable<Text>,
        /// ISO 8601 timestamp of the last frame in the scan.
        ended_at -> Nullable<Text>,
    }
}

diesel::joinable!(scans -> beamtimes (beamtime_id));
diesel::joinable!(scans -> samples (sample_id));

// ---------------------------------------------------------------------------
// Header card registry
// ---------------------------------------------------------------------------

diesel::table! {
    /// Registry of FITS header card names discovered during initial ingestion.
    /// One row per unique card name. Populated automatically on first ingest;
    /// subsequent beamtimes with new card names append rows here without
    /// requiring a schema migration.
    ///
    /// `card_category` classifies the card for UI and query purposes:
    ///   "motor"    - physical positioning motor (Sample X, CCD Theta, etc.)
    ///   "ai"       - analog input channel (Beam Current, TEY signal, etc.)
    ///   "camera"   - CCD / detector configuration (ROI, binning, temp)
    ///   "metadata" - timing, instrument bookkeeping, MCS axes
    header_cards (id) {
        id -> Integer,
        /// Raw card name as it appears in the FITS header (e.g. "AI 3 Izero").
        name -> Text,
        /// Human-readable display name for UI use.
        display_name -> Text,
        /// "motor" | "ai" | "camera" | "metadata"
        card_category -> Text,
    }
}

// ---------------------------------------------------------------------------
// Frames (Data Table)
// ---------------------------------------------------------------------------

diesel::table! {
    /// One row per frame per scan. Contains all first-class reduction-critical
    /// header values as typed columns, plus zarr retrieval keys. All remaining
    /// header cards are stored in `frame_header_values`.
    ///
    /// Zarr retrieval: the monolithic beamtime archive is at
    /// `<beamtime.path>/beamtime.zarr`. Within the archive, images are at
    /// `/<scan_number>/<frame_number>/raw` and
    /// `/<scan_number>/<frame_number>/processed`.
    frames (id) {
        id -> Integer,
        scan_id -> Integer,
        file_id -> Integer,
        frame_number -> Integer,
        /// Group key within the zarr archive, equal to the scan number.
        zarr_group_key -> Integer,
        /// Dataset index within the zarr group, equal to the frame number.
        zarr_frame_index -> Integer,
        /// ISO 8601 acquisition timestamp from the DATE header card.
        acquired_at -> Nullable<Text>,
        // --- first-class motor positions ---
        /// Sample X stage position (mm). FITS card: "Sample X".
        sample_x -> Double,
        /// Sample Y stage position (mm). FITS card: "Sample Y".
        sample_y -> Double,
        /// Sample Z stage position (mm). FITS card: "Sample Z".
        sample_z -> Double,
        /// Sample theta (degrees). FITS card: "Sample Theta".
        sample_theta -> Double,
        /// CCD theta (degrees). FITS card: "CCD Theta".
        ccd_theta -> Double,
        /// Beamline energy (eV). FITS card: "Beamline Energy".
        beamline_energy -> Double,
        // --- first-class AI / beam channels ---
        /// EPU polarization angle (degrees). FITS card: "EPU Polarization".
        epu_polarization -> Double,
        /// CCD exposure time (seconds). FITS card: "EXPOSURE".
        exposure -> Double,
        /// Storage ring current (mA). FITS card: "RINGCRNT".
        ring_current -> Double,
        /// Upstream gold mesh absorption current (V). FITS card: "AI 3 Izero".
        ai3_izero -> Double,
        /// Photodiode beam current (mA). FITS card: "Beam Current".
        beam_current -> Double,
        // --- quality flag ---
        /// NULL when ok. "mislabeled_sample" when stage position deviates
        /// beyond configured tolerance for the attributed sample name.
        quality_flag -> Nullable<Text>,
    }
}

diesel::joinable!(frames -> scans (scan_id));
diesel::joinable!(frames -> files (file_id));

// ---------------------------------------------------------------------------
// Frame header values (EAV for non-critical cards)
// ---------------------------------------------------------------------------

diesel::table! {
    /// Entity-attribute-value store for all FITS header cards not promoted to
    /// first-class columns on `frames`. All card values from the primary HDU
    /// are stored as Double; the card name is resolved through `header_cards`.
    frame_header_values (id) {
        id -> Integer,
        frame_id -> Integer,
        header_card_id -> Integer,
        value -> Double,
    }
}

diesel::joinable!(frame_header_values -> frames (frame_id));
diesel::joinable!(frame_header_values -> header_cards (header_card_id));

// ---------------------------------------------------------------------------
// Profiles
// ---------------------------------------------------------------------------

diesel::table! {
    /// One row per reduced reflectivity profile. A profile is the primary
    /// user-facing unit: a single continuous 1D curve assembled from one or
    /// more stitches collected at a fixed energy (fixed-energy scan) or fixed
    /// angle (fixed-angle scan). Multi-profile scans produce multiple profile
    /// rows sharing the same scan_id, distinguished by profile_index.
    ///
    /// The sample position columns here are the median over all member frames
    /// and are stored for query convenience. The authoritative per-frame
    /// positions remain on `frames`.
    profiles (id) {
        id -> Integer,
        scan_id -> Integer,
        sample_id -> Integer,
        /// Zero-based ordinal position of this profile within the parent scan.
        profile_index -> Integer,
        /// "fixed_energy" or "fixed_angle".
        profile_type -> Text,
        /// Value of the fixed parameter: energy (eV) for fixed_angle profiles,
        /// theta (degrees) for fixed_energy profiles.
        fixed_parameter_value -> Double,
        /// EPU polarization (degrees), constant across the profile.
        epu_polarization -> Double,
        /// Median Sample X (mm) over member frames.
        sample_x -> Double,
        /// Median Sample Y (mm) over member frames.
        sample_y -> Double,
        /// Median Sample Z (mm) over member frames.
        sample_z -> Double,
    }
}

diesel::joinable!(profiles -> scans (scan_id));
diesel::joinable!(profiles -> samples (sample_id));

// ---------------------------------------------------------------------------
// Profile-frame junction
// ---------------------------------------------------------------------------

diesel::table! {
    /// Junction table mapping profiles to their constituent frames, with a
    /// frame_role column classifying each frame's function in the reduction
    /// pipeline. I0 frames appear here multiple times when they serve as the
    /// normalization reference for more than one profile in a multi-profile
    /// scan.
    ///
    /// frame_role values:
    ///   "i0"           - direct beam frame used for I0 normalization
    ///   "stitch"       - first frame of a new stitch segment
    ///   "overlap"      - frame overlapping the preceding stitch for scaling
    ///   "reflectivity" - ordinary reduced reflectivity frame
    profile_frames (id) {
        id -> Integer,
        profile_id -> Integer,
        frame_id -> Integer,
        /// "i0" | "stitch" | "overlap" | "reflectivity"
        frame_role -> Text,
    }
}

diesel::joinable!(profile_frames -> profiles (profile_id));
diesel::joinable!(profile_frames -> frames (frame_id));

// ---------------------------------------------------------------------------
// BeamFinding
// ---------------------------------------------------------------------------

diesel::table! {
    /// Per-frame output of the beamspot localization pipeline. Stores both
    /// the preprocessing parameters applied and the fitted peak result, so the
    /// full reduction provenance is recoverable without re-running the pipeline.
    ///
    /// detection_flag values:
    ///   "ok"                   - credible peak found within detector boundary
    ///   "beam_detection_failed" - peak amplitude below noise threshold or
    ///                             centroid outside detector boundary
    ///   "beam_drift_anomaly"   - centroid deviates from linear drift model
    ///                            by more than the configured threshold
    beam_finding (id) {
        id -> Integer,
        frame_id -> Integer,
        /// 1 if the fixed-width border mask was applied; 0 otherwise.
        edge_removal_applied -> SmallInt,
        /// Number of dark columns used per row for row-wise background subtraction.
        row_bg_dark_cols -> Nullable<Integer>,
        /// Number of dark rows used per column for column-wise background subtraction.
        col_bg_dark_rows -> Nullable<Integer>,
        /// Standard deviation of the Gaussian kernel applied for noise suppression.
        gaussian_kernel_sigma -> Nullable<Double>,
        /// Fitted beamspot centroid row on the post-processed image.
        centroid_row -> Nullable<Double>,
        /// Fitted beamspot centroid column on the post-processed image.
        centroid_col -> Nullable<Double>,
        /// Integrated ROI intensity (counts) after background subtraction.
        roi_intensity -> Nullable<Double>,
        /// Standard deviation of the Gaussian fit to the beamspot.
        fit_std -> Nullable<Double>,
        /// Mean intensity of the designated dark region (counts).
        dark_region_mean -> Nullable<Double>,
        /// Standard deviation of the dark region (counts).
        dark_region_std -> Nullable<Double>,
        /// "ok" | "beam_detection_failed" | "beam_drift_anomaly"
        detection_flag -> Text,
    }
}

diesel::joinable!(beam_finding -> frames (frame_id));

// ---------------------------------------------------------------------------
// StitchCorrection
// ---------------------------------------------------------------------------

diesel::table! {
    /// Per-stitch correction factors computed during the normalization and
    /// stitching pipeline. One row per stitch segment within a profile.
    ///
    /// The Fano factor must always be recorded. A scan processed without a
    /// Fano correction records fano_factor = 1.0 rather than NULL.
    ///
    /// For fixed-angle profiles where I0 is sourced from a separate scan,
    /// i0_source_scan_id is set to that scan's id; otherwise it is NULL and
    /// I0 is derived from the i0 frames within the parent profile.
    stitch_corrections (id) {
        id -> Integer,
        profile_id -> Integer,
        /// Zero-based ordinal index of this stitch within the profile.
        stitch_index -> Integer,
        /// Energy-dependent Fano factor. Always non-null; 1.0 when no
        /// Fano correction was applied.
        fano_factor -> Double,
        /// Weighted-mean overlap scaling factor applied to this stitch.
        /// NULL for the first stitch, which has no preceding stitch to scale against.
        overlap_scale_factor -> Nullable<Double>,
        /// I0 normalization value (counts) used for this stitch.
        i0_normalization_value -> Nullable<Double>,
        /// FK to the scan supplying I0 for fixed-angle profiles. NULL when
        /// I0 comes from frames within the current profile.
        i0_source_scan_id -> Nullable<Integer>,
    }
}

diesel::joinable!(stitch_corrections -> profiles (profile_id));

// ---------------------------------------------------------------------------
// Reflectivity
// ---------------------------------------------------------------------------

diesel::table! {
    /// Frame-level reduced reflectivity data. One row per reduced frame after
    /// normalization and stitching. Frames flagged BEAM_DETECTION_FAILED in
    /// `beam_finding` must not appear here.
    ///
    /// Full stitched profile assembly and parquet export are performed by the
    /// packaging utility in `pyref.reduction`, which joins this table against
    /// `stitch_corrections` filtered by profile_id.
    reflectivity (id) {
        id -> Integer,
        profile_id -> Integer,
        frame_id -> Integer,
        beam_finding_id -> Integer,
        /// Momentum transfer (inverse angstroms).
        q -> Double,
        /// Sample theta (degrees).
        theta -> Double,
        /// Beamline energy (eV).
        energy -> Double,
        /// Normalized reflectivity intensity (dimensionless).
        intensity -> Double,
        /// Propagated one-sigma uncertainty on intensity.
        uncertainty -> Double,
        /// "i0" | "stitch" | "overlap" | "reflectivity"
        frame_type -> Text,
    }
}

diesel::joinable!(reflectivity -> profiles (profile_id));
diesel::joinable!(reflectivity -> frames (frame_id));
diesel::joinable!(reflectivity -> beam_finding (beam_finding_id));

// ---------------------------------------------------------------------------
// Allow tables to appear in the same query
// ---------------------------------------------------------------------------

diesel::allow_tables_to_appear_in_same_query!(
    beamtimes,
    samples,
    tags,
    file_tags,
    files,
    scans,
    header_cards,
    frames,
    frame_header_values,
    profiles,
    profile_frames,
    beam_finding,
    stitch_corrections,
    reflectivity,
);