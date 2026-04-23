//! Diesel `Queryable` rows for catalog queries.

#![allow(dead_code)]

use crate::schema::{files, frames, samples, scans};
use diesel::prelude::*;

#[derive(Debug, Clone, Queryable, Selectable)]
#[diesel(table_name = frames)]
pub struct FrameDb {
    pub id: i32,
    pub scan_id: i32,
    pub file_id: i32,
    pub frame_number: i32,
    pub zarr_group_key: i32,
    pub zarr_frame_index: i32,
    pub zarr_bucket_frame_index: Option<i32>,
    pub acquired_at: Option<String>,
    pub quality_flag: Option<String>,
}

#[derive(Debug, Clone, Queryable, Selectable)]
#[diesel(table_name = files)]
pub struct FileDb {
    pub id: i32,
    pub beamtime_id: i32,
    pub sample_id: i32,
    pub scan_number: i32,
    pub frame_number: i32,
    pub nas_uri: String,
    pub filename: String,
    pub parse_flag: Option<String>,
}

#[derive(Debug, Clone, Queryable, Selectable)]
#[diesel(table_name = scans)]
pub struct ScanDb {
    pub id: i32,
    pub beamtime_id: i32,
    pub sample_id: i32,
    pub scan_number: i32,
    pub scan_type: String,
    pub started_at: Option<String>,
    pub ended_at: Option<String>,
}

#[derive(Debug, Clone, Queryable, Selectable)]
#[diesel(table_name = samples)]
pub struct SampleDb {
    pub id: i32,
    pub beamtime_id: i32,
    pub name: String,
    pub representative_x: f64,
    pub representative_y: f64,
    pub representative_z: f64,
}
