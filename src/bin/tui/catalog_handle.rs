use std::path::{Path, PathBuf};

#[cfg(feature = "catalog")]
use pyref::catalog::{
    list_experimentalists, list_beamtimes_for_expt, catalog_status_for_path,
    list_beamtime_entries_v2, query_scan_points,
    ExptMeta, BeamtimeMeta, DbCatalogStatus, BeamtimeEntries, FileRow, CatalogFilter,
};

#[derive(Debug)]
pub struct CatalogHandle {
    pub db_path: PathBuf,
}

impl CatalogHandle {
    pub fn new(db_path: PathBuf) -> Self {
        Self { db_path }
    }

    #[cfg(feature = "catalog")]
    pub fn list_experimentalists(&self, data_root: &Path) -> Vec<ExptMeta> {
        list_experimentalists(&self.db_path, data_root).unwrap_or_default()
    }

    #[cfg(feature = "catalog")]
    pub fn list_beamtimes(&self, data_root: &Path, experimentalist: &str) -> Vec<BeamtimeMeta> {
        list_beamtimes_for_expt(&self.db_path, data_root, experimentalist).unwrap_or_default()
    }

    #[cfg(feature = "catalog")]
    pub fn catalog_status(&self, beamtime_path: &Path) -> DbCatalogStatus {
        catalog_status_for_path(&self.db_path, beamtime_path)
    }

    #[cfg(feature = "catalog")]
    pub fn query_scan_points(&self, beamtime_path: &Path, filter: Option<&CatalogFilter>) -> Vec<FileRow> {
        query_scan_points(&self.db_path, beamtime_path, filter).unwrap_or_default()
    }

    #[cfg(feature = "catalog")]
    pub fn list_beamtime_entries_v2(&self, beamtime_path: &Path) -> Option<BeamtimeEntries> {
        list_beamtime_entries_v2(&self.db_path, beamtime_path).ok()
    }
}
