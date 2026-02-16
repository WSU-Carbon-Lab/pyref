//! Options for read_fits and scan_fits: header keys, schema, batch size, catalog filter.

use crate::io::schema::FitsMetadataSchema;
use crate::io::source::ResolvePreference;

#[cfg(feature = "catalog")]
use crate::catalog::CatalogFilter;

/// Default header keys for metadata read (matches ingest and Python DEFAULT_HEADER_KEYS).
pub const DEFAULT_HEADER_ITEMS: &[&str] = &[
    "DATE",
    "Beamline Energy",
    "Sample Theta",
    "CCD Theta",
    "Higher Order Suppressor",
    "EPU Polarization",
    "EXPOSURE",
    "Sample Name",
    "Scan ID",
];

/// Options for eager `read_fits`: which headers to read, whether to add Q/Lambda, batch size.
#[derive(Debug, Clone)]
pub struct ReadFitsOptions {
    /// Header keys to read from each FITS file. Defaults to `DEFAULT_HEADER_ITEMS`.
    pub header_items: Vec<String>,
    /// If true, read only headers (no image data). Always true for metadata-only read.
    pub header_only: bool,
    /// If true, add calculated columns (Lambda, Q, etc.) from Beamline Energy and Sample Theta.
    pub add_calculated_domains: bool,
    /// Optional schema override; if None, output follows canonical FitsMetadataSchema.
    pub schema: Option<FitsMetadataSchema>,
    /// Chunk size for batch read when source is multiple files. Default 500 for ingest-style.
    pub batch_size: usize,
    /// When source is a dir with a catalog, whether to use catalog or disk.
    pub resolve_preference: ResolvePreference,
    /// When source resolves to catalog, optional filter. Only used when reading from catalog.
    #[cfg(feature = "catalog")]
    pub catalog_filter: Option<CatalogFilter>,
}

impl Default for ReadFitsOptions {
    fn default() -> Self {
        Self {
            header_items: DEFAULT_HEADER_ITEMS
                .iter()
                .map(|s| (*s).to_string())
                .collect(),
            header_only: true,
            add_calculated_domains: true,
            schema: None,
            batch_size: 500,
            resolve_preference: ResolvePreference::PreferCatalog,
            #[cfg(feature = "catalog")]
            catalog_filter: None,
        }
    }
}

/// Options for lazy `scan_fits`: same as read plus catalog filter when source is catalog.
#[derive(Debug, Clone)]
pub struct ScanFitsOptions {
    /// Header keys to read from each FITS file.
    pub header_items: Vec<String>,
    /// If true, read only headers (no image data).
    pub header_only: bool,
    /// If true, add calculated columns (Lambda, Q, etc.).
    pub add_calculated_domains: bool,
    /// Optional schema override.
    pub schema: Option<FitsMetadataSchema>,
    /// Chunk size for batch read when scanning from disk. Default 50.
    pub batch_size: usize,
    /// When source resolves to catalog, optional filter (sample_name, tag, scan_numbers, energy).
    #[cfg(feature = "catalog")]
    pub catalog_filter: Option<CatalogFilter>,
    /// When source could be catalog or disk, which to use.
    pub resolve_preference: ResolvePreference,
}

impl From<()> for ReadFitsOptions {
    fn from(_: ()) -> Self {
        Self::default()
    }
}

impl From<()> for ScanFitsOptions {
    fn from(_: ()) -> Self {
        Self::default()
    }
}

impl Default for ScanFitsOptions {
    fn default() -> Self {
        Self {
            header_items: DEFAULT_HEADER_ITEMS
                .iter()
                .map(|s| (*s).to_string())
                .collect(),
            header_only: true,
            add_calculated_domains: true,
            schema: None,
            batch_size: 50,
            #[cfg(feature = "catalog")]
            catalog_filter: None,
            resolve_preference: ResolvePreference::PreferCatalog,
        }
    }
}
