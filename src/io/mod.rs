pub mod blur;
pub mod image_mmap;
pub mod options;
pub mod schema;
pub mod source;

#[cfg(feature = "zarr")]
pub mod zarr_store;

use ndarray::{Array2, ArrayBase, Axis, Dim, IxDynImpl, OwnedRepr};
use polars::prelude::*;
use regex::Regex;
use std::fs;
use std::ops::Mul;
use std::path::PathBuf;

use crate::errors::FitsError;
use crate::fits::{ImageHduHeader, PrimaryHdu};

#[derive(Debug, Clone)]
pub struct ImageInfo {
    pub path: PathBuf,
    pub data_offset: u64,
    pub naxis1: usize,
    pub naxis2: usize,
    pub bitpix: i32,
    pub bzero: i64,
}

impl ImageInfo {
    pub fn from_header(path: PathBuf, h: &ImageHduHeader) -> Self {
        let bzero = h
            .header
            .get_card("BZERO")
            .and_then(|c| c.value.as_int())
            .unwrap_or(0);
        ImageInfo {
            path,
            data_offset: h.data_offset,
            naxis1: h.naxis1,
            naxis2: h.naxis2,
            bitpix: h.bitpix,
            bzero,
        }
    }

    pub fn from_dataframe_row(df: &DataFrame, row_index: usize) -> Result<Self, FitsError> {
        let path_str: &str = df
            .column("file_path")
            .map_err(FitsError::from)?
            .str()
            .map_err(FitsError::from)?
            .get(row_index)
            .ok_or_else(|| FitsError::validation("file_path row missing or null"))?;
        let path = PathBuf::from(path_str);
        let data_offset: u64 = df
            .column("data_offset")
            .map_err(FitsError::from)?
            .i64()
            .map_err(FitsError::from)?
            .get(row_index)
            .ok_or_else(|| FitsError::validation("data_offset row missing or null"))?
            as u64;
        let naxis1: usize = df
            .column("naxis1")
            .map_err(FitsError::from)?
            .i64()
            .map_err(FitsError::from)?
            .get(row_index)
            .ok_or_else(|| FitsError::validation("naxis1 row missing or null"))?
            as usize;
        let naxis2: usize = df
            .column("naxis2")
            .map_err(FitsError::from)?
            .i64()
            .map_err(FitsError::from)?
            .get(row_index)
            .ok_or_else(|| FitsError::validation("naxis2 row missing or null"))?
            as usize;
        let bitpix: i32 = df
            .column("bitpix")
            .map_err(FitsError::from)?
            .i64()
            .map_err(FitsError::from)?
            .get(row_index)
            .ok_or_else(|| FitsError::validation("bitpix row missing or null"))?
            as i32;
        let bzero: i64 = df
            .column("bzero")
            .map_err(FitsError::from)?
            .i64()
            .map_err(FitsError::from)?
            .get(row_index)
            .ok_or_else(|| FitsError::validation("bzero row missing or null"))?;
        Ok(ImageInfo {
            path,
            data_offset,
            naxis1,
            naxis2,
            bitpix,
            bzero,
        })
    }
}

#[derive(Debug, Clone)]
pub struct ParsedFitsStem {
    pub file_stem: String,
    pub sample_name: String,
    pub tag: Option<String>,
    pub tags: Vec<String>,
    pub scan_number: i64,
    pub frame_number: i64,
}

pub fn is_polarization_tag(name: &str) -> bool {
    let n = name.trim().to_lowercase();
    n == "s" || n == "p"
}

pub fn parse_fits_stem(stem: &str) -> Option<ParsedFitsStem> {
    let stem = stem.trim();
    let re = Regex::new(r"^(.+?)[\s\-_]?(\d{5})-(\d{5})$").ok()?;
    let cap = re.captures(stem)?;
    let base = cap.get(1)?.as_str().trim();
    let exp_str = cap.get(2)?.as_str();
    let frame_str = cap.get(3)?.as_str();
    let scan_number: i64 = exp_str.parse().ok()?;
    let frame_number: i64 = frame_str.parse().ok()?;
    let mut tags: Vec<String> = Vec::new();
    let sample_name = if base.contains('_') {
        let parts: Vec<&str> = base.split('_').collect();
        if parts.len() >= 2 {
            let last = parts[parts.len() - 1].to_string();
            if !is_polarization_tag(&last) {
                tags.push(last);
            }
            parts[..parts.len() - 1].join("_")
        } else {
            base.to_string()
        }
    } else if base.contains('-') {
        let parts: Vec<&str> = base.split('-').filter(|s| !s.is_empty()).collect();
        if parts.len() >= 2 {
            let rest = &parts[1..];
            for p in rest {
                if !is_polarization_tag(p) {
                    tags.push((*p).to_string());
                }
            }
            parts[0].to_string()
        } else {
            base.to_string()
        }
    } else {
        base.to_string()
    };
    let tag = tags.first().cloned();
    Some(ParsedFitsStem {
        file_stem: stem.to_string(),
        sample_name,
        tag,
        tags,
        scan_number,
        frame_number,
    })
}

pub fn build_fits_stem(
    sample_name: &str,
    tag: Option<&str>,
    scan_number: i64,
    frame_number: i64,
) -> String {
    match tag {
        Some(t) => format!(
            "{}_{}_{:05}-{:05}",
            sample_name, t, scan_number, frame_number
        ),
        None => format!("{}_{:05}-{:05}", sample_name, scan_number, frame_number),
    }
}

pub fn q(lam: f64, theta: f64) -> f64 {
    match 4.0 * std::f64::consts::PI * theta.to_radians().sin() / lam {
        q if q < 0.0 => 0.0,
        q => q,
    }
}

pub fn add_calculated_domains(mut lzf: LazyFrame) -> DataFrame {
    let h = physical_constants::PLANCK_CONSTANT_IN_EV_PER_HZ;
    let c = physical_constants::SPEED_OF_LIGHT_IN_VACUUM * 1e10;

    let schema = lzf.collect_schema().unwrap_or_default();
    let has_column = |name: &str| schema.iter().any(|(col_name, _)| col_name == name);

    let mut lz = lzf;

    if has_column("DATE") {
        lz = lz.sort(["DATE"], Default::default());
    }

    if has_column("file_name") {
        lz = lz.sort(["file_name"], Default::default());
    }

    if has_column("EXPOSURE") {
        lz = lz.with_column(col("EXPOSURE").round(3).alias("EXPOSURE"));
    }

    if has_column("Higher Order Suppressor") {
        lz = lz.with_column(
            col("Higher Order Suppressor")
                .round(2)
                .alias("Higher Order Suppressor"),
        );
    }

    if has_column("Horizontal Exit Slit Size") {
        lz = lz.with_column(
            col("Horizontal Exit Slit Size")
                .round(1)
                .alias("Horizontal Exit Slit Size"),
        );
    }

    if has_column("Beamline Energy") {
        lz = lz.with_column(col("Beamline Energy").round(1).alias("Beamline Energy"));

        lz = lz.with_column(
            col("Beamline Energy")
                .pow(-1)
                .mul(lit(h * c))
                .alias("Lambda"),
        );
    }

    if has_column("Sample Theta") && has_column("Beamline Energy") {
        lz = lz.with_column(
            when(
                col("Sample Theta")
                    .is_not_null()
                    .and(col("Lambda").is_not_null()),
            )
            .then(as_struct(vec![col("Sample Theta"), col("Lambda")]).map(
                move |s| {
                    let struc = s.struct_()?;
                    let th_series = struc.field_by_name("Sample Theta")?;
                    let theta = th_series.f64()?;
                    let lam_series = struc.field_by_name("Lambda")?;
                    let lam = lam_series.f64()?;

                    let out: Float64Chunked = theta
                        .into_iter()
                        .zip(lam.iter())
                        .map(|(theta, lam)| match (theta, lam) {
                            (Some(theta), Some(lam)) => Some(q(lam, theta)),
                            _ => None,
                        })
                        .collect();

                    Ok(Some(out.into_column()))
                },
                GetOutput::from_type(DataType::Float64),
            ))
            .otherwise(lit(NULL))
            .alias("Q"),
        );
    }

    lz.collect().unwrap_or_else(|_| DataFrame::empty())
}

pub const TRIM_ROWS: usize = 5;
pub const TRIM_COLS: usize = 5;
pub const ROW_BG_STRIP_WIDTH: usize = 10;
pub const DARK_BAND_HEIGHT: usize = 10;

pub fn trim_image_interior(
    data: &Array2<i64>,
    trim_rows: usize,
    trim_cols: usize,
) -> Array2<i64> {
    let rows = data.nrows();
    let cols = data.ncols();
    if rows < 2 * trim_rows || cols < 2 * trim_cols {
        return data.clone();
    }
    data.slice(ndarray::s![
        trim_rows..(rows - trim_rows),
        trim_cols..(cols - trim_cols)
    ])
    .to_owned()
}

pub fn subtract_background_row_strips(data: &Array2<i64>) -> Array2<i64> {
    let rows = data.nrows();
    let cols = data.ncols();
    let strip = ROW_BG_STRIP_WIDTH.min(cols / 2);
    if strip == 0 {
        return data.clone();
    }
    let mut result = data.clone();
    for r in 0..rows {
        let left_slice = result.slice(ndarray::s![r, ..strip]);
        let right_slice = result.slice(ndarray::s![r, (cols - strip)..]);
        let left_sum: i64 = left_slice.iter().copied().sum();
        let right_sum: i64 = right_slice.iter().copied().sum();
        let left_mean = left_sum / strip as i64;
        let right_mean = right_sum / strip as i64;
        let bg = left_mean.min(right_mean);
        for c in 0..cols {
            result[[r, c]] -= bg;
        }
    }
    result
}

pub fn subtract_dark_cold_side(data: &Array2<i64>) -> Array2<i64> {
    let rows = data.nrows();
    let cols = data.ncols();
    let band = DARK_BAND_HEIGHT.min(rows / 2);
    if band == 0 {
        return data.clone();
    }
    let top_slice = data.slice(ndarray::s![..band, ..]);
    let bottom_slice = data.slice(ndarray::s![(rows - band).., ..]);
    let top_sum: i64 = top_slice.iter().copied().sum();
    let bottom_sum: i64 = bottom_slice.iter().copied().sum();
    let top_mean = top_sum / (band * cols) as i64;
    let bottom_mean = bottom_sum / (band * cols) as i64;
    let dark = top_mean.min(bottom_mean);
    let mut result = data.clone();
    for v in result.iter_mut() {
        *v -= dark;
    }
    result
}

pub fn subtract_background(
    data: &ArrayBase<OwnedRepr<i64>, Dim<IxDynImpl>>,
) -> ArrayBase<OwnedRepr<i64>, Dim<IxDynImpl>> {
    let rows = data.len_of(Axis(0));
    let cols = data.len_of(Axis(1));
    if rows < 11 || cols < 41 {
        return data.to_owned();
    }
    let view = data.slice(ndarray::s![5..(rows - 5), 5..(cols - 5)]);
    let view_rows = view.len_of(Axis(0));
    let view_cols = view.len_of(Axis(1));

    let left = view.slice(ndarray::s![.., ..20]);
    let right = view.slice(ndarray::s![.., (view_cols - 20)..]);

    let left_sum: i64 = left.iter().copied().sum();
    let right_sum: i64 = right.iter().copied().sum();

    let mut background = ndarray::Array1::zeros(view_rows);

    if left_sum < right_sum {
        for (i, row) in right.axis_iter(Axis(0)).enumerate() {
            background[i] = row.iter().copied().sum::<i64>() / row.len() as i64;
        }
    } else {
        for (i, row) in left.axis_iter(Axis(0)).enumerate() {
            background[i] = row.iter().copied().sum::<i64>() / row.len() as i64;
        }
    }

    let mut result = data.to_owned();
    let mut interior = result.slice_mut(ndarray::s![5..(rows - 5), 5..(cols - 5)]);
    for (i, mut row) in interior.axis_iter_mut(Axis(0)).enumerate() {
        let bg = background[i];
        for val in row.iter_mut() {
            *val -= bg;
        }
    }
    result
}

pub fn subtract_background_edges(
    data: &Array2<i64>,
    bg_rows: usize,
    bg_cols: usize,
) -> Array2<i64> {
    let rows = data.nrows();
    let cols = data.ncols();
    if rows < 2 * bg_rows || cols < 2 * bg_cols {
        return data.clone();
    }
    let mut result = data.clone();
    for r in 0..rows {
        let left_sum: i64 = result
            .slice(ndarray::s![r, ..bg_cols])
            .iter()
            .copied()
            .sum();
        let left_mean = left_sum / bg_cols as i64;
        let right_sum: i64 = result
            .slice(ndarray::s![r, (cols - bg_cols)..])
            .iter()
            .copied()
            .sum();
        let right_mean = right_sum / bg_cols as i64;
        let bg = left_mean.min(right_mean);
        for c in 0..cols {
            result[[r, c]] -= bg;
        }
    }
    for c in 0..cols {
        let top_sum: i64 = result
            .slice(ndarray::s![..bg_rows, c])
            .iter()
            .copied()
            .sum();
        let top_mean = top_sum / bg_rows as i64;
        let bottom_sum: i64 = result
            .slice(ndarray::s![(rows - bg_rows).., c])
            .iter()
            .copied()
            .sum();
        let bottom_mean = bottom_sum / bg_rows as i64;
        let bg = top_mean.min(bottom_mean);
        for r in 0..rows {
            result[[r, c]] -= bg;
        }
    }
    result
}

pub fn process_metadata(hdu: &PrimaryHdu, keys: &[String]) -> Result<Vec<Column>, FitsError> {
    if keys.is_empty() {
        return Ok(Vec::new());
    }
    let mut columns = Vec::new();

    for key in keys {
        if key == "Beamline Energy" {
            if let Some(card) = hdu.header.get_card(key) {
                if let Some(val) = card.value.as_float() {
                    columns.push(Column::new(key.into(), &[val]));
                    continue;
                }
            }

            if let Some(card) = hdu.header.get_card("Beamline Energy Goal") {
                if let Some(val) = card.value.as_float() {
                    columns.push(Column::new(key.into(), &[val]));
                    continue;
                }
            }

            columns.push(Column::new(key.into(), &[0.0]));
            continue;
        }

        if key == "DATE" || key == "Sample Name" {
            if let Some(card) = hdu.header.get_card(key) {
                let val = card.value.to_string();
                columns.push(Column::new(key.into(), &[val]));
                continue;
            }
            columns.push(Column::new(key.into(), &["".to_string()]));
            continue;
        }

        let val = match hdu.header.get_card(key) {
            Some(card) => card.value.as_float().unwrap_or(1.0),
            None => 0.0,
        };

        columns.push(Column::new(key.into(), &[val]));
    }

    Ok(columns)
}

pub fn build_headers_only_columns(
    primary: &PrimaryHdu,
    image_header: &ImageHduHeader,
    path: PathBuf,
    header_items: &[String],
) -> Result<Vec<Column>, FitsError> {
    let mut columns = process_metadata(primary, header_items)?;
    columns.extend(process_file_name(path.clone()));
    let path_str = path
        .to_str()
        .ok_or_else(|| FitsError::validation("Invalid UTF-8 in path"))?
        .to_string();
    columns.push(Column::new("file_path".into(), vec![path_str]));
    let bzero = image_header
        .header
        .get_card("BZERO")
        .and_then(|c| c.value.as_int())
        .unwrap_or(0);
    columns.push(Column::new(
        "data_offset".into(),
        vec![image_header.data_offset as i64],
    ));
    columns.push(Column::new(
        "naxis1".into(),
        vec![image_header.naxis1 as i64],
    ));
    columns.push(Column::new(
        "naxis2".into(),
        vec![image_header.naxis2 as i64],
    ));
    columns.push(Column::new(
        "bitpix".into(),
        vec![image_header.bitpix as i64],
    ));
    let data_size = fs::metadata(&path)
        .map(|metadata| metadata.len())
        .unwrap_or(0) as i64;
    columns.push(Column::new("data_size".into(), vec![data_size]));
    columns.push(Column::new("bzero".into(), vec![bzero]));
    Ok(columns)
}

#[derive(Debug, Clone)]
pub struct BtIngestRow {
    pub file_path: String,
    pub data_offset: i64,
    pub naxis1: i64,
    pub naxis2: i64,
    pub bitpix: i64,
    pub bzero: i64,
    pub file_name: String,
    pub sample_name: String,
    pub tag: Option<String>,
    pub scan_number: i64,
    pub frame_number: i64,
    pub beamline_energy: Option<f64>,
    pub sample_theta: Option<f64>,
    pub ccd_theta: Option<f64>,
    pub epu_polarization: Option<f64>,
    pub exposure: Option<f64>,
}

fn header_float(primary: &PrimaryHdu, key: &str) -> Option<f64> {
    if key == "Beamline Energy" {
        if let Some(card) = primary.header.get_card(key) {
            if let Some(v) = card.value.as_float() {
                return Some(v);
            }
        }
        if let Some(card) = primary.header.get_card("Beamline Energy Goal") {
            return card.value.as_float();
        }
        return Some(0.0);
    }
    if key == "DATE" || key == "Sample Name" {
        return None;
    }
    primary
        .header
        .get_card(key)
        .map(|c| c.value.as_float().unwrap_or(1.0))
}

pub fn build_bt_ingest_row(
    primary: &PrimaryHdu,
    image_header: &ImageHduHeader,
    path: PathBuf,
    header_items: &[String],
) -> Result<BtIngestRow, FitsError> {
    let path_str = path
        .to_str()
        .ok_or_else(|| FitsError::validation("Invalid UTF-8 in path"))?
        .to_string();
    let file_name = path.file_stem().and_then(|s| s.to_str()).unwrap_or("").to_string();
    let (sample_name, tag, scan_number, frame_number) = match parse_fits_stem(&file_name) {
        Some(p) => (p.sample_name, p.tag.clone(), p.scan_number, p.frame_number),
        None => (String::new(), None, 0i64, 0i64),
    };
    let bzero = image_header
        .header
        .get_card("BZERO")
        .and_then(|c| c.value.as_int())
        .unwrap_or(0);
    let mut row = BtIngestRow {
        file_path: path_str,
        data_offset: image_header.data_offset as i64,
        naxis1: image_header.naxis1 as i64,
        naxis2: image_header.naxis2 as i64,
        bitpix: image_header.bitpix as i64,
        bzero,
        file_name,
        sample_name,
        tag,
        scan_number,
        frame_number,
        beamline_energy: None,
        sample_theta: None,
        ccd_theta: None,
        epu_polarization: None,
        exposure: None,
    };
    for key in header_items {
        let v = header_float(primary, key);
        match key.as_str() {
            "Beamline Energy" => row.beamline_energy = v,
            "Sample Theta" => row.sample_theta = v,
            "CCD Theta" => row.ccd_theta = v,
            "EPU Polarization" => row.epu_polarization = v,
            "EXPOSURE" => row.exposure = v,
            _ => {}
        }
    }
    Ok(row)
}

pub fn process_file_name(path: std::path::PathBuf) -> Vec<Column> {
    let file_name = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
    let mut columns = vec![Column::new("file_name".into(), vec![file_name.to_string()])];
    match parse_fits_stem(file_name) {
        Some(p) => {
            columns.push(Column::new("sample_name".into(), vec![p.sample_name.clone()]));
            let tag_series = Series::from_iter(std::iter::once(p.tag.as_deref()))
                .with_name("tag".into())
                .into_column();
            columns.push(tag_series);
            columns.push(Column::new("scan_number".into(), vec![p.scan_number]));
            columns.push(Column::new("frame_number".into(), vec![p.frame_number]));
        }
        None => {
            columns.push(Column::new("sample_name".into(), vec!["".to_string()]));
            columns.push(
                Series::from_iter(std::iter::once(Option::<&str>::None))
                    .with_name("tag".into())
                    .into_column(),
            );
            columns.push(Column::new("scan_number".into(), vec![0i64]));
            columns.push(Column::new("frame_number".into(), vec![0i64]));
        }
    }
    columns
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_fits_stem_znpc_variants() {
        let cases = [
            ("ZnPc_rt81041-00001", "ZnPc", Some("rt"), 81041, 1),
            ("ZnPc_rt_81041-00001", "ZnPc", Some("rt"), 81041, 1),
            ("ZnPc_rt 81041-00001", "ZnPc", Some("rt"), 81041, 1),
            ("ZnPc_rt-81041-00001", "ZnPc", Some("rt"), 81041, 1),
        ];
        for (stem, sample, tag, exp_num, frame_num) in cases {
            let p = parse_fits_stem(stem).expect(stem);
            assert_eq!(p.sample_name, sample, "stem: {}", stem);
            assert_eq!(p.tag.as_deref(), tag, "stem: {}", stem);
            assert_eq!(p.scan_number, exp_num, "stem: {}", stem);
            assert_eq!(p.frame_number, frame_num, "stem: {}", stem);
        }
    }

    #[test]
    fn test_parse_fits_stem_ps_pmma() {
        let p = parse_fits_stem("ps_pmma_rt 81041-00001").expect("should parse");
        assert_eq!(p.sample_name, "ps_pmma");
        assert_eq!(p.tag.as_deref(), Some("rt"));
        assert_eq!(p.scan_number, 81041);
        assert_eq!(p.frame_number, 1);
    }

    #[test]
    fn test_parse_fits_stem_monlayerjune() {
        let p = parse_fits_stem("monlayerjune 81041-00007").expect("should parse");
        assert_eq!(p.sample_name, "monlayerjune");
        assert_eq!(p.tag, None);
        assert_eq!(p.scan_number, 81041);
        assert_eq!(p.frame_number, 7);
        let p2 = parse_fits_stem("monlayerjune 81041-00001").expect("should parse");
        assert_eq!(p2.sample_name, "monlayerjune");
        assert_eq!(p2.tag, None);
        assert_eq!(p2.frame_number, 1);
    }

    #[test]
    fn test_parse_fits_stem_invalid() {
        assert!(parse_fits_stem("notavalidstem").is_none());
        assert!(parse_fits_stem("short1-00001").is_none());
        assert!(parse_fits_stem("sample12345-678").is_none());
    }

    #[test]
    fn test_parse_fits_stem_polarization_not_tag() {
        let p = parse_fits_stem("znpc_s 81041-00001").expect("should parse");
        assert_eq!(p.sample_name, "znpc");
        assert_eq!(p.tag, None);
        let p2 = parse_fits_stem("znpc_P 81041-00002").expect("should parse");
        assert_eq!(p2.sample_name, "znpc");
        assert_eq!(p2.tag, None);
    }

    #[test]
    fn test_parse_fits_stem_hyphen_separated_tags() {
        let p = parse_fits_stem("a-b-c-12345-00001").expect("parse");
        assert_eq!(p.sample_name, "a");
        assert_eq!(p.tags, vec!["b".to_string(), "c".to_string()]);
        assert_eq!(p.tag.as_deref(), Some("b"));
        assert_eq!(p.scan_number, 12345);
        assert_eq!(p.frame_number, 1);
    }

    #[test]
    fn test_subtract_background_edges() {
        let data = Array2::from_shape_vec(
            (4, 6),
            vec![
                10, 10, 20, 20, 10, 10, 10, 10, 20, 20, 10, 10, 10, 10, 20, 20, 10, 10, 10, 10, 20,
                20, 10, 10,
            ],
        )
        .unwrap();
        let result = subtract_background_edges(&data, 1, 2);
        assert_eq!(result.shape(), data.shape());
        assert!(result[[0, 0]] <= 10);
        assert!(result[[1, 2]] <= 20);
    }

    #[test]
    fn test_subtract_background_edges_small_image() {
        let data = Array2::from_shape_vec(
            (4, 4),
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        )
        .unwrap();
        let result = subtract_background_edges(&data, 5, 5);
        assert_eq!(result.shape(), data.shape());
        for (a, b) in data.iter().zip(result.iter()) {
            assert_eq!(a, b);
        }
    }

    #[test]
    fn test_trim_image_interior_shape() {
        let data = Array2::from_shape_vec((20, 30), vec![0i64; 20 * 30]).unwrap();
        let trimmed = trim_image_interior(&data, TRIM_ROWS, TRIM_COLS);
        assert_eq!(trimmed.nrows(), 10);
        assert_eq!(trimmed.ncols(), 20);
    }

    #[test]
    fn test_trim_image_interior_too_small_returns_clone() {
        let data = Array2::from_shape_vec((8, 8), vec![1i64; 64]).unwrap();
        let trimmed = trim_image_interior(&data, TRIM_ROWS, TRIM_COLS);
        assert_eq!(trimmed.shape(), data.shape());
        assert_eq!(trimmed[[0, 0]], 1);
    }

    #[test]
    fn test_trim_image_interior_content_preserved() {
        let mut data = Array2::zeros((15, 25));
        data[[5, 5]] = 100;
        let trimmed = trim_image_interior(&data, TRIM_ROWS, TRIM_COLS);
        assert_eq!(trimmed.nrows(), 5);
        assert_eq!(trimmed.ncols(), 15);
        assert_eq!(trimmed[[0, 0]], 100);
    }

    #[test]
    fn test_subtract_background_row_strips_whole_row_corrected() {
        let data = Array2::from_shape_vec(
            (2, 30),
            (0..60).map(|i| if i < 10 { 5i64 } else if i >= 20 { 5i64 } else { 100 }).collect(),
        )
        .unwrap();
        let result = subtract_background_row_strips(&data);
        for c in 0..30 {
            assert!(result[[0, c]] <= 100 - 5, "row 0 col {} should be corrected", c);
            assert!(result[[1, c]] <= 100 - 5, "row 1 col {} should be corrected", c);
        }
    }

    #[test]
    fn test_subtract_background_row_strips_colder_side_per_row() {
        let mut data = Array2::zeros((2, 30));
        for c in 0..10 {
            data[[0, c]] = 3;
            data[[1, c]] = 20;
        }
        for c in 20..30 {
            data[[0, c]] = 20;
            data[[1, c]] = 3;
        }
        for c in 10..20 {
            data[[0, c]] = 50;
            data[[1, c]] = 50;
        }
        let result = subtract_background_row_strips(&data);
        assert_eq!(result[[0, 0]], 0);
        assert_eq!(result[[0, 15]], 50 - 3);
        assert_eq!(result[[1, 0]], 20 - 3);
        assert_eq!(result[[1, 25]], 0);
    }

    #[test]
    fn test_subtract_background_row_strips_narrow_image() {
        let data = Array2::from_shape_vec((4, 10), vec![10i64; 40]).unwrap();
        let result = subtract_background_row_strips(&data);
        assert_eq!(result.shape(), data.shape());
        for v in result.iter() {
            assert!(*v <= 10);
        }
    }

    #[test]
    fn test_subtract_background_row_strips_exactly_20_cols() {
        let data = Array2::from_shape_vec((2, 20), vec![7i64; 40]).unwrap();
        let result = subtract_background_row_strips(&data);
        assert_eq!(result.shape(), data.shape());
        for v in result.iter() {
            assert_eq!(*v, 0);
        }
    }

    #[test]
    fn test_subtract_dark_cold_side_one_scalar() {
        let data = Array2::from_shape_vec((25, 20), vec![100i64; 500]).unwrap();
        let result = subtract_dark_cold_side(&data);
        assert_eq!(result.shape(), data.shape());
        let expected = 100 - 100;
        for v in result.iter() {
            assert_eq!(*v, expected);
        }
    }

    #[test]
    fn test_subtract_dark_cold_side_colder_chosen() {
        let mut data = Array2::from_shape_vec((30, 15), vec![50i64; 450]).unwrap();
        for r in 0..10 {
            for c in 0..15 {
                data[[r, c]] = 10;
            }
        }
        for r in 20..30 {
            for c in 0..15 {
                data[[r, c]] = 80;
            }
        }
        let result = subtract_dark_cold_side(&data);
        assert_eq!(result[[0, 0]], 0);
        assert!(result[[15, 0]] < 50);
        assert_eq!(result[[25, 0]], 80 - 10);
    }

    #[test]
    fn test_subtract_dark_cold_side_small_height() {
        let data = Array2::from_shape_vec((5, 20), vec![1i64; 100]).unwrap();
        let result = subtract_dark_cold_side(&data);
        assert_eq!(result.shape(), data.shape());
    }

    #[test]
    fn test_raw_pipeline_full_trim_row_dark() {
        let rows = 30usize;
        let cols = 40usize;
        let data = Array2::from_shape_vec((rows, cols), vec![100i64; rows * cols]).unwrap();
        let trimmed = trim_image_interior(&data, TRIM_ROWS, TRIM_COLS);
        assert_eq!(trimmed.nrows(), rows - 2 * TRIM_ROWS);
        assert_eq!(trimmed.ncols(), cols - 2 * TRIM_COLS);
        let row_corrected = subtract_background_row_strips(&trimmed);
        let processed = subtract_dark_cold_side(&row_corrected);
        assert_eq!(processed.shape(), trimmed.shape());
        for v in processed.iter() {
            assert!(*v <= 100);
        }
    }
}
