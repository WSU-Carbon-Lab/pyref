pub mod blur;
pub mod image_mmap;

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
            .ok_or_else(|| FitsError::validation("data_offset row missing or null"))? as u64;
        let naxis1: usize = df
            .column("naxis1")
            .map_err(FitsError::from)?
            .i64()
            .map_err(FitsError::from)?
            .get(row_index)
            .ok_or_else(|| FitsError::validation("naxis1 row missing or null"))? as usize;
        let naxis2: usize = df
            .column("naxis2")
            .map_err(FitsError::from)?
            .i64()
            .map_err(FitsError::from)?
            .get(row_index)
            .ok_or_else(|| FitsError::validation("naxis2 row missing or null"))? as usize;
        let bitpix: i32 = df
            .column("bitpix")
            .map_err(FitsError::from)?
            .i64()
            .map_err(FitsError::from)?
            .get(row_index)
            .ok_or_else(|| FitsError::validation("bitpix row missing or null"))? as i32;
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
    pub experiment_number: i64,
    pub frame_number: i64,
}

pub fn parse_fits_stem(stem: &str) -> Option<ParsedFitsStem> {
    let stem = stem.trim();
    let re = Regex::new(r"^(.+?)[\s\-_]?(\d{5})-(\d{5})$").ok()?;
    let cap = re.captures(stem)?;
    let base = cap.get(1)?.as_str().trim();
    let exp_str = cap.get(2)?.as_str();
    let frame_str = cap.get(3)?.as_str();
    let experiment_number: i64 = exp_str.parse().ok()?;
    let frame_number: i64 = frame_str.parse().ok()?;
    let (sample_name, tag) = if base.contains('_') {
        let parts: Vec<&str> = base.split('_').collect();
        let (last, rest) = parts.split_last()?;
        (rest.join("_"), Some((*last).to_string()))
    } else {
        (base.to_string(), None)
    };
    Some(ParsedFitsStem {
        file_stem: stem.to_string(),
        sample_name,
        tag,
        experiment_number,
        frame_number,
    })
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

    lz.collect().unwrap_or_else(|_| DataFrame::empty())
}

pub fn subtract_background(
    data: &ArrayBase<OwnedRepr<i64>, Dim<IxDynImpl>>,
) -> ArrayBase<OwnedRepr<i64>, Dim<IxDynImpl>> {
    let rows = data.len_of(Axis(0));
    let cols = data.len_of(Axis(1));
    if rows < 11 || cols < 41 {
        return data.to_owned();
    }
    let view = data.slice(ndarray::s![5..-5, 5..-5]);
    let rows = view.len_of(Axis(0));
    let cols = view.len_of(Axis(1));

    let left = view.slice(ndarray::s![.., ..20]);
    let right = view.slice(ndarray::s![.., (cols - 20)..]);

    let left_sum: i64 = left.iter().copied().sum();
    let right_sum: i64 = right.iter().copied().sum();

    let mut background = ndarray::Array1::zeros(rows);

    if left_sum < right_sum {
        for (i, row) in right.axis_iter(Axis(0)).enumerate() {
            background[i] = row.iter().copied().sum::<i64>() / row.len() as i64;
        }
    } else {
        for (i, row) in left.axis_iter(Axis(0)).enumerate() {
            background[i] = row.iter().copied().sum::<i64>() / row.len() as i64;
        }
    }

    let mut result = view.to_owned();

    for (i, mut row) in result.axis_iter_mut(Axis(0)).enumerate() {
        let bg = background[i];
        for val in row.iter_mut() {
            *val -= bg;
        }
    }
    result.into_dyn()
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
        let left_sum: i64 = result.slice(ndarray::s![r, ..bg_cols]).iter().copied().sum();
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

pub fn process_metadata(
    hdu: &PrimaryHdu,
    keys: &[String],
) -> Result<Vec<Column>, FitsError> {
    if keys.is_empty() {
        Ok(hdu
            .header
            .iter()
            .filter(|card| !card.keyword.as_str().to_lowercase().contains("comment"))
            .map(|card| {
                let name = card.keyword.as_str();
                let value = card.value.as_float().unwrap_or(0.0);
                Column::new(name.into(), &[value])
            })
            .collect())
    } else {
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

            if key == "DATE" {
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
    columns.push(Column::new(
        "data_size".into(),
        vec![data_size],
    ));
    columns.push(Column::new("bzero".into(), vec![bzero]));
    Ok(columns)
}

pub fn process_file_name(path: std::path::PathBuf) -> Vec<Column> {
    let file_name = path.file_stem().unwrap().to_str().unwrap_or("");
    let mut columns = vec![Column::new("file_name".into(), vec![file_name.to_string()])];
    match parse_fits_stem(file_name) {
        Some(p) => {
            columns.push(Column::new("sample_name".into(), vec![p.sample_name]));
            let tag_series = Series::from_iter(std::iter::once(p.tag.as_deref()))
                .with_name("tag".into())
                .into_column();
            columns.push(tag_series);
            columns.push(Column::new("experiment_number".into(), vec![p.experiment_number]));
            columns.push(Column::new("frame_number".into(), vec![p.frame_number]));
        }
        None => {
            columns.push(Column::new("sample_name".into(), vec!["".to_string()]));
            columns.push(
                Series::from_iter(std::iter::once(Option::<&str>::None))
                    .with_name("tag".into())
                    .into_column(),
            );
            columns.push(Column::new("experiment_number".into(), vec![0i64]));
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
            assert_eq!(p.experiment_number, exp_num, "stem: {}", stem);
            assert_eq!(p.frame_number, frame_num, "stem: {}", stem);
        }
    }

    #[test]
    fn test_parse_fits_stem_ps_pmma() {
        let p = parse_fits_stem("ps_pmma_rt 81041-00001").expect("should parse");
        assert_eq!(p.sample_name, "ps_pmma");
        assert_eq!(p.tag.as_deref(), Some("rt"));
        assert_eq!(p.experiment_number, 81041);
        assert_eq!(p.frame_number, 1);
    }

    #[test]
    fn test_parse_fits_stem_monlayerjune() {
        let p = parse_fits_stem("monlayerjune 81041-00007").expect("should parse");
        assert_eq!(p.sample_name, "monlayerjune");
        assert_eq!(p.tag, None);
        assert_eq!(p.experiment_number, 81041);
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
    fn test_subtract_background_edges() {
        let data = Array2::from_shape_vec((4, 6), vec![
            10, 10, 20, 20, 10, 10,
            10, 10, 20, 20, 10, 10,
            10, 10, 20, 20, 10, 10,
            10, 10, 20, 20, 10, 10,
        ]).unwrap();
        let result = subtract_background_edges(&data, 1, 2);
        assert_eq!(result.shape(), data.shape());
        assert!(result[[0, 0]] <= 10);
        assert!(result[[1, 2]] <= 20);
    }

    #[test]
    fn test_subtract_background_edges_small_image() {
        let data = Array2::from_shape_vec((4, 4), vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]).unwrap();
        let result = subtract_background_edges(&data, 5, 5);
        assert_eq!(result.shape(), data.shape());
        for (a, b) in data.iter().zip(result.iter()) {
            assert_eq!(a, b);
        }
    }
}
