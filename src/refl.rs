use ndarray::{s, Array2};
use ndarray_ndimage::*;

fn felzenszwalb_segmentation(
    image: &Array2<f32>,
    scale: f32,
    sigma: f32,
    min_size: usize,
) -> (Array2<usize>, usize) {
    let (height, width) = image.dim();
    let num_pixels = width * height;

    // Step 1: Gaussian smoothing
    let smoothed = gaussian_blur(image, sigma);

    // Step 2: Build graph and compute edge weights
    let mut edges = build_graph(&smoothed);

    // Step 3: Sort edges by weight
    edges.sort_by(|a, b| a.weight.partial_cmp(&b.weight).unwrap());

    // Step 4: Initialize disjoint sets
    let mut disjoint_set = DisjointSet::new(num_pixels, scale);

    // Initialize variables to track the smallest component
    let mut smallest_component_root = 0;
    let mut smallest_component_size = usize::MAX;

    // Step 5: Iterate over edges and merge regions
    for edge in &edges {
        let a = disjoint_set.find(edge.a);
        let b = disjoint_set.find(edge.b);

        if a != b {
            let weight = edge.weight;
            let threshold_a = disjoint_set.threshold[a];
            let threshold_b = disjoint_set.threshold[b];

            if weight <= threshold_a && weight <= threshold_b {
                disjoint_set.union(a, b, weight);

                // Update smallest component tracking
                let root = disjoint_set.find(a);
                let size = disjoint_set.size[root];
                if size < smallest_component_size {
                    smallest_component_size = size;
                    smallest_component_root = root;
                }
            }
        }
    }

    // Step 6: Enforce minimum component size
    enforce_min_size(&mut disjoint_set, &edges, min_size);

    // Step 7: Generate labels
    let labels_array = generate_labels(&disjoint_set, height, width);

    // Return labels array and the root of the smallest component
    (labels_array, smallest_component_root)
}

// get the max intensity from the image
pub fn max_idx(array: &Array2<f64>) -> Result<(usize, usize), &str> {
    let mut max = 0.0;
    let mut idx = (0, 0);
    for (i, x) in array.iter().enumerate() {
        if x > &max {
            max = *x;
            idx = (i / array.shape()[1], i % array.shape()[1]);
        }
    }
    // if the index is at the border, panic
    if idx.0 == 0 || idx.0 == array.shape()[0] - 1 || idx.1 == 0 || idx.1 == array.shape()[1] - 1 {
        return Err("Max value is at the border, beam intensity less than max value");
    }
    Ok(idx)
}

// detect the beam spot using sobel filter
pub fn sobel_detect(imf: &Array2<f64>) -> Result<(usize, usize), String> {
    let imf_sobel = sobel(&imf, ndarray::Axis(0), BorderMode::Nearest);
    // draw contours arround the beam spot
}

pub fn detect_beam(img: Array2<u32>, box_size: usize) -> Result<(usize, usize), String> {
    let imf = img.mapv(|x| x as f64);
    let imf_gf = gaussian_filter(&imf, 5.0, 0, BorderMode::Nearest, 5);
    let (x, y) = match max_idx(&imf_gf) {
        Ok(idx) => idx,
        Err(_) => sobel_detect(&imf).unwrap(),
    };
    Ok((x, y))
}

// Integrate the beam intensity, and the
pub fn spec_reflectance(beam_spot: Array2<u32>, background: Array2<u32>) -> f64 {}
