use ndarray::{s, Array2};
use ndarray_ndimage::*;
use std::collections::HashMap;

// ----------------Initial loading and image Processing----------------
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

// Wrapper function for collecting the beam spot.
pub fn detect_beam(img: &Array2<u32>, box_size: usize) -> Result<(usize, usize), String> {
    let imf = img.mapv(|x| x as f64);
    let imf_gf = gaussian_filter(&imf, box_size as f64, 0, BorderMode::Nearest, 5);
    let (x, y) = match max_idx(&imf_gf) {
        Ok(idx) => idx,
        Err(_) => sobel_detect(&imf).unwrap(),
    };
    Ok((x, y))
}

//-----------------Advanced Image Processing-----------------
// detect the beam spot using sobel filter
pub fn sobel_detect(imf: &Array2<f64>) -> Result<(usize, usize), String> {
    let imf_sobel = sobel(&imf, ndarray::Axis(0), BorderMode::Nearest);
    // draw contours arround the beam spot
    let (x, y) = felzenszwalb(imf_sobel);
    Ok((x as usize, y as usize))
}

pub struct UnionFind {
    parent: Vec<usize>,
    size: Vec<usize>,
    threshold: Vec<f64>,
}

impl UnionFind {
    pub fn new(size: usize) -> UnionFind {
        UnionFind {
            parent: (0..size).collect(),
            size: vec![1; size],
            threshold: vec![1.0; size],
        }
    }

    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }

    fn union(&mut self, x: usize, y: usize, cost: f64) {
        let x_root = self.find(x);
        let y_root = self.find(y);
        if x_root == y_root {
            return;
        }
        if self.size[x_root] < self.size[y_root] {
            self.parent[x_root] = y_root;
            self.size[y_root] += self.size[x_root];
            self.threshold[y_root] = cost;
        } else {
            self.parent[y_root] = x_root;
            self.size[x_root] += self.size[y_root];
            self.threshold[x_root] = cost;
        }
    }

    fn size(&mut self, x: usize) -> usize {
        let x_root = self.find(x);
        self.size[x_root]
    }

    fn get_threshold(&mut self, x: usize) -> f64 {
        let x_root = self.find(x);
        self.threshold[x_root]
    }
}

pub fn felzenszwalb(image: Array2<f64>) -> (usize, usize) {
    let (height, width) = image.dim();
    let num_pixels = height * width;

    let mut edges = Vec::new();
    let mut costs = Vec::new();

    // directions for 8 connectivity
    let direction = vec![
        (0, 1),  // Right
        (1, 0),  // Down
        (1, 1),  // Down Right
        (1, -1), // Down Left
    ];

    for &(dy, dx) in &direction {
        for y in 0..height {
            for x in 0..width {
                let nx = x as isize + dx;
                let ny = y as isize + dy;
                if nx >= 0 && nx < width as isize && ny >= 0 && ny < height as isize {
                    let idx1 = y * width + x;
                    let idx2 = (ny as usize) * width + (nx as usize);
                    // calculate the cost
                    let cost = (image[[y, x]] - image[[ny as usize, nx as usize]]).abs();
                    edges.push((idx1, idx2));
                    costs.push(cost);
                }
            }
        }
    }

    let mut edge_cost_pairs: Vec<(f64, (usize, usize))> =
        costs.into_iter().zip(edges.into_iter()).collect();
    edge_cost_pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let mut uf = UnionFind::new(num_pixels);
    // segmentation
    for (cost, (a, b)) in &edge_cost_pairs {
        let a_root = uf.find(*a);
        let b_root = uf.find(*b);
        if a_root == b_root {
            continue;
        }

        let threshold_a = uf.get_threshold(a_root) + 1 as f64 / uf.size(a_root) as f64;
        let threshold_b = uf.get_threshold(b_root) + 1 as f64 / uf.size(b_root) as f64;

        if *cost <= threshold_a.min(threshold_b) {
            uf.union(a_root, b_root, *cost);
        }
    }
    // remove all small component labels
    // Post-processing to remove small components
    for (cost, (a, b)) in &edge_cost_pairs {
        let a_root = uf.find(*a);
        let b_root = uf.find(*b);
        if a_root == b_root {
            continue;
        }
        if uf.size(a_root) < 20 || uf.size(b_root) < 20 {
            uf.union(a_root, b_root, *cost);
        }
    }

    // Assign component labels
    let mut labels = vec![0; num_pixels];
    for i in 0..num_pixels {
        labels[i] = uf.find(i);
    }

    // Compute sizes of each component
    let mut component_sizes = HashMap::new();
    for &label in &labels {
        *component_sizes.entry(label).or_insert(0) += 1;
    }

    // Find the smallest component (label)
    let mut smallest_label = None;
    let mut smallest_size = usize::MAX;
    for (&label, &size) in &component_sizes {
        if size < smallest_size {
            smallest_size = size;
            smallest_label = Some(label);
        }
    }

    let smallest_label = smallest_label.unwrap();

    // Collect coordinates of pixels belonging to the smallest component
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut count = 0.0;

    for i in 0..num_pixels {
        if labels[i] == smallest_label {
            let y = (i / width) as f64;
            let x = (i % width) as f64;
            sum_x += x;
            sum_y += y;
            count += 1.0;
        }
    }

    // Compute centroid
    let centroid_x = sum_x / count;
    let centroid_y = sum_y / count;

    (centroid_x as usize, centroid_y as usize)
}

// -----------------Data Reduction-----------------
/// Calculate the specular reflectance of the beam spot
/// +-------------------------------------------------+
/// |                                                 |
/// |                                                 |
/// |-------------------------------------------------|
/// |   --Bg--      | Beam |     --Bg--               |
/// |-------------------------------------------------|
/// |                                                 |
/// +-------------------------------------------------+
///
/// The specular reflectance is calculated as:
///  R = (I_beam - I_bg)
///
pub fn spec_refl(
    img: &Array2<u32>,
    x: usize,
    y: usize,
    box_size: usize,
) -> (f64, f64, Array2<u32>) {
    // find the beam spot
    let beam_spot = img.slice(s![
        x.saturating_sub(box_size / 2)..(x + box_size / 2).min(img.shape()[0]),
        y.saturating_sub(box_size / 2)..(y + box_size / 2).min(img.shape()[1])
    ]);
    let beam_intensity = beam_spot.iter().sum::<u32>() as f64;
    // Create a mask for the background region
    let mut mask = Array2::<bool>::from_elem(img.dim(), true);
    for i in x.saturating_sub(box_size / 2)..(x + box_size / 2).min(img.shape()[0]) {
        for j in y.saturating_sub(box_size / 2)..(y + box_size / 2).min(img.shape()[1]) {
            mask[[i, j]] = false;
        }
    }

    // Extract the background region using the mask
    let bg_region: Vec<u32> = img
        .iter()
        .zip(mask.iter())
        .filter_map(|(&pixel, &is_bg)| if is_bg { Some(pixel) } else { None })
        .collect();

    // Calculate the background intensity
    let bg_intensity: f64 = bg_region.iter().sum::<u32>() as f64 / bg_region.len() as f64;

    // Calculate the specular reflectance
    let spec_refl = beam_intensity - bg_intensity;

    (spec_refl, bg_intensity, beam_spot.to_owned())
}

// -----------------Stitching-----------------
