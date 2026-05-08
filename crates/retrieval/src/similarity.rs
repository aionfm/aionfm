use serde::{Deserialize, Serialize};

/// Similarity metric used to rank historical analog windows.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SimilarityMetric {
    CenteredCosine,
    Euclidean,
}

impl Default for SimilarityMetric {
    fn default() -> Self {
        Self::CenteredCosine
    }
}

pub fn similarity(metric: SimilarityMetric, left: &[f32], right: &[f32]) -> f32 {
    match metric {
        SimilarityMetric::CenteredCosine => centered_cosine(left, right),
        SimilarityMetric::Euclidean => negative_euclidean(left, right),
    }
}

pub fn centered_cosine(left: &[f32], right: &[f32]) -> f32 {
    if left.is_empty() || right.is_empty() {
        return 0.0;
    }
    let len = left.len().min(right.len());
    let left = &left[..len];
    let right = &right[..len];
    let mean_left = left.iter().sum::<f32>() / len as f32;
    let mean_right = right.iter().sum::<f32>() / len as f32;
    let mut dot = 0.0;
    let mut left_norm = 0.0;
    let mut right_norm = 0.0;
    for (left, right) in left.iter().zip(right) {
        let left = left - mean_left;
        let right = right - mean_right;
        dot += left * right;
        left_norm += left * left;
        right_norm += right * right;
    }
    if left_norm == 0.0 || right_norm == 0.0 {
        0.0
    } else {
        (dot / (left_norm.sqrt() * right_norm.sqrt())).clamp(-1.0, 1.0)
    }
}

pub fn negative_euclidean(left: &[f32], right: &[f32]) -> f32 {
    if left.is_empty() || right.is_empty() {
        return 0.0;
    }
    let distance = left
        .iter()
        .zip(right)
        .map(|(left, right)| {
            let delta = left - right;
            delta * delta
        })
        .sum::<f32>()
        .sqrt();
    -distance
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn centered_cosine_prefers_same_shape() {
        let same = centered_cosine(&[1.0, 2.0, 3.0], &[3.0, 4.0, 5.0]);
        let opposite = centered_cosine(&[1.0, 2.0, 3.0], &[5.0, 4.0, 3.0]);
        assert!(same > opposite);
    }
}
