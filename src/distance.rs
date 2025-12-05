//! Distance and comparison functions for motion embeddings.
//!
//! This module provides functions to compute distances between embeddings
//! and compare trajectory characteristics.

use crate::embedding::MotionEmbedding;
use crate::math::linalg::dot3;

/// Weights for computing embedding distance.
#[derive(Debug, Clone)]
pub struct DistanceWeights {
    /// Weight for scale component.
    pub scale: f64,
    /// Weight for shape components.
    pub shape: f64,
    /// Weight for momentum components.
    pub momentum: f64,
    /// Weight for spectral components.
    pub spectrum: f64,
    /// Weight for heading component.
    pub heading: f64,
    /// Weight for smoothness component.
    pub smoothness: f64,
    /// Weight for velocity components.
    pub velocity: f64,
    /// Weight for safety (g-force) components.
    pub safety: f64,
    /// Weight for chirality component.
    pub chirality: f64,
}

impl Default for DistanceWeights {
    fn default() -> Self {
        Self {
            scale: 1.0,
            shape: 1.0,
            momentum: 1.0,
            spectrum: 0.5,
            heading: 1.0,
            smoothness: 1.0,
            velocity: 1.0,
            safety: 1.0,
            chirality: 1.0,
        }
    }
}

/// Detailed distance breakdown between embeddings.
#[derive(Debug, Clone)]
pub struct DistanceBreakdown {
    /// Total weighted distance.
    pub total: f64,
    /// Scale distance.
    pub scale: f64,
    /// Shape distance.
    pub shape: f64,
    /// Momentum distance.
    pub momentum: f64,
    /// Spectrum distance.
    pub spectrum: f64,
    /// Heading distance (1 - cosine similarity).
    pub heading: f64,
    /// Smoothness distance.
    pub smoothness: f64,
    /// Velocity distance.
    pub velocity: f64,
    /// Safety (g-force) distance.
    pub safety: f64,
    /// Chirality distance.
    pub chirality: f64,
    /// Shape similarity (1 - shape_distance, clamped).
    pub shape_similarity: f64,
    /// Heading similarity (cosine similarity).
    pub heading_similarity: f64,
}

/// Compute distance between two motion embeddings.
///
/// # Arguments
///
/// * `emb1` - First embedding
/// * `emb2` - Second embedding
/// * `weights` - Optional custom weights (None = default weights)
///
/// # Returns
///
/// Weighted sum of component distances.
pub fn compute_embedding_distance(
    emb1: &MotionEmbedding,
    emb2: &MotionEmbedding,
    weights: Option<&DistanceWeights>,
) -> f64 {
    compute_embedding_distance_detailed(emb1, emb2, weights).total
}

/// Compute detailed distance breakdown between embeddings.
///
/// Returns a `DistanceBreakdown` with individual component distances
/// and similarity metrics.
pub fn compute_embedding_distance_detailed(
    emb1: &MotionEmbedding,
    emb2: &MotionEmbedding,
    weights: Option<&DistanceWeights>,
) -> DistanceBreakdown {
    let w = weights.cloned().unwrap_or_default();

    // Scale distance (now compressed to [-1, 1])
    let scale_dist = (emb1.scale_magnitude - emb2.scale_magnitude).abs();

    // Shape distance (L2 norm)
    let shape_dist = ((emb1.shape_entropy[0] - emb2.shape_entropy[0]).powi(2)
        + (emb1.shape_entropy[1] - emb2.shape_entropy[1]).powi(2)
        + (emb1.shape_entropy[2] - emb2.shape_entropy[2]).powi(2))
    .sqrt();

    // Momentum distance (L2 norm)
    let momentum_dist = emb1
        .normalized_momentum
        .iter()
        .zip(emb2.normalized_momentum.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt();

    // Spectral distance (normalized L2)
    let spectrum_dist = compute_spectrum_distance(emb1, emb2);

    // Heading distance (1 - cosine similarity)
    let heading_cos = dot3(&emb1.current_heading, &emb2.current_heading).clamp(-1.0, 1.0);
    let heading_dist = 1.0 - heading_cos;

    // Smoothness distance
    let smoothness_dist = (emb1.smoothness_index - emb2.smoothness_index).abs();

    // Velocity distance (L2 norm)
    let velocity_dist = ((emb1.velocity_normalized[0] - emb2.velocity_normalized[0]).powi(2)
        + (emb1.velocity_normalized[1] - emb2.velocity_normalized[1]).powi(2)
        + (emb1.velocity_normalized[2] - emb2.velocity_normalized[2]).powi(2))
    .sqrt();

    // Safety (g-force) distance (L2 norm)
    let safety_dist = ((emb1.g_force_stats[0] - emb2.g_force_stats[0]).powi(2)
        + (emb1.g_force_stats[1] - emb2.g_force_stats[1]).powi(2)
        + (emb1.g_force_stats[2] - emb2.g_force_stats[2]).powi(2))
    .sqrt();

    // Chirality distance
    let chirality_dist = (emb1.chirality - emb2.chirality).abs();

    // Total weighted distance
    let total = w.scale * scale_dist
        + w.shape * shape_dist
        + w.momentum * momentum_dist
        + w.spectrum * spectrum_dist
        + w.heading * heading_dist
        + w.smoothness * smoothness_dist
        + w.velocity * velocity_dist
        + w.safety * safety_dist
        + w.chirality * chirality_dist;

    DistanceBreakdown {
        total,
        scale: scale_dist,
        shape: shape_dist,
        momentum: momentum_dist,
        spectrum: spectrum_dist,
        heading: heading_dist,
        smoothness: smoothness_dist,
        velocity: velocity_dist,
        safety: safety_dist,
        chirality: chirality_dist,
        shape_similarity: 1.0 - shape_dist.min(1.0),
        heading_similarity: heading_cos,
    }
}

/// Compute normalized spectral distance.
fn compute_spectrum_distance(emb1: &MotionEmbedding, emb2: &MotionEmbedding) -> f64 {
    let spec1 = &emb1.spectral_features;
    let spec2 = &emb2.spectral_features;

    let min_len = spec1.len().min(spec2.len());
    if min_len == 0 {
        return 0.0;
    }

    // Compute magnitudes
    let mag1: Vec<f64> = spec1.iter().take(min_len).map(|c| c.norm()).collect();
    let mag2: Vec<f64> = spec2.iter().take(min_len).map(|c| c.norm()).collect();

    // Normalize
    let norm1: f64 = mag1.iter().map(|x| x * x).sum::<f64>().sqrt() + 1e-7;
    let norm2: f64 = mag2.iter().map(|x| x * x).sum::<f64>().sqrt() + 1e-7;

    // Normalized L2 distance
    mag1.iter()
        .zip(mag2.iter())
        .map(|(&a, &b)| (a / norm1 - b / norm2).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Compute the angle (in degrees) between two trajectory headings.
pub fn heading_angle_between(emb1: &MotionEmbedding, emb2: &MotionEmbedding) -> f64 {
    let cos_angle = dot3(&emb1.current_heading, &emb2.current_heading).clamp(-1.0, 1.0);
    cos_angle.acos().to_degrees()
}

/// Batch compute pairwise distances for a set of embeddings.
///
/// Returns a flattened upper triangular distance matrix.
pub fn compute_pairwise_distances(
    embeddings: &[MotionEmbedding],
    weights: Option<&DistanceWeights>,
) -> Vec<f64> {
    let n = embeddings.len();
    let mut distances = Vec::with_capacity(n * (n - 1) / 2);

    for i in 0..n {
        for j in (i + 1)..n {
            let d = compute_embedding_distance(&embeddings[i], &embeddings[j], weights);
            distances.push(d);
        }
    }

    distances
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedding::{EncodingMetadata, MotionEmbedding};
    use approx::assert_relative_eq;
    use num_complex::Complex64;

    fn default_embedding() -> MotionEmbedding {
        MotionEmbedding {
            scale_magnitude: 0.5,
            shape_entropy: [0.1, 0.2, 0.3],
            normalized_momentum: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            current_heading: [1.0, 0.0, 0.0],
            maneuver_plane: [0.0, 1.0, 0.0],
            velocity_normalized: [0.5, 0.3, 0.1],
            chirality: 0.2,
            g_force_stats: [0.3, 0.5, 0.1],
            smoothness_index: 0.8,
            spectral_features: vec![Complex64::new(1.0, 0.0); 16],
            metadata: EncodingMetadata::default(),
            chirality_threshold: 0.01,
        }
    }

    #[test]
    fn test_self_distance_zero() {
        let emb = default_embedding();
        let dist = compute_embedding_distance(&emb, &emb, None);
        assert_relative_eq!(dist, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_distance_symmetry() {
        let mut emb1 = default_embedding();
        let mut emb2 = default_embedding();

        emb1.scale_magnitude = 0.3;
        emb2.scale_magnitude = 0.7;

        let d12 = compute_embedding_distance(&emb1, &emb2, None);
        let d21 = compute_embedding_distance(&emb2, &emb1, None);

        assert_relative_eq!(d12, d21, epsilon = 1e-10);
    }

    #[test]
    fn test_heading_angle() {
        let mut emb1 = default_embedding();
        let mut emb2 = default_embedding();

        emb1.current_heading = [1.0, 0.0, 0.0];
        emb2.current_heading = [0.0, 1.0, 0.0];

        let angle = heading_angle_between(&emb1, &emb2);
        assert_relative_eq!(angle, 90.0, epsilon = 1e-10);
    }

    #[test]
    fn test_pairwise_distances() {
        let emb = default_embedding();
        let embeddings = vec![emb.clone(), emb.clone(), emb.clone()];

        let distances = compute_pairwise_distances(&embeddings, None);

        // 3 embeddings -> 3 pairs
        assert_eq!(distances.len(), 3);

        // All same embedding -> all distances zero
        for d in distances {
            assert_relative_eq!(d, 0.0, epsilon = 1e-10);
        }
    }
}
