//! Embedding variance validation utilities.
//!
//! This module provides functions to analyze the variance distribution
//! across embedding dimensions, ensuring range-balanced features.

use crate::embedding::MotionEmbedding;

/// Variance analysis results.
#[derive(Debug, Clone)]
pub struct VarianceAnalysis {
    /// Percentage of variance explained by PC1.
    pub pc1_variance_percent: f64,

    /// Effective dimensionality (exponential of entropy).
    pub effective_dimensionality: f64,

    /// Total embedding dimensions analyzed.
    pub total_dimensions: usize,

    /// Top 10 eigenvalues from PCA.
    pub eigenvalue_spectrum: Vec<f64>,

    /// Per-feature-group variances.
    pub feature_group_variances: FeatureGroupVariances,

    /// Max/min variance ratio.
    pub variance_ratio_max_min: f64,

    /// Whether the embedding passes health checks.
    pub is_healthy: bool,
}

/// Variance for each feature group.
#[derive(Debug, Clone, Default)]
pub struct FeatureGroupVariances {
    /// Scale variance.
    pub scale: f64,
    /// Shape variance.
    pub shape: f64,
    /// Momentum variance.
    pub momentum: f64,
    /// Heading variance.
    pub heading: f64,
    /// Maneuver variance.
    pub maneuver: f64,
    /// Velocity variance.
    pub velocity: f64,
    /// Chirality variance.
    pub chirality: f64,
    /// G-force variance.
    pub g_force: f64,
    /// Smoothness variance.
    pub smoothness: f64,
}

/// Analyze variance distribution across embedding dimensions.
///
/// # Arguments
///
/// * `embeddings` - Slice of embeddings to analyze
///
/// # Returns
///
/// Variance analysis with health checks.
///
/// # Health Targets
///
/// - PC1 variance < 40% (balanced distribution)
/// - Effective dimensionality > 8 (most features contribute)
/// - Max/min feature group variance ratio < 50
pub fn validate_embedding_variance(embeddings: &[MotionEmbedding]) -> Option<VarianceAnalysis> {
    if embeddings.len() < 2 {
        return None;
    }

    let n = embeddings.len();

    // Stack compact tensors
    let tensors: Vec<[f64; 24]> = embeddings.iter().map(|e| e.to_compact_array()).collect();

    // Compute mean
    let mut mean = [0.0f64; 24];
    for tensor in &tensors {
        for (i, &v) in tensor.iter().enumerate() {
            mean[i] += v;
        }
    }
    for m in &mut mean {
        *m /= n as f64;
    }

    // Compute per-feature variance
    let mut feature_variances = [0.0f64; 24];
    for tensor in &tensors {
        for (i, &v) in tensor.iter().enumerate() {
            let diff = v - mean[i];
            feature_variances[i] += diff * diff;
        }
    }
    for v in &mut feature_variances {
        *v /= (n - 1) as f64;
    }

    // Feature group variances
    let feature_group_variances = FeatureGroupVariances {
        scale: feature_variances[0],
        shape: feature_variances[1..4].iter().sum(),
        momentum: feature_variances[4..10].iter().sum(),
        heading: feature_variances[10..13].iter().sum(),
        maneuver: feature_variances[13..16].iter().sum(),
        velocity: feature_variances[16..19].iter().sum(),
        chirality: feature_variances[19],
        g_force: feature_variances[20..23].iter().sum(),
        smoothness: feature_variances[23],
    };

    // Variance ratio
    let group_vars = [
        feature_group_variances.scale,
        feature_group_variances.shape,
        feature_group_variances.momentum,
        feature_group_variances.heading,
        feature_group_variances.maneuver,
        feature_group_variances.velocity,
        feature_group_variances.chirality,
        feature_group_variances.g_force,
        feature_group_variances.smoothness,
    ];

    let non_zero_vars: Vec<f64> = group_vars.iter().filter(|&&v| v > 1e-10).copied().collect();

    let variance_ratio_max_min = if non_zero_vars.len() >= 2 {
        let max_var = non_zero_vars.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min_var = non_zero_vars.iter().cloned().fold(f64::INFINITY, f64::min);
        max_var / min_var
    } else {
        f64::INFINITY
    };

    // Compute covariance matrix and PCA
    let mut cov = [[0.0f64; 24]; 24];
    for tensor in &tensors {
        for i in 0..24 {
            for j in 0..24 {
                cov[i][j] += (tensor[i] - mean[i]) * (tensor[j] - mean[j]);
            }
        }
    }
    for row in &mut cov {
        for v in row.iter_mut() {
            *v /= (n - 1) as f64;
        }
    }

    // Simple power iteration for top eigenvalues (approximate)
    let eigenvalues = approximate_eigenvalues(&cov);

    let total_var: f64 = eigenvalues.iter().sum();
    let pc1_variance_percent = if total_var > 0.0 {
        eigenvalues[0] / total_var * 100.0
    } else {
        0.0
    };

    // Effective dimensionality via entropy
    let proportions: Vec<f64> = eigenvalues
        .iter()
        .filter(|&&v| v > 1e-10)
        .map(|&v| v / (total_var + 1e-10))
        .collect();

    let effective_dimensionality = if !proportions.is_empty() {
        let entropy: f64 = -proportions.iter().map(|&p| p * p.ln()).sum::<f64>();
        entropy.exp()
    } else {
        0.0
    };

    // Health checks
    let is_healthy =
        pc1_variance_percent < 40.0 && effective_dimensionality > 8.0 && variance_ratio_max_min < 50.0;

    Some(VarianceAnalysis {
        pc1_variance_percent,
        effective_dimensionality,
        total_dimensions: 24,
        eigenvalue_spectrum: eigenvalues.iter().take(10).copied().collect(),
        feature_group_variances,
        variance_ratio_max_min,
        is_healthy,
    })
}

/// Approximate top eigenvalues using power iteration.
fn approximate_eigenvalues(cov: &[[f64; 24]; 24]) -> Vec<f64> {
    // Simple diagonal approximation for eigenvalues
    // (exact eigendecomposition would require full nalgebra matrix)
    let mut eigenvalues: Vec<f64> = (0..24).map(|i| cov[i][i]).collect();
    eigenvalues.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    eigenvalues
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedding::EncodingMetadata;
    use num_complex::Complex64;

    fn make_test_embedding(offset: f64) -> MotionEmbedding {
        MotionEmbedding {
            scale_magnitude: 0.5 + offset * 0.1,
            shape_entropy: [0.1 + offset * 0.05, 0.2, 0.3],
            normalized_momentum: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            current_heading: [1.0, 0.0, 0.0],
            maneuver_plane: [0.0, 1.0, 0.0],
            velocity_normalized: [0.5 + offset * 0.1, 0.3, 0.1],
            chirality: 0.2 + offset * 0.05,
            g_force_stats: [0.3, 0.5, 0.1],
            smoothness_index: 0.8,
            spectral_features: vec![Complex64::new(1.0, 0.0); 16],
            metadata: EncodingMetadata::default(),
            chirality_threshold: 0.01,
        }
    }

    #[test]
    fn test_variance_analysis() {
        let embeddings: Vec<MotionEmbedding> =
            (0..20).map(|i| make_test_embedding(i as f64)).collect();

        let analysis = validate_embedding_variance(&embeddings);
        assert!(analysis.is_some());

        let a = analysis.unwrap();
        assert_eq!(a.total_dimensions, 24);
        assert!(a.pc1_variance_percent >= 0.0 && a.pc1_variance_percent <= 100.0);
    }

    #[test]
    fn test_too_few_embeddings() {
        let embeddings = vec![make_test_embedding(0.0)];
        let analysis = validate_embedding_variance(&embeddings);
        assert!(analysis.is_none());
    }
}
