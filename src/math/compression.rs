//! Compression utilities for range-balancing embedding features.
//!
//! This module provides functions to compress unbounded values to the `[-1, 1]`
//! range using domain-appropriate transformations.
//!
//! # Compression Methods
//!
//! - **Tanh compression**: `tanh(x / scale)` for unbounded metrics
//! - **Log-tanh compression**: `tanh(log(1 + x) / scale)` for positive metrics
//! - **Sigmoid scaling**: For scale and shape normalization

use crate::config::EmbeddingConfig;

/// Compress a value using tanh with a scale factor.
///
/// Maps any real value to `[-1, 1]` range.
///
/// # Arguments
///
/// * `value` - The value to compress
/// * `scale` - Scale factor (larger = gentler compression)
///
/// # Example
///
/// ```
/// use motion_embedding::math::tanh_compress;
///
/// let compressed = tanh_compress(4.0, 2.0);
/// assert!(compressed > 0.9 && compressed < 1.0);
/// ```
#[inline]
#[must_use]
pub fn tanh_compress(value: f64, scale: f64) -> f64 {
    (value / scale).tanh()
}

/// Inverse of tanh compression.
///
/// Recovers the original value from a tanh-compressed value.
/// Note: This is numerically unstable for values close to ±1.
#[inline]
#[must_use]
pub fn tanh_decompress(compressed: f64, scale: f64) -> f64 {
    // atanh(x) = 0.5 * ln((1 + x) / (1 - x))
    let clamped = compressed.clamp(-0.9999999, 0.9999999);
    clamped.atanh() * scale
}

/// Compress g-force statistics using log-tanh.
///
/// Applies: `tanh(log(1 + x) / scale)`
///
/// This is appropriate for g-force values which are always positive
/// and can span several orders of magnitude.
///
/// # Arguments
///
/// * `raw` - Raw [mean_g, max_g, jerk] values
/// * `config` - Configuration with `g_force_scale`
#[must_use]
pub fn compress_g_force_stats(raw: &[f64; 3], config: &EmbeddingConfig) -> [f64; 3] {
    let log_stats = [(1.0 + raw[0]).ln(), (1.0 + raw[1]).ln(), (1.0 + raw[2]).ln()];

    [
        (log_stats[0] / config.g_force_scale).tanh(),
        (log_stats[1] / config.g_force_scale).tanh(),
        (log_stats[2] / config.g_force_scale).tanh(),
    ]
}

/// Decompress g-force statistics.
///
/// Inverse of `compress_g_force_stats`.
#[must_use]
pub fn decompress_g_force_stats(compressed: &[f64; 3], config: &EmbeddingConfig) -> [f64; 3] {
    [
        (tanh_decompress(compressed[0], config.g_force_scale).exp() - 1.0).max(0.0),
        (tanh_decompress(compressed[1], config.g_force_scale).exp() - 1.0).max(0.0),
        (tanh_decompress(compressed[2], config.g_force_scale).exp() - 1.0).max(0.0),
    ]
}

/// Compress scale magnitude using centered tanh.
///
/// Applies: `tanh((log_scale - center) / width)`
///
/// This centers the compression around typical trajectory scales.
#[must_use]
pub fn compress_scale(raw_log_scale: f64, config: &EmbeddingConfig) -> f64 {
    ((raw_log_scale - config.scale_center) / config.scale_width).tanh()
}

/// Decompress scale magnitude.
///
/// Inverse of `compress_scale`.
#[must_use]
pub fn decompress_scale(compressed: f64, config: &EmbeddingConfig) -> f64 {
    tanh_decompress(compressed, config.scale_width) + config.scale_center
}

/// Compress chirality value.
///
/// Applies: `tanh(chirality / chirality_scale)`
#[inline]
#[must_use]
pub fn compress_chirality(raw_chirality: f64, config: &EmbeddingConfig) -> f64 {
    (raw_chirality / config.chirality_scale).tanh()
}

/// Decompress chirality value.
#[inline]
#[must_use]
pub fn decompress_chirality(compressed: f64, config: &EmbeddingConfig) -> f64 {
    tanh_decompress(compressed, config.chirality_scale)
}

/// Compress velocity by mean speed with tanh.
///
/// Applies: `tanh(velocity / mean_speed)` component-wise.
#[must_use]
pub fn compress_velocity(velocity: &[f64; 3], mean_speed: f64, eps: f64) -> [f64; 3] {
    let scale = if mean_speed > eps { mean_speed } else { eps };
    [
        (velocity[0] / scale).tanh(),
        (velocity[1] / scale).tanh(),
        (velocity[2] / scale).tanh(),
    ]
}

/// Compute log-space scale and shape from eigenvalues.
///
/// Returns (scale_magnitude, shape_entropy) where:
/// - scale_magnitude: log10 of total extent (optionally tanh-compressed)
/// - shape_entropy: normalized log-ratios of eigenvalues (3D)
#[must_use]
pub fn compute_log_space_inertia(
    eigenvalues: &[f64; 4],
    config: &EmbeddingConfig,
) -> (f64, [f64; 3]) {
    let eps = config.numerical_eps;
    let base = config.log_scale_base;

    // Ensure positive eigenvalues
    let clamped = [
        eigenvalues[0].max(eps),
        eigenvalues[1].max(eps),
        eigenvalues[2].max(eps),
        eigenvalues[3].max(eps),
    ];

    // Square root of eigenvalues (standard deviations)
    let sqrt_eigs = [
        clamped[0].sqrt(),
        clamped[1].sqrt(),
        clamped[2].sqrt(),
        clamped[3].sqrt(),
    ];

    // Total scale = sum of sqrt eigenvalues
    let total = sqrt_eigs[0] + sqrt_eigs[1] + sqrt_eigs[2] + sqrt_eigs[3];
    let raw_scale = (total + eps).log(base);

    // Optionally compress scale
    let scale_magnitude = if config.scale_compression {
        compress_scale(raw_scale, config)
    } else {
        raw_scale
    };

    // Normalized proportions
    let proportions = [
        sqrt_eigs[0] / total,
        sqrt_eigs[1] / total,
        sqrt_eigs[2] / total,
        sqrt_eigs[3] / total,
    ];

    // Log-ratios relative to first component
    let log_proportions = [
        (proportions[0] + eps).log(base),
        (proportions[1] + eps).log(base),
        (proportions[2] + eps).log(base),
        (proportions[3] + eps).log(base),
    ];

    // Center to sum to 0
    let mean_log =
        (log_proportions[0] + log_proportions[1] + log_proportions[2] + log_proportions[3]) / 4.0;

    // Take last 3 components (skip first for redundancy)
    let mut shape_entropy = [
        log_proportions[1] - mean_log,
        log_proportions[2] - mean_log,
        log_proportions[3] - mean_log,
    ];

    // Optionally normalize to [-1, 1]
    if config.shape_normalize {
        shape_entropy = [
            shape_entropy[0].tanh(),
            shape_entropy[1].tanh(),
            shape_entropy[2].tanh(),
        ];
    }

    (scale_magnitude, shape_entropy)
}

/// Compute smoothness index from high/low frequency energy ratio.
///
/// Returns a value in [0, 1] where 1 = perfectly smooth.
#[must_use]
pub fn compute_smoothness_index(hf_ratio: f64, sensitivity: f64) -> f64 {
    if hf_ratio < 1e-10 {
        return 1.0;
    }

    let log_ratio = (hf_ratio / sensitivity).ln();
    1.0 / (1.0 + (log_ratio / 2.0).exp())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_tanh_compress() {
        // tanh(0) = 0
        assert_relative_eq!(tanh_compress(0.0, 1.0), 0.0);

        // Large positive values approach 1
        assert!(tanh_compress(10.0, 2.0) > 0.99);

        // Large negative values approach -1
        assert!(tanh_compress(-10.0, 2.0) < -0.99);

        // Scale affects compression rate
        let small_scale = tanh_compress(2.0, 1.0);
        let large_scale = tanh_compress(2.0, 4.0);
        assert!(small_scale > large_scale);
    }

    #[test]
    fn test_tanh_round_trip() {
        let original = 3.5;
        let scale = 2.0;
        let compressed = tanh_compress(original, scale);
        let decompressed = tanh_decompress(compressed, scale);
        assert_relative_eq!(original, decompressed, epsilon = 1e-10);
    }

    #[test]
    fn test_g_force_compression() {
        let config = EmbeddingConfig::default();
        let raw = [1.0, 3.0, 0.5];
        let compressed = compress_g_force_stats(&raw, &config);

        // All values should be in [-1, 1]
        for c in compressed {
            assert!(c >= -1.0 && c <= 1.0);
        }

        // Round trip
        let decompressed = decompress_g_force_stats(&compressed, &config);
        for (r, d) in raw.iter().zip(decompressed.iter()) {
            assert_relative_eq!(r, d, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_scale_compression() {
        let config = EmbeddingConfig::default();

        // At center, should be ~0
        let at_center = compress_scale(config.scale_center, &config);
        assert_relative_eq!(at_center, 0.0, epsilon = 1e-10);

        // Round trip
        let raw = 2.5;
        let compressed = compress_scale(raw, &config);
        let decompressed = decompress_scale(compressed, &config);
        assert_relative_eq!(raw, decompressed, epsilon = 1e-6);
    }

    #[test]
    fn test_velocity_compression() {
        let velocity = [10.0, -5.0, 2.0];
        let mean_speed = 8.0;
        let compressed = compress_velocity(&velocity, mean_speed, 1e-7);

        // All values should be in [-1, 1]
        for c in compressed {
            assert!(c >= -1.0 && c <= 1.0);
        }
    }

    #[test]
    fn test_smoothness_index() {
        // Very low HF ratio = very smooth
        let smooth = compute_smoothness_index(1e-6, 0.001);
        assert!(smooth > 0.9);

        // High HF ratio = not smooth
        let jerky = compute_smoothness_index(0.5, 0.001);
        assert!(jerky < 0.5);

        // Zero HF ratio = perfect smoothness
        let perfect = compute_smoothness_index(0.0, 0.001);
        assert_relative_eq!(perfect, 1.0);
    }

    #[test]
    fn test_log_space_inertia() {
        let config = EmbeddingConfig::default();

        // Equal eigenvalues = balanced shape
        let equal_eigs = [1.0, 1.0, 1.0, 1.0];
        let (_scale, shape) = compute_log_space_inertia(&equal_eigs, &config);

        // Shape entropy should be near zero for equal eigenvalues
        for s in shape {
            assert!(s.abs() < 0.1);
        }

        // Unequal eigenvalues = elongated shape
        let unequal_eigs = [10.0, 1.0, 0.1, 0.01];
        let (_, unequal_shape) = compute_log_space_inertia(&unequal_eigs, &config);

        // Should have larger variance in shape
        let max_diff = unequal_shape
            .iter()
            .map(|x| x.abs())
            .fold(0.0f64, f64::max);
        assert!(max_diff > 0.3);
    }
}
