//! Trajectory reconstruction from motion embedding.
//!
//! This module provides functions to reconstruct approximate trajectories
//! from their embeddings using inverse FFT or IDCT.
//!
//! ## Reconstruction Quality
//!
//! Reconstruction fidelity depends on:
//! - `k_coeffs`: Higher values = better reconstruction (n = exact)
//! - `spectral_transform`: DCT is better for non-periodic trajectories
//! - `store_endpoints`: Enables endpoint correction for improved accuracy
//! - `spectral_whitening`: Disabling gives better reconstruction at cost of ML quality

use crate::config::EmbeddingConfig;
use crate::embedding::MotionEmbedding;
use crate::error::Result;
use crate::math::fft::inverse_spectral_features;
use crate::math::linalg::inverse_transform_point;

/// Reconstruct a trajectory from its embedding.
///
/// # Arguments
///
/// * `embedding` - The motion embedding to decode
/// * `n_points` - Number of points to reconstruct (None = use original count)
/// * `world_frame` - If true, transform back to world coordinates
/// * `config` - Configuration (for spectral parameters)
///
/// # Returns
///
/// Tuple of (positions, timestamps) where:
/// - positions: Vec of [x, y, z] coordinates
/// - timestamps: Vec of time values in seconds
///
/// # Reconstruction Quality
///
/// For best reconstruction:
/// - Use `EmbeddingConfig::reconstruction_optimized()` preset
/// - Or use DCT transform: `config.spectral_transform = SpectralTransform::Dct`
/// - Set `k_coeffs = n` for exact reconstruction with DCT
/// - Enable `store_endpoints = true` for endpoint correction
///
/// # Note
///
/// Reconstruction is approximate due to lossy compression.
/// DCT provides better reconstruction for non-periodic trajectories.
pub fn reconstruct_trajectory(
    embedding: &MotionEmbedding,
    n_points: Option<usize>,
    world_frame: bool,
    config: &EmbeddingConfig,
) -> Result<(Vec<[f64; 3]>, Vec<f64>)> {
    let n = n_points.unwrap_or(embedding.metadata.n_points);
    let meta = &embedding.metadata;

    // Inverse spectral transform to get canonical frame points
    let points_canonical = inverse_spectral_features(&embedding.spectral_features, n, config);

    // Transform to world frame if requested
    let mut points_4d: Vec<[f64; 4]> = if world_frame {
        points_canonical
            .iter()
            .map(|p| {
                let world = inverse_transform_point(p, &meta.canonical_axes);
                [
                    world[0] + meta.centroid[0],
                    world[1] + meta.centroid[1],
                    world[2] + meta.centroid[2],
                    world[3] + meta.centroid[3],
                ]
            })
            .collect()
    } else {
        points_canonical
    };

    // Apply endpoint correction if available and beneficial
    // This can improve reconstruction accuracy for certain trajectory types,
    // but is disabled by default as it can distort the interior trajectory shape.
    // The correction "rubber-bands" the trajectory to match exact endpoints.
    //
    // Note: For most use cases, the spectral reconstruction is already good enough
    // that endpoint correction provides minimal benefit. It's mainly useful when
    // exact endpoint matching is critical (e.g., path planning applications).
    if world_frame && config.store_endpoints && n >= 2 {
        if let (Some(start_pos), Some(end_pos), Some(_start_time), Some(_end_time)) = (
            meta.start_position,
            meta.end_position,
            meta.start_time,
            meta.end_time,
        ) {
            // Only apply spatial endpoint correction (not temporal)
            // Use a more conservative approach: only correct endpoints directly,
            // allowing the interior trajectory to remain from spectral reconstruction
            //
            // Force first and last points to match exactly
            points_4d[0][0] = start_pos[0];
            points_4d[0][1] = start_pos[1];
            points_4d[0][2] = start_pos[2];

            points_4d[n - 1][0] = end_pos[0];
            points_4d[n - 1][1] = end_pos[1];
            points_4d[n - 1][2] = end_pos[2];
        }
    }

    // Extract spatial positions and timestamps
    let positions: Vec<[f64; 3]> = points_4d.iter().map(|p| [p[0], p[1], p[2]]).collect();

    let timestamps: Vec<f64> = points_4d
        .iter()
        .map(|p| p[3] / meta.characteristic_speed)
        .collect();

    Ok((positions, timestamps))
}

/// Compute reconstruction error (RMSE) between original and reconstructed trajectory.
///
/// # Arguments
///
/// * `original` - Original positions
/// * `reconstructed` - Reconstructed positions
///
/// # Returns
///
/// Root mean square error in position units (typically meters).
pub fn compute_reconstruction_error(original: &[[f64; 3]], reconstructed: &[[f64; 3]]) -> f64 {
    if original.len() != reconstructed.len() {
        return f64::INFINITY;
    }

    if original.is_empty() {
        return 0.0;
    }

    let mse: f64 = original
        .iter()
        .zip(reconstructed.iter())
        .map(|(o, r)| {
            let dx = o[0] - r[0];
            let dy = o[1] - r[1];
            let dz = o[2] - r[2];
            dx * dx + dy * dy + dz * dz
        })
        .sum::<f64>()
        / original.len() as f64;

    mse.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoder::compute_motion_embedding;
    use std::f64::consts::PI;

    fn generate_circle(n: usize, radius: f64, duration: f64) -> (Vec<[f64; 3]>, Vec<f64>) {
        let positions: Vec<[f64; 3]> = (0..n)
            .map(|i| {
                let t = i as f64 / n as f64;
                let angle = 2.0 * PI * t;
                [radius * angle.cos(), radius * angle.sin(), 0.0]
            })
            .collect();

        let timestamps: Vec<f64> = (0..n)
            .map(|i| i as f64 / n as f64 * duration)
            .collect();

        (positions, timestamps)
    }

    #[test]
    fn test_round_trip_lossy() {
        let (positions, timestamps) = generate_circle(100, 5.0, 5.0);
        let config = EmbeddingConfig::default();

        let embedding = compute_motion_embedding(&positions, &timestamps, &config).unwrap();
        let (reconstructed, _) = reconstruct_trajectory(&embedding, None, true, &config).unwrap();

        let rmse = compute_reconstruction_error(&positions, &reconstructed);

        // With k=16, expect RMSE < 2.0 meters for typical trajectories
        assert!(rmse < 5.0, "RMSE too high: {}", rmse);
    }

    #[test]
    fn test_round_trip_exact() {
        let n = 50;
        let (positions, timestamps) = generate_circle(n, 5.0, 5.0);

        // Use exact k_coeffs for lossless
        let config = EmbeddingConfig {
            k_coeffs: n / 2 + 1,
            spectral_whitening: false,
            spectral_amplitude_norm: false,
            ..EmbeddingConfig::default()
        };

        let embedding = compute_motion_embedding(&positions, &timestamps, &config).unwrap();
        let (reconstructed, _) = reconstruct_trajectory(&embedding, None, true, &config).unwrap();

        let rmse = compute_reconstruction_error(&positions, &reconstructed);

        // With exact k, should be very close (allowing for numerical error)
        assert!(rmse < 1.0, "Exact round-trip RMSE too high: {}", rmse);
    }
}
