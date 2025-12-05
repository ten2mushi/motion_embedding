//! Core encoding algorithm for motion embedding.
//!
//! This module implements the main `compute_motion_embedding` function that
//! transforms a trajectory into a range-balanced embedding.
//!
//! # Pipeline Overview
//!
//! 1. Construct 4D spacetime manifold (w = c × t)
//! 2. Compute acceleration-weighted masses + g-force stats
//! 3. Calculate weighted centroid and center points
//! 4. Compute 4D velocities
//! 5. Calculate chirality index with tanh compression
//! 6. Determine sign-corrected principal axes (PCA)
//! 7. Extract log-space inertia (scale + shape)
//! 8. Extract world-frame navigation state
//! 9. Transform to canonical frame
//! 10. Compute 4D angular momentum (bivector)
//! 11. Compute FFT spectral coefficients
//! 12. Apply spectral whitening + smoothness

use crate::config::{EmbeddingConfig, VelocityCompression};
use crate::embedding::{EncodingMetadata, MotionEmbedding};
use crate::error::{EmbeddingError, Result};
use crate::math::{
    bivector::Bivector6,
    compression::{
        compress_chirality, compress_g_force_stats, compress_velocity, compute_log_space_inertia,
    },
    fft::compute_spectral_features,
    linalg::{
        apply_sign_correction, compute_weighted_pca, normalize3, norm3, transform_point,
    },
};

/// Minimum number of points required for embedding.
pub const MIN_POINTS: usize = 2;

/// Compute motion embedding from positions and timestamps.
///
/// This is the main entry point for embedding computation.
///
/// # Arguments
///
/// * `positions` - Slice of 3D positions [x, y, z] in meters
/// * `timestamps` - Slice of timestamps in seconds (monotonically increasing)
/// * `config` - Embedding configuration
///
/// # Returns
///
/// A `MotionEmbedding` with all components range-balanced.
///
/// # Errors
///
/// Returns an error if:
/// - Less than 2 points provided
/// - Position and timestamp lengths don't match
/// - Timestamps not monotonically increasing
/// - Numerical computation fails
///
/// # Example
///
/// ```
/// use motion_embedding::{compute_motion_embedding, EmbeddingConfig};
///
/// let positions = vec![
///     [0.0, 0.0, 0.0],
///     [1.0, 0.0, 0.0],
///     [2.0, 1.0, 0.0],
///     [3.0, 1.0, 1.0],
/// ];
/// let timestamps = vec![0.0, 0.1, 0.2, 0.3];
/// let config = EmbeddingConfig::drone(false);
///
/// let embedding = compute_motion_embedding(&positions, &timestamps, &config)?;
/// let compact = embedding.to_compact_array(); // 24D
/// # Ok::<(), motion_embedding::EmbeddingError>(())
/// ```
pub fn compute_motion_embedding(
    positions: &[[f64; 3]],
    timestamps: &[f64],
    config: &EmbeddingConfig,
) -> Result<MotionEmbedding> {
    // Validate inputs
    validate_inputs(positions, timestamps)?;
    config.validate()?;

    let n = positions.len();
    let eps = config.numerical_eps;

    // =========================================================================
    // 1. CONSTRUCT 4D SPACETIME MANIFOLD
    // =========================================================================
    let points_4d: Vec<[f64; 4]> = positions
        .iter()
        .zip(timestamps.iter())
        .map(|(p, &t)| [p[0], p[1], p[2], t * config.characteristic_speed])
        .collect();

    // =========================================================================
    // 2. COMPUTE ACCELERATION-WEIGHTED MASSES
    // =========================================================================
    let (masses, g_force_stats_raw, velocities_3d, mean_speed) =
        compute_acceleration_weights(positions, timestamps, config);

    // Compress g-force stats
    let g_force_stats = compress_g_force_stats(&g_force_stats_raw, config);

    // =========================================================================
    // 3. WEIGHTED CENTROID AND CENTERING
    // =========================================================================
    let centroid = compute_weighted_centroid(&points_4d, &masses);
    let centered_points: Vec<[f64; 4]> = points_4d
        .iter()
        .map(|p| {
            [
                p[0] - centroid[0],
                p[1] - centroid[1],
                p[2] - centroid[2],
                p[3] - centroid[3],
            ]
        })
        .collect();

    // =========================================================================
    // 4. COMPUTE 4D VELOCITIES
    // =========================================================================
    let dt: Vec<f64> = timestamps
        .windows(2)
        .map(|w| (w[1] - w[0]).max(eps))
        .collect();

    let velocities_4d: Vec<[f64; 4]> = centered_points
        .windows(2)
        .zip(dt.iter())
        .map(|(pts, &d)| {
            [
                (pts[1][0] - pts[0][0]) / d,
                (pts[1][1] - pts[0][1]) / d,
                (pts[1][2] - pts[0][2]) / d,
                (pts[1][3] - pts[0][3]) / d,
            ]
        })
        .collect();

    // =========================================================================
    // 5. COMPUTE CHIRALITY INDEX
    // =========================================================================
    let (chirality, raw_chirality) =
        compute_chirality_index(&centered_points, &masses, timestamps, config);

    // =========================================================================
    // 6. SIGN-CORRECTED PRINCIPAL AXES (PCA)
    // =========================================================================
    let mut pca = compute_weighted_pca(&centered_points, &masses)?;
    apply_sign_correction(&mut pca, &centered_points, &velocities_4d, &dt, config);

    // =========================================================================
    // 7. LOG-SPACE INERTIA
    // =========================================================================
    let (scale_magnitude, shape_entropy) =
        compute_log_space_inertia(&pca.eigenvalues, config);

    // Store raw scale for metadata
    let total_sqrt_eigs: f64 = pca.eigenvalues.iter().map(|&e| e.max(eps).sqrt()).sum();
    let raw_scale = (total_sqrt_eigs + eps).log(config.log_scale_base);

    // =========================================================================
    // 8. WORLD-FRAME NAVIGATION STATE
    // =========================================================================
    // Current heading: 3D spatial component of major axis
    let major_axis = pca.axis(0);
    let current_heading = normalize3(&[major_axis[0], major_axis[1], major_axis[2]]);

    // Maneuver plane: 3D spatial component of second axis
    let minor_axis = pca.axis(1);
    let maneuver_plane = normalize3(&[minor_axis[0], minor_axis[1], minor_axis[2]]);

    // Velocity normalization
    let velocity_normalized = if !velocities_3d.is_empty() {
        let last_velocity = velocities_3d[velocities_3d.len() - 1];
        match config.velocity_compression {
            VelocityCompression::Mean => {
                compress_velocity(&last_velocity, mean_speed, eps)
            }
            VelocityCompression::Characteristic => {
                [
                    last_velocity[0] / config.characteristic_speed,
                    last_velocity[1] / config.characteristic_speed,
                    last_velocity[2] / config.characteristic_speed,
                ]
            }
        }
    } else {
        [0.0, 0.0, 0.0]
    };

    // =========================================================================
    // 9. TRANSFORM TO CANONICAL FRAME
    // =========================================================================
    let points_canonical: Vec<[f64; 4]> = centered_points
        .iter()
        .map(|p| transform_point(p, &pca.eigenvectors))
        .collect();

    // =========================================================================
    // 10. COMPUTE 4D ANGULAR MOMENTUM
    // =========================================================================
    let total_duration = (timestamps[n - 1] - timestamps[0]).max(eps);

    // Canonical frame velocities
    let velocities_canonical: Vec<[f64; 4]> = points_canonical
        .windows(2)
        .zip(dt.iter())
        .map(|(pts, &d)| {
            [
                (pts[1][0] - pts[0][0]) / d,
                (pts[1][1] - pts[0][1]) / d,
                (pts[1][2] - pts[0][2]) / d,
                (pts[1][3] - pts[0][3]) / d,
            ]
        })
        .collect();

    // Midpoints
    let r_mid: Vec<[f64; 4]> = points_canonical
        .windows(2)
        .map(|pts| {
            [
                (pts[0][0] + pts[1][0]) / 2.0,
                (pts[0][1] + pts[1][1]) / 2.0,
                (pts[0][2] + pts[1][2]) / 2.0,
                (pts[0][3] + pts[1][3]) / 2.0,
            ]
        })
        .collect();

    // Compute angular momentum
    let mut l_net = Bivector6::zero();
    for i in 0..r_mid.len() {
        let l_segment = Bivector6::wedge(&r_mid[i], &velocities_canonical[i]);
        let segment_mass = (masses[i] + masses[i + 1]) / 2.0;
        l_net += l_segment * segment_mass;
    }

    // Normalize momentum
    let normalized_momentum = if config.momentum_frequency_normalize {
        let characteristic_length_sq: f64 =
            pca.eigenvalues[0..3].iter().map(|&e| e.max(eps)).sum();

        let mut momentum = if characteristic_length_sq > eps {
            l_net.scale(total_duration / characteristic_length_sq)
        } else {
            l_net.scale(1.0 / eps)
        };

        momentum = momentum.scale(config.momentum_scale_factor);

        if config.momentum_tanh_compress {
            [
                momentum.components[0].tanh(),
                momentum.components[1].tanh(),
                momentum.components[2].tanh(),
                momentum.components[3].tanh(),
                momentum.components[4].tanh(),
                momentum.components[5].tanh(),
            ]
        } else {
            momentum.components
        }
    } else {
        let characteristic_length = 10.0f64.powf(raw_scale / 2.0);
        let scaled = l_net.scale(1.0 / (total_duration * characteristic_length + eps));
        scaled.components
    };

    // =========================================================================
    // 11. SPECTRAL COEFFICIENTS
    // =========================================================================
    let (spectral_features, smoothness_index) =
        compute_spectral_features(&points_canonical, config.k_coeffs, config);

    // =========================================================================
    // 12. ASSEMBLE EMBEDDING
    // =========================================================================
    // Store endpoints if configured (for improved reconstruction)
    let (start_position, end_position, start_time, end_time) = if config.store_endpoints {
        (
            Some(positions[0]),
            Some(positions[n - 1]),
            Some(timestamps[0]),
            Some(timestamps[n - 1]),
        )
    } else {
        (None, None, None, None)
    };

    let metadata = EncodingMetadata {
        canonical_axes: pca.eigenvectors,
        centroid,
        n_points: n,
        characteristic_speed: config.characteristic_speed,
        total_duration,
        raw_scale,
        mean_speed,
        raw_chirality,
        raw_g_force_stats: g_force_stats_raw,
        start_position,
        end_position,
        start_time,
        end_time,
    };

    Ok(MotionEmbedding {
        scale_magnitude,
        shape_entropy,
        normalized_momentum,
        current_heading,
        maneuver_plane,
        velocity_normalized,
        chirality,
        g_force_stats,
        smoothness_index,
        spectral_features,
        metadata,
        chirality_threshold: config.chirality_threshold,
    })
}

/// Validate input positions and timestamps.
fn validate_inputs(positions: &[[f64; 3]], timestamps: &[f64]) -> Result<()> {
    let n = positions.len();

    if n < MIN_POINTS {
        return Err(EmbeddingError::trajectory_too_short(MIN_POINTS, n));
    }

    if timestamps.len() != n {
        return Err(EmbeddingError::length_mismatch(n, timestamps.len()));
    }

    // Check monotonicity
    for i in 1..timestamps.len() {
        if timestamps[i] <= timestamps[i - 1] {
            return Err(EmbeddingError::NonMonotonicTimestamps { index: i });
        }
    }

    // Check for NaN/Inf
    for (i, pos) in positions.iter().enumerate() {
        if !pos[0].is_finite() || !pos[1].is_finite() || !pos[2].is_finite() {
            return Err(EmbeddingError::numerical_instability(format!(
                "Non-finite position at index {}",
                i
            )));
        }
    }

    for (i, &t) in timestamps.iter().enumerate() {
        if !t.is_finite() {
            return Err(EmbeddingError::numerical_instability(format!(
                "Non-finite timestamp at index {}",
                i
            )));
        }
    }

    Ok(())
}

/// Compute acceleration-weighted masses and g-force statistics.
fn compute_acceleration_weights(
    positions: &[[f64; 3]],
    timestamps: &[f64],
    config: &EmbeddingConfig,
) -> (Vec<f64>, [f64; 3], Vec<[f64; 3]>, f64) {
    let n = positions.len();
    let eps = config.numerical_eps;

    if n < 3 {
        let uniform_mass = 1.0 / n as f64;
        return (
            vec![uniform_mass; n],
            [0.0, 0.0, 0.0],
            vec![[0.0; 3]; 1.max(n.saturating_sub(1))],
            0.0,
        );
    }

    // Time intervals
    let dt: Vec<f64> = timestamps
        .windows(2)
        .map(|w| (w[1] - w[0]).max(eps))
        .collect();

    // Velocities (N-1 points)
    let velocities: Vec<[f64; 3]> = positions
        .windows(2)
        .zip(dt.iter())
        .map(|(ps, &d)| {
            [
                (ps[1][0] - ps[0][0]) / d,
                (ps[1][1] - ps[0][1]) / d,
                (ps[1][2] - ps[0][2]) / d,
            ]
        })
        .collect();

    // Speed statistics
    let speeds: Vec<f64> = velocities.iter().map(|v| norm3(v)).collect();
    let mean_speed = speeds.iter().sum::<f64>() / speeds.len() as f64;

    // Accelerations (N-2 points)
    let dt_acc: Vec<f64> = dt.windows(2).map(|w| (w[0] + w[1]) / 2.0).collect();

    let accelerations: Vec<[f64; 3]> = velocities
        .windows(2)
        .zip(dt_acc.iter())
        .map(|(vs, &d)| {
            [
                (vs[1][0] - vs[0][0]) / d,
                (vs[1][1] - vs[0][1]) / d,
                (vs[1][2] - vs[0][2]) / d,
            ]
        })
        .collect();

    let acc_magnitudes: Vec<f64> = accelerations.iter().map(|a| norm3(a)).collect();

    // G-forces (raw)
    let g_forces: Vec<f64> = acc_magnitudes
        .iter()
        .map(|&a| a / config.gravity)
        .collect();

    let mean_g = if !g_forces.is_empty() {
        g_forces.iter().sum::<f64>() / g_forces.len() as f64
    } else {
        0.0
    };

    let max_g = g_forces.iter().cloned().fold(0.0f64, f64::max);

    // Jerk (N-3 points)
    let jerk_raw = if n >= 4 {
        let dt_jerk: Vec<f64> = dt_acc.windows(2).map(|w| (w[0] + w[1]) / 2.0).collect();

        let jerks: Vec<f64> = accelerations
            .windows(2)
            .zip(dt_jerk.iter())
            .map(|(accs, &d)| {
                let da = [
                    (accs[1][0] - accs[0][0]) / d,
                    (accs[1][1] - accs[0][1]) / d,
                    (accs[1][2] - accs[0][2]) / d,
                ];
                norm3(&da)
            })
            .collect();

        if !jerks.is_empty() {
            jerks.iter().sum::<f64>() / jerks.len() as f64 / (config.gravity + eps)
        } else {
            0.0
        }
    } else {
        0.0
    };

    let g_force_stats_raw = [mean_g, max_g, jerk_raw];

    // Compute mass weights
    let dt_avg = dt.iter().sum::<f64>() / dt.len() as f64;
    let mut masses = vec![dt_avg; n];
    masses[0] = dt_avg;
    masses[n - 1] = dt_avg;

    for i in 1..n - 1 {
        if i - 1 < g_forces.len() && i - 1 < dt_acc.len() {
            masses[i] = (1.0 + config.alpha * g_forces[i - 1]) * dt_acc[i - 1];
        }
    }

    // Normalize masses
    let total_mass: f64 = masses.iter().sum();
    if total_mass > 0.0 {
        for m in &mut masses {
            *m /= total_mass;
        }
    }

    (masses, g_force_stats_raw, velocities, mean_speed)
}

/// Compute weighted centroid of 4D points.
fn compute_weighted_centroid(points: &[[f64; 4]], weights: &[f64]) -> [f64; 4] {
    let mut centroid = [0.0; 4];

    for (p, &w) in points.iter().zip(weights.iter()) {
        centroid[0] += p[0] * w;
        centroid[1] += p[1] * w;
        centroid[2] += p[2] * w;
        centroid[3] += p[3] * w;
    }

    centroid
}

/// Compute mass array from g-forces.
#[allow(dead_code)]
fn compute_masses(n: usize, g_forces: &[f64], dt_acc: &[f64], alpha: f64, dt_avg: f64) -> Vec<f64> {
    let mut masses = vec![dt_avg; n];

    if g_forces.len() + 2 == n && !dt_acc.is_empty() {
        for i in 1..n - 1 {
            masses[i] = (1.0 + alpha * g_forces[i - 1]) * dt_acc[i - 1];
        }
    }

    // Normalize
    let total: f64 = masses.iter().sum();
    if total > 0.0 {
        for m in &mut masses {
            *m /= total;
        }
    }

    masses
}

/// Compute chirality (helicity) index.
fn compute_chirality_index(
    centered_points: &[[f64; 4]],
    masses: &[f64],
    timestamps: &[f64],
    config: &EmbeddingConfig,
) -> (f64, f64) {
    let n = centered_points.len();
    let eps = config.numerical_eps;

    if n < 3 {
        return (0.0, 0.0);
    }

    // Net displacement
    let net_displacement = [
        centered_points[n - 1][0] - centered_points[0][0],
        centered_points[n - 1][1] - centered_points[0][1],
        centered_points[n - 1][2] - centered_points[0][2],
    ];
    let displacement_norm = norm3(&net_displacement);

    // Displacement direction
    let displacement_dir = if displacement_norm < eps * 1000.0 {
        // Closed loop - use cumulative velocity
        let dt: Vec<f64> = timestamps
            .windows(2)
            .map(|w| (w[1] - w[0]).max(eps))
            .collect();

        let mut cumulative_vel = [0.0; 3];
        for i in 0..n - 1 {
            let vel = [
                (centered_points[i + 1][0] - centered_points[i][0]) / dt[i],
                (centered_points[i + 1][1] - centered_points[i][1]) / dt[i],
                (centered_points[i + 1][2] - centered_points[i][2]) / dt[i],
            ];
            cumulative_vel[0] += vel[0] * dt[i];
            cumulative_vel[1] += vel[1] * dt[i];
            cumulative_vel[2] += vel[2] * dt[i];
        }

        let cum_norm = norm3(&cumulative_vel);
        if cum_norm < eps * 1000.0 {
            return (0.0, 0.0);
        }

        [
            cumulative_vel[0] / cum_norm,
            cumulative_vel[1] / cum_norm,
            cumulative_vel[2] / cum_norm,
        ]
    } else {
        [
            net_displacement[0] / displacement_norm,
            net_displacement[1] / displacement_norm,
            net_displacement[2] / displacement_norm,
        ]
    };

    // Spatial angular momentum
    let dt: Vec<f64> = timestamps
        .windows(2)
        .map(|w| (w[1] - w[0]).max(eps))
        .collect();

    let mut l_total = [0.0; 3]; // [L_yz, -L_xz, L_xy] (Hodge dual of bivector)

    for i in 0..n - 1 {
        let r_mid = [
            (centered_points[i][0] + centered_points[i + 1][0]) / 2.0,
            (centered_points[i][1] + centered_points[i + 1][1]) / 2.0,
            (centered_points[i][2] + centered_points[i + 1][2]) / 2.0,
        ];

        let vel = [
            (centered_points[i + 1][0] - centered_points[i][0]) / dt[i],
            (centered_points[i + 1][1] - centered_points[i][1]) / dt[i],
            (centered_points[i + 1][2] - centered_points[i][2]) / dt[i],
        ];

        let segment_mass = (masses[i] + masses[i + 1]) / 2.0;

        // L_xy = x*vy - y*vx
        // L_xz = x*vz - z*vx
        // L_yz = y*vz - z*vy
        let l_xy = (r_mid[0] * vel[1] - r_mid[1] * vel[0]) * segment_mass;
        let l_xz = (r_mid[0] * vel[2] - r_mid[2] * vel[0]) * segment_mass;
        let l_yz = (r_mid[1] * vel[2] - r_mid[2] * vel[1]) * segment_mass;

        // Hodge dual: spin = [L_yz, -L_xz, L_xy]
        l_total[0] += l_yz;
        l_total[1] += -l_xz;
        l_total[2] += l_xy;
    }

    // Helicity = spin · direction
    let chirality_raw =
        l_total[0] * displacement_dir[0]
        + l_total[1] * displacement_dir[1]
        + l_total[2] * displacement_dir[2];

    // Normalize by scale
    let raw_chirality = chirality_raw / (displacement_norm + eps);

    // Compress
    let compressed = compress_chirality(raw_chirality, config);

    (compressed, raw_chirality)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn generate_helix(n: usize, radius: f64, pitch: f64, duration: f64) -> (Vec<[f64; 3]>, Vec<f64>) {
        let positions: Vec<[f64; 3]> = (0..n)
            .map(|i| {
                let t = i as f64 / (n - 1) as f64;
                let angle = 2.0 * PI * t;
                [radius * angle.cos(), radius * angle.sin(), pitch * t]
            })
            .collect();

        let timestamps: Vec<f64> = (0..n)
            .map(|i| i as f64 / (n - 1) as f64 * duration)
            .collect();

        (positions, timestamps)
    }

    fn generate_straight_line(n: usize, length: f64, duration: f64) -> (Vec<[f64; 3]>, Vec<f64>) {
        let positions: Vec<[f64; 3]> = (0..n)
            .map(|i| {
                let t = i as f64 / (n - 1) as f64;
                [length * t, 0.0, 0.0]
            })
            .collect();

        let timestamps: Vec<f64> = (0..n)
            .map(|i| i as f64 / (n - 1) as f64 * duration)
            .collect();

        (positions, timestamps)
    }

    #[test]
    fn test_basic_embedding() {
        let (positions, timestamps) = generate_helix(100, 5.0, 10.0, 5.0);
        let config = EmbeddingConfig::drone(false);

        let result = compute_motion_embedding(&positions, &timestamps, &config);
        assert!(result.is_ok());

        let emb = result.unwrap();
        let compact = emb.to_compact_array();
        assert_eq!(compact.len(), 24);

        // Check range bounds
        assert!(emb.scale_magnitude >= -1.0 && emb.scale_magnitude <= 1.0);
        assert!(emb.chirality >= -1.0 && emb.chirality <= 1.0);
        assert!(emb.smoothness_index >= 0.0 && emb.smoothness_index <= 1.0);
    }

    #[test]
    fn test_chirality_detection() {
        let config = EmbeddingConfig::drone(false);

        // Right-handed helix
        let (pos_right, ts_right) = generate_helix(100, 5.0, 10.0, 5.0);
        let emb_right = compute_motion_embedding(&pos_right, &ts_right, &config).unwrap();

        // Left-handed helix (negative radius for opposite winding)
        let pos_left: Vec<[f64; 3]> = pos_right
            .iter()
            .map(|p| [p[0], -p[1], p[2]])  // Mirror y
            .collect();
        let emb_left = compute_motion_embedding(&pos_left, &ts_right, &config).unwrap();

        // Chiralities should have opposite signs
        assert!(emb_right.chirality * emb_left.chirality < 0.0);
    }

    #[test]
    fn test_straight_line() {
        let (positions, timestamps) = generate_straight_line(50, 100.0, 10.0);
        let config = EmbeddingConfig::drone(false);

        let result = compute_motion_embedding(&positions, &timestamps, &config);
        assert!(result.is_ok());

        let emb = result.unwrap();

        // Straight line should have ~zero chirality
        assert!(emb.chirality.abs() < 0.1);

        // Note: Straight lines can appear "jerky" in FFT due to Gibbs phenomenon
        // (non-periodic boundary conditions). This is expected behavior.
        // The embedding still captures the linear shape correctly via shape_entropy.
    }

    #[test]
    fn test_validation_too_short() {
        let positions = vec![[0.0, 0.0, 0.0]];
        let timestamps = vec![0.0];
        let config = EmbeddingConfig::default();

        let result = compute_motion_embedding(&positions, &timestamps, &config);
        assert!(matches!(
            result,
            Err(EmbeddingError::TrajectoryTooShort { .. })
        ));
    }

    #[test]
    fn test_validation_length_mismatch() {
        let positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let timestamps = vec![0.0, 1.0, 2.0];
        let config = EmbeddingConfig::default();

        let result = compute_motion_embedding(&positions, &timestamps, &config);
        assert!(matches!(result, Err(EmbeddingError::LengthMismatch { .. })));
    }

    #[test]
    fn test_validation_non_monotonic() {
        let positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
        let timestamps = vec![0.0, 2.0, 1.0]; // Not monotonic
        let config = EmbeddingConfig::default();

        let result = compute_motion_embedding(&positions, &timestamps, &config);
        assert!(matches!(
            result,
            Err(EmbeddingError::NonMonotonicTimestamps { .. })
        ));
    }

    #[test]
    fn test_semantic_accessors() {
        let (positions, timestamps) = generate_helix(100, 5.0, 10.0, 5.0);
        let config = EmbeddingConfig::drone(false);

        let emb = compute_motion_embedding(&positions, &timestamps, &config).unwrap();

        // Check semantic accessors return sensible values
        assert!(emb.mean_g_force() >= 0.0);
        assert!(emb.max_g_force() >= emb.mean_g_force());
        assert!(emb.aspect_ratio() >= 1.0);
    }
}
