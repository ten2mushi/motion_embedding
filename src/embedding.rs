//! Motion embedding data structures.
//!
//! This module defines the core [`MotionEmbedding`] struct and its associated
//! metadata for trajectory reconstruction.
//!
//! # Embedding Layout (24D Compact)
//!
//! | Index | Feature | Dims | Range | Description |
//! |-------|---------|------|-------|-------------|
//! | 0 | scale_magnitude | 1 | [-1,1] | tanh-compressed log-scale |
//! | 1:4 | shape_entropy | 3 | [-1,1] | Eigenvalue ratios |
//! | 4:10 | normalized_momentum | 6 | [-1,1] | 4D angular momentum |
//! | 10:13 | current_heading | 3 | [-1,1] | Unit direction vector |
//! | 13:16 | maneuver_plane | 3 | [-1,1] | Unit normal vector |
//! | 16:19 | velocity_normalized | 3 | [-1,1] | tanh(vel/mean_speed) |
//! | 19 | chirality | 1 | [-1,1] | Handedness |
//! | 20:23 | g_force_stats | 3 | [-1,1] | [mean_g, max_g, jerk] |
//! | 23 | smoothness_index | 1 | [0,1] | Spectral decay |

use num_complex::Complex64;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Handedness classification of trajectory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Handedness {
    /// Left-handed (counter-clockwise when viewed from above).
    Left,
    /// Right-handed (clockwise when viewed from above).
    Right,
    /// No significant chirality.
    #[default]
    Neutral,
}

impl Handedness {
    /// Determine handedness from raw chirality value.
    #[must_use]
    pub fn from_chirality(raw_chirality: f64, threshold: f64) -> Self {
        if raw_chirality > threshold {
            Self::Right
        } else if raw_chirality < -threshold {
            Self::Left
        } else {
            Self::Neutral
        }
    }
}

/// Metadata required for trajectory reconstruction.
///
/// This struct stores the information needed to invert the embedding
/// back to a trajectory approximation.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct EncodingMetadata {
    /// 4x4 rotation matrix (canonical axes).
    pub canonical_axes: [[f64; 4]; 4],

    /// Spacetime centroid [x, y, z, w].
    pub centroid: [f64; 4],

    /// Number of points in original trajectory.
    pub n_points: usize,

    /// Characteristic speed used for encoding.
    pub characteristic_speed: f64,

    /// Total duration of trajectory (seconds).
    pub total_duration: f64,

    /// Raw scale before compression (log10 of extent).
    pub raw_scale: f64,

    /// Mean speed for velocity denormalization.
    pub mean_speed: f64,

    /// Raw chirality before tanh compression.
    pub raw_chirality: f64,

    /// Raw g-force stats [mean_g, max_g, jerk] before compression.
    pub raw_g_force_stats: [f64; 3],

    /// Start position [x, y, z] in world frame (for endpoint preservation).
    pub start_position: Option<[f64; 3]>,

    /// End position [x, y, z] in world frame (for endpoint preservation).
    pub end_position: Option<[f64; 3]>,

    /// Start timestamp (for endpoint preservation).
    pub start_time: Option<f64>,

    /// End timestamp (for endpoint preservation).
    pub end_time: Option<f64>,
}

impl Default for EncodingMetadata {
    fn default() -> Self {
        Self {
            canonical_axes: [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            centroid: [0.0; 4],
            n_points: 0,
            characteristic_speed: 15.0,
            total_duration: 0.0,
            raw_scale: 0.0,
            mean_speed: 0.0,
            raw_chirality: 0.0,
            raw_g_force_stats: [0.0; 3],
            start_position: None,
            end_position: None,
            start_time: None,
            end_time: None,
        }
    }
}

/// Range-balanced motion embedding.
///
/// This embedding achieves ML-readiness by compressing all features
/// to the `[-1, 1]` range using domain-appropriate sigmoidal compression.
///
/// # Embedding Groups
///
/// - **Group A (Scale, 1D)**: `scale_magnitude` - tanh-compressed log-scale
/// - **Group B (Shape, 3D)**: `shape_entropy` - normalized eigenvalue ratios
/// - **Group C (Dynamics, 6D)**: `normalized_momentum` - 4D angular momentum bivector
/// - **Group D (Navigation, 10D)**: heading, maneuver plane, velocity, chirality
/// - **Group E (Safety, 4D)**: g-force stats, smoothness
/// - **Group F (Spectral, variable)**: log-whitened FFT coefficients
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct MotionEmbedding {
    // Group A: Scale (1D)
    /// Log-scale, tanh-compressed to `[-1, 1]`.
    pub scale_magnitude: f64,

    // Group B: Shape (3D)
    /// Normalized log-eigenvalue ratios.
    pub shape_entropy: [f64; 3],

    // Group C: Dynamics (6D)
    /// Properly nondimensionalized 4D angular momentum bivector.
    /// Components: [L_xy, L_xz, L_xw, L_yz, L_yw, L_zw].
    pub normalized_momentum: [f64; 6],

    // Group D: Navigation State (10D)
    /// Unit world-frame heading direction.
    pub current_heading: [f64; 3],

    /// Unit world-frame maneuver plane normal.
    pub maneuver_plane: [f64; 3],

    /// Tanh-compressed relative velocity.
    pub velocity_normalized: [f64; 3],

    /// Tanh-compressed helicity value.
    pub chirality: f64,

    // Group E: Safety/Quality (4D)
    /// Tanh-compressed [mean_g, max_g, jerk].
    pub g_force_stats: [f64; 3],

    /// Recalibrated spectral decay `[0, 1]`.
    pub smoothness_index: f64,

    // Group F: Spectral (variable)
    /// Log-whitened complex FFT coefficients.
    /// Shape: (k_coeffs * 4) stored as flat Vec for each dimension.
    pub spectral_features: Vec<Complex64>,

    /// Metadata for reconstruction.
    pub metadata: EncodingMetadata,

    /// Threshold for handedness classification.
    pub chirality_threshold: f64,
}

impl MotionEmbedding {
    /// Compact embedding dimension (always 24).
    pub const COMPACT_DIM: usize = 24;

    /// Rotation-invariant embedding dimension (always 11).
    pub const INVARIANT_DIM: usize = 11;

    /// Scale-invariant embedding dimension (always 23).
    pub const SCALE_INVARIANT_DIM: usize = 23;

    /// Convert to compact 24D array.
    ///
    /// All components are range-balanced (`[-1, 1]` or `[0, 1]`).
    #[must_use]
    pub fn to_compact_array(&self) -> [f64; 24] {
        let mut arr = [0.0; 24];

        arr[0] = self.scale_magnitude;

        arr[1] = self.shape_entropy[0];
        arr[2] = self.shape_entropy[1];
        arr[3] = self.shape_entropy[2];

        arr[4] = self.normalized_momentum[0];
        arr[5] = self.normalized_momentum[1];
        arr[6] = self.normalized_momentum[2];
        arr[7] = self.normalized_momentum[3];
        arr[8] = self.normalized_momentum[4];
        arr[9] = self.normalized_momentum[5];

        arr[10] = self.current_heading[0];
        arr[11] = self.current_heading[1];
        arr[12] = self.current_heading[2];

        arr[13] = self.maneuver_plane[0];
        arr[14] = self.maneuver_plane[1];
        arr[15] = self.maneuver_plane[2];

        arr[16] = self.velocity_normalized[0];
        arr[17] = self.velocity_normalized[1];
        arr[18] = self.velocity_normalized[2];

        arr[19] = self.chirality;

        arr[20] = self.g_force_stats[0];
        arr[21] = self.g_force_stats[1];
        arr[22] = self.g_force_stats[2];

        arr[23] = self.smoothness_index;

        arr
    }

    /// Convert to compact Vec.
    #[must_use]
    pub fn to_compact_vec(&self) -> Vec<f64> {
        self.to_compact_array().to_vec()
    }

    /// Convert to full embedding Vec.
    ///
    /// # Arguments
    ///
    /// * `include_spectrum` - Whether to include spectral coefficients.
    #[must_use]
    pub fn to_full_vec(&self, include_spectrum: bool) -> Vec<f64> {
        let mut vec = self.to_compact_vec();

        if include_spectrum {
            // Flatten complex spectrum to real (interleaved real/imag)
            for c in &self.spectral_features {
                vec.push(c.re);
                vec.push(c.im);
            }
        }

        vec
    }

    /// Convert to rotation-invariant 11D array.
    ///
    /// Use this when absolute heading doesn't matter.
    /// Includes: scale, shape, momentum magnitude, speed, chirality, safety metrics.
    #[must_use]
    pub fn to_invariant_array(&self) -> [f64; 11] {
        let momentum_magnitude = (self.normalized_momentum[0].powi(2)
            + self.normalized_momentum[1].powi(2)
            + self.normalized_momentum[2].powi(2)
            + self.normalized_momentum[3].powi(2)
            + self.normalized_momentum[4].powi(2)
            + self.normalized_momentum[5].powi(2))
        .sqrt();

        let speed_normalized = (self.velocity_normalized[0].powi(2)
            + self.velocity_normalized[1].powi(2)
            + self.velocity_normalized[2].powi(2))
        .sqrt();

        [
            self.scale_magnitude,
            self.shape_entropy[0],
            self.shape_entropy[1],
            self.shape_entropy[2],
            momentum_magnitude,
            speed_normalized,
            self.chirality,
            self.g_force_stats[0],
            self.g_force_stats[1],
            self.g_force_stats[2],
            self.smoothness_index,
        ]
    }

    /// Convert to rotation-invariant Vec.
    #[must_use]
    pub fn to_invariant_vec(&self) -> Vec<f64> {
        self.to_invariant_array().to_vec()
    }

    /// Convert to scale-invariant 23D Vec.
    ///
    /// Excludes `scale_magnitude` for comparing trajectory shapes regardless of size.
    #[must_use]
    pub fn to_scale_invariant_vec(&self) -> Vec<f64> {
        let vel_norm = (self.velocity_normalized[0].powi(2)
            + self.velocity_normalized[1].powi(2)
            + self.velocity_normalized[2].powi(2))
        .sqrt()
            + 1e-7;

        let vel_direction = [
            self.velocity_normalized[0] / vel_norm,
            self.velocity_normalized[1] / vel_norm,
            self.velocity_normalized[2] / vel_norm,
        ];

        vec![
            // Shape (3)
            self.shape_entropy[0],
            self.shape_entropy[1],
            self.shape_entropy[2],
            // Momentum (6)
            self.normalized_momentum[0],
            self.normalized_momentum[1],
            self.normalized_momentum[2],
            self.normalized_momentum[3],
            self.normalized_momentum[4],
            self.normalized_momentum[5],
            // Heading (3)
            self.current_heading[0],
            self.current_heading[1],
            self.current_heading[2],
            // Maneuver plane (3)
            self.maneuver_plane[0],
            self.maneuver_plane[1],
            self.maneuver_plane[2],
            // Velocity direction (3)
            vel_direction[0],
            vel_direction[1],
            vel_direction[2],
            // Chirality (1)
            self.chirality,
            // G-force (3)
            self.g_force_stats[0],
            self.g_force_stats[1],
            self.g_force_stats[2],
            // Smoothness (1)
            self.smoothness_index,
        ]
    }

    /// Full embedding dimension including spectrum.
    #[must_use]
    pub fn full_dim(&self) -> usize {
        Self::COMPACT_DIM + self.spectral_features.len() * 2
    }

    // Semantic accessors

    /// Mean G-force (uncompressed for human readability).
    #[must_use]
    pub fn mean_g_force(&self) -> f64 {
        self.metadata.raw_g_force_stats[0]
    }

    /// Max G-force (uncompressed for human readability).
    #[must_use]
    pub fn max_g_force(&self) -> f64 {
        self.metadata.raw_g_force_stats[1]
    }

    /// Jerk index (uncompressed for human readability).
    #[must_use]
    pub fn jerk_index(&self) -> f64 {
        self.metadata.raw_g_force_stats[2]
    }

    /// Raw chirality (uncompressed for human readability).
    #[must_use]
    pub fn raw_chirality(&self) -> f64 {
        self.metadata.raw_chirality
    }

    /// Speed in m/s (denormalized).
    #[must_use]
    pub fn speed(&self) -> f64 {
        let vel_norm = (self.velocity_normalized[0].powi(2)
            + self.velocity_normalized[1].powi(2)
            + self.velocity_normalized[2].powi(2))
        .sqrt();
        vel_norm * self.metadata.mean_speed
    }

    /// Human-readable handedness.
    #[must_use]
    pub fn handedness(&self) -> Handedness {
        Handedness::from_chirality(self.metadata.raw_chirality, self.chirality_threshold)
    }

    /// Whether trajectory is considered smooth (smoothness_index > 0.5).
    #[must_use]
    pub fn is_smooth(&self) -> bool {
        self.smoothness_index > 0.5
    }

    /// Geometric aspect ratio from shape entropy.
    ///
    /// High values = elongated (needle-like), low values = compact (sphere-like).
    #[must_use]
    pub fn aspect_ratio(&self) -> f64 {
        let ratios = [
            self.shape_entropy[0].exp(),
            self.shape_entropy[1].exp(),
            self.shape_entropy[2].exp(),
        ];
        let max = ratios.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min = ratios.iter().cloned().fold(f64::INFINITY, f64::min);
        max / (min + 1e-7)
    }

    /// Purely spatial turning components: L_xy, L_xz, L_yz.
    #[must_use]
    pub fn spatial_momentum(&self) -> [f64; 3] {
        [
            self.normalized_momentum[0], // L_xy
            self.normalized_momentum[1], // L_xz
            self.normalized_momentum[3], // L_yz
        ]
    }

    /// Time-involving momentum components: L_xw, L_yw, L_zw.
    #[must_use]
    pub fn temporal_momentum(&self) -> [f64; 3] {
        [
            self.normalized_momentum[2], // L_xw
            self.normalized_momentum[4], // L_yw
            self.normalized_momentum[5], // L_zw
        ]
    }
}

impl Default for MotionEmbedding {
    fn default() -> Self {
        Self {
            scale_magnitude: 0.0,
            shape_entropy: [0.0; 3],
            normalized_momentum: [0.0; 6],
            current_heading: [1.0, 0.0, 0.0],
            maneuver_plane: [0.0, 1.0, 0.0],
            velocity_normalized: [0.0; 3],
            chirality: 0.0,
            g_force_stats: [0.0; 3],
            smoothness_index: 1.0,
            spectral_features: Vec::new(),
            metadata: EncodingMetadata::default(),
            chirality_threshold: 0.01,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compact_array() {
        let emb = MotionEmbedding::default();
        let arr = emb.to_compact_array();
        assert_eq!(arr.len(), 24);
    }

    #[test]
    fn test_invariant_array() {
        let emb = MotionEmbedding::default();
        let arr = emb.to_invariant_array();
        assert_eq!(arr.len(), 11);
    }

    #[test]
    fn test_scale_invariant_vec() {
        let emb = MotionEmbedding::default();
        let vec = emb.to_scale_invariant_vec();
        assert_eq!(vec.len(), 23);
    }

    #[test]
    fn test_handedness() {
        assert_eq!(Handedness::from_chirality(0.5, 0.01), Handedness::Right);
        assert_eq!(Handedness::from_chirality(-0.5, 0.01), Handedness::Left);
        assert_eq!(Handedness::from_chirality(0.005, 0.01), Handedness::Neutral);
    }

    #[test]
    fn test_semantic_accessors() {
        let mut emb = MotionEmbedding::default();
        emb.metadata.raw_g_force_stats = [1.5, 3.0, 0.5];
        emb.metadata.raw_chirality = 0.5;
        emb.metadata.mean_speed = 10.0;
        emb.velocity_normalized = [0.6, 0.8, 0.0];

        assert_eq!(emb.mean_g_force(), 1.5);
        assert_eq!(emb.max_g_force(), 3.0);
        assert_eq!(emb.jerk_index(), 0.5);
        assert_eq!(emb.raw_chirality(), 0.5);
        assert!((emb.speed() - 10.0).abs() < 1e-6);
    }
}
