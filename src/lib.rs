//! Motion Embedding Library
//!
//! Range-balanced trajectory embedding for ML applications.
//!
//! This library provides a 24-dimensional embedding for 3D trajectories
//! (position + time), designed for machine learning tasks like classification,
//! clustering, and similarity search.
//!
//! # Features
//!
//! - **Range-balanced**: All features compressed to `[-1, 1]` for equal weighting
//! - **Physically meaningful**: Captures scale, shape, momentum, chirality, dynamics
//! - **FFT-based**: Spectral decomposition for trajectory texture
//! - **Streaming support**: Rolling horizon embedder for real-time applications
//!
//! # Quick Start
//!
//! ```
//! use motion_embedding::{compute_motion_embedding, EmbeddingConfig};
//!
//! let positions = vec![
//!     [0.0, 0.0, 0.0],
//!     [1.0, 0.5, 0.0],
//!     [2.0, 1.0, 0.5],
//!     [3.0, 1.0, 1.0],
//!     [4.0, 0.5, 1.5],
//! ];
//! let timestamps = vec![0.0, 0.1, 0.2, 0.3, 0.4];
//!
//! let config = EmbeddingConfig::drone(false);
//! let embedding = compute_motion_embedding(&positions, &timestamps, &config)?;
//!
//! // Get 24D compact embedding for ML
//! let compact = embedding.to_compact_array();
//! # Ok::<(), motion_embedding::EmbeddingError>(())
//! ```
//!
//! # Embedding Dimensions
//!
//! | Mode | Dimensions | Use Case |
//! |------|------------|----------|
//! | `to_compact_array()` | 24 | Default ML input |
//! | `to_full_vec(true)` | 24 + k*8 | With spectral detail |
//! | `to_invariant_array()` | 11 | Rotation-invariant |
//! | `to_scale_invariant_vec()` | 23 | Scale-invariant |
//!
//! # Presets
//!
//! Domain-specific configurations are available:
//!
//! ```
//! use motion_embedding::EmbeddingConfig;
//!
//! let drone_config = EmbeddingConfig::drone(false);
//! let racing_config = EmbeddingConfig::racing();
//! let pedestrian_config = EmbeddingConfig::pedestrian();
//! let vehicle_config = EmbeddingConfig::vehicle();
//! ```

#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::similar_names)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

pub mod config;
pub mod decoder;
pub mod distance;
pub mod embedding;
pub mod encoder;
pub mod error;
pub mod math;
pub mod streaming;
pub mod validation;

// Re-exports for convenient access
pub use config::{EmbeddingConfig, SpectralTransform, VelocityCompression, WindowFunction};
pub use decoder::{compute_reconstruction_error, reconstruct_trajectory};
pub use distance::{
    compute_embedding_distance, compute_embedding_distance_detailed, compute_pairwise_distances,
    heading_angle_between, DistanceBreakdown, DistanceWeights,
};
pub use embedding::{EncodingMetadata, Handedness, MotionEmbedding};
pub use encoder::compute_motion_embedding;
pub use error::{EmbeddingError, Result};
pub use streaming::RollingHorizonEmbedder;
pub use validation::{validate_embedding_variance, FeatureGroupVariances, VarianceAnalysis};

/// Library version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Compact embedding dimension.
pub const COMPACT_DIM: usize = 24;

/// Rotation-invariant embedding dimension.
pub const INVARIANT_DIM: usize = 11;

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn generate_helix(n: usize) -> (Vec<[f64; 3]>, Vec<f64>) {
        let positions: Vec<[f64; 3]> = (0..n)
            .map(|i| {
                let t = i as f64 / (n - 1) as f64;
                let angle = 2.0 * PI * t;
                [5.0 * angle.cos(), 5.0 * angle.sin(), 10.0 * t]
            })
            .collect();

        let timestamps: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64 * 5.0).collect();

        (positions, timestamps)
    }

    #[test]
    fn test_full_pipeline() {
        let (positions, timestamps) = generate_helix(100);
        let config = EmbeddingConfig::drone(false);

        // Encode
        let embedding = compute_motion_embedding(&positions, &timestamps, &config).unwrap();

        // Check dimensions
        assert_eq!(embedding.to_compact_array().len(), COMPACT_DIM);
        assert_eq!(embedding.to_invariant_array().len(), INVARIANT_DIM);

        // Check range bounds
        let compact = embedding.to_compact_array();
        for (i, &v) in compact.iter().enumerate() {
            // Smoothness can be 0-1, others should be roughly -1 to 1
            if i != 23 {
                assert!(
                    v >= -1.5 && v <= 1.5,
                    "Component {} out of range: {}",
                    i,
                    v
                );
            }
        }

        // Decode
        let (recon_pos, _recon_ts) =
            reconstruct_trajectory(&embedding, None, true, &config).unwrap();

        // Check reconstruction
        let rmse = compute_reconstruction_error(&positions, &recon_pos);
        assert!(rmse < 10.0, "RMSE too high: {}", rmse);
    }

    #[test]
    fn test_distance_functions() {
        let (positions, timestamps) = generate_helix(50);
        let config = EmbeddingConfig::drone(false);

        let emb1 = compute_motion_embedding(&positions, &timestamps, &config).unwrap();
        let emb2 = compute_motion_embedding(&positions, &timestamps, &config).unwrap();

        // Self distance should be zero
        let dist = compute_embedding_distance(&emb1, &emb2, None);
        assert!(dist < 1e-10);
    }

    #[test]
    fn test_streaming_embedder() {
        let config = EmbeddingConfig::drone(false);
        let mut embedder = RollingHorizonEmbedder::new(config, 2.0);

        let (positions, timestamps) = generate_helix(100);

        for (&pos, &ts) in positions.iter().zip(timestamps.iter()) {
            embedder.update(pos, ts);
        }

        assert!(embedder.is_ready());
        let embedding = embedder.get_embedding();
        assert!(embedding.is_some());
    }
}
