//! Real-time streaming embedder for continuous trajectories.
//!
//! This module provides [`RollingHorizonEmbedder`] for computing embeddings
//! on a sliding window of incoming position data.

use std::collections::VecDeque;

use crate::config::EmbeddingConfig;
use crate::embedding::MotionEmbedding;
use crate::encoder::compute_motion_embedding;

/// Real-time streaming embedder using a rolling horizon window.
///
/// Maintains a buffer of recent positions and timestamps, automatically
/// trimming old data outside the horizon window.
///
/// # Example
///
/// ```
/// use motion_embedding::{EmbeddingConfig, RollingHorizonEmbedder};
///
/// let config = EmbeddingConfig::drone(false);
/// let mut embedder = RollingHorizonEmbedder::new(config, 2.0);
///
/// // Stream in position updates
/// embedder.update([0.0, 0.0, 0.0], 0.0);
/// embedder.update([1.0, 0.0, 0.0], 0.1);
/// // ... more updates ...
///
/// if embedder.is_ready() {
///     if let Some(embedding) = embedder.get_embedding() {
///         let compact = embedding.to_compact_array();
///     }
/// }
/// ```
#[derive(Debug)]
pub struct RollingHorizonEmbedder {
    /// Configuration for embedding computation.
    config: EmbeddingConfig,

    /// Time window in seconds.
    horizon_seconds: f64,

    /// Maximum buffer size.
    max_points: usize,

    /// Minimum points required for valid embedding.
    min_points: usize,

    /// Position buffer.
    positions: VecDeque<[f64; 3]>,

    /// Timestamp buffer.
    timestamps: VecDeque<f64>,

    /// Cached embedding (lazily computed).
    cached_embedding: Option<MotionEmbedding>,

    /// Whether cache is valid.
    cache_valid: bool,
}

impl RollingHorizonEmbedder {
    /// Create a new rolling horizon embedder.
    ///
    /// # Arguments
    ///
    /// * `config` - Embedding configuration
    /// * `horizon_seconds` - Time window size in seconds
    pub fn new(config: EmbeddingConfig, horizon_seconds: f64) -> Self {
        Self {
            horizon_seconds,
            max_points: 500,
            min_points: 10,
            positions: VecDeque::with_capacity(500),
            timestamps: VecDeque::with_capacity(500),
            cached_embedding: None,
            cache_valid: false,
            config,
        }
    }

    /// Create with custom buffer parameters.
    ///
    /// # Arguments
    ///
    /// * `config` - Embedding configuration
    /// * `horizon_seconds` - Time window size
    /// * `max_points` - Maximum buffer size
    /// * `min_points` - Minimum points for valid embedding
    pub fn with_buffer(
        config: EmbeddingConfig,
        horizon_seconds: f64,
        max_points: usize,
        min_points: usize,
    ) -> Self {
        Self {
            horizon_seconds,
            max_points,
            min_points,
            positions: VecDeque::with_capacity(max_points),
            timestamps: VecDeque::with_capacity(max_points),
            cached_embedding: None,
            cache_valid: false,
            config,
        }
    }

    /// Update with a new position measurement.
    ///
    /// # Arguments
    ///
    /// * `position` - 3D position [x, y, z]
    /// * `timestamp` - Time in seconds
    pub fn update(&mut self, position: [f64; 3], timestamp: f64) {
        // Check for monotonicity
        if let Some(&last_ts) = self.timestamps.back() {
            if timestamp <= last_ts {
                // Skip non-monotonic timestamps
                return;
            }
        }

        self.positions.push_back(position);
        self.timestamps.push_back(timestamp);
        self.cache_valid = false;

        self.trim_to_horizon();
        self.enforce_max_size();
    }

    /// Update with a batch of positions.
    ///
    /// # Arguments
    ///
    /// * `positions` - Slice of 3D positions
    /// * `timestamps` - Slice of timestamps
    pub fn update_batch(&mut self, positions: &[[f64; 3]], timestamps: &[f64]) {
        for (&pos, &ts) in positions.iter().zip(timestamps.iter()) {
            self.update(pos, ts);
        }
    }

    /// Trim old data outside the horizon window.
    fn trim_to_horizon(&mut self) {
        if self.timestamps.is_empty() {
            return;
        }

        let current_time = *self.timestamps.back().unwrap();
        let cutoff_time = current_time - self.horizon_seconds;

        while !self.timestamps.is_empty() && self.timestamps[0] < cutoff_time {
            self.positions.pop_front();
            self.timestamps.pop_front();
        }
    }

    /// Enforce maximum buffer size.
    fn enforce_max_size(&mut self) {
        while self.positions.len() > self.max_points {
            self.positions.pop_front();
            self.timestamps.pop_front();
        }
    }

    /// Check if enough data for valid embedding.
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.positions.len() >= self.min_points
    }

    /// Get the current embedding (computes if cache invalid).
    ///
    /// Returns None if not enough data points.
    pub fn get_embedding(&mut self) -> Option<&MotionEmbedding> {
        if !self.is_ready() {
            return None;
        }

        if !self.cache_valid {
            self.recompute_embedding();
        }

        self.cached_embedding.as_ref()
    }

    /// Force recomputation of embedding.
    fn recompute_embedding(&mut self) {
        let positions: Vec<[f64; 3]> = self.positions.iter().copied().collect();
        let timestamps: Vec<f64> = self.timestamps.iter().copied().collect();

        // Add epsilon noise for smoothness sensitivity on simulated data
        let positions_noised: Vec<[f64; 3]> = if self.config.position_noise_eps > 0.0 {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};

            positions
                .iter()
                .enumerate()
                .map(|(i, p)| {
                    // Deterministic pseudo-random noise
                    let mut hasher = DefaultHasher::new();
                    i.hash(&mut hasher);
                    let h = hasher.finish();
                    let noise_scale = self.config.position_noise_eps;
                    [
                        p[0] + ((h & 0xFF) as f64 / 255.0 - 0.5) * noise_scale,
                        p[1] + (((h >> 8) & 0xFF) as f64 / 255.0 - 0.5) * noise_scale,
                        p[2] + (((h >> 16) & 0xFF) as f64 / 255.0 - 0.5) * noise_scale,
                    ]
                })
                .collect()
        } else {
            positions
        };

        match compute_motion_embedding(&positions_noised, &timestamps, &self.config) {
            Ok(emb) => {
                self.cached_embedding = Some(emb);
                self.cache_valid = true;
            }
            Err(_) => {
                self.cached_embedding = None;
                self.cache_valid = false;
            }
        }
    }

    /// Get embedding tensor for ML inference.
    ///
    /// Returns None if not ready, otherwise the 24D compact embedding.
    pub fn get_compact_embedding(&mut self) -> Option<[f64; 24]> {
        self.get_embedding().map(|e| e.to_compact_array())
    }

    /// Reset the embedder state.
    pub fn reset(&mut self) {
        self.positions.clear();
        self.timestamps.clear();
        self.cached_embedding = None;
        self.cache_valid = false;
    }

    /// Current number of points in buffer.
    #[must_use]
    pub fn n_points(&self) -> usize {
        self.positions.len()
    }

    /// Current time span covered by buffer.
    #[must_use]
    pub fn time_span(&self) -> f64 {
        if self.timestamps.len() < 2 {
            return 0.0;
        }
        self.timestamps.back().unwrap() - self.timestamps.front().unwrap()
    }

    /// Get reference to configuration.
    #[must_use]
    pub const fn config(&self) -> &EmbeddingConfig {
        &self.config
    }

    /// Set new horizon window.
    pub fn set_horizon(&mut self, horizon_seconds: f64) {
        self.horizon_seconds = horizon_seconds;
        self.trim_to_horizon();
        self.cache_valid = false;
    }
}

impl Default for RollingHorizonEmbedder {
    fn default() -> Self {
        Self::new(EmbeddingConfig::default(), 2.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn generate_streaming_data(n: usize, dt: f64) -> (Vec<[f64; 3]>, Vec<f64>) {
        let positions: Vec<[f64; 3]> = (0..n)
            .map(|i| {
                let t = i as f64 * dt;
                let angle = 2.0 * PI * t / 5.0;
                [5.0 * angle.cos(), 5.0 * angle.sin(), t * 0.5]
            })
            .collect();

        let timestamps: Vec<f64> = (0..n).map(|i| i as f64 * dt).collect();

        (positions, timestamps)
    }

    #[test]
    fn test_streaming_basic() {
        let config = EmbeddingConfig::drone(false);
        let mut embedder = RollingHorizonEmbedder::new(config, 2.0);

        let (positions, timestamps) = generate_streaming_data(100, 0.05);

        // Stream in data
        for (&pos, &ts) in positions.iter().zip(timestamps.iter()) {
            embedder.update(pos, ts);
        }

        assert!(embedder.is_ready());

        let embedding = embedder.get_embedding();
        assert!(embedding.is_some());
    }

    #[test]
    fn test_horizon_trimming() {
        let config = EmbeddingConfig::drone(false);
        let mut embedder = RollingHorizonEmbedder::new(config, 1.0);

        // Add data spanning 3 seconds
        for i in 0..60 {
            let t = i as f64 * 0.05;
            embedder.update([t, 0.0, 0.0], t);
        }

        // Should only have ~1 second of data
        assert!(embedder.time_span() <= 1.1);
    }

    #[test]
    fn test_batch_update() {
        let config = EmbeddingConfig::drone(false);
        // Use a horizon longer than the trajectory duration
        let mut embedder = RollingHorizonEmbedder::new(config, 5.0);

        let (positions, timestamps) = generate_streaming_data(50, 0.05);
        // Total duration = 50 * 0.05 = 2.5 seconds, within 5.0s horizon

        embedder.update_batch(&positions, &timestamps);

        assert!(embedder.is_ready());
        assert_eq!(embedder.n_points(), 50);
    }

    #[test]
    fn test_reset() {
        let config = EmbeddingConfig::drone(false);
        let mut embedder = RollingHorizonEmbedder::new(config, 2.0);

        let (positions, timestamps) = generate_streaming_data(50, 0.05);
        embedder.update_batch(&positions, &timestamps);

        assert!(embedder.is_ready());

        embedder.reset();

        assert!(!embedder.is_ready());
        assert_eq!(embedder.n_points(), 0);
    }

    #[test]
    fn test_compact_embedding() {
        let config = EmbeddingConfig::drone(false);
        let mut embedder = RollingHorizonEmbedder::new(config, 2.0);

        let (positions, timestamps) = generate_streaming_data(50, 0.05);
        embedder.update_batch(&positions, &timestamps);

        let compact = embedder.get_compact_embedding();
        assert!(compact.is_some());

        let arr = compact.unwrap();
        assert_eq!(arr.len(), 24);
    }
}
