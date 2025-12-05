//! Configuration for motion embedding computation.
//!
//! This module provides the [`EmbeddingConfig`] struct which centralizes all
//! tunable parameters for trajectory embedding, along with domain-specific presets.
//!
//! # Example
//!
//! ```
//! use motion_embedding::EmbeddingConfig;
//!
//! // Use default configuration
//! let config = EmbeddingConfig::default();
//!
//! // Use domain preset
//! let drone_config = EmbeddingConfig::drone(false);
//! let racing_config = EmbeddingConfig::racing();
//! ```

use crate::error::{EmbeddingError, Result};

/// Configuration for motion embedding computation.
///
/// This struct centralizes all tunable parameters with sensible defaults
/// optimized for ML training stability. All embedding dimensions are
/// compressed to `[-1, 1]` range using domain-appropriate scaling.
///
/// # Core Parameters
///
/// - `characteristic_speed`: Scaling factor (m/s) to map time to spatial units.
/// - `alpha`: Acceleration sensitivity for mass weighting (0=uniform, 3+=high-G focus).
/// - `k_coeffs`: Number of spectral coefficients (16=lossy, 51=exact for 100 points).
///
/// # Compression Parameters
///
/// - `chirality_scale`: Scale for chirality tanh compression (default: 2.0).
/// - `g_force_scale`: Scale for g-force tanh compression (default: 2.0).
/// - `scale_center`/`scale_width`: Parameters for scale tanh compression.
#[derive(Debug, Clone, PartialEq)]
pub struct EmbeddingConfig {
    // Core parameters
    /// Scaling factor (m/s) to map time to spatial units (w = c * t).
    /// - Quadcopter: 15-25 m/s
    /// - Human walking: 1.5 m/s
    /// - Ground vehicle: 20-40 m/s
    pub characteristic_speed: f64,

    /// Acceleration sensitivity for mass weighting.
    /// - 0.0-0.5: Navigation/surveillance (uniform weight)
    /// - 1.0: General purpose (balanced)
    /// - 2.0-5.0: Racing/aerobatics (high-G maneuvers dominate)
    pub alpha: f64,

    /// Number of spectral coefficients to keep.
    /// - 8-16: Smooth trajectories
    /// - 20-40: Jerky/complex motion
    /// - n/2+1: Lossless (for n points)
    pub k_coeffs: usize,

    // Physical constants
    /// Gravitational acceleration (m/s²).
    pub gravity: f64,

    // Compression parameters
    /// Scale factor for chirality tanh compression.
    /// `tanh(chirality / chirality_scale)` maps to `[-1, 1]`.
    pub chirality_scale: f64,

    /// Scale factor for g-force tanh compression (after log).
    pub g_force_scale: f64,

    /// Velocity normalization method: `"mean"` or `"characteristic"`.
    pub velocity_compression: VelocityCompression,

    /// Whether to normalize momentum by frequency (1/duration).
    pub momentum_frequency_normalize: bool,

    /// Minimum detectable high-frequency ratio for smoothness.
    pub smoothness_sensitivity: f64,

    /// Whether to apply tanh to scale_magnitude.
    pub scale_compression: bool,

    /// Center point for scale compression.
    pub scale_center: f64,

    /// Width for scale compression.
    pub scale_width: f64,

    /// Whether to normalize shape_entropy to `[-1, 1]`.
    pub shape_normalize: bool,

    // Numerical thresholds
    /// Epsilon for singularity detection in PCA.
    pub singularity_eps: f64,

    /// Chirality threshold for handedness classification.
    pub chirality_threshold: f64,

    /// General numerical epsilon.
    pub numerical_eps: f64,

    /// Base for logarithmic scaling.
    pub log_scale_base: f64,

    /// Whether to apply spectral whitening.
    pub spectral_whitening: bool,

    /// Whether to apply log to safety metrics.
    pub log_safety_metrics: bool,

    /// Whether to normalize spectral amplitude by trajectory length.
    pub spectral_amplitude_norm: bool,

    /// Epsilon noise injection for smoothness sensitivity.
    pub position_noise_eps: f64,

    /// Scale factor for momentum normalization.
    pub momentum_scale_factor: f64,

    /// Whether to apply tanh to momentum for bounded range.
    pub momentum_tanh_compress: bool,

    /// Rolling horizon window in seconds (for streaming).
    pub rolling_horizon_seconds: f64,

    /// Spectral transform type (FFT or DCT).
    /// DCT is better for non-periodic trajectories like lines and spirals.
    pub spectral_transform: SpectralTransform,

    /// Window function to apply before FFT to reduce Gibbs phenomenon.
    /// Only used when spectral_transform is FFT.
    pub window_function: WindowFunction,

    /// Whether to store endpoints for improved reconstruction.
    /// Adds 6 floats to metadata (start and end 3D positions).
    pub store_endpoints: bool,
}

/// Velocity compression method.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VelocityCompression {
    /// Normalize by mean speed: `vel / (mean_speed + eps)`.
    /// Captures velocity relative to current energy state.
    #[default]
    Mean,
    /// Normalize by characteristic speed.
    Characteristic,
}

/// Spectral transform type for trajectory encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SpectralTransform {
    /// Standard FFT - best for periodic trajectories (circles, figure-8s).
    /// Can suffer from Gibbs phenomenon on non-periodic signals.
    #[default]
    Fft,
    /// Discrete Cosine Transform - best for non-periodic trajectories.
    /// Implicitly assumes signal is mirrored at boundaries, avoiding discontinuities.
    /// Better energy compaction for lines, spirals, and general motion.
    Dct,
}

/// Window function to apply before FFT to reduce Gibbs phenomenon.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WindowFunction {
    /// No windowing (rectangular window).
    #[default]
    None,
    /// Hanning window - good general-purpose choice.
    /// Reduces Gibbs ringing but slightly reduces frequency resolution.
    Hanning,
    /// Tukey window with alpha=0.5 - tapers only the endpoints.
    /// Preserves more of the signal while reducing edge discontinuities.
    Tukey,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            // Core parameters
            characteristic_speed: 15.0,
            alpha: 1.0,
            k_coeffs: 16,

            // Physical constants
            gravity: 9.81,

            // Compression parameters
            chirality_scale: 1.0,    // Reduced from 2.0 for more sensitivity
            g_force_scale: 2.0,
            velocity_compression: VelocityCompression::Mean,
            momentum_frequency_normalize: true,
            smoothness_sensitivity: 0.1,  // Increased from 0.001 for better spread
            scale_compression: true,
            scale_center: 1.0,       // Changed from 0.5 for wider range
            scale_width: 2.5,        // Changed from 1.0 for wider spread
            shape_normalize: true,

            // Numerical thresholds
            singularity_eps: 1e-6,
            chirality_threshold: 0.01,
            numerical_eps: 1e-7,
            log_scale_base: 10.0,

            // Processing flags
            spectral_whitening: true,
            log_safety_metrics: true,
            spectral_amplitude_norm: true,
            position_noise_eps: 1e-6,
            momentum_scale_factor: 1.0,
            momentum_tanh_compress: true,
            rolling_horizon_seconds: 2.0,
            spectral_transform: SpectralTransform::Fft,
            window_function: WindowFunction::None,
            store_endpoints: true,
        }
    }
}

impl EmbeddingConfig {
    /// Create a new configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Validate the configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if any parameter is out of valid range.
    pub fn validate(&self) -> Result<()> {
        if self.characteristic_speed <= 0.0 {
            return Err(EmbeddingError::invalid_config(
                "characteristic_speed must be positive",
            ));
        }
        if self.alpha < 0.0 {
            return Err(EmbeddingError::invalid_config("alpha must be non-negative"));
        }
        if self.k_coeffs < 1 {
            return Err(EmbeddingError::invalid_config(
                "k_coeffs must be at least 1",
            ));
        }
        if self.gravity <= 0.0 {
            return Err(EmbeddingError::invalid_config("gravity must be positive"));
        }
        if self.log_scale_base <= 1.0 {
            return Err(EmbeddingError::invalid_config(
                "log_scale_base must be > 1",
            ));
        }
        if self.chirality_scale <= 0.0 {
            return Err(EmbeddingError::invalid_config(
                "chirality_scale must be positive",
            ));
        }
        if self.g_force_scale <= 0.0 {
            return Err(EmbeddingError::invalid_config(
                "g_force_scale must be positive",
            ));
        }
        Ok(())
    }

    /// Preset for quadcopter/multirotor drones.
    ///
    /// # Arguments
    ///
    /// * `aggressive` - If true, emphasizes high-G maneuvers.
    #[must_use]
    pub fn drone(aggressive: bool) -> Self {
        Self {
            characteristic_speed: 20.0,
            alpha: if aggressive { 2.0 } else { 1.0 },
            k_coeffs: 16,
            chirality_scale: 1.0,
            g_force_scale: 2.0,
            ..Self::default()
        }
    }

    /// Preset optimized for ML training with balanced feature variance.
    ///
    /// This configuration is tuned to produce embeddings where all feature
    /// groups have similar variance, which is important for neural network
    /// training and distance-based algorithms.
    #[must_use]
    pub fn ml_training() -> Self {
        Self {
            characteristic_speed: 15.0,
            alpha: 1.0,
            k_coeffs: 16,
            smoothness_sensitivity: 0.1,
            scale_center: 1.0,
            scale_width: 2.5,
            chirality_scale: 1.0,
            g_force_scale: 1.5,
            ..Self::default()
        }
    }

    /// Preset for human walking/running motion.
    #[must_use]
    pub fn pedestrian() -> Self {
        Self {
            characteristic_speed: 1.5,
            alpha: 0.5,
            k_coeffs: 32,
            chirality_scale: 1.0,
            g_force_scale: 1.5,
            ..Self::default()
        }
    }

    /// Preset for ground vehicles.
    #[must_use]
    pub fn vehicle() -> Self {
        Self {
            characteristic_speed: 30.0,
            alpha: 1.0,
            k_coeffs: 16,
            chirality_scale: 3.0,
            g_force_scale: 2.0,
            ..Self::default()
        }
    }

    /// Preset for high-performance racing scenarios.
    #[must_use]
    pub fn racing() -> Self {
        Self {
            characteristic_speed: 25.0,
            alpha: 3.0,
            k_coeffs: 24,
            chirality_scale: 2.5,
            g_force_scale: 3.0,
            ..Self::default()
        }
    }

    /// Preset for surveillance/loitering patterns.
    #[must_use]
    pub fn surveillance() -> Self {
        Self {
            characteristic_speed: 15.0,
            alpha: 0.0,
            k_coeffs: 16,
            chirality_scale: 2.0,
            g_force_scale: 1.5,
            ..Self::default()
        }
    }

    /// Set the number of spectral coefficients.
    #[must_use]
    pub const fn with_k_coeffs(mut self, k: usize) -> Self {
        self.k_coeffs = k;
        self
    }

    /// Set the characteristic speed.
    #[must_use]
    pub const fn with_characteristic_speed(mut self, speed: f64) -> Self {
        self.characteristic_speed = speed;
        self
    }

    /// Set the alpha parameter.
    #[must_use]
    pub const fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    /// Calculate the k_coeffs needed for exact round-trip reconstruction.
    #[must_use]
    pub const fn exact_k_coeffs(n_points: usize) -> usize {
        n_points / 2 + 1
    }

    /// Set the spectral transform type.
    #[must_use]
    pub const fn with_spectral_transform(mut self, transform: SpectralTransform) -> Self {
        self.spectral_transform = transform;
        self
    }

    /// Set the window function.
    #[must_use]
    pub const fn with_window_function(mut self, window: WindowFunction) -> Self {
        self.window_function = window;
        self
    }

    /// Enable/disable endpoint storage for reconstruction.
    #[must_use]
    pub const fn with_store_endpoints(mut self, store: bool) -> Self {
        self.store_endpoints = store;
        self
    }

    /// Preset optimized for faithful reconstruction of arbitrary trajectory shapes.
    ///
    /// Uses DCT (better for non-periodic signals), stores endpoints, and uses
    /// higher k_coeffs for better fidelity. Ideal when round-trip reconstruction
    /// quality matters more than embedding compactness.
    #[must_use]
    pub fn reconstruction_optimized() -> Self {
        Self {
            spectral_transform: SpectralTransform::Dct,
            window_function: WindowFunction::None,
            store_endpoints: true,
            k_coeffs: 24,
            spectral_whitening: false,  // Whitening is lossy
            spectral_amplitude_norm: false,
            ..Self::default()
        }
    }

    /// Preset optimized for mixed use (ML + reconstruction).
    ///
    /// Uses DCT with moderate k_coeffs, balancing ML embedding quality
    /// with reconstruction fidelity for any trajectory shape.
    #[must_use]
    pub fn balanced() -> Self {
        Self {
            spectral_transform: SpectralTransform::Dct,
            window_function: WindowFunction::None,
            store_endpoints: true,
            k_coeffs: 16,
            spectral_whitening: true,  // Keep for ML
            ..Self::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = EmbeddingConfig::default();
        assert!(config.validate().is_ok());
        assert_eq!(config.characteristic_speed, 15.0);
        assert_eq!(config.k_coeffs, 16);
    }

    #[test]
    fn test_drone_preset() {
        let config = EmbeddingConfig::drone(false);
        assert!(config.validate().is_ok());
        assert_eq!(config.characteristic_speed, 20.0);
        assert_eq!(config.alpha, 1.0);

        let aggressive = EmbeddingConfig::drone(true);
        assert_eq!(aggressive.alpha, 2.0);
    }

    #[test]
    fn test_pedestrian_preset() {
        let config = EmbeddingConfig::pedestrian();
        assert!(config.validate().is_ok());
        assert_eq!(config.characteristic_speed, 1.5);
        assert_eq!(config.k_coeffs, 32);
    }

    #[test]
    fn test_validation() {
        let mut config = EmbeddingConfig::default();

        config.characteristic_speed = 0.0;
        assert!(config.validate().is_err());

        config.characteristic_speed = 15.0;
        config.alpha = -1.0;
        assert!(config.validate().is_err());

        config.alpha = 1.0;
        config.k_coeffs = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_builder_pattern() {
        let config = EmbeddingConfig::drone(false)
            .with_k_coeffs(32)
            .with_alpha(2.5);
        assert_eq!(config.k_coeffs, 32);
        assert_eq!(config.alpha, 2.5);
    }

    #[test]
    fn test_exact_k_coeffs() {
        assert_eq!(EmbeddingConfig::exact_k_coeffs(100), 51);
        assert_eq!(EmbeddingConfig::exact_k_coeffs(50), 26);
    }
}
