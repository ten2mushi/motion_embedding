//! Error types for motion embedding operations.
//!
//! This module provides a comprehensive error hierarchy for all operations
//! in the motion embedding library.

use thiserror::Error;

/// Main error type for motion embedding operations.
#[derive(Error, Debug)]
pub enum EmbeddingError {
    /// Input validation errors.
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// Trajectory is too short for embedding.
    #[error("Trajectory too short: need at least {min} points, got {actual}")]
    TrajectoryTooShort { min: usize, actual: usize },

    /// Timestamps and positions have mismatched lengths.
    #[error("Length mismatch: {positions} positions vs {timestamps} timestamps")]
    LengthMismatch { positions: usize, timestamps: usize },

    /// Timestamps are not monotonically increasing.
    #[error("Timestamps must be monotonically increasing at index {index}")]
    NonMonotonicTimestamps { index: usize },

    /// Linear algebra computation failed.
    #[error("Linear algebra error: {0}")]
    LinalgError(String),

    /// FFT computation failed.
    #[error("FFT error: {0}")]
    FftError(String),

    /// Numerical computation resulted in NaN or Inf.
    #[error("Numerical instability: {context}")]
    NumericalInstability { context: String },

    /// Configuration validation failed.
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// Reconstruction failed.
    #[error("Reconstruction error: {0}")]
    ReconstructionError(String),
}

/// Result type alias for motion embedding operations.
pub type Result<T> = std::result::Result<T, EmbeddingError>;

impl EmbeddingError {
    /// Create an invalid input error.
    #[must_use]
    pub fn invalid_input(msg: impl Into<String>) -> Self {
        Self::InvalidInput(msg.into())
    }

    /// Create a trajectory too short error.
    #[must_use]
    pub const fn trajectory_too_short(min: usize, actual: usize) -> Self {
        Self::TrajectoryTooShort { min, actual }
    }

    /// Create a length mismatch error.
    #[must_use]
    pub const fn length_mismatch(positions: usize, timestamps: usize) -> Self {
        Self::LengthMismatch {
            positions,
            timestamps,
        }
    }

    /// Create a numerical instability error.
    #[must_use]
    pub fn numerical_instability(context: impl Into<String>) -> Self {
        Self::NumericalInstability {
            context: context.into(),
        }
    }

    /// Create a linear algebra error.
    #[must_use]
    pub fn linalg(msg: impl Into<String>) -> Self {
        Self::LinalgError(msg.into())
    }

    /// Create an FFT error.
    #[must_use]
    pub fn fft(msg: impl Into<String>) -> Self {
        Self::FftError(msg.into())
    }

    /// Create an invalid configuration error.
    #[must_use]
    pub fn invalid_config(msg: impl Into<String>) -> Self {
        Self::InvalidConfig(msg.into())
    }

    /// Create a reconstruction error.
    #[must_use]
    pub fn reconstruction(msg: impl Into<String>) -> Self {
        Self::ReconstructionError(msg.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = EmbeddingError::trajectory_too_short(10, 5);
        assert!(err.to_string().contains("10"));
        assert!(err.to_string().contains("5"));
    }

    #[test]
    fn test_error_constructors() {
        let _ = EmbeddingError::invalid_input("test");
        let _ = EmbeddingError::length_mismatch(10, 20);
        let _ = EmbeddingError::numerical_instability("nan in computation");
        let _ = EmbeddingError::linalg("eigendecomposition failed");
        let _ = EmbeddingError::fft("invalid fft size");
    }
}
