//! Mathematical utilities for motion embedding.
//!
//! This module provides:
//! - [`bivector`]: 4D wedge product (geometric algebra)
//! - [`compression`]: tanh/log compression utilities
//! - [`linalg`]: PCA and eigendecomposition
//! - [`fft`]: FFT spectral analysis

pub mod bivector;
pub mod compression;
pub mod fft;
pub mod linalg;

pub use bivector::Bivector6;
pub use compression::{compress_g_force_stats, tanh_compress};
pub use fft::{compute_spectral_features, inverse_spectral_features};
pub use linalg::{apply_sign_correction, compute_weighted_pca, PcaResult};
