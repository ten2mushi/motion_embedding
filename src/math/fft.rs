//! FFT/DCT spectral analysis for trajectory embedding.
//!
//! This module provides spectral decomposition with support for both FFT and DCT,
//! along with optional window functions and log-whitening.
//!
//! ## Transform Comparison
//!
//! - **FFT**: Best for periodic signals (circles, figure-8s). Can exhibit Gibbs
//!   phenomenon (ringing artifacts) on non-periodic signals.
//! - **DCT**: Best for non-periodic signals (lines, spirals, random walks).
//!   Implicitly mirrors the signal at boundaries, eliminating discontinuities.

use crate::config::{EmbeddingConfig, SpectralTransform, WindowFunction};
use num_complex::Complex64;
use rustfft::{num_complex::Complex, FftPlanner};
use std::f64::consts::PI;

/// Compute smoothness based on normalized acceleration variance.
///
/// This metric captures human-intuitive "jerkiness" better than spectral analysis.
/// Returns a value in [0, 1] where 1 = perfectly smooth (constant velocity).
fn compute_acceleration_smoothness(points: &[[f64; 4]], config: &EmbeddingConfig) -> f64 {
    let n = points.len();
    if n < 3 {
        return 1.0; // Not enough points to compute acceleration
    }

    let eps = config.numerical_eps;

    // Compute velocities (first differences)
    let velocities: Vec<[f64; 4]> = points
        .windows(2)
        .map(|w| [
            w[1][0] - w[0][0],
            w[1][1] - w[0][1],
            w[1][2] - w[0][2],
            w[1][3] - w[0][3],
        ])
        .collect();

    // Compute accelerations (second differences)
    let accelerations: Vec<[f64; 4]> = velocities
        .windows(2)
        .map(|w| [
            w[1][0] - w[0][0],
            w[1][1] - w[0][1],
            w[1][2] - w[0][2],
            w[1][3] - w[0][3],
        ])
        .collect();

    if accelerations.is_empty() {
        return 1.0;
    }

    // Compute mean speed for normalization
    let mean_speed: f64 = velocities
        .iter()
        .map(|v| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3]).sqrt())
        .sum::<f64>()
        / velocities.len() as f64;

    if mean_speed < eps {
        return 1.0; // Stationary trajectory
    }

    // Compute acceleration magnitudes, normalized by mean speed
    let normalized_acc_mags: Vec<f64> = accelerations
        .iter()
        .map(|a| {
            let mag = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2] + a[3] * a[3]).sqrt();
            mag / mean_speed
        })
        .collect();

    // Compute coefficient of variation of acceleration
    let mean_acc = normalized_acc_mags.iter().sum::<f64>() / normalized_acc_mags.len() as f64;
    let variance_acc = normalized_acc_mags
        .iter()
        .map(|a| (a - mean_acc).powi(2))
        .sum::<f64>()
        / normalized_acc_mags.len() as f64;

    // Use mean + std as a "jerkiness" measure
    let jerkiness = mean_acc + variance_acc.sqrt();

    // Compress to [0, 1] using exponential decay
    // Higher jerkiness = lower smoothness
    // smoothness = exp(-jerkiness / sensitivity)
    // This maps: 0 -> 1.0 (perfectly smooth), high -> 0 (very jerky)
    (-jerkiness / config.smoothness_sensitivity).exp()
}

/// Generate window function coefficients.
///
/// Window functions taper the signal at endpoints to reduce spectral leakage
/// and Gibbs phenomenon when using FFT on non-periodic signals.
fn generate_window(n: usize, window: WindowFunction) -> Vec<f64> {
    match window {
        WindowFunction::None => vec![1.0; n],
        WindowFunction::Hanning => {
            (0..n)
                .map(|i| {
                    let t = i as f64 / (n - 1).max(1) as f64;
                    0.5 * (1.0 - (2.0 * PI * t).cos())
                })
                .collect()
        }
        WindowFunction::Tukey => {
            let alpha = 0.5; // Taper ratio
            (0..n)
                .map(|i| {
                    let t = i as f64 / (n - 1).max(1) as f64;
                    if t < alpha / 2.0 {
                        0.5 * (1.0 + (2.0 * PI * t / alpha - PI).cos())
                    } else if t > 1.0 - alpha / 2.0 {
                        0.5 * (1.0 + (2.0 * PI * (t - 1.0) / alpha + PI).cos())
                    } else {
                        1.0
                    }
                })
                .collect()
        }
    }
}

/// Compute DCT-II (forward DCT) of a real signal.
///
/// DCT-II is the most common DCT variant, defined as:
/// X[k] = sum_{n=0}^{N-1} x[n] * cos(pi * k * (2n + 1) / (2N))
///
/// DCT is better than FFT for non-periodic signals because it implicitly
/// mirrors the signal at boundaries, avoiding discontinuities.
fn compute_dct(signal: &[f64]) -> Vec<f64> {
    let n = signal.len();
    if n == 0 {
        return vec![];
    }

    let mut result = vec![0.0; n];

    for k in 0..n {
        let mut sum = 0.0;
        for (i, &x) in signal.iter().enumerate() {
            sum += x * (PI * k as f64 * (2.0 * i as f64 + 1.0) / (2.0 * n as f64)).cos();
        }
        // Apply orthonormal scaling
        let scale = if k == 0 {
            (1.0 / n as f64).sqrt()
        } else {
            (2.0 / n as f64).sqrt()
        };
        result[k] = sum * scale;
    }

    result
}

/// Compute IDCT-II (inverse DCT) to reconstruct signal.
///
/// IDCT-II is the inverse of DCT-II:
/// x[n] = sum_{k=0}^{N-1} w[k] * X[k] * cos(pi * k * (2n + 1) / (2N))
/// where w[0] = 1/sqrt(N), w[k>0] = sqrt(2/N)
fn compute_idct(spectrum: &[f64], n_points: usize) -> Vec<f64> {
    let k_max = spectrum.len();
    if k_max == 0 {
        return vec![0.0; n_points];
    }

    let mut result = vec![0.0; n_points];

    for i in 0..n_points {
        let mut sum = 0.0;
        for (k, &coeff) in spectrum.iter().enumerate() {
            // Apply orthonormal scaling (inverse of forward)
            let scale = if k == 0 {
                (1.0 / n_points as f64).sqrt()
            } else {
                (2.0 / n_points as f64).sqrt()
            };
            sum += scale * coeff * (PI * k as f64 * (2.0 * i as f64 + 1.0) / (2.0 * n_points as f64)).cos();
        }
        result[i] = sum;
    }

    result
}

/// Compute spectral features from canonical-frame points.
///
/// Supports both FFT and DCT transforms, with optional windowing for FFT.
///
/// # Arguments
///
/// * `points_canonical` - Points in canonical (PCA) frame
/// * `k_coeffs` - Number of spectral coefficients to keep
/// * `config` - Configuration with spectral parameters
///
/// # Returns
///
/// Tuple of (whitened_spectrum, smoothness_index) where:
/// - whitened_spectrum: Vec of Complex64, length k_coeffs * 4
///   (for DCT, imaginary parts are zero)
/// - smoothness_index: f64 in [0, 1]
pub fn compute_spectral_features(
    points_canonical: &[[f64; 4]],
    k_coeffs: usize,
    config: &EmbeddingConfig,
) -> (Vec<Complex64>, f64) {
    let n = points_canonical.len();

    if n < 2 {
        return (vec![Complex64::new(0.0, 0.0); k_coeffs * 4], 1.0);
    }

    let mut all_coeffs = Vec::with_capacity(k_coeffs * 4);

    match config.spectral_transform {
        SpectralTransform::Dct => {
            // DCT path - produces real coefficients (no imaginary)
            for dim in 0..4 {
                let signal: Vec<f64> = points_canonical.iter().map(|p| p[dim]).collect();
                let dct_coeffs = compute_dct(&signal);

                // Take k coefficients
                let k_actual = k_coeffs.min(dct_coeffs.len());

                // Apply amplitude normalization if enabled
                let norm_factor = if config.spectral_amplitude_norm {
                    (n as f64).sqrt()
                } else {
                    1.0
                };

                // Apply log-whitening if enabled and convert to Complex64
                let whitened: Vec<Complex64> = dct_coeffs.iter()
                    .take(k_actual)
                    .map(|&c| {
                        let normalized = c / norm_factor;
                        if config.spectral_whitening {
                            let sign = normalized.signum();
                            let log_mag = (1.0 + normalized.abs()).ln();
                            Complex64::new(sign * log_mag, 0.0)
                        } else {
                            Complex64::new(normalized, 0.0)
                        }
                    })
                    .collect();

                // Pad if needed
                let mut dim_coeffs = whitened;
                while dim_coeffs.len() < k_coeffs {
                    dim_coeffs.push(Complex64::new(0.0, 0.0));
                }

                all_coeffs.extend(dim_coeffs);
            }
        }
        SpectralTransform::Fft => {
            // FFT path - produces complex coefficients
            let window = generate_window(n, config.window_function);

            let mut planner = FftPlanner::new();
            let fft = planner.plan_fft_forward(n);

            let n_freq = n / 2 + 1;
            let k_actual = k_coeffs.min(n_freq);

            for dim in 0..4 {
                // Extract dimension, apply window, and convert to complex
                let mut buffer: Vec<Complex<f64>> = points_canonical
                    .iter()
                    .zip(window.iter())
                    .map(|(p, &w)| Complex::new(p[dim] * w, 0.0))
                    .collect();

                // Perform FFT
                fft.process(&mut buffer);

                // Normalize (orthonormal FFT)
                let norm_factor = (n as f64).sqrt();
                for c in &mut buffer {
                    *c /= norm_factor;
                }

                // Amplitude normalization by trajectory length
                if config.spectral_amplitude_norm {
                    let length_factor = (n as f64).sqrt();
                    for c in &mut buffer {
                        *c /= length_factor;
                    }
                }

                // Apply log-whitening if enabled
                let whitened: Vec<Complex64> = if config.spectral_whitening {
                    buffer.iter().take(k_actual).map(|c| {
                        let mag = c.norm();
                        let phase = c.arg();
                        let log_mag = (1.0 + mag).ln();
                        Complex64::from_polar(log_mag, phase)
                    }).collect()
                } else {
                    buffer.iter().take(k_actual)
                        .map(|c| Complex64::new(c.re, c.im))
                        .collect()
                };

                // Pad if needed
                let mut dim_coeffs = whitened;
                while dim_coeffs.len() < k_coeffs {
                    dim_coeffs.push(Complex64::new(0.0, 0.0));
                }

                all_coeffs.extend(dim_coeffs);
            }
        }
    }

    // Compute smoothness index using ACCELERATION VARIANCE
    // This captures human-intuitive "jerkiness" better than spectral analysis
    let smoothness_index = compute_acceleration_smoothness(points_canonical, config);

    (all_coeffs, smoothness_index)
}

/// Inverse spectral features - reconstruct points from spectrum.
///
/// Supports both FFT and DCT inverse transforms.
///
/// # Arguments
///
/// * `spectrum` - Whitened spectrum (k_coeffs * 4 complex values)
/// * `n_points` - Number of points to reconstruct
/// * `config` - Configuration
///
/// # Returns
///
/// Reconstructed points in canonical frame.
pub fn inverse_spectral_features(
    spectrum: &[Complex64],
    n_points: usize,
    config: &EmbeddingConfig,
) -> Vec<[f64; 4]> {
    let k_coeffs = spectrum.len() / 4;
    let mut result = vec![[0.0; 4]; n_points];

    match config.spectral_transform {
        SpectralTransform::Dct => {
            // DCT inverse path
            for dim in 0..4 {
                let dim_start = dim * k_coeffs;
                let dim_spectrum = &spectrum[dim_start..dim_start + k_coeffs];

                // Undo log-whitening and extract real parts
                let raw_coeffs: Vec<f64> = dim_spectrum
                    .iter()
                    .map(|c| {
                        let val = c.re; // DCT produces real coefficients
                        if config.spectral_whitening {
                            let sign = val.signum();
                            let log_mag = val.abs();
                            sign * (log_mag.exp() - 1.0)
                        } else {
                            val
                        }
                    })
                    .collect();

                // Undo amplitude normalization
                let norm_factor = if config.spectral_amplitude_norm {
                    (n_points as f64).sqrt()
                } else {
                    1.0
                };

                let denormalized: Vec<f64> = raw_coeffs.iter().map(|&c| c * norm_factor).collect();

                // Perform IDCT
                let reconstructed = compute_idct(&denormalized, n_points);

                for (i, &val) in reconstructed.iter().enumerate() {
                    result[i][dim] = val;
                }
            }
        }
        SpectralTransform::Fft => {
            // FFT inverse path
            let n_freq = n_points / 2 + 1;

            let mut planner = FftPlanner::new();
            let ifft = planner.plan_fft_inverse(n_points);

            for dim in 0..4 {
                let dim_start = dim * k_coeffs;
                let dim_spectrum = &spectrum[dim_start..dim_start + k_coeffs];

                // Undo log-whitening
                let raw_spectrum: Vec<Complex<f64>> = dim_spectrum
                    .iter()
                    .map(|c| {
                        if config.spectral_whitening {
                            let log_mag = c.norm();
                            let phase = c.arg();
                            let mag = log_mag.exp() - 1.0;
                            Complex::from_polar(mag.max(0.0), phase)
                        } else {
                            Complex::new(c.re, c.im)
                        }
                    })
                    .collect();

                // Pad to full spectrum size
                let mut buffer = vec![Complex::new(0.0, 0.0); n_freq];
                for (i, c) in raw_spectrum.iter().enumerate().take(n_freq) {
                    buffer[i] = *c;
                }

                // Undo amplitude normalization
                if config.spectral_amplitude_norm {
                    let length_factor = (n_points as f64).sqrt();
                    for c in &mut buffer {
                        *c *= length_factor;
                    }
                }

                // Build full spectrum for IFFT (conjugate symmetry for real signal)
                let mut full_buffer = vec![Complex::new(0.0, 0.0); n_points];
                for (i, c) in buffer.iter().enumerate() {
                    full_buffer[i] = *c;
                }
                // Mirror for conjugate symmetry (skip DC and Nyquist)
                for i in 1..n_freq - 1 {
                    if n_freq + i - 1 < n_points {
                        full_buffer[n_points - i] = buffer[i].conj();
                    }
                }

                // Perform IFFT
                ifft.process(&mut full_buffer);

                // Normalize
                let norm_factor = n_points as f64;
                for (i, c) in full_buffer.iter().enumerate() {
                    result[i][dim] = c.re / norm_factor * (n_points as f64).sqrt();
                }
            }
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::SpectralTransform;
    use approx::assert_relative_eq;

    fn generate_sine_trajectory(n: usize, freq: f64) -> Vec<[f64; 4]> {
        (0..n)
            .map(|i| {
                let t = i as f64 / n as f64;
                [
                    (2.0 * PI * freq * t).sin(),
                    (2.0 * PI * freq * t).cos(),
                    t,
                    0.0,
                ]
            })
            .collect()
    }

    fn generate_line_trajectory(n: usize) -> Vec<[f64; 4]> {
        (0..n)
            .map(|i| {
                let t = i as f64 / n as f64;
                [t * 10.0, t * 5.0, t * 2.0, t]
            })
            .collect()
    }

    #[test]
    fn test_spectral_features_basic() {
        let config = EmbeddingConfig::default();
        let points = generate_sine_trajectory(64, 2.0);

        let (spectrum, smoothness) = compute_spectral_features(&points, 16, &config);

        // Should have k_coeffs * 4 values
        assert_eq!(spectrum.len(), 64);

        // Smoothness should be in valid range [0, 1]
        assert!(smoothness >= 0.0 && smoothness <= 1.0);
    }

    #[test]
    fn test_fft_round_trip() {
        let mut config = EmbeddingConfig::default();
        config.spectral_whitening = false;
        config.spectral_amplitude_norm = false;
        config.spectral_transform = SpectralTransform::Fft;

        let n = 32;
        let points = generate_sine_trajectory(n, 1.0);

        // Use full spectrum for exact reconstruction
        let k = n / 2 + 1;
        let (spectrum, _) = compute_spectral_features(&points, k, &config);
        let reconstructed = inverse_spectral_features(&spectrum, n, &config);

        // Check reconstruction quality
        for i in 0..n {
            for d in 0..4 {
                assert_relative_eq!(points[i][d], reconstructed[i][d], epsilon = 0.5);
            }
        }
    }

    #[test]
    fn test_dct_round_trip_exact() {
        let mut config = EmbeddingConfig::default();
        config.spectral_whitening = false;
        config.spectral_amplitude_norm = false;
        config.spectral_transform = SpectralTransform::Dct;

        let n = 32;
        let points = generate_line_trajectory(n);

        // Use full DCT coefficients for exact reconstruction
        let k = n;
        let (spectrum, _) = compute_spectral_features(&points, k, &config);
        let reconstructed = inverse_spectral_features(&spectrum, n, &config);

        // DCT should reconstruct lines exactly
        for i in 0..n {
            for d in 0..4 {
                assert_relative_eq!(points[i][d], reconstructed[i][d], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_dct_better_for_lines() {
        // Compare FFT vs DCT reconstruction of a line
        let n = 64;
        let points = generate_line_trajectory(n);
        let k = 16; // Lossy compression

        // FFT reconstruction
        let mut fft_config = EmbeddingConfig::default();
        fft_config.spectral_whitening = false;
        fft_config.spectral_amplitude_norm = false;
        fft_config.spectral_transform = SpectralTransform::Fft;

        let (fft_spectrum, _) = compute_spectral_features(&points, k, &fft_config);
        let fft_recon = inverse_spectral_features(&fft_spectrum, n, &fft_config);

        // DCT reconstruction
        let mut dct_config = EmbeddingConfig::default();
        dct_config.spectral_whitening = false;
        dct_config.spectral_amplitude_norm = false;
        dct_config.spectral_transform = SpectralTransform::Dct;

        let (dct_spectrum, _) = compute_spectral_features(&points, k, &dct_config);
        let dct_recon = inverse_spectral_features(&dct_spectrum, n, &dct_config);

        // Compute RMSEs
        let fft_rmse: f64 = points.iter().zip(fft_recon.iter())
            .map(|(o, r)| {
                let mut sum = 0.0;
                for d in 0..4 {
                    sum += (o[d] - r[d]).powi(2);
                }
                sum
            })
            .sum::<f64>().sqrt() / (n as f64).sqrt();

        let dct_rmse: f64 = points.iter().zip(dct_recon.iter())
            .map(|(o, r)| {
                let mut sum = 0.0;
                for d in 0..4 {
                    sum += (o[d] - r[d]).powi(2);
                }
                sum
            })
            .sum::<f64>().sqrt() / (n as f64).sqrt();

        // DCT should be significantly better for lines
        assert!(dct_rmse < fft_rmse, "DCT RMSE ({}) should be < FFT RMSE ({}) for lines", dct_rmse, fft_rmse);
    }

    #[test]
    fn test_smoothness_jerky() {
        let config = EmbeddingConfig::default();

        // Create a jerky trajectory with sharp changes
        let points: Vec<[f64; 4]> = (0..64)
            .map(|i| {
                let t = i as f64 / 64.0;
                let noise = if i % 4 == 0 { 0.5 } else { 0.0 };
                [t + noise, 0.0, 0.0, 0.0]
            })
            .collect();

        let (_, smoothness) = compute_spectral_features(&points, 16, &config);

        // Jerky trajectory should have lower smoothness
        assert!(smoothness < 0.9);
    }

    #[test]
    fn test_window_functions() {
        let n = 100;

        // Test window generation
        let rect = generate_window(n, WindowFunction::None);
        assert!(rect.iter().all(|&w| (w - 1.0).abs() < 1e-10));

        let hanning = generate_window(n, WindowFunction::Hanning);
        assert!((hanning[0]).abs() < 1e-10); // Starts at 0
        assert!((hanning[n/2] - 1.0).abs() < 0.01); // Peak near middle
        assert!((hanning[n-1]).abs() < 1e-10); // Ends at 0

        let tukey = generate_window(n, WindowFunction::Tukey);
        assert!(tukey[n/2] > 0.99); // Flat in the middle
        assert!(tukey[0] < tukey[n/4]); // Tapers at start
    }

    #[test]
    fn test_dct_basic() {
        // Test DCT of simple signal
        let signal = vec![1.0, 2.0, 3.0, 4.0];
        let dct = compute_dct(&signal);

        // DCT should preserve energy
        let signal_energy: f64 = signal.iter().map(|x| x * x).sum();
        let dct_energy: f64 = dct.iter().map(|x| x * x).sum();
        assert_relative_eq!(signal_energy, dct_energy, epsilon = 1e-10);

        // Round-trip should be exact
        let idct = compute_idct(&dct, signal.len());
        for (i, &orig) in signal.iter().enumerate() {
            assert_relative_eq!(orig, idct[i], epsilon = 1e-10);
        }
    }
}
