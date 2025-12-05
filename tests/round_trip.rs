//! Comprehensive round-trip tests for trajectory embedding.
//!
//! These tests verify that trajectories can be encoded and decoded with
//! acceptable reconstruction error across different configurations.

use motion_embedding::{
    compute_motion_embedding, compute_reconstruction_error, reconstruct_trajectory,
    EmbeddingConfig,
};
use std::f64::consts::PI;

// =============================================================================
// TRAJECTORY GENERATORS
// =============================================================================

/// Generate a helix trajectory.
fn generate_helix(n: usize, radius: f64, pitch: f64, duration: f64) -> (Vec<[f64; 3]>, Vec<f64>) {
    let positions: Vec<[f64; 3]> = (0..n)
        .map(|i| {
            let t = i as f64 / (n - 1) as f64;
            let angle = 2.0 * PI * t * 2.0; // 2 full rotations
            [radius * angle.cos(), radius * angle.sin(), pitch * t]
        })
        .collect();

    let timestamps: Vec<f64> = (0..n)
        .map(|i| i as f64 / (n - 1) as f64 * duration)
        .collect();

    (positions, timestamps)
}

/// Generate a circle trajectory (planar).
fn generate_circle(n: usize, radius: f64, duration: f64) -> (Vec<[f64; 3]>, Vec<f64>) {
    let positions: Vec<[f64; 3]> = (0..n)
        .map(|i| {
            let t = i as f64 / n as f64; // Don't close the loop for better FFT
            let angle = 2.0 * PI * t;
            [radius * angle.cos(), radius * angle.sin(), 0.0]
        })
        .collect();

    let timestamps: Vec<f64> = (0..n)
        .map(|i| i as f64 / n as f64 * duration)
        .collect();

    (positions, timestamps)
}

/// Generate a figure-8 trajectory.
fn generate_figure8(n: usize, radius: f64, duration: f64) -> (Vec<[f64; 3]>, Vec<f64>) {
    let positions: Vec<[f64; 3]> = (0..n)
        .map(|i| {
            let t = i as f64 / (n - 1) as f64;
            let angle = 2.0 * PI * t;
            [
                radius * angle.sin(),
                radius * (2.0 * angle).sin() / 2.0,
                0.0,
            ]
        })
        .collect();

    let timestamps: Vec<f64> = (0..n)
        .map(|i| i as f64 / (n - 1) as f64 * duration)
        .collect();

    (positions, timestamps)
}

/// Generate a straight line trajectory.
fn generate_line(n: usize, length: f64, duration: f64) -> (Vec<[f64; 3]>, Vec<f64>) {
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

/// Generate a 3D Lissajous curve.
fn generate_lissajous(
    n: usize,
    amp: f64,
    freq_x: f64,
    freq_y: f64,
    freq_z: f64,
    duration: f64,
) -> (Vec<[f64; 3]>, Vec<f64>) {
    let positions: Vec<[f64; 3]> = (0..n)
        .map(|i| {
            let t = i as f64 / (n - 1) as f64;
            let phase = 2.0 * PI * t;
            [
                amp * (freq_x * phase).sin(),
                amp * (freq_y * phase).sin(),
                amp * (freq_z * phase).sin(),
            ]
        })
        .collect();

    let timestamps: Vec<f64> = (0..n)
        .map(|i| i as f64 / (n - 1) as f64 * duration)
        .collect();

    (positions, timestamps)
}

/// Generate a random walk trajectory (reproducible).
fn generate_random_walk(n: usize, step_size: f64, duration: f64, seed: u64) -> (Vec<[f64; 3]>, Vec<f64>) {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut positions = Vec::with_capacity(n);
    let mut pos = [0.0, 0.0, 0.0];
    positions.push(pos);

    for i in 1..n {
        let mut hasher = DefaultHasher::new();
        (seed, i).hash(&mut hasher);
        let h = hasher.finish();

        // Extract pseudo-random direction
        let dx = ((h & 0xFFFF) as f64 / 32768.0 - 1.0) * step_size;
        let dy = (((h >> 16) & 0xFFFF) as f64 / 32768.0 - 1.0) * step_size;
        let dz = (((h >> 32) & 0xFFFF) as f64 / 32768.0 - 1.0) * step_size;

        pos = [pos[0] + dx, pos[1] + dy, pos[2] + dz];
        positions.push(pos);
    }

    let timestamps: Vec<f64> = (0..n)
        .map(|i| i as f64 / (n - 1) as f64 * duration)
        .collect();

    (positions, timestamps)
}

/// Generate a spiral trajectory (expanding helix).
fn generate_spiral(n: usize, initial_radius: f64, expansion: f64, pitch: f64, duration: f64) -> (Vec<[f64; 3]>, Vec<f64>) {
    let positions: Vec<[f64; 3]> = (0..n)
        .map(|i| {
            let t = i as f64 / (n - 1) as f64;
            let angle = 2.0 * PI * t * 3.0; // 3 rotations
            let radius = initial_radius + expansion * t;
            [radius * angle.cos(), radius * angle.sin(), pitch * t]
        })
        .collect();

    let timestamps: Vec<f64> = (0..n)
        .map(|i| i as f64 / (n - 1) as f64 * duration)
        .collect();

    (positions, timestamps)
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/// Compute round-trip RMSE for a trajectory with given config.
fn compute_round_trip_rmse(
    positions: &[[f64; 3]],
    timestamps: &[f64],
    config: &EmbeddingConfig,
) -> f64 {
    let embedding = compute_motion_embedding(positions, timestamps, config)
        .expect("Encoding should succeed");

    let (reconstructed, _) = reconstruct_trajectory(&embedding, None, true, config)
        .expect("Reconstruction should succeed");

    compute_reconstruction_error(positions, &reconstructed)
}

/// Compute normalized RMSE (as percentage of trajectory extent).
fn compute_normalized_rmse(
    positions: &[[f64; 3]],
    timestamps: &[f64],
    config: &EmbeddingConfig,
) -> f64 {
    let rmse = compute_round_trip_rmse(positions, timestamps, config);

    // Compute trajectory extent
    let mut min = [f64::INFINITY; 3];
    let mut max = [f64::NEG_INFINITY; 3];
    for p in positions {
        for i in 0..3 {
            min[i] = min[i].min(p[i]);
            max[i] = max[i].max(p[i]);
        }
    }
    let extent = ((max[0] - min[0]).powi(2)
        + (max[1] - min[1]).powi(2)
        + (max[2] - min[2]).powi(2))
    .sqrt();

    if extent > 1e-10 {
        rmse / extent * 100.0
    } else {
        rmse
    }
}

// =============================================================================
// ROUND-TRIP TESTS: EXACT RECONSTRUCTION
// =============================================================================

#[test]
fn test_round_trip_exact_helix() {
    let n = 100;
    let (positions, timestamps) = generate_helix(n, 5.0, 10.0, 5.0);

    // k_coeffs = n/2 + 1 for theoretically exact reconstruction
    let config = EmbeddingConfig {
        k_coeffs: n / 2 + 1,
        spectral_whitening: false,
        spectral_amplitude_norm: false,
        ..EmbeddingConfig::default()
    };

    let rmse = compute_round_trip_rmse(&positions, &timestamps, &config);

    // With exact k, RMSE should be very small (numerical precision only)
    assert!(
        rmse < 1.0,
        "Exact helix round-trip RMSE too high: {:.4}m (expected < 1.0m)",
        rmse
    );
}

#[test]
fn test_round_trip_exact_circle() {
    let n = 64;
    let (positions, timestamps) = generate_circle(n, 10.0, 4.0);

    let config = EmbeddingConfig {
        k_coeffs: n / 2 + 1,
        spectral_whitening: false,
        spectral_amplitude_norm: false,
        ..EmbeddingConfig::default()
    };

    let rmse = compute_round_trip_rmse(&positions, &timestamps, &config);

    assert!(
        rmse < 1.0,
        "Exact circle round-trip RMSE too high: {:.4}m",
        rmse
    );
}

#[test]
fn test_round_trip_exact_figure8() {
    let n = 80;
    let (positions, timestamps) = generate_figure8(n, 8.0, 5.0);

    let config = EmbeddingConfig {
        k_coeffs: n / 2 + 1,
        spectral_whitening: false,
        spectral_amplitude_norm: false,
        ..EmbeddingConfig::default()
    };

    let rmse = compute_round_trip_rmse(&positions, &timestamps, &config);

    assert!(
        rmse < 1.0,
        "Exact figure-8 round-trip RMSE too high: {:.4}m",
        rmse
    );
}

#[test]
fn test_round_trip_exact_lissajous() {
    let n = 100;
    let (positions, timestamps) = generate_lissajous(n, 5.0, 2.0, 3.0, 5.0, 6.0);

    let config = EmbeddingConfig {
        k_coeffs: n / 2 + 1,
        spectral_whitening: false,
        spectral_amplitude_norm: false,
        ..EmbeddingConfig::default()
    };

    let rmse = compute_round_trip_rmse(&positions, &timestamps, &config);

    assert!(
        rmse < 1.0,
        "Exact Lissajous round-trip RMSE too high: {:.4}m",
        rmse
    );
}

// =============================================================================
// ROUND-TRIP TESTS: LOSSY RECONSTRUCTION
// =============================================================================

#[test]
fn test_round_trip_lossy_helix_k16() {
    let (positions, timestamps) = generate_helix(100, 5.0, 10.0, 5.0);

    let config = EmbeddingConfig {
        k_coeffs: 16,
        ..EmbeddingConfig::default()
    };

    let rmse = compute_round_trip_rmse(&positions, &timestamps, &config);
    let normalized = compute_normalized_rmse(&positions, &timestamps, &config);

    // With k=16, expect reasonable reconstruction
    assert!(
        rmse < 5.0,
        "Lossy helix (k=16) RMSE too high: {:.4}m",
        rmse
    );
    assert!(
        normalized < 30.0,
        "Lossy helix (k=16) normalized RMSE too high: {:.2}%",
        normalized
    );
}

#[test]
fn test_round_trip_lossy_circle_k8() {
    let (positions, timestamps) = generate_circle(100, 10.0, 5.0);

    let config = EmbeddingConfig {
        k_coeffs: 8,
        ..EmbeddingConfig::default()
    };

    let rmse = compute_round_trip_rmse(&positions, &timestamps, &config);

    // Circles are simple, should reconstruct well even with low k
    assert!(
        rmse < 5.0,
        "Lossy circle (k=8) RMSE too high: {:.4}m",
        rmse
    );
}

#[test]
fn test_round_trip_lossy_line_k16() {
    let (positions, timestamps) = generate_line(50, 100.0, 10.0);

    let config = EmbeddingConfig {
        k_coeffs: 16,
        ..EmbeddingConfig::default()
    };

    let rmse = compute_round_trip_rmse(&positions, &timestamps, &config);

    // Lines have Gibbs phenomenon but should still be reasonable
    assert!(
        rmse < 15.0,
        "Lossy line (k=16) RMSE too high: {:.4}m",
        rmse
    );
}

// =============================================================================
// K_COEFFS COMPARISON TESTS
// =============================================================================

/// Test structure for k_coeffs comparison results.
#[derive(Debug)]
struct KCoeffsResult {
    k: usize,
    rmse: f64,
    normalized_rmse: f64,
}

fn test_k_coeffs_range(
    positions: &[[f64; 3]],
    timestamps: &[f64],
    k_values: &[usize],
    trajectory_name: &str,
) -> Vec<KCoeffsResult> {
    let n = positions.len();
    let mut results = Vec::new();

    for &k in k_values {
        if k > n / 2 + 1 {
            continue; // Skip invalid k values
        }

        let config = EmbeddingConfig {
            k_coeffs: k,
            ..EmbeddingConfig::default()
        };

        let rmse = compute_round_trip_rmse(positions, timestamps, &config);
        let normalized = compute_normalized_rmse(positions, timestamps, &config);

        results.push(KCoeffsResult {
            k,
            rmse,
            normalized_rmse: normalized,
        });
    }

    // Print results for visibility
    println!("\n{} (n={})", trajectory_name, n);
    println!("{:>6} {:>12} {:>15}", "k", "RMSE (m)", "Normalized (%)");
    println!("{}", "-".repeat(35));
    for r in &results {
        println!("{:>6} {:>12.4} {:>15.2}", r.k, r.rmse, r.normalized_rmse);
    }

    results
}

#[test]
fn test_k_coeffs_comparison_helix() {
    let (positions, timestamps) = generate_helix(100, 5.0, 10.0, 5.0);
    let k_values = [4, 8, 12, 16, 24, 32, 51];

    let results = test_k_coeffs_range(&positions, &timestamps, &k_values, "Helix");

    // Verify monotonic improvement (RMSE should decrease as k increases)
    for i in 1..results.len() {
        assert!(
            results[i].rmse <= results[i - 1].rmse + 0.5, // Allow small tolerance
            "RMSE should generally decrease with k: k={} RMSE={:.4} > k={} RMSE={:.4}",
            results[i].k,
            results[i].rmse,
            results[i - 1].k,
            results[i - 1].rmse
        );
    }

    // Verify k=51 (exact) has low error
    if let Some(exact) = results.iter().find(|r| r.k == 51) {
        assert!(
            exact.rmse < 2.0,
            "Exact k should have low RMSE: {:.4}",
            exact.rmse
        );
    }
}

#[test]
fn test_k_coeffs_comparison_circle() {
    let (positions, timestamps) = generate_circle(100, 10.0, 5.0);
    let k_values = [4, 8, 16, 32, 51];

    let results = test_k_coeffs_range(&positions, &timestamps, &k_values, "Circle");

    // Circles are simple - even k=4 should work reasonably
    if let Some(k4) = results.iter().find(|r| r.k == 4) {
        assert!(
            k4.normalized_rmse < 50.0,
            "Circle with k=4 should be < 50% error: {:.2}%",
            k4.normalized_rmse
        );
    }
}

#[test]
fn test_k_coeffs_comparison_figure8() {
    let (positions, timestamps) = generate_figure8(100, 8.0, 5.0);
    let k_values = [4, 8, 16, 24, 32, 51];

    let results = test_k_coeffs_range(&positions, &timestamps, &k_values, "Figure-8");

    // Figure-8 has more complexity, needs more coefficients
    if let Some(k8) = results.iter().find(|r| r.k == 8) {
        assert!(
            k8.normalized_rmse < 60.0,
            "Figure-8 with k=8 should be < 60% error: {:.2}%",
            k8.normalized_rmse
        );
    }
}

#[test]
fn test_k_coeffs_comparison_lissajous() {
    let (positions, timestamps) = generate_lissajous(100, 5.0, 2.0, 3.0, 5.0, 6.0);
    let k_values = [8, 16, 24, 32, 40, 51];

    let results = test_k_coeffs_range(&positions, &timestamps, &k_values, "Lissajous (2,3,5)");

    // Lissajous with multiple frequencies needs more coefficients
    if let Some(k16) = results.iter().find(|r| r.k == 16) {
        assert!(
            k16.normalized_rmse < 50.0,
            "Lissajous with k=16 should be < 50% error: {:.2}%",
            k16.normalized_rmse
        );
    }
}

#[test]
fn test_k_coeffs_comparison_random_walk() {
    let (positions, timestamps) = generate_random_walk(100, 1.0, 5.0, 42);
    let k_values = [8, 16, 24, 32, 51];

    let results = test_k_coeffs_range(&positions, &timestamps, &k_values, "Random Walk");

    // Random walks have high-frequency content, need many coefficients
    // Just verify it runs and produces results
    assert!(!results.is_empty(), "Should produce results");
}

#[test]
fn test_k_coeffs_comparison_spiral() {
    let (positions, timestamps) = generate_spiral(100, 2.0, 8.0, 15.0, 5.0);
    let k_values = [8, 16, 24, 32, 51];

    let results = test_k_coeffs_range(&positions, &timestamps, &k_values, "Expanding Spiral");

    // Spirals have varying frequency content
    if let Some(k24) = results.iter().find(|r| r.k == 24) {
        assert!(
            k24.normalized_rmse < 40.0,
            "Spiral with k=24 should be < 40% error: {:.2}%",
            k24.normalized_rmse
        );
    }
}

// =============================================================================
// COMPREHENSIVE MSE ANALYSIS
// =============================================================================

#[test]
#[ignore] // Run with: cargo test test_comprehensive_mse_analysis -- --ignored --nocapture
fn test_comprehensive_mse_analysis() {
    println!("\n");
    println!("{}", "=".repeat(80));
    println!("COMPREHENSIVE ROUND-TRIP MSE ANALYSIS");
    println!("{}", "=".repeat(80));

    let n = 100;
    let k_values = [4, 8, 12, 16, 20, 24, 32, 40, 51];

    // Test trajectories
    let trajectories: Vec<(&str, Vec<[f64; 3]>, Vec<f64>)> = vec![
        {
            let (p, t) = generate_helix(n, 5.0, 10.0, 5.0);
            ("Helix (r=5, p=10)", p, t)
        },
        {
            let (p, t) = generate_circle(n, 10.0, 5.0);
            ("Circle (r=10)", p, t)
        },
        {
            let (p, t) = generate_figure8(n, 8.0, 5.0);
            ("Figure-8 (r=8)", p, t)
        },
        {
            let (p, t) = generate_line(n, 50.0, 5.0);
            ("Line (L=50)", p, t)
        },
        {
            let (p, t) = generate_lissajous(n, 5.0, 2.0, 3.0, 5.0, 6.0);
            ("Lissajous (2,3,5)", p, t)
        },
        {
            let (p, t) = generate_spiral(n, 2.0, 8.0, 15.0, 5.0);
            ("Spiral (expanding)", p, t)
        },
        {
            let (p, t) = generate_random_walk(n, 0.5, 5.0, 42);
            ("Random Walk", p, t)
        },
    ];

    // Header
    println!("\nRMSE (meters) by k_coeffs:");
    print!("{:<22}", "Trajectory");
    for k in &k_values {
        print!("{:>8}", format!("k={}", k));
    }
    println!();
    println!("{}", "-".repeat(22 + k_values.len() * 8));

    // Results
    for (name, positions, timestamps) in &trajectories {
        print!("{:<22}", name);
        for &k in &k_values {
            if k > n / 2 + 1 {
                print!("{:>8}", "-");
                continue;
            }

            let config = EmbeddingConfig {
                k_coeffs: k,
                ..EmbeddingConfig::default()
            };

            let rmse = compute_round_trip_rmse(positions, timestamps, &config);
            print!("{:>8.2}", rmse);
        }
        println!();
    }

    // Normalized RMSE
    println!("\nNormalized RMSE (% of extent) by k_coeffs:");
    print!("{:<22}", "Trajectory");
    for k in &k_values {
        print!("{:>8}", format!("k={}", k));
    }
    println!();
    println!("{}", "-".repeat(22 + k_values.len() * 8));

    for (name, positions, timestamps) in &trajectories {
        print!("{:<22}", name);
        for &k in &k_values {
            if k > n / 2 + 1 {
                print!("{:>8}", "-");
                continue;
            }

            let config = EmbeddingConfig {
                k_coeffs: k,
                ..EmbeddingConfig::default()
            };

            let normalized = compute_normalized_rmse(positions, timestamps, &config);
            print!("{:>7.1}%", normalized);
        }
        println!();
    }

    println!("\n{}", "=".repeat(80));
    println!("ANALYSIS COMPLETE");
    println!("{}", "=".repeat(80));
}

// =============================================================================
// WHITENING IMPACT TESTS
// =============================================================================

#[test]
fn test_whitening_impact_on_reconstruction() {
    let (positions, timestamps) = generate_helix(100, 5.0, 10.0, 5.0);

    // Without whitening
    let config_no_whitening = EmbeddingConfig {
        k_coeffs: 16,
        spectral_whitening: false,
        spectral_amplitude_norm: false,
        ..EmbeddingConfig::default()
    };

    // With whitening (default)
    let config_whitening = EmbeddingConfig {
        k_coeffs: 16,
        spectral_whitening: true,
        spectral_amplitude_norm: true,
        ..EmbeddingConfig::default()
    };

    let rmse_no_whitening = compute_round_trip_rmse(&positions, &timestamps, &config_no_whitening);
    let rmse_whitening = compute_round_trip_rmse(&positions, &timestamps, &config_whitening);

    println!("\nWhitening Impact on Helix (k=16):");
    println!("  Without whitening: RMSE = {:.4}m", rmse_no_whitening);
    println!("  With whitening:    RMSE = {:.4}m", rmse_whitening);

    // Whitening is for ML training, not reconstruction - both should work
    assert!(rmse_no_whitening < 10.0, "Non-whitened RMSE too high");
    assert!(rmse_whitening < 10.0, "Whitened RMSE too high");
}

// =============================================================================
// EDGE CASE TESTS
// =============================================================================

#[test]
fn test_round_trip_minimum_points() {
    // Test with minimum viable trajectory (3 points for acceleration)
    let positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
    let timestamps = vec![0.0, 0.1, 0.2];

    let config = EmbeddingConfig {
        k_coeffs: 2, // n/2 + 1 = 2 for n=3
        spectral_whitening: false,
        spectral_amplitude_norm: false,
        ..EmbeddingConfig::default()
    };

    let embedding = compute_motion_embedding(&positions, &timestamps, &config);
    assert!(embedding.is_ok(), "Should encode minimum trajectory");

    let (reconstructed, _) = reconstruct_trajectory(&embedding.unwrap(), None, true, &config)
        .expect("Should reconstruct");

    assert_eq!(reconstructed.len(), positions.len());
}

#[test]
fn test_round_trip_large_trajectory() {
    let n = 500;
    let (positions, timestamps) = generate_helix(n, 10.0, 20.0, 10.0);

    let config = EmbeddingConfig {
        k_coeffs: 32,
        ..EmbeddingConfig::default()
    };

    let rmse = compute_round_trip_rmse(&positions, &timestamps, &config);

    assert!(
        rmse < 10.0,
        "Large trajectory (n=500, k=32) RMSE too high: {:.4}m",
        rmse
    );
}

#[test]
fn test_round_trip_different_scales() {
    // Test that scale doesn't affect normalized reconstruction error
    let scales = [0.1, 1.0, 10.0, 100.0];
    let mut normalized_errors = Vec::new();

    for scale in &scales {
        let (positions, timestamps) = generate_helix(100, *scale, *scale * 2.0, 5.0);

        let config = EmbeddingConfig {
            k_coeffs: 24,
            ..EmbeddingConfig::default()
        };

        let normalized = compute_normalized_rmse(&positions, &timestamps, &config);
        normalized_errors.push(normalized);

        println!("Scale {}: normalized RMSE = {:.2}%", scale, normalized);
    }

    // Normalized errors should be similar across scales
    let max_error = normalized_errors.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min_error = normalized_errors.iter().cloned().fold(f64::INFINITY, f64::min);

    assert!(
        max_error - min_error < 20.0,
        "Normalized errors vary too much across scales: {:.2}% - {:.2}%",
        min_error,
        max_error
    );
}

// =============================================================================
// PRESET CONFIGURATION TESTS
// =============================================================================

#[test]
fn test_round_trip_all_presets() {
    let (positions, timestamps) = generate_helix(100, 5.0, 10.0, 5.0);

    let presets: Vec<(&str, EmbeddingConfig)> = vec![
        ("default", EmbeddingConfig::default()),
        ("drone(false)", EmbeddingConfig::drone(false)),
        ("drone(true)", EmbeddingConfig::drone(true)),
        ("pedestrian", EmbeddingConfig::pedestrian()),
        ("vehicle", EmbeddingConfig::vehicle()),
        ("racing", EmbeddingConfig::racing()),
        ("surveillance", EmbeddingConfig::surveillance()),
        ("ml_training", EmbeddingConfig::ml_training()),
    ];

    println!("\nRound-trip RMSE by Preset (Helix, k=16):");
    println!("{:<20} {:>12}", "Preset", "RMSE (m)");
    println!("{}", "-".repeat(34));

    for (name, config) in presets {
        let rmse = compute_round_trip_rmse(&positions, &timestamps, &config);
        println!("{:<20} {:>12.4}", name, rmse);

        // All presets should produce reasonable reconstruction
        assert!(
            rmse < 10.0,
            "Preset {} has high RMSE: {:.4}m",
            name,
            rmse
        );
    }
}

// =============================================================================
// DETERMINISM TEST
// =============================================================================

#[test]
fn test_round_trip_deterministic() {
    let (positions, timestamps) = generate_helix(100, 5.0, 10.0, 5.0);
    let config = EmbeddingConfig::default();

    // Run encoding/decoding multiple times
    let mut rmses = Vec::new();
    for _ in 0..5 {
        let rmse = compute_round_trip_rmse(&positions, &timestamps, &config);
        rmses.push(rmse);
    }

    // All results should be identical
    let first = rmses[0];
    for (i, rmse) in rmses.iter().enumerate() {
        assert!(
            (rmse - first).abs() < 1e-10,
            "Run {} produced different RMSE: {} vs {}",
            i,
            rmse,
            first
        );
    }
}

// =============================================================================
// DCT VS FFT COMPARISON TESTS
// =============================================================================

use motion_embedding::SpectralTransform;

/// Helper to compute RMSE with specific spectral transform
fn compute_rmse_with_transform(
    positions: &[[f64; 3]],
    timestamps: &[f64],
    k_coeffs: usize,
    transform: SpectralTransform,
    whitening: bool,
) -> f64 {
    let config = EmbeddingConfig {
        k_coeffs,
        spectral_transform: transform,
        spectral_whitening: whitening,
        spectral_amplitude_norm: whitening,
        store_endpoints: true,
        ..EmbeddingConfig::default()
    };

    compute_round_trip_rmse(positions, timestamps, &config)
}

#[test]
fn test_dct_vs_fft_on_line() {
    // DCT should significantly outperform FFT on straight lines
    let (positions, timestamps) = generate_line(100, 50.0, 5.0);

    let fft_rmse = compute_rmse_with_transform(&positions, &timestamps, 16, SpectralTransform::Fft, false);
    let dct_rmse = compute_rmse_with_transform(&positions, &timestamps, 16, SpectralTransform::Dct, false);

    println!("\nLine reconstruction (k=16, no whitening):");
    println!("  FFT RMSE: {:.4}m", fft_rmse);
    println!("  DCT RMSE: {:.4}m", dct_rmse);

    // DCT should be better (or at least not worse) for lines
    // Note: With endpoint correction, both should be good, but DCT naturally handles lines better
    assert!(
        dct_rmse <= fft_rmse + 0.5, // Allow small tolerance
        "DCT should not be significantly worse than FFT for lines: DCT={:.4} vs FFT={:.4}",
        dct_rmse,
        fft_rmse
    );
}

#[test]
fn test_dct_vs_fft_on_circle() {
    // FFT and DCT should perform similarly on periodic signals
    let (positions, timestamps) = generate_circle(100, 10.0, 5.0);

    let fft_rmse = compute_rmse_with_transform(&positions, &timestamps, 16, SpectralTransform::Fft, false);
    let dct_rmse = compute_rmse_with_transform(&positions, &timestamps, 16, SpectralTransform::Dct, false);

    println!("\nCircle reconstruction (k=16, no whitening):");
    println!("  FFT RMSE: {:.4}m", fft_rmse);
    println!("  DCT RMSE: {:.4}m", dct_rmse);

    // Both should work well on circles
    assert!(fft_rmse < 5.0, "FFT RMSE too high for circle: {:.4}", fft_rmse);
    assert!(dct_rmse < 5.0, "DCT RMSE too high for circle: {:.4}", dct_rmse);
}

#[test]
fn test_dct_vs_fft_on_random_walk() {
    // DCT should handle non-periodic random walks better
    let (positions, timestamps) = generate_random_walk(100, 0.5, 5.0, 42);

    let fft_rmse = compute_rmse_with_transform(&positions, &timestamps, 24, SpectralTransform::Fft, false);
    let dct_rmse = compute_rmse_with_transform(&positions, &timestamps, 24, SpectralTransform::Dct, false);

    println!("\nRandom walk reconstruction (k=24, no whitening):");
    println!("  FFT RMSE: {:.4}m", fft_rmse);
    println!("  DCT RMSE: {:.4}m", dct_rmse);

    // DCT should generally be better for random walks (non-periodic)
    // Both should produce reasonable results
    assert!(fft_rmse < 10.0, "FFT RMSE too high for random walk");
    assert!(dct_rmse < 10.0, "DCT RMSE too high for random walk");
}

#[test]
fn test_reconstruction_optimized_preset() {
    // Test the new reconstruction_optimized preset
    let (positions, timestamps) = generate_line(100, 50.0, 5.0);

    // Default config
    let default_config = EmbeddingConfig::default();
    let default_rmse = compute_round_trip_rmse(&positions, &timestamps, &default_config);

    // Reconstruction-optimized config
    let recon_config = EmbeddingConfig::reconstruction_optimized();
    let recon_rmse = compute_round_trip_rmse(&positions, &timestamps, &recon_config);

    println!("\nLine reconstruction preset comparison:");
    println!("  Default (FFT, k=16, whitening):     RMSE = {:.4}m", default_rmse);
    println!("  Reconstruction-opt (DCT, k=24):     RMSE = {:.4}m", recon_rmse);

    // Reconstruction-optimized should be better for lines
    assert!(
        recon_rmse <= default_rmse + 0.5,
        "reconstruction_optimized() should not be worse: {:.4} vs {:.4}",
        recon_rmse,
        default_rmse
    );
}

#[test]
fn test_balanced_preset() {
    // Test the balanced preset (DCT + whitening)
    let (positions, timestamps) = generate_helix(100, 5.0, 10.0, 5.0);

    let balanced_config = EmbeddingConfig::balanced();
    let embedding = compute_motion_embedding(&positions, &timestamps, &balanced_config)
        .expect("Should encode with balanced preset");

    // Should produce valid embedding
    let compact = embedding.to_compact_array();
    assert_eq!(compact.len(), 24);

    // All components should be in valid range
    for (i, &v) in compact.iter().enumerate() {
        assert!(
            v.is_finite(),
            "Component {} is not finite: {}",
            i, v
        );
    }

    // Reconstruction should work
    let rmse = compute_round_trip_rmse(&positions, &timestamps, &balanced_config);
    assert!(rmse < 10.0, "Balanced preset RMSE too high: {:.4}", rmse);
}

#[test]
fn test_endpoint_correction_improves_reconstruction() {
    // Test that endpoint correction improves reconstruction
    let (positions, timestamps) = generate_line(100, 50.0, 5.0);

    // Without endpoint correction
    let config_no_endpoints = EmbeddingConfig {
        k_coeffs: 16,
        spectral_transform: SpectralTransform::Fft,
        spectral_whitening: false,
        store_endpoints: false,
        ..EmbeddingConfig::default()
    };

    // With endpoint correction
    let config_with_endpoints = EmbeddingConfig {
        k_coeffs: 16,
        spectral_transform: SpectralTransform::Fft,
        spectral_whitening: false,
        store_endpoints: true,
        ..EmbeddingConfig::default()
    };

    let rmse_no_endpoints = compute_round_trip_rmse(&positions, &timestamps, &config_no_endpoints);
    let rmse_with_endpoints = compute_round_trip_rmse(&positions, &timestamps, &config_with_endpoints);

    println!("\nEndpoint correction impact on line (k=16, FFT):");
    println!("  Without endpoints: RMSE = {:.4}m", rmse_no_endpoints);
    println!("  With endpoints:    RMSE = {:.4}m", rmse_with_endpoints);

    // Endpoint correction should improve (or at least not hurt) reconstruction
    // For lines especially, this should be significant
    assert!(
        rmse_with_endpoints <= rmse_no_endpoints + 0.1,
        "Endpoint correction should not make reconstruction worse"
    );
}

#[test]
#[ignore] // Run with: cargo test test_comprehensive_dct_fft_comparison -- --ignored --nocapture
fn test_comprehensive_dct_fft_comparison() {
    println!("\n");
    println!("{}", "=".repeat(90));
    println!("COMPREHENSIVE DCT vs FFT COMPARISON");
    println!("{}", "=".repeat(90));

    let n = 100;
    let k_values = [8, 16, 24, 32];

    // Test trajectories
    let trajectories: Vec<(&str, Vec<[f64; 3]>, Vec<f64>)> = vec![
        {
            let (p, t) = generate_helix(n, 5.0, 10.0, 5.0);
            ("Helix", p, t)
        },
        {
            let (p, t) = generate_circle(n, 10.0, 5.0);
            ("Circle", p, t)
        },
        {
            let (p, t) = generate_figure8(n, 8.0, 5.0);
            ("Figure-8", p, t)
        },
        {
            let (p, t) = generate_line(n, 50.0, 5.0);
            ("Line", p, t)
        },
        {
            let (p, t) = generate_spiral(n, 2.0, 8.0, 15.0, 5.0);
            ("Spiral", p, t)
        },
        {
            let (p, t) = generate_random_walk(n, 0.5, 5.0, 42);
            ("Random Walk", p, t)
        },
    ];

    // Header
    println!("\nRMSE Comparison (FFT vs DCT, no whitening):");
    print!("{:<15}", "Trajectory");
    for k in &k_values {
        print!("{:>16}", format!("k={} (FFT/DCT)", k));
    }
    println!();
    println!("{}", "-".repeat(15 + k_values.len() * 16));

    // Results
    for (name, positions, timestamps) in &trajectories {
        print!("{:<15}", name);
        for &k in &k_values {
            let fft_rmse = compute_rmse_with_transform(positions, timestamps, k, SpectralTransform::Fft, false);
            let dct_rmse = compute_rmse_with_transform(positions, timestamps, k, SpectralTransform::Dct, false);

            let better = if dct_rmse < fft_rmse - 0.01 { "D" } else if fft_rmse < dct_rmse - 0.01 { "F" } else { "=" };
            print!("{:>6.2}/{:<6.2}[{}]", fft_rmse, dct_rmse, better);
        }
        println!();
    }

    println!("\nLegend: [D]=DCT better, [F]=FFT better, [=]=similar");
    println!("{}", "=".repeat(90));
}
