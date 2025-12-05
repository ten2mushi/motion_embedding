//! Test that exports embeddings to JSON for Python analysis.
//!
//! Run with: cargo test --test export_embeddings -- --ignored --nocapture

use motion_embedding::{compute_motion_embedding, EmbeddingConfig, SpectralTransform};
use serde::Serialize;
use std::f64::consts::PI;
use std::fs::File;
use std::io::Write;

#[derive(Serialize)]
struct EmbeddingExport {
    compact: [f64; 24],
    full: Vec<f64>,  // 24 + spectral features (for FFT vs DCT comparison)
    label: String,
    trajectory_type: String,
    spectral_transform: String,
    params: TrajectoryParams,
}

#[derive(Serialize, Clone)]
struct TrajectoryParams {
    n_points: usize,
    duration: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    radius: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pitch: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    speed: Option<f64>,
}

#[derive(Serialize)]
struct ExportData {
    embeddings: Vec<EmbeddingExport>,
    config: ConfigExport,
}

#[derive(Serialize)]
struct ConfigExport {
    characteristic_speed: f64,
    k_coeffs: usize,
    alpha: f64,
    spectral_transform: String,
}

/// Generate a helix trajectory (like a drone spiraling up).
fn generate_helix(n: usize, radius: f64, pitch: f64, duration: f64) -> (Vec<[f64; 3]>, Vec<f64>) {
    let positions: Vec<[f64; 3]> = (0..n)
        .map(|i| {
            let t = i as f64 / (n - 1) as f64;
            let angle = 2.0 * PI * t * 2.0; // 2 full rotations
            [
                radius * angle.cos(),
                radius * angle.sin(),
                pitch * t * duration,
            ]
        })
        .collect();

    let timestamps: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64 * duration).collect();
    (positions, timestamps)
}

/// Generate a circular trajectory (horizontal loop).
fn generate_circle(n: usize, radius: f64, duration: f64) -> (Vec<[f64; 3]>, Vec<f64>) {
    let positions: Vec<[f64; 3]> = (0..n)
        .map(|i| {
            let t = i as f64 / (n - 1) as f64;
            let angle = 2.0 * PI * t;
            [radius * angle.cos(), radius * angle.sin(), 0.0]
        })
        .collect();

    let timestamps: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64 * duration).collect();
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

    let timestamps: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64 * duration).collect();
    (positions, timestamps)
}

/// Generate a straight line trajectory.
fn generate_line(n: usize, length: f64, duration: f64, direction: [f64; 3]) -> (Vec<[f64; 3]>, Vec<f64>) {
    let mag = (direction[0].powi(2) + direction[1].powi(2) + direction[2].powi(2)).sqrt();
    let dir = [direction[0] / mag, direction[1] / mag, direction[2] / mag];

    let positions: Vec<[f64; 3]> = (0..n)
        .map(|i| {
            let t = i as f64 / (n - 1) as f64;
            [
                dir[0] * length * t,
                dir[1] * length * t,
                dir[2] * length * t,
            ]
        })
        .collect();

    let timestamps: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64 * duration).collect();
    (positions, timestamps)
}

/// Generate a random walk trajectory.
fn generate_random_walk(n: usize, step_size: f64, duration: f64, seed: u64) -> (Vec<[f64; 3]>, Vec<f64>) {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut pos = [0.0, 0.0, 0.0];
    let mut positions = Vec::with_capacity(n);
    positions.push(pos);

    for i in 1..n {
        let mut hasher = DefaultHasher::new();
        (seed, i).hash(&mut hasher);
        let h = hasher.finish();

        // Generate pseudo-random direction
        let theta = ((h & 0xFFFF) as f64 / 65535.0) * 2.0 * PI;
        let phi = (((h >> 16) & 0xFFFF) as f64 / 65535.0) * PI;

        pos[0] += step_size * phi.sin() * theta.cos();
        pos[1] += step_size * phi.sin() * theta.sin();
        pos[2] += step_size * phi.cos();
        positions.push(pos);
    }

    let timestamps: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64 * duration).collect();
    (positions, timestamps)
}

/// Generate an aggressive maneuver (sharp turns).
fn generate_aggressive(n: usize, radius: f64, duration: f64) -> (Vec<[f64; 3]>, Vec<f64>) {
    let positions: Vec<[f64; 3]> = (0..n)
        .map(|i| {
            let t = i as f64 / (n - 1) as f64;
            // Square wave-ish pattern with sharp reversals
            let phase = (t * 4.0).floor() as i32 % 4;
            let local_t = (t * 4.0).fract();

            match phase {
                0 => [radius * local_t, 0.0, local_t * 2.0],
                1 => [radius, radius * local_t, 2.0 + local_t * 2.0],
                2 => [radius * (1.0 - local_t), radius, 4.0 + local_t * 2.0],
                _ => [0.0, radius * (1.0 - local_t), 6.0 + local_t * 2.0],
            }
        })
        .collect();

    let timestamps: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64 * duration).collect();
    (positions, timestamps)
}

/// Generate hover/station-keeping trajectory (small movements around a point).
fn generate_hover(n: usize, jitter: f64, duration: f64, seed: u64) -> (Vec<[f64; 3]>, Vec<f64>) {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let positions: Vec<[f64; 3]> = (0..n)
        .map(|i| {
            let mut hasher = DefaultHasher::new();
            (seed, i).hash(&mut hasher);
            let h = hasher.finish();

            [
                ((h & 0xFF) as f64 / 255.0 - 0.5) * jitter,
                (((h >> 8) & 0xFF) as f64 / 255.0 - 0.5) * jitter,
                (((h >> 16) & 0xFF) as f64 / 255.0 - 0.5) * jitter,
            ]
        })
        .collect();

    let timestamps: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64 * duration).collect();
    (positions, timestamps)
}

/// Generate scaled helix at different sizes (for scale diversity).
fn generate_scaled_helix(n: usize, scale: f64, duration: f64) -> (Vec<[f64; 3]>, Vec<f64>) {
    let positions: Vec<[f64; 3]> = (0..n)
        .map(|i| {
            let t = i as f64 / (n - 1) as f64;
            let angle = 2.0 * PI * t * 2.0;
            [
                scale * angle.cos(),
                scale * angle.sin(),
                scale * t * 2.0,
            ]
        })
        .collect();

    let timestamps: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64 * duration).collect();
    (positions, timestamps)
}

/// Generate jerky trajectory with HIGH-FREQUENCY NOISE (for smoothness diversity).
fn generate_jerky_trajectory(n: usize, radius: f64, duration: f64, noise_amplitude: f64, seed: u64) -> (Vec<[f64; 3]>, Vec<f64>) {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let positions: Vec<[f64; 3]> = (0..n)
        .map(|i| {
            let t = i as f64 / (n - 1) as f64;
            let angle = 2.0 * PI * t * 2.0;

            // Add high-frequency random noise at EVERY point
            let mut hasher = DefaultHasher::new();
            (seed, i).hash(&mut hasher);
            let h = hasher.finish();

            // Extract 3 independent noise values
            let noise_x = ((h & 0xFFFF) as f64 / 65535.0 - 0.5) * noise_amplitude;
            let noise_y = (((h >> 16) & 0xFFFF) as f64 / 65535.0 - 0.5) * noise_amplitude;
            let noise_z = (((h >> 32) & 0xFFFF) as f64 / 65535.0 - 0.5) * noise_amplitude;

            [
                radius * angle.cos() + noise_x,
                radius * angle.sin() + noise_y,
                radius * t * 2.0 + noise_z,
            ]
        })
        .collect();

    let timestamps: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64 * duration).collect();
    (positions, timestamps)
}

/// Generate ultra-smooth trajectory (for smoothness diversity - smooth end).
fn generate_smooth_arc(n: usize, radius: f64, duration: f64) -> (Vec<[f64; 3]>, Vec<f64>) {
    // Simple smooth parabolic arc - minimal high-frequency content
    let positions: Vec<[f64; 3]> = (0..n)
        .map(|i| {
            let t = i as f64 / (n - 1) as f64;
            let x = t * radius * 2.0;
            let y = -4.0 * radius * t * (t - 1.0); // Parabola
            let z = t * radius * 0.5;
            [x, y, z]
        })
        .collect();

    let timestamps: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64 * duration).collect();
    (positions, timestamps)
}

/// Generate left-handed helix (for chirality diversity).
fn generate_left_helix(n: usize, radius: f64, pitch: f64, duration: f64) -> (Vec<[f64; 3]>, Vec<f64>) {
    let positions: Vec<[f64; 3]> = (0..n)
        .map(|i| {
            let t = i as f64 / (n - 1) as f64;
            let angle = -2.0 * PI * t * 2.0; // Negative angle for left-handed
            [
                radius * angle.cos(),
                radius * angle.sin(),
                pitch * t,
            ]
        })
        .collect();

    let timestamps: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64 * duration).collect();
    (positions, timestamps)
}

/// Generate an expanding spiral (radius increases with time).
fn generate_spiral(n: usize, start_radius: f64, end_radius: f64, pitch: f64, duration: f64) -> (Vec<[f64; 3]>, Vec<f64>) {
    let positions: Vec<[f64; 3]> = (0..n)
        .map(|i| {
            let t = i as f64 / (n - 1) as f64;
            let radius = start_radius + (end_radius - start_radius) * t;
            let angle = 2.0 * PI * t * 3.0; // 3 full rotations
            [
                radius * angle.cos(),
                radius * angle.sin(),
                pitch * t,
            ]
        })
        .collect();

    let timestamps: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64 * duration).collect();
    (positions, timestamps)
}

/// Generate a 3D Lissajous curve.
fn generate_lissajous(n: usize, amp: f64, freq_x: f64, freq_y: f64, freq_z: f64, duration: f64) -> (Vec<[f64; 3]>, Vec<f64>) {
    let positions: Vec<[f64; 3]> = (0..n)
        .map(|i| {
            let t = i as f64 / (n - 1) as f64;
            let phase = 2.0 * PI * t;
            [
                amp * (freq_x * phase).sin(),
                amp * (freq_y * phase + PI / 4.0).sin(),
                amp * (freq_z * phase + PI / 2.0).sin(),
            ]
        })
        .collect();

    let timestamps: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64 * duration).collect();
    (positions, timestamps)
}

/// Helper to encode a trajectory with a specific config and return the export struct.
fn encode_trajectory(
    positions: &[[f64; 3]],
    timestamps: &[f64],
    config: &EmbeddingConfig,
    label: String,
    trajectory_type: String,
    params: TrajectoryParams,
) -> Option<EmbeddingExport> {
    compute_motion_embedding(positions, timestamps, config)
        .ok()
        .map(|emb| EmbeddingExport {
            compact: emb.to_compact_array(),
            full: emb.to_full_vec(true),  // Include spectral features
            label,
            trajectory_type,
            spectral_transform: match config.spectral_transform {
                SpectralTransform::Fft => "FFT".to_string(),
                SpectralTransform::Dct => "DCT".to_string(),
            },
            params,
        })
}

#[test]
#[ignore] // Run manually with: cargo test --test export_embeddings -- --ignored --nocapture
fn export_embeddings_to_json() {
    // Create both FFT and DCT configs for comparison
    let fft_config = EmbeddingConfig::default();
    let dct_config = EmbeddingConfig::balanced(); // Uses DCT

    let mut embeddings = Vec::new();
    let n_points = 100;

    // ========== HELIX ==========
    for (i, (radius, pitch)) in [(5.0, 10.0), (10.0, 5.0), (3.0, 15.0), (8.0, 8.0), (6.0, 12.0)].iter().enumerate() {
        let duration = 5.0;
        let (positions, timestamps) = generate_helix(n_points, *radius, *pitch, duration);
        let params = TrajectoryParams { n_points, duration, radius: Some(*radius), pitch: Some(*pitch), speed: None };

        if let Some(exp) = encode_trajectory(&positions, &timestamps, &fft_config, format!("helix_fft_{}", i), "helix".to_string(), params.clone()) {
            embeddings.push(exp);
        }
        if let Some(exp) = encode_trajectory(&positions, &timestamps, &dct_config, format!("helix_dct_{}", i), "helix".to_string(), params) {
            embeddings.push(exp);
        }
    }

    // ========== CIRCLE ==========
    for (i, radius) in [3.0, 5.0, 8.0, 12.0, 15.0].iter().enumerate() {
        let duration = 4.0;
        let (positions, timestamps) = generate_circle(n_points, *radius, duration);
        let params = TrajectoryParams { n_points, duration, radius: Some(*radius), pitch: None, speed: None };

        if let Some(exp) = encode_trajectory(&positions, &timestamps, &fft_config, format!("circle_fft_{}", i), "circle".to_string(), params.clone()) {
            embeddings.push(exp);
        }
        if let Some(exp) = encode_trajectory(&positions, &timestamps, &dct_config, format!("circle_dct_{}", i), "circle".to_string(), params) {
            embeddings.push(exp);
        }
    }

    // ========== FIGURE-8 ==========
    for (i, radius) in [5.0, 8.0, 10.0, 15.0].iter().enumerate() {
        let duration = 6.0;
        let (positions, timestamps) = generate_figure8(n_points, *radius, duration);
        let params = TrajectoryParams { n_points, duration, radius: Some(*radius), pitch: None, speed: None };

        if let Some(exp) = encode_trajectory(&positions, &timestamps, &fft_config, format!("figure8_fft_{}", i), "figure8".to_string(), params.clone()) {
            embeddings.push(exp);
        }
        if let Some(exp) = encode_trajectory(&positions, &timestamps, &dct_config, format!("figure8_dct_{}", i), "figure8".to_string(), params) {
            embeddings.push(exp);
        }
    }

    // ========== LINE (where DCT really shines) ==========
    for (i, dir) in [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [1.0, 1.0, 1.0],
    ].iter().enumerate() {
        let duration = 3.0;
        let length = 50.0;
        let (positions, timestamps) = generate_line(n_points, length, duration, *dir);
        let params = TrajectoryParams { n_points, duration, radius: None, pitch: None, speed: Some(length / duration) };

        if let Some(exp) = encode_trajectory(&positions, &timestamps, &fft_config, format!("line_fft_{}", i), "line".to_string(), params.clone()) {
            embeddings.push(exp);
        }
        if let Some(exp) = encode_trajectory(&positions, &timestamps, &dct_config, format!("line_dct_{}", i), "line".to_string(), params) {
            embeddings.push(exp);
        }
    }

    // ========== SPIRAL ==========
    for (i, (start_r, end_r)) in [(2.0, 8.0), (1.0, 10.0), (5.0, 15.0)].iter().enumerate() {
        let duration = 5.0;
        let (positions, timestamps) = generate_spiral(n_points, *start_r, *end_r, 10.0, duration);
        let params = TrajectoryParams { n_points, duration, radius: Some(*end_r), pitch: Some(10.0), speed: None };

        if let Some(exp) = encode_trajectory(&positions, &timestamps, &fft_config, format!("spiral_fft_{}", i), "spiral".to_string(), params.clone()) {
            embeddings.push(exp);
        }
        if let Some(exp) = encode_trajectory(&positions, &timestamps, &dct_config, format!("spiral_dct_{}", i), "spiral".to_string(), params) {
            embeddings.push(exp);
        }
    }

    // ========== LISSAJOUS ==========
    for (i, (fx, fy, fz)) in [(2.0, 3.0, 5.0), (1.0, 2.0, 3.0), (3.0, 4.0, 7.0)].iter().enumerate() {
        let duration = 6.0;
        let (positions, timestamps) = generate_lissajous(n_points, 5.0, *fx, *fy, *fz, duration);
        let params = TrajectoryParams { n_points, duration, radius: Some(5.0), pitch: None, speed: None };

        if let Some(exp) = encode_trajectory(&positions, &timestamps, &fft_config, format!("lissajous_fft_{}", i), "lissajous".to_string(), params.clone()) {
            embeddings.push(exp);
        }
        if let Some(exp) = encode_trajectory(&positions, &timestamps, &dct_config, format!("lissajous_dct_{}", i), "lissajous".to_string(), params) {
            embeddings.push(exp);
        }
    }

    // ========== RANDOM WALK ==========
    for i in 0..5 {
        let duration = 5.0;
        let (positions, timestamps) = generate_random_walk(n_points, 1.0, duration, i as u64);
        let params = TrajectoryParams { n_points, duration, radius: None, pitch: None, speed: None };

        if let Some(exp) = encode_trajectory(&positions, &timestamps, &fft_config, format!("random_fft_{}", i), "random_walk".to_string(), params.clone()) {
            embeddings.push(exp);
        }
        if let Some(exp) = encode_trajectory(&positions, &timestamps, &dct_config, format!("random_dct_{}", i), "random_walk".to_string(), params) {
            embeddings.push(exp);
        }
    }

    // ========== AGGRESSIVE MANEUVERS ==========
    for (i, radius) in [5.0, 8.0, 10.0, 12.0].iter().enumerate() {
        let duration = 4.0;
        let (positions, timestamps) = generate_aggressive(n_points, *radius, duration);
        let params = TrajectoryParams { n_points, duration, radius: Some(*radius), pitch: None, speed: None };

        if let Some(exp) = encode_trajectory(&positions, &timestamps, &fft_config, format!("aggressive_fft_{}", i), "aggressive".to_string(), params.clone()) {
            embeddings.push(exp);
        }
        if let Some(exp) = encode_trajectory(&positions, &timestamps, &dct_config, format!("aggressive_dct_{}", i), "aggressive".to_string(), params) {
            embeddings.push(exp);
        }
    }

    // ========== HOVER ==========
    for i in 0..5 {
        let duration = 10.0;
        let jitter = 0.5 + i as f64 * 0.3;
        let (positions, timestamps) = generate_hover(n_points, jitter, duration, i as u64 + 100);
        let params = TrajectoryParams { n_points, duration, radius: Some(jitter), pitch: None, speed: None };

        if let Some(exp) = encode_trajectory(&positions, &timestamps, &fft_config, format!("hover_fft_{}", i), "hover".to_string(), params.clone()) {
            embeddings.push(exp);
        }
        if let Some(exp) = encode_trajectory(&positions, &timestamps, &dct_config, format!("hover_dct_{}", i), "hover".to_string(), params) {
            embeddings.push(exp);
        }
    }

    // ========== SCALED HELICES (scale diversity) ==========
    for (i, scale) in [0.1, 1.0, 10.0, 100.0, 1000.0].iter().enumerate() {
        let duration = 5.0;
        let (positions, timestamps) = generate_scaled_helix(n_points, *scale, duration);
        let params = TrajectoryParams { n_points, duration, radius: Some(*scale), pitch: Some(*scale * 2.0), speed: None };

        if let Some(exp) = encode_trajectory(&positions, &timestamps, &fft_config, format!("scaled_fft_{}", i), "scaled_helix".to_string(), params.clone()) {
            embeddings.push(exp);
        }
        if let Some(exp) = encode_trajectory(&positions, &timestamps, &dct_config, format!("scaled_dct_{}", i), "scaled_helix".to_string(), params) {
            embeddings.push(exp);
        }
    }

    // ========== JERKY (low smoothness) ==========
    for (i, noise_amp) in [0.5, 1.0, 2.0, 5.0].iter().enumerate() {
        let duration = 5.0;
        let (positions, timestamps) = generate_jerky_trajectory(n_points, 5.0, duration, *noise_amp, i as u64);
        let params = TrajectoryParams { n_points, duration, radius: Some(5.0), pitch: Some(*noise_amp), speed: None };

        if let Some(exp) = encode_trajectory(&positions, &timestamps, &fft_config, format!("jerky_fft_{}", i), "jerky".to_string(), params.clone()) {
            embeddings.push(exp);
        }
        if let Some(exp) = encode_trajectory(&positions, &timestamps, &dct_config, format!("jerky_dct_{}", i), "jerky".to_string(), params) {
            embeddings.push(exp);
        }
    }

    // ========== SMOOTH ARCS (high smoothness) ==========
    for (i, radius) in [3.0, 5.0, 8.0, 12.0].iter().enumerate() {
        let duration = 4.0;
        let (positions, timestamps) = generate_smooth_arc(n_points, *radius, duration);
        let params = TrajectoryParams { n_points, duration, radius: Some(*radius), pitch: None, speed: None };

        if let Some(exp) = encode_trajectory(&positions, &timestamps, &fft_config, format!("smooth_fft_{}", i), "smooth_arc".to_string(), params.clone()) {
            embeddings.push(exp);
        }
        if let Some(exp) = encode_trajectory(&positions, &timestamps, &dct_config, format!("smooth_dct_{}", i), "smooth_arc".to_string(), params) {
            embeddings.push(exp);
        }
    }

    // ========== LEFT HELIX (chirality) ==========
    for (i, (radius, pitch)) in [(5.0, 10.0), (10.0, 5.0), (3.0, 15.0)].iter().enumerate() {
        let duration = 5.0;
        let (positions, timestamps) = generate_left_helix(n_points, *radius, *pitch, duration);
        let params = TrajectoryParams { n_points, duration, radius: Some(*radius), pitch: Some(*pitch), speed: None };

        if let Some(exp) = encode_trajectory(&positions, &timestamps, &fft_config, format!("left_helix_fft_{}", i), "left_helix".to_string(), params.clone()) {
            embeddings.push(exp);
        }
        if let Some(exp) = encode_trajectory(&positions, &timestamps, &dct_config, format!("left_helix_dct_{}", i), "left_helix".to_string(), params) {
            embeddings.push(exp);
        }
    }

    // Export to JSON
    let export_data = ExportData {
        embeddings,
        config: ConfigExport {
            characteristic_speed: fft_config.characteristic_speed,
            k_coeffs: fft_config.k_coeffs,
            alpha: fft_config.alpha,
            spectral_transform: "FFT (default) and DCT (balanced)".to_string(),
        },
    };

    let json = serde_json::to_string_pretty(&export_data).expect("Failed to serialize");

    let output_path = "python/embeddings.json";
    let mut file = File::create(output_path).expect("Failed to create file");
    file.write_all(json.as_bytes()).expect("Failed to write file");

    println!("Exported {} embeddings to {}", export_data.embeddings.len(), output_path);

    // Print summary by type
    let mut type_counts: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    for emb in &export_data.embeddings {
        *type_counts.entry(&emb.trajectory_type).or_insert(0) += 1;
    }

    println!("\nTrajectory types:");
    let mut sorted: Vec<_> = type_counts.iter().collect();
    sorted.sort_by_key(|(k, _)| *k);
    for (t, count) in sorted {
        println!("  {}: {} ({} FFT + {} DCT)", t, count, count / 2, count / 2);
    }

    // Print summary by transform
    let fft_count = export_data.embeddings.iter().filter(|e| e.spectral_transform == "FFT").count();
    let dct_count = export_data.embeddings.iter().filter(|e| e.spectral_transform == "DCT").count();
    println!("\nSpectral transforms:");
    println!("  FFT: {}", fft_count);
    println!("  DCT: {}", dct_count);
}

// ============================================================================
// MSE vs k_coeffs Analysis
// ============================================================================

use motion_embedding::{reconstruct_trajectory, compute_reconstruction_error};

#[derive(Serialize)]
struct MseDataPoint {
    trajectory_type: String,
    spectral_transform: String,
    k_coeffs: usize,
    rmse: f64,
    n_points: usize,
}

#[derive(Serialize)]
struct MseExportData {
    data_points: Vec<MseDataPoint>,
    k_values: Vec<usize>,
    trajectory_types: Vec<String>,
}

/// Compute RMSE for a trajectory with given config
fn compute_trajectory_rmse(
    positions: &[[f64; 3]],
    timestamps: &[f64],
    config: &EmbeddingConfig,
) -> Option<f64> {
    let embedding = compute_motion_embedding(positions, timestamps, config).ok()?;
    let (reconstructed, _) = reconstruct_trajectory(&embedding, None, true, config).ok()?;
    Some(compute_reconstruction_error(positions, &reconstructed))
}

#[test]
#[ignore] // Run manually with: cargo test --test export_embeddings export_mse -- --ignored --nocapture
fn export_mse_vs_kcoeffs() {
    let n_points = 100;
    let k_values: Vec<usize> = vec![4, 8, 12, 16, 20, 24, 32, 40, 48];

    // Define trajectory generators
    let trajectory_generators: Vec<(&str, Box<dyn Fn() -> (Vec<[f64; 3]>, Vec<f64>)>)> = vec![
        ("helix", Box::new(|| generate_helix(n_points, 5.0, 10.0, 5.0))),
        ("circle", Box::new(|| generate_circle(n_points, 5.0, 4.0))),
        ("figure8", Box::new(|| generate_figure8(n_points, 5.0, 6.0))),
        ("line", Box::new(|| generate_line(n_points, 50.0, 3.0, [1.0, 0.5, 0.2]))),
        ("spiral", Box::new(|| generate_spiral(n_points, 2.0, 10.0, 8.0, 5.0))),
        ("lissajous", Box::new(|| generate_lissajous(n_points, 5.0, 2.0, 3.0, 5.0, 6.0))),
        ("random_walk", Box::new(|| generate_random_walk(n_points, 1.0, 5.0, 42))),
        ("aggressive", Box::new(|| generate_aggressive(n_points, 8.0, 4.0))),
        ("hover", Box::new(|| generate_hover(n_points, 0.5, 10.0, 100))),
        ("smooth_arc", Box::new(|| generate_smooth_arc(n_points, 8.0, 4.0))),
        ("left_helix", Box::new(|| generate_left_helix(n_points, 5.0, 10.0, 5.0))),
        ("jerky", Box::new(|| generate_jerky_trajectory(n_points, 5.0, 5.0, 1.0, 42))),
    ];

    let mut data_points = Vec::new();
    let mut trajectory_types = Vec::new();

    println!("Computing MSE for different k_coeffs values...\n");
    println!("{:<15} {:>6} {:>10} {:>10}", "Trajectory", "k", "FFT RMSE", "DCT RMSE");
    println!("{}", "-".repeat(45));

    for (traj_type, generator) in &trajectory_generators {
        trajectory_types.push(traj_type.to_string());
        let (positions, timestamps) = generator();

        for &k in &k_values {
            // FFT config
            let fft_config = EmbeddingConfig {
                k_coeffs: k,
                spectral_transform: SpectralTransform::Fft,
                spectral_whitening: false,
                spectral_amplitude_norm: false,
                store_endpoints: true,
                ..EmbeddingConfig::default()
            };

            // DCT config
            let dct_config = EmbeddingConfig {
                k_coeffs: k,
                spectral_transform: SpectralTransform::Dct,
                spectral_whitening: false,
                spectral_amplitude_norm: false,
                store_endpoints: true,
                ..EmbeddingConfig::default()
            };

            let fft_rmse = compute_trajectory_rmse(&positions, &timestamps, &fft_config)
                .unwrap_or(f64::NAN);
            let dct_rmse = compute_trajectory_rmse(&positions, &timestamps, &dct_config)
                .unwrap_or(f64::NAN);

            data_points.push(MseDataPoint {
                trajectory_type: traj_type.to_string(),
                spectral_transform: "FFT".to_string(),
                k_coeffs: k,
                rmse: fft_rmse,
                n_points,
            });

            data_points.push(MseDataPoint {
                trajectory_type: traj_type.to_string(),
                spectral_transform: "DCT".to_string(),
                k_coeffs: k,
                rmse: dct_rmse,
                n_points,
            });

            println!("{:<15} {:>6} {:>10.4} {:>10.4}", traj_type, k, fft_rmse, dct_rmse);
        }
        println!();
    }

    // Export to JSON
    let export_data = MseExportData {
        data_points,
        k_values: k_values.clone(),
        trajectory_types,
    };

    let json = serde_json::to_string_pretty(&export_data).expect("Failed to serialize");

    let output_path = "python/mse_data.json";
    let mut file = File::create(output_path).expect("Failed to create file");
    file.write_all(json.as_bytes()).expect("Failed to write file");

    println!("\nExported MSE data to {}", output_path);
    println!("k values tested: {:?}", k_values);
}
