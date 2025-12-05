# Motion Embedding Library: Deep Dive

A comprehensive technical guide to the `motion_embedding` Rust library for transforming 3D trajectories into compact, ML-ready embeddings.

## Table of Contents

1. [Overview](#overview)
2. [Embedding Architecture](#embedding-architecture)
3. [Feature Groups Explained](#feature-groups-explained)
4. [Configuration System](#configuration-system)
5. [Spectral Transforms: FFT vs DCT](#spectral-transforms-fft-vs-dct)
6. [Reconstruction Pipeline](#reconstruction-pipeline)
7. [Streaming API](#streaming-api)
8. [Distance & Comparison](#distance--comparison)
9. [Variance Validation](#variance-validation)
10. [Mathematical Foundations](#mathematical-foundations)

---

## Overview

The `motion_embedding` library transforms 3D position + timestamp trajectories into compact vector representations suitable for machine learning. The core design principles are:

1. **Range-balanced**: All features compressed to `[-1, 1]` or `[0, 1]` for stable ML training
2. **Physically meaningful**: Features correspond to intuitive properties (scale, shape, dynamics, safety)
3. **Invertible**: Trajectories can be approximately reconstructed from embeddings
4. **Configurable**: Domain presets and fine-grained parameter control

### Input/Output

```rust
// Input: 3D positions + timestamps
let positions: Vec<[f64; 3]> = vec![...];  // [x, y, z] in meters
let timestamps: Vec<f64> = vec![...];       // seconds, monotonically increasing

// Output: 24D compact embedding (or more with spectral features)
let embedding = compute_motion_embedding(&positions, &timestamps, &config)?;
let compact: [f64; 24] = embedding.to_compact_array();
```

---

## Embedding Architecture

### Compact Embedding (24D)

The standard ML-ready representation has exactly 24 dimensions:

| Index | Feature | Dims | Range | Description |
|-------|---------|------|-------|-------------|
| 0 | `scale_magnitude` | 1 | [-1,1] | Log-scale of trajectory extent |
| 1-3 | `shape_entropy` | 3 | [-1,1] | PCA eigenvalue ratios (geometry) |
| 4-9 | `normalized_momentum` | 6 | [-1,1] | 4D angular momentum bivector |
| 10-12 | `current_heading` | 3 | [-1,1] | Unit direction vector |
| 13-15 | `maneuver_plane` | 3 | [-1,1] | Unit maneuver normal |
| 16-18 | `velocity_normalized` | 3 | [-1,1] | Relative velocity |
| 19 | `chirality` | 1 | [-1,1] | Handedness (helicity) |
| 20-22 | `g_force_stats` | 3 | [-1,1] | [mean_g, max_g, jerk] |
| 23 | `smoothness_index` | 1 | [0,1] | Motion smoothness |

### Output Variants

```rust
// Standard ML embedding (24D)
let compact: [f64; 24] = embedding.to_compact_array();
let compact_vec: Vec<f64> = embedding.to_compact_vec();

// Full embedding with spectral coefficients (24 + k*8 dimensions)
// Where k = k_coeffs from config (default 16)
// Spectral: k*4 complex values = k*8 real values (interleaved re/im)
let full: Vec<f64> = embedding.to_full_vec(true);  // 24 + 128 = 152D with k=16

// Rotation-invariant (11D) - ignores absolute heading
let invariant: [f64; 11] = embedding.to_invariant_array();

// Scale-invariant (23D) - ignores trajectory size
let scale_inv: Vec<f64> = embedding.to_scale_invariant_vec();
```

### Dimension Breakdown

```
Compact (24D):
├── Scale (1D)           → indices 0
├── Shape (3D)           → indices 1-3
├── Momentum (6D)        → indices 4-9
├── Heading (3D)         → indices 10-12
├── Maneuver Plane (3D)  → indices 13-15
├── Velocity (3D)        → indices 16-18
├── Chirality (1D)       → index 19
├── G-Force Stats (3D)   → indices 20-22
└── Smoothness (1D)      → index 23

Full (24 + k*8 D):
├── Compact (24D)
└── Spectral (k*8 D)     → Complex FFT/DCT coefficients
    ├── X dimension (k*2)
    ├── Y dimension (k*2)
    ├── Z dimension (k*2)
    └── W dimension (k*2)  (time * characteristic_speed)
```

---

## Feature Groups Explained

### Group A: Scale (1D)

**What it captures**: The overall spatial extent of the trajectory.

```rust
pub scale_magnitude: f64  // [-1, 1]
```

**Computation**:
1. Perform 4D PCA on spacetime trajectory
2. Sum square roots of eigenvalues to get raw scale
3. Apply log transform: `raw_scale = log10(sum_sqrt_eigs)`
4. Compress with tanh: `scale_magnitude = tanh((raw_scale - center) / width)`

**Interpretation**:
- `-1`: Very small trajectory (millimeters)
- `0`: Medium trajectory (~1 meter)
- `+1`: Very large trajectory (kilometers)

---

### Group B: Shape (3D)

**What it captures**: The geometric aspect ratios of the trajectory envelope.

```rust
pub shape_entropy: [f64; 3]  // [-1, 1] each
```

**Computation**:
1. Extract eigenvalues `[λ1, λ2, λ3, λ4]` from 4D PCA (sorted descending)
2. Compute log-ratios: `shape[i] = log(λ[i+1] / λ[0])` for i=0,1,2
3. Optionally normalize to [-1, 1]

**Interpretation**:
- Equal values → spherical/isotropic motion
- Large differences → elongated/linear motion
- `shape[0]` ≈ 0: trajectory extends equally in first two directions
- `shape[0]` << 0: trajectory is needle-like (one dominant direction)

**Accessor**:
```rust
let aspect = embedding.aspect_ratio();  // exp(max_shape) / exp(min_shape)
```

---

### Group C: Dynamics/Momentum (6D)

**What it captures**: The rotational dynamics in 4D spacetime using geometric algebra.

```rust
pub normalized_momentum: [f64; 6]  // [-1, 1] each
// Components: [L_xy, L_xz, L_xw, L_yz, L_yw, L_zw]
```

**Computation**:
1. Compute 4D angular momentum using wedge products: `L = Σ mᵢ (rᵢ ∧ vᵢ)`
2. The wedge product in 4D produces a bivector with 6 components
3. Normalize by trajectory scale and duration
4. Apply tanh compression

**Physical Meaning**:
- **Spatial components** `[L_xy, L_xz, L_yz]`: Rotation in spatial planes
  - `L_xy`: Rotation around Z axis (yaw)
  - `L_xz`: Rotation around Y axis (pitch)
  - `L_yz`: Rotation around X axis (roll)
- **Temporal components** `[L_xw, L_yw, L_zw]`: Spacetime "boost" components
  - Capture how the trajectory accelerates/decelerates in each spatial direction

**Accessors**:
```rust
let spatial = embedding.spatial_momentum();   // [L_xy, L_xz, L_yz]
let temporal = embedding.temporal_momentum(); // [L_xw, L_yw, L_zw]
```

---

### Group D: Navigation State (10D)

#### Heading (3D)
```rust
pub current_heading: [f64; 3]  // Unit vector
```

The primary direction of motion, extracted as the spatial component of the first principal axis.

#### Maneuver Plane (3D)
```rust
pub maneuver_plane: [f64; 3]  // Unit vector
```

The normal to the plane of primary turning, from the second principal axis.

#### Velocity (3D)
```rust
pub velocity_normalized: [f64; 3]  // [-1, 1] each
```

Current velocity normalized by mean speed, with tanh compression.

**Accessor**:
```rust
let speed_mps = embedding.speed();  // Denormalized speed in m/s
```

#### Chirality (1D)
```rust
pub chirality: f64  // [-1, 1]
```

**What it captures**: The "handedness" or helicity of the trajectory.

**Computation**:
1. Compute spatial angular momentum vector (Hodge dual of bivector)
2. Project onto net displacement direction
3. Normalize by displacement magnitude
4. Apply tanh compression

**Interpretation**:
- `-1`: Strong left-handed spiral (counter-clockwise when viewed from ahead)
- `0`: Neutral (straight line or symmetric)
- `+1`: Strong right-handed spiral (clockwise when viewed from ahead)

**Accessor**:
```rust
let handedness = embedding.handedness();  // Handedness::Left, Right, or Neutral
```

---

### Group E: Safety/Quality (4D)

#### G-Force Statistics (3D)
```rust
pub g_force_stats: [f64; 3]  // [-1, 1] each: [mean_g, max_g, jerk]
```

**Computation**:
1. Compute accelerations from position differences
2. Convert to g-forces: `g = |acceleration| / 9.81`
3. Compute jerk (rate of change of acceleration)
4. Apply log-tanh compression: `tanh(log(1 + raw_value) / g_force_scale)`

**Accessors** (return raw, uncompressed values):
```rust
embedding.mean_g_force()  // Average g-force experienced
embedding.max_g_force()   // Peak g-force
embedding.jerk_index()    // Jerkiness measure
```

#### Smoothness Index (1D)
```rust
pub smoothness_index: f64  // [0, 1]
```

**What it captures**: How smooth/continuous the motion is.

**Computation**:
1. Compute acceleration variance normalized by mean speed
2. Apply exponential decay: `smoothness = exp(-jerkiness / sensitivity)`

**Interpretation**:
- `1.0`: Perfectly smooth (constant velocity)
- `0.5`: Moderate smoothness
- `0.0`: Very jerky motion

**Accessor**:
```rust
embedding.is_smooth()  // true if smoothness_index > 0.5
```

---

## Configuration System

### EmbeddingConfig Structure

```rust
pub struct EmbeddingConfig {
    // === Core Parameters ===
    pub characteristic_speed: f64,  // m/s, maps time to space (w = c*t)
    pub alpha: f64,                 // Acceleration weighting (0=uniform, 3+=high-G focus)
    pub k_coeffs: usize,            // Spectral coefficients (16=lossy, n/2+1=exact)

    // === Spectral Transform ===
    pub spectral_transform: SpectralTransform,  // FFT or DCT
    pub window_function: WindowFunction,         // None, Hanning, or Tukey
    pub spectral_whitening: bool,               // Log-whitening for ML
    pub spectral_amplitude_norm: bool,          // Normalize by trajectory length

    // === Reconstruction ===
    pub store_endpoints: bool,  // Store start/end for improved reconstruction

    // === Compression Parameters ===
    pub chirality_scale: f64,       // Chirality tanh scale
    pub g_force_scale: f64,         // G-force tanh scale
    pub scale_center: f64,          // Scale compression center
    pub scale_width: f64,           // Scale compression width
    pub smoothness_sensitivity: f64, // Smoothness detection threshold

    // ... and more
}
```

### Domain Presets

```rust
// General purpose (FFT, k=16, balanced parameters)
EmbeddingConfig::default()

// Quadcopter/drone
EmbeddingConfig::drone(false)  // Normal flight
EmbeddingConfig::drone(true)   // Aggressive/racing

// Human motion
EmbeddingConfig::pedestrian()  // Walking/running (characteristic_speed=1.5 m/s)

// Ground vehicles
EmbeddingConfig::vehicle()     // Cars/trucks (characteristic_speed=30 m/s)

// Racing
EmbeddingConfig::racing()      // High-G maneuvers (alpha=3.0)

// Surveillance
EmbeddingConfig::surveillance() // Loitering patterns (alpha=0)

// ML optimized
EmbeddingConfig::ml_training() // Balanced variance across features

// Reconstruction optimized
EmbeddingConfig::reconstruction_optimized() // DCT, no whitening, k=24

// Balanced (ML + reconstruction)
EmbeddingConfig::balanced()    // DCT with whitening
```

### Key Configuration Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `characteristic_speed` | 15.0 m/s | Maps time to space in 4D manifold |
| `alpha` | 1.0 | High-G maneuver emphasis (0=none, 3+=strong) |
| `k_coeffs` | 16 | Spectral resolution (higher = better reconstruction) |
| `spectral_transform` | FFT | FFT for periodic, DCT for non-periodic |
| `spectral_whitening` | true | Log-compress spectrum for ML |
| `store_endpoints` | true | Store exact endpoints for reconstruction |
| `smoothness_sensitivity` | 0.1 | Smoothness detection threshold |

### Builder Pattern

```rust
let config = EmbeddingConfig::drone(false)
    .with_k_coeffs(32)
    .with_spectral_transform(SpectralTransform::Dct)
    .with_store_endpoints(true);
```

---

## Spectral Transforms: FFT vs DCT

### The Problem: Gibbs Phenomenon

FFT assumes periodic signals. For non-periodic trajectories (straight lines, spirals), this creates discontinuities at boundaries, causing **Gibbs phenomenon** (ringing artifacts).

### Solution: Discrete Cosine Transform (DCT)

DCT implicitly mirrors the signal at boundaries, eliminating discontinuities. This makes it ideal for:
- Straight lines
- Spirals
- Random walks
- Any non-periodic trajectory

### Transform Comparison

```rust
pub enum SpectralTransform {
    Fft,  // Default - best for periodic (circles, figure-8s)
    Dct,  // Best for non-periodic (lines, spirals)
}
```

### Reconstruction Quality (RMSE in meters, k=16, 100 points)

| Trajectory | FFT | DCT | Winner |
|------------|-----|-----|--------|
| Circle | 0.00m | 0.08m | FFT |
| Figure-8 | 0.02m | 0.10m | FFT |
| Helix | 0.25m | 0.09m | DCT |
| **Line** | **1.26m** | **0.07m** | **DCT (18x better)** |
| Spiral | 0.43m | 0.22m | DCT |
| Random Walk | 0.26m | 0.39m | FFT |

**Key insight**: DCT dramatically outperforms FFT on straight lines because it doesn't suffer from Gibbs phenomenon.

### Recommended Presets

```rust
// For ML only (reconstruction not needed)
EmbeddingConfig::default()  // FFT with whitening

// For mixed trajectories (ML + reconstruction)
EmbeddingConfig::balanced()  // DCT with whitening

// For best reconstruction fidelity
EmbeddingConfig::reconstruction_optimized()  // DCT without whitening, k=24
```

### Window Functions (FFT only)

For FFT, window functions can reduce Gibbs phenomenon:

```rust
pub enum WindowFunction {
    None,     // No windowing (default)
    Hanning,  // Smooth taper, good general purpose
    Tukey,    // Flat center with tapered edges (alpha=0.5)
}
```

---

## Reconstruction Pipeline

The library supports round-trip encoding: trajectory → embedding → trajectory.

### Basic Reconstruction

```rust
use motion_embedding::{compute_motion_embedding, reconstruct_trajectory, EmbeddingConfig};

// Encode
let config = EmbeddingConfig::balanced();
let embedding = compute_motion_embedding(&positions, &timestamps, &config)?;

// Decode
let (recon_positions, recon_timestamps) = reconstruct_trajectory(
    &embedding,
    None,       // Use original point count (or specify custom)
    true,       // Transform back to world frame
    &config,
)?;
```

### How Reconstruction Works

1. **Inverse Spectral Transform**: Apply IFFT or IDCT to reconstruct canonical-frame points
2. **Coordinate Transform**: Rotate from canonical frame back to world frame using stored PCA axes
3. **Translation**: Add back the centroid
4. **Endpoint Correction** (if enabled): Force exact start/end positions

### Endpoint Storage

When `config.store_endpoints = true`, the embedding metadata stores:
- `start_position: Option<[f64; 3]>` - Exact starting position
- `end_position: Option<[f64; 3]>` - Exact ending position
- `start_time: Option<f64>` - Starting timestamp
- `end_time: Option<f64>` - Ending timestamp

This enables precise endpoint matching during reconstruction.

### Configuration for Best Reconstruction

```rust
// Option 1: Reconstruction-optimized preset
let config = EmbeddingConfig::reconstruction_optimized();
// - Uses DCT (better for non-periodic)
// - Disables whitening (preserves amplitude)
// - Higher k_coeffs (24)
// - Stores endpoints

// Option 2: Exact reconstruction (lossless)
let config = EmbeddingConfig {
    k_coeffs: EmbeddingConfig::exact_k_coeffs(n_points), // n/2+1
    spectral_whitening: false,
    spectral_amplitude_norm: false,
    spectral_transform: SpectralTransform::Dct,
    store_endpoints: true,
    ..EmbeddingConfig::default()
};
```

### Reconstruction Error Measurement

```rust
use motion_embedding::compute_reconstruction_error;

let rmse = compute_reconstruction_error(&original_positions, &reconstructed_positions);
// Returns RMSE in position units (meters)
```

### How Spectral Parameters Affect Embedding Shape

| Configuration | Compact Dims | Full Dims (with spectrum) |
|---------------|--------------|---------------------------|
| `k_coeffs=16` (default) | 24 | 24 + 16×4×2 = 152 |
| `k_coeffs=24` | 24 | 24 + 24×4×2 = 216 |
| `k_coeffs=32` | 24 | 24 + 32×4×2 = 280 |
| `k_coeffs=51` (exact for 100 points) | 24 | 24 + 51×4×2 = 432 |

The compact embedding is always 24D. Spectral features are stored separately and can be included via `to_full_vec(true)`.

---

## Streaming API

For real-time applications, the `RollingHorizonEmbedder` maintains a sliding window of recent positions.

### Basic Usage

```rust
use motion_embedding::{EmbeddingConfig, RollingHorizonEmbedder};

let config = EmbeddingConfig::drone(false);
let mut embedder = RollingHorizonEmbedder::new(config, 2.0); // 2-second window

// Stream in position updates
loop {
    let (position, timestamp) = get_sensor_data();
    embedder.update(position, timestamp);

    if embedder.is_ready() {
        if let Some(embedding) = embedder.get_embedding() {
            let compact = embedding.to_compact_array();
            // Use for inference...
        }
    }
}
```

### Configuration

```rust
// Custom buffer parameters
let embedder = RollingHorizonEmbedder::with_buffer(
    config,
    horizon_seconds: 3.0,  // Time window
    max_points: 1000,      // Max buffer size
    min_points: 20,        // Minimum for valid embedding
);
```

### Key Methods

```rust
embedder.update(position, timestamp);      // Single update
embedder.update_batch(&positions, &timestamps);  // Batch update
embedder.is_ready() -> bool;               // Has enough data?
embedder.get_embedding() -> Option<&MotionEmbedding>;  // Get (cached) embedding
embedder.get_compact_embedding() -> Option<[f64; 24]>; // Direct tensor access
embedder.reset();                          // Clear buffer
embedder.n_points() -> usize;              // Current buffer size
embedder.time_span() -> f64;               // Current time window
embedder.set_horizon(seconds);             // Change window size
```

### Caching Behavior

- Embedding is lazily computed on first `get_embedding()` call
- Cache is invalidated on each `update()`
- Subsequent `get_embedding()` calls return cached result

---

## Distance & Comparison

### Basic Distance

```rust
use motion_embedding::compute_embedding_distance;

let dist = compute_embedding_distance(&emb1, &emb2, None);  // Default weights
```

### Custom Weights

```rust
use motion_embedding::DistanceWeights;

let weights = DistanceWeights {
    scale: 1.0,
    shape: 2.0,      // Emphasize shape similarity
    momentum: 1.0,
    spectrum: 0.5,
    heading: 0.5,    // De-emphasize heading
    smoothness: 1.0,
    velocity: 1.0,
    safety: 1.0,
    chirality: 1.0,
};

let dist = compute_embedding_distance(&emb1, &emb2, Some(&weights));
```

### Detailed Breakdown

```rust
use motion_embedding::compute_embedding_distance_detailed;

let breakdown = compute_embedding_distance_detailed(&emb1, &emb2, None);

println!("Total distance: {}", breakdown.total);
println!("Shape distance: {}", breakdown.shape);
println!("Heading similarity: {}", breakdown.heading_similarity);
// ... access individual component distances
```

### Heading Angle

```rust
use motion_embedding::heading_angle_between;

let angle_degrees = heading_angle_between(&emb1, &emb2);
```

### Batch Pairwise Distances

```rust
use motion_embedding::compute_pairwise_distances;

let distances = compute_pairwise_distances(&embeddings, None);
// Returns upper triangular matrix as flat Vec
```

---

## Variance Validation

Ensure embeddings have good variance distribution for ML training.

### Health Targets

| Metric | Target | Meaning |
|--------|--------|---------|
| PC1 Variance | < 40% | No single dimension dominates |
| Effective Dimensionality | > 8 | Most features contribute |
| Max/Min Variance Ratio | < 50 | Balanced feature groups |

### Usage

```rust
use motion_embedding::validate_embedding_variance;

let embeddings: Vec<MotionEmbedding> = /* ... */;

if let Some(analysis) = validate_embedding_variance(&embeddings) {
    println!("PC1 Variance: {:.1}%", analysis.pc1_variance_percent);
    println!("Effective Dim: {:.1}", analysis.effective_dimensionality);
    println!("Healthy: {}", analysis.is_healthy);

    // Per-group variances
    println!("Scale variance: {}", analysis.feature_group_variances.scale);
    println!("Shape variance: {}", analysis.feature_group_variances.shape);
    // ...
}
```

---

## Mathematical Foundations

### 4D Spacetime Manifold

The trajectory is embedded in 4D spacetime:
```
p_4d = [x, y, z, w]  where w = characteristic_speed × t
```

This maps time to a spatial dimension, enabling unified treatment of position and time.

### Weighted PCA

Points are weighted by acceleration:
```
mass[i] = (1 + alpha × g_force[i]) × dt[i]
```

This emphasizes high-G maneuvers when `alpha > 0`.

### Bivector Angular Momentum

In 4D, angular momentum is a bivector (antisymmetric tensor) with 6 independent components:
```
L = Σ mᵢ (rᵢ ∧ vᵢ)

Components:
- L_xy, L_xz, L_yz: spatial rotation (standard 3D angular momentum)
- L_xw, L_yw, L_zw: spacetime "boost" (acceleration coupling)
```

### Chirality Computation

Chirality measures helicity by projecting angular momentum onto displacement:
```
chirality = (L⃗ · d̂) / |d|

where:
- L⃗ = Hodge dual of spatial angular momentum bivector
- d̂ = unit net displacement vector
```

### Smoothness via Acceleration Variance

```
jerkiness = mean(|a|/v_mean) + std(|a|/v_mean)
smoothness = exp(-jerkiness / sensitivity)
```

### DCT-II Definition

```
X[k] = Σ_{n=0}^{N-1} x[n] × cos(π × k × (2n + 1) / (2N))
```

DCT implicitly mirrors the signal at boundaries, avoiding the discontinuity that causes Gibbs phenomenon in FFT.

---

## Quick Reference

### Embedding Dimensions

| Method | Dimensions | Use Case |
|--------|------------|----------|
| `to_compact_array()` | 24 | Standard ML input |
| `to_full_vec(true)` | 24 + k×8 | Include spectral features |
| `to_invariant_array()` | 11 | Rotation-invariant comparison |
| `to_scale_invariant_vec()` | 23 | Scale-invariant comparison |

### Configuration Presets

| Preset | Spectral | k_coeffs | Use Case |
|--------|----------|----------|----------|
| `default()` | FFT | 16 | General ML |
| `balanced()` | DCT | 16 | ML + reconstruction |
| `reconstruction_optimized()` | DCT | 24 | Best reconstruction |
| `drone(false)` | FFT | 16 | Quadcopter flight |
| `drone(true)` | FFT | 16 | Aggressive drone racing |
| `pedestrian()` | FFT | 32 | Human walking |
| `vehicle()` | FFT | 16 | Ground vehicles |
| `racing()` | FFT | 24 | High-performance racing |

### Semantic Accessors

```rust
embedding.mean_g_force()      // f64: Average g-force
embedding.max_g_force()       // f64: Peak g-force
embedding.jerk_index()        // f64: Jerk magnitude
embedding.speed()             // f64: Current speed (m/s)
embedding.handedness()        // Handedness enum
embedding.is_smooth()         // bool: smoothness > 0.5
embedding.aspect_ratio()      // f64: Shape elongation
embedding.spatial_momentum()  // [f64; 3]: Rotation components
embedding.temporal_momentum() // [f64; 3]: Boost components
```
