# motion_embedding

Range-balanced 24D trajectory embedding for ML applications.

## Overview

`motion_embedding` transforms 3D position + timestamp trajectories into compact, ML-ready embeddings. All features are range-balanced to `[-1, 1]` for optimal neural network training.

**Key Features:**
- 24D compact embedding capturing scale, shape, dynamics, navigation, and safety
- Physically meaningful features (chirality, g-force, momentum, smoothness)
- FFT or DCT spectral decomposition (DCT better for non-periodic trajectories)
- Round-trip reconstruction: trajectory → embedding → trajectory
- Streaming support for real-time applications
- Domain presets (drone, vehicle, pedestrian, racing)

## Quick Start

```rust
use motion_embedding::{compute_motion_embedding, EmbeddingConfig};

let positions = vec![
    [0.0, 0.0, 0.0],
    [1.0, 0.5, 0.0],
    [2.0, 1.0, 0.5],
    [3.0, 1.0, 1.0],
];
let timestamps = vec![0.0, 0.1, 0.2, 0.3];

let config = EmbeddingConfig::drone(false);
let embedding = compute_motion_embedding(&positions, &timestamps, &config)?;

// 24D compact embedding for ML
let compact = embedding.to_compact_array();
```

## Installation

```toml
[dependencies]
motion_embedding = "0.1.0"

# Optional: Enable serde for JSON serialization
motion_embedding = { version = "0.1.0", features = ["serde"] }
```

## Embedding Structure (24D)

| Index | Feature | Dims | Range | Description |
|-------|---------|------|-------|-------------|
| 0 | `scale_magnitude` | 1 | [-1,1] | Log-scale of trajectory extent |
| 1-3 | `shape_entropy` | 3 | [-1,1] | PCA eigenvalue ratios (elongation) |
| 4-9 | `normalized_momentum` | 6 | [-1,1] | 4D angular momentum bivector |
| 10-12 | `current_heading` | 3 | [-1,1] | Unit direction vector |
| 13-15 | `maneuver_plane` | 3 | [-1,1] | Unit maneuver normal |
| 16-18 | `velocity_normalized` | 3 | [-1,1] | Relative velocity |
| 19 | `chirality` | 1 | [-1,1] | Handedness (helicity) |
| 20-22 | `g_force_stats` | 3 | [-1,1] | [mean_g, max_g, jerk] |
| 23 | `smoothness_index` | 1 | [0,1] | Motion smoothness |

## API Reference

### Core Functions

#### `compute_motion_embedding`
```rust
pub fn compute_motion_embedding(
    positions: &[[f64; 3]],
    timestamps: &[f64],
    config: &EmbeddingConfig,
) -> Result<MotionEmbedding>
```
Main entry point. Transforms a trajectory into a `MotionEmbedding`.

#### `reconstruct_trajectory`
```rust
pub fn reconstruct_trajectory(
    embedding: &MotionEmbedding,
    n_points: Option<usize>,
    world_frame: bool,
    config: &EmbeddingConfig,
) -> Result<(Vec<[f64; 3]>, Vec<f64>)>
```
Reconstructs an approximate trajectory from its embedding.

#### `compute_embedding_distance`
```rust
pub fn compute_embedding_distance(
    emb1: &MotionEmbedding,
    emb2: &MotionEmbedding,
    weights: Option<&DistanceWeights>,
) -> f64
```
Computes weighted distance between two embeddings.

### Embedding Output Methods

| Method | Dimensions | Use Case |
|--------|------------|----------|
| `to_compact_array()` | 24 | Default ML input |
| `to_compact_vec()` | 24 | Same as above, returns Vec |
| `to_full_vec(true)` | 24 + k*8 | With spectral coefficients |
| `to_invariant_array()` | 11 | Rotation-invariant comparison |
| `to_scale_invariant_vec()` | 23 | Scale-invariant comparison |

### Semantic Accessors

```rust
let emb = compute_motion_embedding(&positions, &timestamps, &config)?;

emb.mean_g_force()     // -> f64: Average g-force
emb.max_g_force()      // -> f64: Peak g-force
emb.jerk_index()       // -> f64: Jerk magnitude
emb.speed()            // -> f64: Current speed (m/s)
emb.handedness()       // -> Handedness: Left/Right/Neutral
emb.is_smooth()        // -> bool: smoothness_index > 0.5
emb.aspect_ratio()     // -> f64: Shape elongation
emb.spatial_momentum() // -> [f64; 3]: Rotation components
emb.temporal_momentum()// -> [f64; 3]: Spacetime boost components
```

### Configuration

#### Presets

```rust
EmbeddingConfig::default()                // General purpose (FFT, k=16)
EmbeddingConfig::drone(false)             // Quadcopter (non-aggressive)
EmbeddingConfig::drone(true)              // Quadcopter (aggressive/racing)
EmbeddingConfig::pedestrian()             // Human walking/running
EmbeddingConfig::vehicle()                // Ground vehicles
EmbeddingConfig::racing()                 // High-performance racing
EmbeddingConfig::surveillance()           // Loitering/surveillance patterns
EmbeddingConfig::ml_training()            // Optimized for balanced variance
EmbeddingConfig::balanced()               // DCT + whitening (ML + reconstruction)
EmbeddingConfig::reconstruction_optimized() // Best reconstruction fidelity (DCT, no whitening)
```

#### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `characteristic_speed` | 15.0 | Velocity scale (m/s) for spacetime mapping |
| `alpha` | 1.0 | Acceleration weighting (0=uniform, 3+=high-G focus) |
| `k_coeffs` | 16 | Spectral coefficients (16=lossy, n=exact for DCT) |
| `spectral_transform` | FFT | FFT (periodic) or DCT (non-periodic trajectories) |
| `spectral_whitening` | true | Log-whitening for ML (disable for reconstruction) |
| `store_endpoints` | true | Store endpoints for reconstruction correction |
| `smoothness_sensitivity` | 0.1 | Smoothness detection threshold |
| `chirality_scale` | 1.0 | Chirality compression scale |
| `g_force_scale` | 2.0 | G-force compression scale |

#### Spectral Transform Selection

The library supports two spectral transforms:

- **FFT (default)**: Best for periodic trajectories (circles, figure-8s). Can exhibit Gibbs phenomenon (ringing artifacts) at boundaries.
- **DCT**: Best for non-periodic trajectories (lines, spirals, random walks). Implicitly mirrors the signal at boundaries, eliminating discontinuities.

**Recommendation**: Use `EmbeddingConfig::balanced()` for mixed trajectory types that need both ML-quality embeddings and reasonable reconstruction.

#### DCT vs FFT Reconstruction Quality (RMSE in meters, k=16, 100 points)

| Trajectory | FFT | DCT | Winner |
|------------|-----|-----|--------|
| Helix | 0.25 | 0.09 | DCT |
| Circle | 0.00 | 0.08 | FFT |
| Figure-8 | 0.02 | 0.10 | FFT |
| Line | 1.26 | 0.07 | **DCT** |
| Spiral | 0.43 | 0.22 | DCT |
| Random Walk | 0.26 | 0.39 | FFT |

Key insight: **DCT dramatically outperforms FFT on straight lines** (the most problematic case for FFT due to Gibbs phenomenon).

### Streaming API

```rust
use motion_embedding::{EmbeddingConfig, RollingHorizonEmbedder};

let config = EmbeddingConfig::drone(false);
let mut embedder = RollingHorizonEmbedder::new(config, 2.0); // 2-second window

// Stream position updates
embedder.update([x, y, z], timestamp);

// Get embedding when ready
if embedder.is_ready() {
    if let Some(embedding) = embedder.get_embedding() {
        let compact = embedding.to_compact_array();
    }
}

// Batch update
embedder.update_batch(&positions, &timestamps);

// Reset state
embedder.reset();
```

### Distance & Comparison

```rust
use motion_embedding::{compute_embedding_distance, DistanceWeights, heading_angle_between};

// Default weighted distance
let dist = compute_embedding_distance(&emb1, &emb2, None);

// Custom weights
let weights = DistanceWeights {
    scale: 1.0,
    shape: 2.0,      // Emphasize shape similarity
    momentum: 1.0,
    heading: 0.5,    // De-emphasize heading
    ..Default::default()
};
let dist = compute_embedding_distance(&emb1, &emb2, Some(&weights));

// Detailed breakdown
let breakdown = compute_embedding_distance_detailed(&emb1, &emb2, None);
println!("Shape distance: {}", breakdown.shape);
println!("Heading similarity: {}", breakdown.heading_similarity);

// Heading angle (degrees)
let angle = heading_angle_between(&emb1, &emb2);
```

### Validation

```rust
use motion_embedding::validate_embedding_variance;

let embeddings: Vec<MotionEmbedding> = /* ... */;
if let Some(analysis) = validate_embedding_variance(&embeddings) {
    println!("PC1 Variance: {:.1}%", analysis.pc1_variance_percent);
    println!("Effective Dim: {:.1}", analysis.effective_dimensionality);
    println!("Healthy: {}", analysis.is_healthy);
}
```

**Health Targets:**
- PC1 Variance < 40%
- Effective Dimensionality > 8
- Max/Min Variance Ratio < 50

## Feature Descriptions

### Scale (1D)
Logarithmic measure of trajectory spatial extent. A helix with 5m radius vs 50m radius will have different scale values. Tanh-compressed for range balance.

### Shape (3D)
Normalized eigenvalue ratios from 4D PCA. Captures trajectory geometry:
- Equal ratios: spherical/compact motion
- Unequal ratios: elongated/linear motion

### Momentum (6D)
4D angular momentum bivector components representing rotational dynamics in spacetime:
- Spatial: `[L_xy, L_xz, L_yz]` - rotation in spatial planes
- Temporal: `[L_xw, L_yw, L_zw]` - spacetime boost components

### Navigation (10D)
- **Heading (3D)**: Unit vector of primary motion direction
- **Maneuver Plane (3D)**: Normal to the plane of turning
- **Velocity (3D)**: Current velocity normalized by mean speed
- **Chirality (1D)**: Handedness/helicity of the trajectory (-1=left, +1=right)

### Safety (4D)
- **Mean G-Force**: Average acceleration load
- **Max G-Force**: Peak acceleration
- **Jerk Index**: Rate of change of acceleration
- **Smoothness**: Motion smoothness based on acceleration variance (1=smooth, 0=jerky)

## Architecture

```
motion_embedding/
├── lib.rs           # Public API re-exports
├── config.rs        # EmbeddingConfig + presets
├── embedding.rs     # MotionEmbedding struct
├── encoder.rs       # Main compute_motion_embedding()
├── decoder.rs       # Trajectory reconstruction
├── streaming.rs     # RollingHorizonEmbedder
├── distance.rs      # Embedding comparison
├── validation.rs    # Variance analysis
├── error.rs         # Error types
└── math/
    ├── bivector.rs    # 4D wedge product
    ├── compression.rs # Tanh/log compression
    ├── fft.rs         # Spectral analysis
    └── linalg.rs      # PCA with sign correction
```

## Performance

- **Encoding**: ~50μs for 100-point trajectory (release mode)
- **Streaming**: Lazy evaluation with caching
- **Memory**: 24 * 8 = 192 bytes per compact embedding

## Examples

### Trajectory Classification

```rust
let embeddings: Vec<[f64; 24]> = trajectories
    .iter()
    .map(|(pos, ts)| {
        compute_motion_embedding(pos, ts, &config)
            .unwrap()
            .to_compact_array()
    })
    .collect();

// Feed to ML model
let predictions = model.predict(&embeddings);
```

### Similarity Search

```rust
let query_emb = compute_motion_embedding(&query_pos, &query_ts, &config)?;

let mut similarities: Vec<(usize, f64)> = database
    .iter()
    .enumerate()
    .map(|(i, emb)| (i, compute_embedding_distance(&query_emb, emb, None)))
    .collect();

similarities.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
let top_k = &similarities[..10];
```

### Real-Time Drone Tracking

```rust
let config = EmbeddingConfig::drone(true);
let mut embedder = RollingHorizonEmbedder::new(config, 2.0);

loop {
    let (pos, ts) = get_drone_position();
    embedder.update(pos, ts);

    if let Some(emb) = embedder.get_embedding() {
        if emb.max_g_force() > 3.0 {
            alert("High-G maneuver detected!");
        }
        if emb.handedness() == Handedness::Left {
            log("Left-turning pattern");
        }
    }
}
```

## Testing

```bash
# Run all tests
cargo test

# Run with output
cargo test -- --nocapture

# Run specific test
cargo test test_chirality_detection

# Run benchmarks
cargo bench
```

