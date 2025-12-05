# Motion Embedding Variance Analysis Report

## Executive Summary

The motion embedding library produces 24-dimensional embeddings with significant variance imbalance across feature groups. While PC1 variance (33.7%) meets the <40% health target, the **max/min variance ratio of 1095:1** far exceeds the <50 target, and effective dimensionality (7.0) falls slightly below the >8 target.

This report identifies root causes and provides actionable solutions.

---

## Variance Analysis Results

### Feature Group Variances

| Feature Group | Variance | Status |
|--------------|----------|--------|
| **Smoothness** | 0.0008 | CRITICAL - Near zero |
| **Scale** | 0.0029 | CRITICAL - Very low |
| **Chirality** | 0.0793 | WARNING - Low |
| Momentum | 0.4188 | OK |
| G-Force | 0.3468 | OK |
| Shape | 0.7362 | OK |
| Velocity | 0.7567 | OK |
| Maneuver | 0.7876 | OK |
| Heading | 0.8416 | OK |

### Health Metrics

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| PC1 Variance | 33.7% | <40% | PASS |
| Effective Dimensionality | 7.0 | >8 | FAIL |
| Max/Min Variance Ratio | 1095.4 | <50 | FAIL |

---

## Root Cause Analysis

### 1. Smoothness (Variance: 0.0008) - CRITICAL

**Observed Values in Test Data:**
```
helix:      0.100 - 0.102
circle:     0.101 - 0.114
figure8:    0.101 - 0.107
line:       0.101 (all identical)
random:     0.100 - 0.101
aggressive: 0.181 - 0.190
hover:      0.100 - 0.101
```

**Root Cause:** The smoothness calculation in `math/fft.rs:109-112`:

```rust
let hf_ratio = total_high_freq_energy / total_energy;
let smoothness_index = compute_smoothness_index(hf_ratio, config.smoothness_sensitivity);
```

Where `compute_smoothness_index` (compression.rs:213-220):

```rust
pub fn compute_smoothness_index(hf_ratio: f64, sensitivity: f64) -> f64 {
    if hf_ratio < 1e-10 {
        return 1.0;
    }
    let log_ratio = (hf_ratio / sensitivity).ln();
    1.0 / (1.0 + (log_ratio / 2.0).exp())  // Sigmoid in [0, 1]
}
```

**Problem:** The sigmoid function with `smoothness_sensitivity=0.001` maps most trajectories to ~0.1:
- Default sensitivity is too low
- The function saturates quickly
- Output range [0.1, 0.2] instead of full [0, 1]

### 2. Scale (Variance: 0.0029) - CRITICAL

**Observed Values:**
```
helix:      0.797 - 0.813
circle:     0.729 - 0.806
figure8:    0.794 - 0.822
line:       0.695 (all identical)
random:     0.760 - 0.770
aggressive: 0.689 - 0.740
hover:      0.851 - 0.858
```

**Root Cause:** Scale compression in `compression.rs:89-91`:

```rust
pub fn compress_scale(raw_log_scale: f64, config: &EmbeddingConfig) -> f64 {
    ((raw_log_scale - config.scale_center) / config.scale_width).tanh()
}
```

With defaults: `scale_center=0.5`, `scale_width=1.0`

**Problem:** All test trajectories have similar spatial extent:
- 100 points over 3-10 seconds
- Radii of 3-15 meters
- Log-scale values cluster around 0.5-0.8
- After tanh compression, range becomes [0.69, 0.86]

### 3. Chirality (Variance: 0.0793) - WARNING

**Observed Values:**
```
helix:      0.149 - 0.999 (varies with pitch/radius ratio)
circle:     0.0 (all)
figure8:    0.0 (all)
line:       0.0 (all)
random:    -0.118 to 0.268
aggressive: 0.0 (all)
hover:     -0.144 to 0.179
```

**Root Cause:** Many trajectory types have zero or near-zero chirality by definition:
- Circles are planar → no helicity
- Figure-8s are symmetric → no net helicity
- Lines have no rotation
- Aggressive patterns are box-like → no helicity

The chirality compression (tanh with scale=2.0) is appropriate, but the test data lacks chirality diversity.

---

## Proposed Solutions

### Solution 1: Fix Smoothness Calculation (HIGH PRIORITY)

**Option A: Adjust smoothness sensitivity**

```rust
// In config.rs, change default
pub smoothness_sensitivity: f64,  // Change from 0.001 to 0.1
```

**Option B: Transform to full [0, 1] range**

```rust
// In compression.rs, new function
pub fn compute_smoothness_index_v2(hf_ratio: f64, sensitivity: f64) -> f64 {
    if hf_ratio < 1e-10 {
        return 1.0;
    }
    // Use log-ratio directly with tanh for full [-1, 1] then rescale to [0, 1]
    let log_ratio = (hf_ratio / sensitivity).ln();
    let compressed = (-log_ratio / 4.0).tanh();  // Flip sign: low HF = high smoothness
    (compressed + 1.0) / 2.0  // Map to [0, 1]
}
```

**Option C: Use quantile-based normalization**

Instead of absolute HF ratio, use percentile within expected distribution:

```rust
pub fn compute_smoothness_quantile(hf_ratio: f64) -> f64 {
    // Based on empirical distribution of HF ratios
    // Map to CDF-like transformation
    let log_hf = (hf_ratio + 1e-10).ln();
    let z = (log_hf - (-4.0)) / 2.0;  // Center around typical log(HF) values
    1.0 / (1.0 + z.exp())
}
```

### Solution 2: Fix Scale Compression (HIGH PRIORITY)

**Option A: Widen the scale range**

```rust
// In config.rs
pub scale_center: f64,  // Change from 0.5 to 1.0 (log10 of ~10m typical)
pub scale_width: f64,   // Change from 1.0 to 2.0 (wider compression)
```

**Option B: Use relative scale within batch**

For ML applications, consider z-score normalization at batch level:

```rust
impl MotionEmbedding {
    pub fn normalize_scale_batch(embeddings: &mut [Self]) {
        let scales: Vec<f64> = embeddings.iter().map(|e| e.scale_magnitude).collect();
        let mean = scales.iter().sum::<f64>() / scales.len() as f64;
        let std = (scales.iter().map(|s| (s - mean).powi(2)).sum::<f64>()
                   / scales.len() as f64).sqrt();

        for emb in embeddings {
            emb.scale_magnitude = ((emb.scale_magnitude - mean) / (std + 1e-7)).tanh();
        }
    }
}
```

**Option C: Log-ratio against characteristic size**

```rust
pub fn compress_scale_v2(raw_log_scale: f64, characteristic_size: f64, width: f64) -> f64 {
    let char_log = characteristic_size.ln();
    let ratio = (raw_log_scale - char_log) / width;
    ratio.tanh()
}
```

### Solution 3: Improve Test Data Diversity (MEDIUM PRIORITY)

Add trajectory generators that exercise low-variance features:

```rust
// In export_embeddings.rs

/// Generate trajectories with varying smoothness
fn generate_jerky_trajectory(n: usize, jerk_frequency: usize) -> (Vec<[f64; 3]>, Vec<f64>) {
    // Add sharp direction changes at jerk_frequency intervals
}

/// Generate trajectories at different scales
fn generate_scaled_helix(n: usize, scale: f64) -> (Vec<[f64; 3]>, Vec<f64>) {
    // scale ranges from 0.1 (tiny) to 1000 (huge)
}

/// Generate left-handed and right-handed helices
fn generate_helix_with_chirality(n: usize, handedness: i32) -> (Vec<[f64; 3]>, Vec<f64>) {
    // handedness: -1 (left), +1 (right)
}
```

### Solution 4: Post-hoc Variance Balancing (OPTIONAL)

For ML training, apply feature-group-wise normalization:

```rust
pub struct VarianceBalancer {
    group_means: [f64; 9],
    group_stds: [f64; 9],
}

impl VarianceBalancer {
    pub fn fit(embeddings: &[MotionEmbedding]) -> Self {
        // Compute per-group statistics
    }

    pub fn transform(&self, embedding: &MotionEmbedding) -> [f64; 24] {
        // Z-score normalize each feature group
    }
}
```

---

## Recommended Configuration Changes

### Immediate Fixes (config.rs)

```rust
impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            // ... existing fields ...

            // CHANGED: Increase smoothness sensitivity for better spread
            smoothness_sensitivity: 0.1,  // was 0.001

            // CHANGED: Wider scale compression
            scale_center: 1.0,   // was 0.5
            scale_width: 2.5,    // was 1.0

            // CHANGED: Reduce chirality scale for more sensitivity
            chirality_scale: 1.0,  // was 2.0

            // ... rest unchanged ...
        }
    }
}
```

### New Preset for Variance-Balanced Training

```rust
impl EmbeddingConfig {
    /// Preset optimized for ML training with balanced variance
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
}
```

---

## Validation Procedure

After implementing fixes, run this validation:

```bash
# 1. Export embeddings with new config
cargo test --test export_embeddings -- --ignored --nocapture

# 2. Run Python analysis
cd python
poetry run python analyze_embeddings.py embeddings.json

# 3. Check health targets:
#    - PC1 Variance < 40%
#    - Effective Dimensionality > 8
#    - Max/Min Variance Ratio < 50
```

---

## Appendix: Feature Index Reference

| Index | Feature | Group | Expected Range |
|-------|---------|-------|----------------|
| 0 | scale_magnitude | Scale | [-1, 1] |
| 1-3 | shape_entropy[0-2] | Shape | [-1, 1] |
| 4-9 | normalized_momentum[0-5] | Momentum | [-1, 1] |
| 10-12 | current_heading[0-2] | Heading | [-1, 1] (unit vec) |
| 13-15 | maneuver_plane[0-2] | Maneuver | [-1, 1] (unit vec) |
| 16-18 | velocity_normalized[0-2] | Velocity | [-1, 1] |
| 19 | chirality | Chirality | [-1, 1] |
| 20-22 | g_force_stats[0-2] | G-Force | [-1, 1] |
| 23 | smoothness_index | Smoothness | [0, 1] |

---

## Conclusion

The primary issues are:
1. **Smoothness**: Sigmoid saturation due to low sensitivity parameter
2. **Scale**: Narrow log-scale range after tanh compression
3. **Chirality**: Test data lacks diversity (acceptable)

Implementing Solutions 1 and 2 (config parameter changes) should bring the variance ratio below 50 and effective dimensionality above 8.
