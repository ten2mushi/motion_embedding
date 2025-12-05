//! Linear algebra utilities for motion embedding.
//!
//! This module provides PCA with sign correction and related operations
//! using nalgebra for eigendecomposition.

use crate::config::EmbeddingConfig;
use crate::error::{EmbeddingError, Result};
use nalgebra::{Matrix4, SymmetricEigen, Vector4};

/// Result of PCA computation.
#[derive(Debug, Clone)]
pub struct PcaResult {
    /// Eigenvalues sorted in descending order.
    pub eigenvalues: [f64; 4],

    /// Eigenvectors as columns, sorted by corresponding eigenvalue.
    /// `eigenvectors[i]` is the i-th column (i-th principal component).
    pub eigenvectors: [[f64; 4]; 4],
}

impl PcaResult {
    /// Get the i-th principal axis as a 4D vector.
    #[must_use]
    pub fn axis(&self, i: usize) -> [f64; 4] {
        debug_assert!(i < 4);
        [
            self.eigenvectors[0][i],
            self.eigenvectors[1][i],
            self.eigenvectors[2][i],
            self.eigenvectors[3][i],
        ]
    }

    /// Get the transformation matrix (eigenvectors as columns).
    #[must_use]
    pub fn to_matrix(&self) -> [[f64; 4]; 4] {
        self.eigenvectors
    }

    /// Compute determinant of eigenvector matrix.
    #[must_use]
    pub fn determinant(&self) -> f64 {
        let m = Matrix4::from_iterator(self.eigenvectors.iter().flatten().copied());
        m.determinant()
    }
}

/// Compute weighted PCA of 4D points.
///
/// # Arguments
///
/// * `points` - Slice of 4D points
/// * `weights` - Weight for each point (must sum to 1 for proper centroid)
///
/// # Returns
///
/// PCA result with eigenvalues and eigenvectors sorted descending.
pub fn compute_weighted_pca(points: &[[f64; 4]], weights: &[f64]) -> Result<PcaResult> {
    if points.len() != weights.len() {
        return Err(EmbeddingError::length_mismatch(points.len(), weights.len()));
    }

    if points.len() < 2 {
        return Err(EmbeddingError::trajectory_too_short(2, points.len()));
    }

    // Compute weighted covariance matrix
    // C = sum_i w_i * x_i * x_i^T (points should already be centered)
    let mut cov = [[0.0f64; 4]; 4];

    for (point, &w) in points.iter().zip(weights.iter()) {
        let sqrt_w = w.sqrt();
        let weighted = [
            point[0] * sqrt_w,
            point[1] * sqrt_w,
            point[2] * sqrt_w,
            point[3] * sqrt_w,
        ];

        for i in 0..4 {
            for j in 0..4 {
                cov[i][j] += weighted[i] * weighted[j];
            }
        }
    }

    // Convert to nalgebra matrix
    let cov_matrix = Matrix4::from_row_slice(&[
        cov[0][0], cov[0][1], cov[0][2], cov[0][3], cov[1][0], cov[1][1], cov[1][2], cov[1][3],
        cov[2][0], cov[2][1], cov[2][2], cov[2][3], cov[3][0], cov[3][1], cov[3][2], cov[3][3],
    ]);

    // Symmetric eigendecomposition
    let eigen = SymmetricEigen::new(cov_matrix);

    // Collect eigenvalue-eigenvector pairs and sort descending
    let mut pairs: Vec<(f64, Vector4<f64>)> = eigen
        .eigenvalues
        .iter()
        .enumerate()
        .map(|(i, &v)| (v, eigen.eigenvectors.column(i).into_owned()))
        .collect();

    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    // Extract sorted results
    let eigenvalues = [pairs[0].0, pairs[1].0, pairs[2].0, pairs[3].0];

    // Build eigenvector matrix (columns are eigenvectors)
    let eigenvectors = [
        [pairs[0].1[0], pairs[1].1[0], pairs[2].1[0], pairs[3].1[0]],
        [pairs[0].1[1], pairs[1].1[1], pairs[2].1[1], pairs[3].1[1]],
        [pairs[0].1[2], pairs[1].1[2], pairs[2].1[2], pairs[3].1[2]],
        [pairs[0].1[3], pairs[1].1[3], pairs[2].1[3], pairs[3].1[3]],
    ];

    Ok(PcaResult {
        eigenvalues,
        eigenvectors,
    })
}

/// Apply sign correction to principal axes for deterministic orientation.
///
/// This ensures consistent sign conventions across different trajectories
/// using a hierarchy of heuristics:
///
/// 1. Net displacement alignment
/// 2. Cumulative velocity flow
/// 3. Arrow of time (positive temporal component)
/// 4. Largest component positive
/// 5. Right-handed coordinate system
pub fn apply_sign_correction(
    pca: &mut PcaResult,
    centered_points: &[[f64; 4]],
    velocities_4d: &[[f64; 4]],
    dt: &[f64],
    config: &EmbeddingConfig,
) {
    let eps = config.singularity_eps;

    // Net displacement vector
    let n = centered_points.len();
    let net_displacement = [
        centered_points[n - 1][0] - centered_points[0][0],
        centered_points[n - 1][1] - centered_points[0][1],
        centered_points[n - 1][2] - centered_points[0][2],
        centered_points[n - 1][3] - centered_points[0][3],
    ];

    for i in 0..4 {
        let axis = [
            pca.eigenvectors[0][i],
            pca.eigenvectors[1][i],
            pca.eigenvectors[2][i],
            pca.eigenvectors[3][i],
        ];

        // Heuristic 1: Net displacement
        let displacement_score = dot4(&axis, &net_displacement);
        let should_flip = if displacement_score.abs() > eps {
            displacement_score < 0.0
        } else {
            // Heuristic 2: Cumulative velocity flow
            let flow_score: f64 = velocities_4d
                .iter()
                .zip(dt.iter())
                .map(|(v, &d)| dot4(&axis, v) * d)
                .sum();

            if flow_score.abs() > eps {
                flow_score < 0.0
            } else {
                // Heuristic 3: Arrow of time
                let temporal_component = axis[3];
                if temporal_component.abs() > eps {
                    temporal_component < 0.0
                } else {
                    // Heuristic 4: Largest component positive
                    let max_idx = axis
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap())
                        .map(|(i, _)| i)
                        .unwrap_or(0);

                    axis[max_idx] < 0.0
                }
            }
        };

        if should_flip {
            for j in 0..4 {
                pca.eigenvectors[j][i] = -pca.eigenvectors[j][i];
            }
        }
    }

    // Heuristic 5: Ensure right-handed coordinate system
    if pca.determinant() < 0.0 {
        // Flip the last axis
        for j in 0..4 {
            pca.eigenvectors[j][3] = -pca.eigenvectors[j][3];
        }
    }
}

/// Dot product of two 4D vectors.
#[inline]
fn dot4(a: &[f64; 4], b: &[f64; 4]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
}

/// Transform a 4D point using the PCA axes.
#[must_use]
pub fn transform_point(point: &[f64; 4], axes: &[[f64; 4]; 4]) -> [f64; 4] {
    // result = axes^T * point (project onto principal components)
    [
        axes[0][0] * point[0] + axes[1][0] * point[1] + axes[2][0] * point[2] + axes[3][0] * point[3],
        axes[0][1] * point[0] + axes[1][1] * point[1] + axes[2][1] * point[2] + axes[3][1] * point[3],
        axes[0][2] * point[0] + axes[1][2] * point[1] + axes[2][2] * point[2] + axes[3][2] * point[3],
        axes[0][3] * point[0] + axes[1][3] * point[1] + axes[2][3] * point[2] + axes[3][3] * point[3],
    ]
}

/// Inverse transform (from canonical back to world frame).
#[must_use]
pub fn inverse_transform_point(point: &[f64; 4], axes: &[[f64; 4]; 4]) -> [f64; 4] {
    // result = axes * point (axes columns are basis vectors)
    [
        axes[0][0] * point[0] + axes[0][1] * point[1] + axes[0][2] * point[2] + axes[0][3] * point[3],
        axes[1][0] * point[0] + axes[1][1] * point[1] + axes[1][2] * point[2] + axes[1][3] * point[3],
        axes[2][0] * point[0] + axes[2][1] * point[1] + axes[2][2] * point[2] + axes[2][3] * point[3],
        axes[3][0] * point[0] + axes[3][1] * point[1] + axes[3][2] * point[2] + axes[3][3] * point[3],
    ]
}

/// Normalize a 3D vector to unit length.
#[must_use]
pub fn normalize3(v: &[f64; 3]) -> [f64; 3] {
    let norm = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if norm < 1e-10 {
        return [0.0, 0.0, 0.0];
    }
    [v[0] / norm, v[1] / norm, v[2] / norm]
}

/// Compute the norm of a 3D vector.
#[must_use]
#[inline]
pub fn norm3(v: &[f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

/// Dot product of two 3D vectors.
#[must_use]
#[inline]
pub fn dot3(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn identity_pca() -> PcaResult {
        PcaResult {
            eigenvalues: [1.0, 1.0, 1.0, 1.0],
            eigenvectors: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }

    #[test]
    fn test_pca_simple() {
        // Points along x-axis should have major eigenvalue in x direction
        let points = vec![
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 0.0],
        ];
        let weights = vec![1.0 / 3.0; 3];

        // Center the points
        let centroid = [1.0, 0.0, 0.0, 0.0];
        let centered: Vec<_> = points
            .iter()
            .map(|p| [p[0] - centroid[0], p[1], p[2], p[3]])
            .collect();

        let pca = compute_weighted_pca(&centered, &weights).unwrap();

        // First eigenvalue should be largest
        assert!(pca.eigenvalues[0] >= pca.eigenvalues[1]);
        assert!(pca.eigenvalues[1] >= pca.eigenvalues[2]);
        assert!(pca.eigenvalues[2] >= pca.eigenvalues[3]);
    }

    #[test]
    fn test_transform_round_trip() {
        let pca = identity_pca();
        let point = [1.0, 2.0, 3.0, 4.0];

        let transformed = transform_point(&point, &pca.eigenvectors);
        let recovered = inverse_transform_point(&transformed, &pca.eigenvectors);

        for i in 0..4 {
            assert_relative_eq!(point[i], recovered[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_normalize3() {
        let v = [3.0, 4.0, 0.0];
        let n = normalize3(&v);
        assert_relative_eq!(n[0], 0.6);
        assert_relative_eq!(n[1], 0.8);
        assert_relative_eq!(n[2], 0.0);
    }

    #[test]
    fn test_determinant() {
        let pca = identity_pca();
        assert_relative_eq!(pca.determinant(), 1.0, epsilon = 1e-10);
    }
}
