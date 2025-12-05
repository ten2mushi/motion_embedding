//! 4D bivector operations using geometric algebra.
//!
//! In 4D space, the wedge product of two vectors produces a bivector with
//! 6 independent components (C(4,2) = 6). This represents the oriented
//! plane spanned by the two vectors.
//!
//! # Component Ordering
//!
//! The 6 bivector components are ordered as:
//! - `e01` (L_xy): rotation in xy-plane
//! - `e02` (L_xz): rotation in xz-plane
//! - `e03` (L_xw): boost in x-time plane
//! - `e12` (L_yz): rotation in yz-plane
//! - `e13` (L_yw): boost in y-time plane
//! - `e23` (L_zw): boost in z-time plane

use std::ops::{Add, AddAssign, Mul, Sub};

/// 4D bivector with 6 components.
///
/// Represents an oriented 2D plane in 4D spacetime.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Bivector6 {
    /// The 6 bivector components [e01, e02, e03, e12, e13, e23].
    pub components: [f64; 6],
}

impl Bivector6 {
    /// Create a new bivector from components.
    #[must_use]
    pub const fn new(components: [f64; 6]) -> Self {
        Self { components }
    }

    /// Create a zero bivector.
    #[must_use]
    pub const fn zero() -> Self {
        Self {
            components: [0.0; 6],
        }
    }

    /// Compute the wedge product of two 4D vectors: u ∧ v.
    ///
    /// The wedge product produces a bivector representing the oriented
    /// parallelogram spanned by u and v.
    ///
    /// # Formula
    ///
    /// For 4D vectors u = [u0, u1, u2, u3] and v = [v0, v1, v2, v3]:
    /// - e_ij = u_i * v_j - u_j * v_i
    #[must_use]
    pub fn wedge(u: &[f64; 4], v: &[f64; 4]) -> Self {
        Self {
            components: [
                u[0] * v[1] - u[1] * v[0], // e01 (L_xy)
                u[0] * v[2] - u[2] * v[0], // e02 (L_xz)
                u[0] * v[3] - u[3] * v[0], // e03 (L_xw)
                u[1] * v[2] - u[2] * v[1], // e12 (L_yz)
                u[1] * v[3] - u[3] * v[1], // e13 (L_yw)
                u[2] * v[3] - u[3] * v[2], // e23 (L_zw)
            ],
        }
    }

    /// Compute the wedge product for 3D vectors embedded in 4D (w=0).
    ///
    /// Only produces spatial rotation components (e01, e02, e12).
    #[must_use]
    pub fn wedge_3d(u: &[f64; 3], v: &[f64; 3]) -> Self {
        Self {
            components: [
                u[0] * v[1] - u[1] * v[0], // e01 (L_xy)
                u[0] * v[2] - u[2] * v[0], // e02 (L_xz)
                0.0,                        // e03 (L_xw) - no time component
                u[1] * v[2] - u[2] * v[1], // e12 (L_yz)
                0.0,                        // e13 (L_yw) - no time component
                0.0,                        // e23 (L_zw) - no time component
            ],
        }
    }

    /// L_xy component (rotation in xy-plane).
    #[must_use]
    #[inline]
    pub const fn l_xy(&self) -> f64 {
        self.components[0]
    }

    /// L_xz component (rotation in xz-plane).
    #[must_use]
    #[inline]
    pub const fn l_xz(&self) -> f64 {
        self.components[1]
    }

    /// L_xw component (boost in x-time plane).
    #[must_use]
    #[inline]
    pub const fn l_xw(&self) -> f64 {
        self.components[2]
    }

    /// L_yz component (rotation in yz-plane).
    #[must_use]
    #[inline]
    pub const fn l_yz(&self) -> f64 {
        self.components[3]
    }

    /// L_yw component (boost in y-time plane).
    #[must_use]
    #[inline]
    pub const fn l_yw(&self) -> f64 {
        self.components[4]
    }

    /// L_zw component (boost in z-time plane).
    #[must_use]
    #[inline]
    pub const fn l_zw(&self) -> f64 {
        self.components[5]
    }

    /// Extract spatial rotation components [L_xy, L_xz, L_yz].
    #[must_use]
    pub const fn spatial(&self) -> [f64; 3] {
        [self.components[0], self.components[1], self.components[3]]
    }

    /// Extract spacetime boost components [L_xw, L_yw, L_zw].
    #[must_use]
    pub const fn temporal(&self) -> [f64; 3] {
        [self.components[2], self.components[4], self.components[5]]
    }

    /// Compute the squared magnitude of the bivector.
    #[must_use]
    pub fn magnitude_squared(&self) -> f64 {
        self.components.iter().map(|x| x * x).sum()
    }

    /// Compute the magnitude of the bivector.
    #[must_use]
    pub fn magnitude(&self) -> f64 {
        self.magnitude_squared().sqrt()
    }

    /// Normalize the bivector to unit magnitude.
    #[must_use]
    pub fn normalize(&self) -> Self {
        let mag = self.magnitude();
        if mag < 1e-10 {
            return Self::zero();
        }
        Self {
            components: [
                self.components[0] / mag,
                self.components[1] / mag,
                self.components[2] / mag,
                self.components[3] / mag,
                self.components[4] / mag,
                self.components[5] / mag,
            ],
        }
    }

    /// Scale the bivector by a scalar.
    #[must_use]
    pub fn scale(&self, s: f64) -> Self {
        Self {
            components: [
                self.components[0] * s,
                self.components[1] * s,
                self.components[2] * s,
                self.components[3] * s,
                self.components[4] * s,
                self.components[5] * s,
            ],
        }
    }

    /// Convert to array.
    #[must_use]
    pub const fn to_array(&self) -> [f64; 6] {
        self.components
    }

    /// Create from array.
    #[must_use]
    pub const fn from_array(arr: [f64; 6]) -> Self {
        Self { components: arr }
    }
}

impl Add for Bivector6 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            components: [
                self.components[0] + rhs.components[0],
                self.components[1] + rhs.components[1],
                self.components[2] + rhs.components[2],
                self.components[3] + rhs.components[3],
                self.components[4] + rhs.components[4],
                self.components[5] + rhs.components[5],
            ],
        }
    }
}

impl AddAssign for Bivector6 {
    fn add_assign(&mut self, rhs: Self) {
        for i in 0..6 {
            self.components[i] += rhs.components[i];
        }
    }
}

impl Sub for Bivector6 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            components: [
                self.components[0] - rhs.components[0],
                self.components[1] - rhs.components[1],
                self.components[2] - rhs.components[2],
                self.components[3] - rhs.components[3],
                self.components[4] - rhs.components[4],
                self.components[5] - rhs.components[5],
            ],
        }
    }
}

impl Mul<f64> for Bivector6 {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self::Output {
        self.scale(rhs)
    }
}

impl Mul<Bivector6> for f64 {
    type Output = Bivector6;

    fn mul(self, rhs: Bivector6) -> Self::Output {
        rhs.scale(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_wedge_product() {
        // x ∧ y should give e01
        let x = [1.0, 0.0, 0.0, 0.0];
        let y = [0.0, 1.0, 0.0, 0.0];
        let bv = Bivector6::wedge(&x, &y);

        assert_relative_eq!(bv.l_xy(), 1.0);
        assert_relative_eq!(bv.l_xz(), 0.0);
        assert_relative_eq!(bv.l_xw(), 0.0);
    }

    #[test]
    fn test_wedge_antisymmetry() {
        // u ∧ v = -(v ∧ u)
        let u = [1.0, 2.0, 3.0, 4.0];
        let v = [5.0, 6.0, 7.0, 8.0];

        let uv = Bivector6::wedge(&u, &v);
        let vu = Bivector6::wedge(&v, &u);

        for i in 0..6 {
            assert_relative_eq!(uv.components[i], -vu.components[i]);
        }
    }

    #[test]
    fn test_wedge_self_zero() {
        // u ∧ u = 0
        let u = [1.0, 2.0, 3.0, 4.0];
        let bv = Bivector6::wedge(&u, &u);

        for c in bv.components {
            assert_relative_eq!(c, 0.0);
        }
    }

    #[test]
    fn test_magnitude() {
        let bv = Bivector6::new([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert_relative_eq!(bv.magnitude(), 1.0);

        let bv2 = Bivector6::new([3.0, 4.0, 0.0, 0.0, 0.0, 0.0]);
        assert_relative_eq!(bv2.magnitude(), 5.0);
    }

    #[test]
    fn test_normalize() {
        let bv = Bivector6::new([3.0, 4.0, 0.0, 0.0, 0.0, 0.0]);
        let normalized = bv.normalize();

        assert_relative_eq!(normalized.magnitude(), 1.0);
        assert_relative_eq!(normalized.components[0], 0.6);
        assert_relative_eq!(normalized.components[1], 0.8);
    }

    #[test]
    fn test_add() {
        let a = Bivector6::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = Bivector6::new([6.0, 5.0, 4.0, 3.0, 2.0, 1.0]);
        let c = a + b;

        assert_relative_eq!(c.components[0], 7.0);
        assert_relative_eq!(c.components[5], 7.0);
    }

    #[test]
    fn test_spatial_temporal() {
        let bv = Bivector6::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let spatial = bv.spatial();
        assert_eq!(spatial, [1.0, 2.0, 4.0]);

        let temporal = bv.temporal();
        assert_eq!(temporal, [3.0, 5.0, 6.0]);
    }
}
