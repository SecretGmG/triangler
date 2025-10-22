use std::f64::consts::PI;

use crate::{vectors::{Vec3}, IntegrandArgs};
use enum_dispatch::enum_dispatch;
use num_complex::Complex64;

// Common energy computation helper
#[enum_dispatch]
pub trait ComplexIntegrand: Sync + Send {
    fn evaluate(&self, k: Vec3) -> Complex64;
} // Trait defining the common interface for any integrand mode

pub struct ImprovedLTD {
    args: IntegrandArgs<Complex64>,
}
impl ImprovedLTD{
    pub fn new(args : IntegrandArgs<Complex64>) -> Self{
        Self{args}
    }
}

impl ComplexIntegrand for ImprovedLTD {
    fn evaluate(&self, k: Vec3) -> Complex64 {
        let e: [Complex64; 3] = self.args.energies(k);

        let mut elliptical = [[Complex64::ZERO; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                elliptical[i][j] = e[i] + e[j] - (self.args.qs[i].t() - self.args.qs[j].t());
            }
        }
        let term = |i: usize, j: usize, k: usize| {
            1.0 / (elliptical[i][j] * elliptical[i][k]) + 1.0 / (elliptical[j][i] * elliptical[k][i])
        };
        let sum = term(0, 1, 2) + term(1, 2, 0) + term(2, 0, 1);
        (1.0 / ((4.0 * PI).powi(3)*e.iter().product::<Complex64>())) * sum
    }
}