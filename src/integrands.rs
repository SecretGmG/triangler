use num_complex::Complex64;

use crate::{vectors::Vec3, LVec};
use std::f64::consts::PI;

pub struct TriangleIntegrandBuilder{
    pub p1: LVec,
    pub p2: LVec,
    pub m: Complex64,
}
impl Default for TriangleIntegrandBuilder{
    fn default() -> Self {
        Self{
            p1: LVec::ZERO,
            p2: LVec::ZERO,
            m: Complex64::new(0.0,0.0),
        }
    }
}
impl TriangleIntegrandBuilder{
    pub fn new(p1: LVec, p2: LVec, m: Complex64) -> Self{
        Self{p1, p2, m}
    }
    pub fn with_mass(&self, m: Complex64) -> Self{
        Self{
            p1: self.p1,
            p2: self.p2,
            m,
        }
    }
    pub fn with_momenta(&self, p1: LVec, p2: LVec) -> Self{
        Self{
            p1,
            p2,
            m: self.m,
        }
    }

    // Common energy computation helper
    fn energy(k: Vec3, qi: Vec3, m: Complex64) -> Complex64 {
        let kq = k + qi;
        (kq.dot(&kq) + m * m).sqrt()
    }
    pub fn energies(&self, k: Vec3, qs : &[LVec; 3]) -> [ Complex64; 3 ] {
        [Self::energy(k, qs[0].spatial(), self.m), Self::energy(k, qs[1].spatial(), self.m),Self::energy(k, qs[2].spatial(), self.m)]
    }
    pub fn basic_ltd(&self) -> impl Fn(Vec3) -> Complex64 + '_ {
        let qs = [LVec::ZERO, self.p1, -self.p2];
        move |k: Vec3| {
            let e = self.energies(k, &qs);
            let term = |i: usize, j: usize, k: usize| {
            let (ei, ej, ek) = (e[i], e[j], e[k]);
            let (qi_t, qj_t, qk_t) = (qs[i].t(), qs[j].t(), qs[k].t());
            1.0 / (2.0
                * ei
                * (ei + ej + (qi_t - qj_t))
                * (ei - ej + (qi_t - qj_t))
                * (ei + ek + (qi_t - qk_t))
                * (ei - ek + (qi_t - qk_t)))
        };

        let sum = term(0, 1, 2) + term(1, 0, 2) + term(2, 0, 1);
        (1.0 / (2.0 * PI).powi(3)) * sum
        }
    }
    pub fn real_basic_ltd(&self) -> Option<impl Fn(Vec3) -> f64 + '_> {
        let func = self.basic_ltd();
        if self.m.im != 0.0 {
            return None;
        }
        Some(move |k: Vec3| {
            func(k).re
        })
    }
    pub fn improved_ltd(&self) -> impl Fn(Vec3) -> Complex64 + '_ {
        let qs = [LVec::ZERO, self.p1, -self.p2];
        move |k: Vec3| {
            let e = self.energies(k, &qs);

            let mut elliptical = [[Complex64::ZERO; 3]; 3];
            for i in 0..3 {
                for j in 0..3 {
                    elliptical[i][j] = e[i] + e[j] - (qs[i].t() - qs[j].t());
                }
            }
            let term = |i: usize, j: usize, k: usize| {
                1.0 / (elliptical[i][j] * elliptical[i][k]) + 1.0 / (elliptical[j][i] * elliptical[k][i])
            };
            let sum = term(0, 1, 2) + term(1, 2, 0) + term(2, 0, 1);
            (1.0 / ((4.0 * PI).powi(3)*e.iter().product::<Complex64>())) * sum
        }
    }
}