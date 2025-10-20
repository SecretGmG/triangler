use crate::vectors::{LVec, Vec3};
use enum_dispatch::enum_dispatch;
use std::f64::consts::PI;

// Common energy computation helper
fn energy(k: Vec3, qi: LVec, m_psi: f64) -> f64 {
    let spatial = qi.spatial();
    let kq = k + spatial;
    (kq.dot(&kq) + m_psi * m_psi).sqrt()
}

#[enum_dispatch]
pub trait RealIntegrandTrait: Sync + Send {
    fn evaluate(&self, k: Vec3) -> f64;
} // Trait defining the common interface for any integrand mode

// ---------- BASIC LTD IMPLEMENTATION ----------
pub struct BasicLTD {
    p: LVec,
    q: LVec,
    m_psi: f64,
}

impl BasicLTD{
    pub fn new(p: LVec, q: LVec, m_psi: f64) -> Self{
        Self{p,q,m_psi}
    }
}

impl RealIntegrandTrait for BasicLTD {
    fn evaluate(&self, k: Vec3) -> f64 {
        let qv = [LVec::new(0.0, 0.0, 0.0, 0.0), -self.q, self.p];
        let e: Vec<f64> = qv.iter().map(|qi| energy(k, *qi, self.m_psi)).collect();

        let term = |i: usize, j: usize, k: usize| {
            let (ei, ej, ek) = (e[i], e[j], e[k]);
            let (qi_t, qj_t, qk_t) = (qv[i].t(), qv[j].t(), qv[k].t());
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

// ---------- IMPROVED LTD IMPLEMENTATION ----------
pub struct ImprovedLTD {
    p: LVec,
    q: LVec,
    m_psi: f64,
}
impl ImprovedLTD{
    pub fn new(p: LVec, q: LVec, m_psi: f64) -> Self{
        Self{p,q,m_psi}
    }
}

impl RealIntegrandTrait for ImprovedLTD {
    fn evaluate(&self, k: Vec3) -> f64 {
        let qv = [LVec::new(0.0, 0.0, 0.0, 0.0), -self.q, self.p];
        let e: Vec<f64> = qv.iter().map(|qi| energy(k, *qi, self.m_psi)).collect();

        let mut eta = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                eta[i][j] = e[i] + e[j];
            }
        }

        let sum = 1.0 / ((eta[0][2] - qv[0].t() + qv[2].t()) * (eta[1][2] - qv[1].t() + qv[2].t()))
            + 1.0 / ((eta[0][1] + qv[0].t() - qv[1].t()) * (eta[1][2] - qv[1].t() + qv[2].t()))
            + 1.0 / ((eta[0][1] + qv[0].t() - qv[1].t()) * (eta[0][2] + qv[0].t() - qv[2].t()))
            + 1.0 / ((eta[0][2] + qv[0].t() - qv[2].t()) * (eta[1][2] + qv[1].t() - qv[2].t()))
            + 1.0 / ((eta[0][1] - qv[0].t() + qv[1].t()) * (eta[1][2] + qv[1].t() - qv[2].t()))
            + 1.0 / ((eta[0][1] - qv[0].t() + qv[1].t()) * (eta[0][2] - qv[0].t() + qv[2].t()));

        (2.0 * PI).powi(-3) / (2.0 * e[0] * 2.0 * e[1] * 2.0 * e[2]) * sum
    }
}