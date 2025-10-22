pub mod real;
pub mod parametrization;
pub mod vectors;
pub mod logger;
pub mod complex;
mod traits;
use num_complex::Complex64;
pub use traits::*;
pub use crate::vectors::{LVec, Vec3};

#[derive(Clone, Copy, Debug)]
pub struct IntegrationResult {
    pub mean: f64,
    pub err: f64,
}
#[derive(Clone, Copy, Debug)]
pub struct IntegrandArgs<T>{
    pub p1: LVec,
    pub p2: LVec,
    pub qs : [LVec;3],
    pub m_psi: T,
}
impl<T> IntegrandArgs<T>{
    pub fn new(p1: LVec, p2: LVec, m_psi: T)->Self{
        Self::new_with_offset(p1, p2, LVec::ZERO, m_psi)
    }
    pub fn new_with_offset(p1: LVec, p2: LVec, offset: LVec, m_psi: T)->Self{
        Self{
            p1, p2,
            qs: [offset, p1 + offset, -p2 + offset],
            m_psi
        }
    }
}
impl IntegrandArgs<f64>{
    fn energy(k: Vec3, q: &LVec, m_psi: f64) -> f64 {
        let spatial = q.spatial();
        let kq = k - spatial;
        (kq.dot(&kq) + m_psi * m_psi).sqrt()
    }
    pub fn energies(&self, k: Vec3)->[f64;3]{
        [Self::energy(k, &self.qs[0], self.m_psi),Self::energy(k, &self.qs[1], self.m_psi),Self::energy(k, &self.qs[2], self.m_psi)]
    }
}
impl IntegrandArgs<Complex64>{
    fn energy(k: Vec3, q: &LVec, m_psi: Complex64) -> Complex64 {
        let spatial = q.spatial();
        let kq = k - spatial;
        (kq.dot(&kq) + m_psi * m_psi).sqrt()
    }
    pub fn energies(&self, k: Vec3)->[Complex64;3]{
        [Self::energy(k, &self.qs[0], self.m_psi),Self::energy(k, &self.qs[1], self.m_psi),Self::energy(k, &self.qs[2], self.m_psi)]
    }
    pub fn try_into_real(&self) -> Option<IntegrandArgs<f64>> {
        if self.m_psi.im == 0.0 {
            Some(IntegrandArgs{
                p1: self.p1, p2: self.p2,
                qs: self.qs,
                m_psi: self.m_psi.re,
            })
        } else {
            None
        }
    }
}