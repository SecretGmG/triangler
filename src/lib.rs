pub mod implementations;
pub mod parametrization;
pub mod logger;
pub mod traits;
mod vectors;
pub use traits::{Integrator, Integrand, Parametrization};
pub use vectors::{LVec, Vec3};
use num_complex::{Complex64};

pub mod prelude{
    pub use crate::{Integrator, Integrand, Parametrization, Vec3, LVec, IntegrationResult, IntegrandArgs};
}
#[derive(Clone, Copy, Debug)]
pub struct IntegrationResult {
    pub mean: f64,
    pub err: f64,
}
#[derive(Clone, Copy, Debug)]
pub struct IntegrandArgs{
    pub p1: LVec,
    pub p2: LVec,
    pub qs : [LVec;3],
    pub m_psi: Complex64,
}
impl IntegrandArgs{
    pub fn new(p1: LVec, p2: LVec, m_psi: Complex64)->Self{
        // Per default center the singularities around the origin
        Self::new_with_offset(p1, p2, (p1-p2)*(1.0/3.0), m_psi)
    }
    pub fn new_with_offset(p1: LVec, p2: LVec, offset: LVec, m_psi: Complex64)->Self{
        Self{
            p1, p2,
            qs: [-offset, p1 - offset, -p2 - offset],
            m_psi
        }
    }
    fn energy(k: Vec3, q: &LVec, m_psi: Complex64) -> Complex64 {
        let spatial = q.spatial();
        let kq = k - spatial;
        (kq.dot(&kq) + m_psi.powi(2)).sqrt()
    }
    fn real_energy(k: Vec3, q: &LVec, m_psi: f64) -> f64 {
        let spatial = q.spatial();
        let kq = k - spatial;
        (kq.dot(&kq) + m_psi.powi(2)).sqrt()
    }
    pub fn energies(&self, k: Vec3)->[Complex64;3]{
        [Self::energy(k, &self.qs[0], self.m_psi),Self::energy(k, &self.qs[1], self.m_psi),Self::energy(k, &self.qs[2], self.m_psi)]
    }
    pub fn real_energies(&self, k: Vec3)->[f64;3]{
        [Self::real_energy(k, &self.qs[0], self.m_psi.re),Self::real_energy(k, &self.qs[1], self.m_psi.re),Self::real_energy(k, &self.qs[2], self.m_psi.re)]
    }
    pub fn is_real(&self) -> bool{
        self.m_psi.im == 0.0
    }
}