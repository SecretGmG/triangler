use crate::{vectors::Vec3, Integrand, IntegrandArgs};
use std::f64::consts::PI;

pub struct BasicLTD {
    args: IntegrandArgs<f64>,
}

impl BasicLTD{
    pub fn new(args: IntegrandArgs<f64>) -> Self{
        Self{args}
    }
}

impl Integrand<f64> for BasicLTD {
    fn evaluate(&self, k: Vec3) -> f64 {
        let e = self.args.energies(k);

        let term = |i: usize, j: usize, k: usize| {
            let (ei, ej, ek) = (e[i], e[j], e[k]);
            let (qi_t, qj_t, qk_t) = (self.args.qs[i].t(), self.args.qs[j].t(), self.args.qs[k].t());
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

pub struct ImprovedLTD {
    args: IntegrandArgs<f64>,
}
impl ImprovedLTD{
    pub fn new(args: IntegrandArgs<f64>) -> Self{
        Self{args}
    }
}

impl Integrand<f64> for ImprovedLTD {
    fn evaluate(&self, k: Vec3) -> f64 {
        let e  = self.args.energies(k);

        let mut elliptical = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                elliptical[i][j] = e[i] + e[j] - (self.args.qs[i].t() - self.args.qs[j].t());
            }
        }
        let term = |i: usize, j: usize, k: usize| {
            1.0 / (elliptical[i][j] * elliptical[i][k]) + 1.0 / (elliptical[j][i] * elliptical[k][i])
        };
        let sum = term(0, 1, 2) + term(1, 2, 0) + term(2, 0, 1);
        (1.0 / ((4.0 * PI).powi(3)*e.iter().product::<f64>())) * sum
    }
}