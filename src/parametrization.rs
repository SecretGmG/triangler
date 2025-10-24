use std::f64::consts::PI;
use crate::{vectors::Vec3};

pub trait Parametrization : Clone + Sync + Send {
    fn transform(&self, xs: Vec3) -> (Vec3, f64);
}

#[derive(Clone)]
pub struct CartesianParam {
    scale: f64,
}
impl CartesianParam{
    pub fn new(scale : f64)-> Self{
        return Self{scale}
    }
}

impl Parametrization for CartesianParam {
    fn transform(&self, xs: Vec3) -> (Vec3, f64) {
        // Polynomial map: [0,1] → (-∞, ∞)
        fn poly_map(x: f64) -> f64 {
            1.0 / (1.0 - x) - 1.0 / x
        }
        fn poly_map_jac(x: f64) -> f64 {
            1.0 / (1.0 - x).powi(2) + 1.0 / x.powi(2)
        }

        let mapped = Vec3 {
            x: self.scale * poly_map(xs.x),
            y: self.scale * poly_map(xs.y),
            z: self.scale * poly_map(xs.z),
        };

        let jac = poly_map_jac(xs.x)
            * poly_map_jac(xs.y)
            * poly_map_jac(xs.z)
            * self.scale.powi(3);

        (mapped, jac)
    }
}


#[derive(Clone)]
pub struct SphericalParam {
    scale: f64,
}

impl SphericalParam{
    pub fn new(scale : f64)-> Self{
        return Self{scale}
    }
}
impl Parametrization for SphericalParam {
    fn transform(&self, xs: Vec3) -> (Vec3, f64) {
        let r = xs.x / (1.0 - xs.x);
        let r_jac = 1.0 / (1.0 - xs.x).powi(2);

        let th = xs.y * 2.0 * PI;
        let th_jac = 2.0 * PI;

        let phi = xs.z * PI;
        let phi_jac = PI;

        let (sin_phi, cos_phi) = phi.sin_cos();

        let v = Vec3 {
            x: self.scale * r * sin_phi * th.cos(),
            y: self.scale * r * sin_phi * th.sin(),
            z: self.scale * r * cos_phi,
        };

        let jac = r_jac * r.powi(2) * th_jac * phi_jac * sin_phi * self.scale.powi(3);

        (v, jac)
    }
}