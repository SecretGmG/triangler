use crate::Vec3;

pub trait Integrand<T>: Sync + Send {
    fn evaluate(&self, k: Vec3) -> T;
}


pub trait Integrator<T, R> {
    fn integrate(
        &self,
        integrand: &impl Integrand<T>,
        parametrization: &impl Parametrization,
    ) -> R;
}

pub trait Parametrization: Sync + Send {
    fn transform(&self, xs: Vec3) -> (Vec3, f64);
}