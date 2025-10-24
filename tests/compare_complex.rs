mod common;
use triangler::{
    integrator::{MultiIntegrator},
    parametrization::SphericalParam,
    sampler::{RealR3SamplerAggregator},
};
use common::test_real_integrator;
#[test]
fn test_havana_complex_improved_ltd() {
    test_real_integrator(
        &MultiIntegrator::new(50, 50_000, 10),
        &mut RealR3SamplerAggregator::new(SphericalParam::new(1.0), 42, 20, 1000),
    );
}
