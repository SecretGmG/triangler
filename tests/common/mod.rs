use num_complex::Complex64;
use olo_rust::{olo_3_point_complex, TO_FEYNMAN};
use triangler::{integrands::TriangleIntegrandBuilder, integrator::Integrator, sampler::SamplerAggregator, LVec, Vec3};

pub fn test_real_integrator(
    integrator: &impl Integrator<Vec3, f64>,
    sampler: &mut impl SamplerAggregator<Vec3, f64>,
) {
    for (builder, reference) in get_builder_result_pairs() {
        if let Some(integrand) = builder.real_basic_ltd() {
            integrator.integrate(&integrand, sampler);
            println!("{}", sampler.get_report(Some(reference.re)));
        }
    }
}
pub fn test_complex_integrator(
    integrator: &impl Integrator<Vec3, Complex64>,
    sampler: &mut impl SamplerAggregator<Vec3, Complex64>,
) {
    for (builder, reference) in get_builder_result_pairs() {
        let integrand = builder.basic_ltd();
        integrator.integrate(&integrand, sampler);
        println!("{}", sampler.get_report(Some(reference)));
    }
}


pub fn get_reference_value(p1: LVec, p2: LVec, m: Complex64) -> Complex64 {
    let m_psi_2 = m.powi(2);
    let result = olo_3_point_complex(
        Complex64::from(p1.squared()),
        Complex64::from(p2.squared()),
        Complex64::from((p1 + p2).squared()),
        m_psi_2,
        m_psi_2,
        m_psi_2,
    );
    result.epsilon_0() * TO_FEYNMAN
}
pub fn get_builders() -> Vec<TriangleIntegrandBuilder> {
    vec![TriangleIntegrandBuilder::new(
        LVec::new(0.005, 0.0, 0.0,  0.005),
        LVec::new(0.005, 0.0, 0.0, -0.005),
        Complex64::from(0.02),
    ),TriangleIntegrandBuilder::new(
        LVec::new(0.005, 0.0, 0.0,  0.005),
        LVec::new(0.005, 0.0, 0.0, -0.005),
        Complex64::new(0.02,-0.01),
    ),
    TriangleIntegrandBuilder::new(
        LVec::new(5.0, 0.2, 0.1,  0.01),
        LVec::new(1.0, 0.1, 0.01, -0.01),
        Complex64::new(0.1,-0.04),
    ),
    TriangleIntegrandBuilder::new(
        LVec::new(1.0, 0.0, 0.0,  1.0),
        LVec::new(1.0, 0.0, 0.0, -1.0),
        Complex64::new(0.5,-0.0001),
    ),
    ]
}
pub fn get_builder_result_pairs() -> Vec<(TriangleIntegrandBuilder, Complex64)> {
    let builders = get_builders();
    builders.into_iter()
        .map(|builder| {
            let res = get_reference_value(builder.p1, builder.p2, builder.m);
            (builder, res)
        })
        .collect()
}