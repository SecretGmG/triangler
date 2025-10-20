use num_complex::{Complex64, ComplexFloat};
use olo_rust::{olo_3_point_complex, TO_FEYNMAN};
use triangler::{
    basic::{
        integrands::ImprovedLTD,
        integrators::{RealIntegratorTrait, VegasMultiIntegrator},
    }, logger::BasicLogger, parametrization::SphericalParam, vectors::LVec
};

pub fn check_triangle(p1: LVec, p2: LVec, m_psi: Complex64) -> Option<Complex64> {
    let m_psi_2 = m_psi * m_psi;
    let p3 = p1 + p2;
    let result = olo_3_point_complex(
        Complex64::from(p1.squared()),
        Complex64::from(p2.squared()),
        Complex64::from(p3.squared()),
        m_psi_2,
        m_psi_2,
        m_psi_2,
    );
    if result.epsilon_minus_1().abs() != 0.0 || result.epsilon_minus_2().abs() != 0.0 {
        None
    } else {
        Some(result.epsilon_0() * TO_FEYNMAN)
    }
}

fn get_inputs() -> Vec<(LVec, LVec, f64)> {
    vec![(
        LVec::new(0.005, 0.0, 0.0,  0.005),
        LVec::new(0.005, 0.0, 0.0, -0.005),
        0.02,
    )]
}

#[test]
fn test_havana_basic() {
    for (p1, p2, m_psi) in get_inputs() {

        let integrand = ImprovedLTD::new(p1, p2, m_psi);
        let integrator = VegasMultiIntegrator::new(10,100_000,42,100);
        let parametrization = SphericalParam::new(1.0);

        let res = integrator.integrate(&integrand, &parametrization, &Option::<BasicLogger>::None);

        let reference = check_triangle(p1, p2, Complex64::from(m_psi)).expect("the test inputs should all converge");

        assert!((res.mean-reference.re()) < res.err);
        println!("{res:?}");
        println!("{reference:?}");
    }
}
