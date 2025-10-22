use num_complex::Complex64;
use olo_rust::{olo_3_point_complex, TO_FEYNMAN};
use triangler::{IntegrandArgs, LVec};

pub fn get_reference_value(args: IntegrandArgs<Complex64>) -> Complex64 {
    let m_psi_2 = args.m_psi.powi(2);
    let result = olo_3_point_complex(
        Complex64::from(args.p1.squared()),
        Complex64::from(args.p2.squared()),
        Complex64::from((args.p1 + args.p2).squared()),
        m_psi_2,
        m_psi_2,
        m_psi_2,
    );
    result.epsilon_0() * TO_FEYNMAN
}
pub fn get_args() -> Vec<IntegrandArgs<Complex64>> {
    vec![IntegrandArgs::new(
        LVec::new(0.005, 0.0, 0.0,  0.005),
        LVec::new(0.005, 0.0, 0.0, -0.005),
        Complex64::from(0.02),
    ),IntegrandArgs::new(
        LVec::new(0.005, 0.0, 0.0,  0.005),
        LVec::new(0.005, 0.0, 0.0, -0.005),
        Complex64::new(0.02,-0.01),
    )]
}
pub fn get_args_result_pairs() -> Vec<(IntegrandArgs<Complex64>, Complex64)> {
    let args = get_args();
    args.into_iter()
        .map(|arg| {
            let res = get_reference_value(arg);
            (arg, res)
        })
        .collect()
}
pub fn get_real_args_result_pairs() -> Vec<(IntegrandArgs<f64>, f64)> {
    get_args_result_pairs()
        .into_iter()
        .filter_map(|(arg, res)| {
            if let Some(real_arg) = arg.try_into_real() {
                Some((real_arg, res.re))
            } else {
                None
            }
        })
        .collect()
}