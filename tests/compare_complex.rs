mod common;
use common::{get_args_result_pairs};
use triangler::prelude::*;
use triangler::{implementations::{HavanaIntegrator, ImprovedLTD}, parametrization::SphericalParam};

#[test]
fn test_havana_complex_improved_ltd() {
    for (args, reference) in get_args_result_pairs() {

        let integrand = ImprovedLTD::new(args);
        let integrator = HavanaIntegrator::new(10,50_000,42);
        let parametrization = SphericalParam::new(1.0);

        let (res_re, res_im) = integrator.integrate(&integrand, &parametrization);

        println!("re: {res_re:?}, im: {res_im:?}");
        println!("{reference:?}");
        assert!((res_re.mean-reference.re).abs() < res_re.err*2.0);
        assert!((res_im.mean-reference.im).abs() < res_im.err*2.0);
    }
}