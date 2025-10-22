mod common;
use common::get_real_args_result_pairs;
use triangler::{parametrization::SphericalParam, real::{integrands::ImprovedLTD, integrators::HavanaMultiIntegrator}, Integrator};

#[test]
fn test_havana_real_improved_ltd() {
    for (args, reference) in get_real_args_result_pairs(){

        let integrand = ImprovedLTD::new(args);
        let integrator = HavanaMultiIntegrator::new(20,100_000,42,100);
        let parametrization = SphericalParam::new(1.0);

        let res = integrator.integrate(&integrand, &parametrization);

        println!("{res:?}");
        println!("{reference:?}");
        assert!((res.mean-reference).abs() < res.err*2.0);

    }
}