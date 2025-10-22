use crate::{
    complex::integrands::ComplexIntegrand, vectors::Vec3, IntegrationResult, Parametrization
};
use rand::{rngs::StdRng, SeedableRng};
use symbolica::numerical_integration::{ContinuousGrid, Sample, StatisticsAccumulator};
use enum_dispatch::enum_dispatch;

#[enum_dispatch]
pub trait ComplexIntegrator {
    fn integrate(
        &self,
        integrand: &impl ComplexIntegrand,
        parametrization: &impl Parametrization,
    ) -> (IntegrationResult, IntegrationResult);
}
pub struct ComplexVegasIntegrator {
    epochs: usize,
    iters_per_epoch: usize,
    seed: u64,
}
impl ComplexVegasIntegrator{
    pub fn new(epochs: usize, iters_per_epoch: usize, seed: u64) -> Self{
        Self { epochs, iters_per_epoch, seed }
    }
}
impl ComplexIntegrator for ComplexVegasIntegrator {
    fn integrate(
        &self,
        integrand: &impl ComplexIntegrand,
        param: &impl Parametrization,
    ) -> (IntegrationResult, IntegrationResult) {
        let mut grid = ContinuousGrid::new(3, 50, 1000, None, false);
        let mut acc_re = StatisticsAccumulator::new();
        let mut acc_im = StatisticsAccumulator::new();

        let mut rng = StdRng::seed_from_u64(self.seed);
        let mut sample = Sample::new();

        for _ in 0..self.epochs {
            for _ in 0..self.iters_per_epoch {
                grid.sample(&mut rng, &mut sample);

                if let Sample::Continuous(_wgt, xs) = &sample {
                    let xs = Vec3::new(xs[0], xs[1], xs[2]);
                    let (k, jac) = param.transform(xs);
                    let eval = jac*integrand.evaluate(k);
                    grid.add_training_sample(&sample, eval.re).unwrap();
                    // Unintuitive api, that i have to multiply by the weight myself here, even though the sample already has it
                    acc_re.add_sample(eval.re*sample.get_weight(), Some(&sample));
                    acc_im.add_sample(eval.im*sample.get_weight(), Some(&sample));
                }
            }

            grid.update(1.0);
        }

        acc_re.update_iter(true);
        acc_im.update_iter(true);
        (IntegrationResult {
            mean: acc_re.avg,
            err: acc_re.err,
        },IntegrationResult {
            mean: acc_im.avg,
            err: acc_im.err,
        })
    }
}