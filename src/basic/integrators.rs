use crate::{
    basic::integrands::RealIntegrandTrait, logger::Logger, parametrization::{ParametrizationTrait}, vectors::Vec3, IntegrationResult
};
use rand::{rngs::StdRng, SeedableRng};
use rayon::prelude::*;
use symbolica::numerical_integration::{ContinuousGrid, Sample};
use enum_dispatch::enum_dispatch;

#[enum_dispatch]
pub trait RealIntegratorTrait {
    fn integrate(
        &self,
        integrand: &impl RealIntegrandTrait,
        parametrization: &impl ParametrizationTrait,
        logger: &impl Logger,
    ) -> IntegrationResult;
}

pub struct VegasIntegrator {
    epochs: usize,
    iters_per_epoch: usize,
    seed: u64,
}
impl VegasIntegrator{
    pub fn new(epochs: usize, iters_per_epoch: usize, seed: u64) -> Self{
        Self { epochs, iters_per_epoch, seed }
    }
}
impl RealIntegratorTrait for VegasIntegrator {
    fn integrate(
        &self,
        integrand: &impl RealIntegrandTrait,
        param: &impl ParametrizationTrait,
        logger: &impl Logger,
    ) -> IntegrationResult {
        let mut grid = ContinuousGrid::new(3, 50, 1000, None, false);
        let mut rng = StdRng::seed_from_u64(self.seed);
        let mut sample = Sample::new();

        for _ in 0..self.epochs {
            for _ in 0..self.iters_per_epoch {
                grid.sample(&mut rng, &mut sample);

                if let Sample::Continuous(_wgt, xs) = &sample {
                    let xs = Vec3::new(xs[0], xs[1], xs[2]);
                    let (k, jac) = param.transform(xs);
                    let res = integrand.evaluate(k);
                    grid.add_training_sample(&sample, jac * res).unwrap();
                }
            }

            grid.update(1.0);

            let (mean, err, _) = grid.accumulator.get_live_estimate();
            logger.info(format!("mean: {mean}, err: {err}"));
        }

        let (mean, err, _) = grid.accumulator.get_live_estimate();
        IntegrationResult { mean, err }
    }
}

pub struct VegasMultiIntegrator {
    epochs: usize,
    iters_per_epoch: usize,
    seed: u64,
    batches: usize,
}
impl VegasMultiIntegrator{
    pub fn new(epochs: usize, iters_per_epoch: usize, seed: u64, batches: usize) -> Self{
        Self { epochs, iters_per_epoch, seed, batches }
    }
}
impl RealIntegratorTrait for VegasMultiIntegrator {
    fn integrate(
        &self,
        integrand: &impl RealIntegrandTrait,
        param: &impl ParametrizationTrait,
        logger: &impl Logger,
    ) -> IntegrationResult {
        let mut grid = ContinuousGrid::new(3, 50, 1000, None, false);

        for i in 0..self.epochs {
            let subgrids: Vec<ContinuousGrid<f64>> = (0..self.batches)
                .into_par_iter()
                .map(|j| {
                    let mut grid_clone = grid.clone();
                    let unique_seed_offset = i * self.batches + j;
                    let mut rng = StdRng::seed_from_u64(self.seed + unique_seed_offset as u64);
                    let mut sample = Sample::new();

                    for _ in 0..self.iters_per_epoch / self.batches {
                        grid_clone.sample(&mut rng, &mut sample);

                        if let Sample::Continuous(_wgt, xs) = &sample {
                            let xs_vec = Vec3::new(xs[0], xs[1], xs[2]);
                            let (k, jac) = param.transform(xs_vec);
                            let res = integrand.evaluate(k);
                            grid_clone.add_training_sample(&sample, jac * res).unwrap();
                        }
                    }

                    grid_clone
                })
                .collect();

            subgrids.iter().for_each(|subgrid| grid.merge(subgrid).unwrap());
            grid.update(1.0);

            let (mean, err, _) = grid.accumulator.get_live_estimate();
            logger.info(format!("mean: {mean}, err: {err}"));
        }

        let (mean, err, _) = grid.accumulator.get_live_estimate();
        IntegrationResult { mean, err }
    }
}