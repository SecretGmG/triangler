use crate::{
    basic::integrands::Integrand, parametrization::Parametrization, vectors::Vec3, IntegrationResult, Logger
};
use pyo3::{exceptions::PyValueError, PyResult};
use rand::{rngs::StdRng, SeedableRng};
use rayon::prelude::*;
use symbolica::numerical_integration::{ContinuousGrid, Sample};

pub enum Integrator{
    Vegas(VegasIntegrator),
    VegasMulti(VegasMultiIntegrator)
}
impl Integrator{
    pub fn from_args(integrator: &str, epochs: usize, iters_per_epoch: usize, seed: u64, batches: usize) -> PyResult<Self>{
        match integrator {
            "vegas" => Ok(Self::Vegas(VegasIntegrator { epochs, iters_per_epoch, seed})),
            "vegas_multi" => Ok(Self::VegasMulti(VegasMultiIntegrator { epochs, iters_per_epoch, seed, batches })),
            other => Err(PyValueError::new_err(format!("{other} is an invalid argument for parametrization")))
        }
    }
    pub fn integrate(&self, integrand: &Integrand, parametrization: &Parametrization, logger : &Logger) -> IntegrationResult{
        match self {
            Integrator::Vegas(vegas_integrator) => vegas_integrator.integrate(integrand, parametrization, logger),
            Integrator::VegasMulti(vegas_multi_integrator) => vegas_multi_integrator.integrate(integrand, parametrization, logger),
        }
    }
}
pub struct VegasIntegrator {
    epochs: usize,
    iters_per_epoch: usize,
    seed: u64,
}
impl VegasIntegrator {
    fn integrate(
        &self,
        integrand: &Integrand,
        param: &Parametrization,
        logger: &Logger,
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
impl VegasMultiIntegrator {
    fn integrate(
        &self,
        integrand: &Integrand,
        param: &Parametrization,
        logger: &Logger,
    ) -> IntegrationResult {
        let mut grid = ContinuousGrid::new(3, 50, 1000, None, false);
        for i in 0..self.epochs {
            let subgrids: Vec<ContinuousGrid<f64>> = (0..self.batches)
                .into_par_iter()
                .map(|j| {
                    let mut grid_clone = grid.clone();
                    let unique_seed_offset = i*self.batches+j;
                    let mut rng = StdRng::seed_from_u64(self.seed+unique_seed_offset as u64);
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
            subgrids
                .iter()
                .for_each(|subgrid| grid.merge(subgrid).unwrap());
            grid.update(1.0);

            let (mean, err, _) = grid.accumulator.get_live_estimate();
            logger.info(format!("mean: {mean}, err: {err}"));
        }
        let (mean, err, _) = grid.accumulator.get_live_estimate();
        IntegrationResult { mean, err }
    }
}
