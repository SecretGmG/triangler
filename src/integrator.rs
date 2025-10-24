use crate::{sampler::SamplerAggregator};

pub trait Integrator<T, U> {
    fn integrate(
        &self,
        integrand: &(impl Fn(T) -> U+Sync),
        sampler: &mut impl SamplerAggregator<T, U>,
    );
}

pub struct BasicIntegrator{
    pub n_epochs: usize,
    pub n_samples_per_epoch: usize,
}

impl BasicIntegrator{
    pub fn new(n_epochs: usize, n_samples_per_epoch: usize) -> Self{
        Self{n_epochs, n_samples_per_epoch}
    }
}
impl<T, U> Integrator<T, U> for BasicIntegrator{
    fn integrate(
        &self,
        integrand: &impl Fn(T) -> U,
        sampler: &mut impl SamplerAggregator<T, U>,
    ) {
        for _ in 0..self.n_epochs{
            for _ in 0..self.n_samples_per_epoch{
                sampler.process_sample(integrand);
            }
        }
    }
}
pub struct MultiIntegrator{
    pub n_epochs: usize,
    pub n_samples_per_epoch: usize,
    pub n_workers: usize,

}

impl MultiIntegrator{
    pub fn new(n_epochs: usize, n_samples_per_epoch: usize, n_workers: usize) -> Self{
        Self{n_epochs, n_samples_per_epoch, n_workers}
    }
}
impl<T, U> Integrator<T, U> for MultiIntegrator{
    fn integrate(
        &self,
        integrand: &(impl (Fn(T) -> U) + Sync),
        sampler: &mut impl SamplerAggregator<T, U>,
    ) {
        use rayon::prelude::*;
        for _ in 0..self.n_epochs{
            let samplers: Vec<_> = (0..self.n_workers).map(|_| SamplerAggregator::split_off_worker(sampler)).par_bridge().map(|mut sampler|{
                let mut worker_sampler = sampler.split_off_worker();
                for _ in 0..self.n_samples_per_epoch / self.n_workers{
                    worker_sampler.process_sample(integrand);
                }
                worker_sampler
            }).collect();
            for s in samplers.into_iter(){
                sampler.combine(&s);
            }
            sampler.update();
        }
    }
}