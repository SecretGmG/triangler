use rand::{rngs::StdRng, RngCore, SeedableRng};

use crate::{parametrization::Parametrization, Vec3};


pub trait SamplerAggregator<T, U>: Sync + Send{
    fn process_sample(&mut self, integrand: &impl Fn(T) -> U) -> Sample<T>;
    fn update(&mut self);
    fn combine(&mut self, other: &Self);
    fn split_off_worker(&mut self) -> Self;
    fn get_report(&self, reference_value: Option<U>) -> String;
}

pub struct Sample<T>{
    pub point: T,
    pub weight: f64,
}

pub struct RealR3SamplerAggregator<P: Parametrization>{
    rng: StdRng,
    param: P,
    grid: symbolica::numerical_integration::ContinuousGrid<f64>,
}
impl<P: Parametrization> RealR3SamplerAggregator<P>{
    pub fn new(param: P, seed: u64, n_bins: usize, min_samples_for_update: usize)-> Self{
        let rng = StdRng::seed_from_u64(seed);
        let grid = symbolica::numerical_integration::ContinuousGrid::new(3, n_bins, min_samples_for_update, None, false);
        Self{rng, param, grid}
    }
}

impl<P: Parametrization> SamplerAggregator<Vec3, f64> for RealR3SamplerAggregator<P>{
    fn update(&mut self) {
        self.grid.update(1.0);
    }

    fn combine(&mut self, other: &Self) {
        _ = self.grid.merge(&other.grid);
    }

    fn process_sample(&mut self, integrand: &impl Fn(Vec3) -> f64) -> Sample<Vec3> {
        let mut sample = symbolica::numerical_integration::Sample::new();
        self.grid.sample(&mut self.rng, &mut sample);

        if let symbolica::numerical_integration::Sample::Continuous(_wgt, xs) = &sample {
            let xs = Vec3::new(xs[0], xs[1], xs[2]);
            let (k, jac) = self.param.transform(xs);
            let eval = jac*integrand(k);
            self.grid.add_training_sample(&sample, eval).unwrap();
            Sample{
                point: k,
                weight: eval*sample.get_weight(),
            }
        }else{
            panic!("Expected continuous sample");
        }
    }
    fn split_off_worker(&mut self) -> Self {
        Self { rng: StdRng::seed_from_u64(self.rng.next_u64()), param: self.param.clone(), grid: self.grid.clone() }
    }

    fn get_report(&self, reference_value: Option<f64>) -> String{
        let mean = self.grid.accumulator.avg;
        let std = self.grid.accumulator.err;
        let r_std = std / mean.abs();

        if let Some(r) = reference_value{
            let err = mean-r;
            let r_err = err / std;
            format!("Reference: {:.6e}\nMean: {:.6e} Error: {:.6e} ({:.2}%)\nDeviation: {:.6e} ({:.2} sigma){}", r,mean, std, r_std*100.0, err, r_err, if r_err.abs() > 3.0 { " <--- HIGH DEVIATION !" } else { "" } )
        }
        else{
            format!("Mean: {:.6e} Error: {:.6e} ({:.2}%)", mean, std, r_std*100.0)
        }
    }
}