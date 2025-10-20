pub mod basic;
pub mod parametrization;
pub mod vectors;
pub mod logger;
#[cfg(feature = "python")]
pub mod python;


pub use crate::vectors::{LVec, Vec3};

#[cfg(feature = "python")]
use pyo3::prelude::*;


#[cfg(feature = "python")]
use pyo3::prelude::*;


#[cfg_attr(feature = "python", pyclass)]
#[derive(Debug)]
pub struct IntegrationResult {
    pub mean: f64,
    pub err: f64,
}
#[cfg(feature = "python")]
#[pymethods]
impl IntegrationResult {
    fn __str__(&self) -> PyResult<String> {
        Ok(format!("mean = {}, error = {}", self.mean, self.err))
    }
}