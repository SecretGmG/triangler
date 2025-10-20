use enum_dispatch::enum_dispatch;
use pyo3::prelude::*;

use crate::{basic::{integrands::{BasicLTD, ImprovedLTD, RealIntegrandTrait}, integrators::{RealIntegratorTrait, VegasIntegrator, VegasMultiIntegrator}}, logger::Logger, parametrization::{ParametrizationTrait, CartesianParam, SphericalParam}, IntegrationResult, LVec, Vec3};
#[pymodule]
fn triangler(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Triangle>()?;
    m.add_class::<Vec3>()?;
    m.add_class::<LVec>()?;
    Ok(())
}

///Wrapper to expose Basic integration to python
#[pyclass]
pub struct Triangle {
    integrand: RealIntegrand,
    integrator: RealIntegrator,
    parametrization: Parametrization,
    logger: Option<PyLogger>
}
#[pymethods]
impl Triangle {
    #[new]
    #[pyo3(signature = (
        p,
        q,
        m_psi,
        integrand_mode = "improved",
        integrator_mode = "havana",
        parametrization_mode = "spherical",
        scale = 1.0,
        epochs = 20,
        iters_per_epoch = 200_000,
        seed = 42,
        batches = 100,
        logger = None
    ))]
    pub fn new(
        p: LVec,
        q: LVec,
        m_psi: f64,
        integrand_mode: &str,
        integrator_mode: &str,
        parametrization_mode: &str,
        scale: f64,
        epochs: usize,
        iters_per_epoch: usize,
        seed: u64,
        batches: usize,
        logger: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        let integrand = RealIntegrand::from_args(integrand_mode, p, q, m_psi)?;
        let integrator = RealIntegrator::from_args(integrator_mode, epochs, iters_per_epoch, seed, batches)?;
        let parametrization = Parametrization::from_args(parametrization_mode, scale)?;
        let py_logger = logger.and_then(PyLogger::new);

        Ok(Self {
            integrand,
            integrator,
            parametrization,
            logger: py_logger,
        })
    }

    /// Performs the integration over the triangle.
    pub fn integrate(&self) -> IntegrationResult {
        self.integrator
            .integrate(&self.integrand, &self.parametrization, &self.logger)
    }

    /// Evaluates the integrand at a given momentum vector `k`.
    pub fn evaluate(&self, k: Vec3) -> f64 {
        self.integrand.evaluate(k)
    }

    /// Evaluates the integrand using a parameterized coordinate `xs`.
    pub fn evaluate_parameterized(&self, xs: Vec3) -> f64 {
        self.evaluate(self.parametrization.transform(xs).0)
    }
}
#[enum_dispatch(ParametrizationTrait)]
pub enum Parametrization {
    CartesianParam,
    SphericalParam,
}
impl Parametrization {
    pub fn from_args(mode: &str, scale: f64) -> pyo3::PyResult<Self> {
        match mode {
            "cartesian" => Ok(Self::CartesianParam(CartesianParam::new(scale))),
            "spherical" => Ok(Self::SphericalParam(SphericalParam::new(scale))),
            other => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "{other} is an invalid argument for parametrization"
            ))),
        }
    }
}
#[enum_dispatch(RealIntegratorTrait)]
pub enum RealIntegrator {
    VegasIntegrator,
    VegasMultiIntegrator,
}
impl RealIntegrator {
    pub fn from_args(
        integrator: &str,
        epochs: usize,
        iters_per_epoch: usize,
        seed: u64,
        batches: usize,
    ) -> pyo3::PyResult<Self> {
        match integrator {
            "havana" => Ok(Self::VegasIntegrator(VegasIntegrator::new(
                epochs,
                iters_per_epoch,
                seed,
            ))),
            "havana_multi" => Ok(Self::VegasMultiIntegrator(VegasMultiIntegrator::new(
                epochs,
                iters_per_epoch,
                seed,
                batches,
            ))),
            other => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "{other} is an invalid argument for integrator"
            ))),
        }
    }
}
#[enum_dispatch(RealIntegrandTrait)]
pub enum RealIntegrand {
    BasicLTD,
    ImprovedLTD,
}
impl RealIntegrand {
    pub fn from_args(mode: &str, p: LVec, q: LVec, m_psi: f64) -> pyo3::PyResult<Self> {
        match mode {
            "basic" => Ok(Self::BasicLTD(BasicLTD::new(p, q, m_psi ))),
            "improved" => Ok(Self::ImprovedLTD(ImprovedLTD::new(p, q, m_psi ))),
            other => {
                return Err(pyo3::PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "{other} is not a valid Integrand mode. Use 'basic' or 'improved'."
                )));
            }
        }
    }
}

pub struct PyLogger {
    logger: Py<PyAny>,
}
impl PyLogger{
    pub fn new(logger: Py<PyAny>) -> Option<Self> {
        Python::attach(|py| {
            let bound = logger.bind(py);
            let has_methods = bound.hasattr("info").unwrap_or(false)
                && bound.hasattr("debug").unwrap_or(false)
                && bound.hasattr("critical").unwrap_or(false);
            if has_methods {
                Some(Self { logger })
            } else {
                None
            }
        })
    }
}
/// Thin wrapper around a python logger
impl Logger for PyLogger{
    fn debug(&self, msg: String) {
        Python::attach(|py| {
            _ = self.logger.call_method1(py, "debug", (msg,));
        })
    }
    fn info(&self, msg: String) {
        Python::attach(|py| {
            _ = self.logger.call_method1(py, "info", (msg,));
        })
    }
    fn critical(&self, msg: String) {
        Python::attach(|py| {
            _ = self.logger.call_method1(py, "critical", (msg,));
        })
    }
}
