from typing import Callable
import symbolica
import numpy as np


class ComplexIntegrator:
    """
    Integrator for complex functions
    
    keeps track of the real and imaginary parts separately
    """

    def __init__(self, n_dims):
        self.sampler = symbolica.NumericalIntegrator.continuous(n_dims=n_dims)
        self.real_integral = symbolica.NumericalIntegrator.continuous(n_dims=n_dims)
        self.imag_integral = symbolica.NumericalIntegrator.continuous(n_dims=n_dims)

    @staticmethod
    def samples_to_np(samples: list[symbolica.Sample]):
        return np.array(list(map(lambda s: s.c, samples)))

    def integrate(
        self,
        integrand: Callable,
        parametrization: Callable,
        n_epochs: int = 100,
        samples_per_epoch: int = 1_000,
        rng = None,
        learning_weight = 1.5,
        weighter = None
    ):
        """
        integrates a complex function
        
        integrand: the function to integrate
        parametrization: a function that maps samples to points and jacobians
        n_epochs: the number of epochs to train for
        samples_per_epoch: the number of samples to generate per epoch
        rng: the random number generator to use
        learning_weight: the learning weight to use
        weighter: the weighter to use, defaults to np.abs
        
        returns a ComplexIntegrationResult
        """
        if rng is None:
            rng=symbolica.RandomNumberGenerator(0, 0)

        if weighter is None:
            weighter = np.abs
        
        for _ in range(n_epochs):
            samples = self.sampler.sample(
                samples_per_epoch, rng
            )
            xs = ComplexIntegrator.samples_to_np(samples)

            points, jacs = parametrization(xs)

            values : np.typing.ArrayLike = integrand(points) * jacs
            
            mask = np.isfinite(values)
            
            values = values[mask]
            samples = np.array(samples)[mask]
            
            self.real_integral.add_training_samples(samples, values.real)
            self.imag_integral.add_training_samples(samples, values.imag)
            self.sampler.add_training_samples(samples, weighter(values))

            self.sampler.update(0, learning_weight)
        
        return ComplexIntegrationResult.from_live_estimates(
            real_live_estimate=self.real_integral.get_live_estimate(),
            imag_live_estimate=self.imag_integral.get_live_estimate(),
        )


class ComplexIntegrationResult:
    """
    represents the result of a complex monte carlo integration
    keeps track of the real and imaginary values and errors, and the total number of iterations
    """
    
    real_avg = 0
    imag_avg = 0
    real_err = 0
    imag_err = 0
    iters = 0


class ComplexIntegrationResult:
    def __init__(self, real_avg, imag_avg, real_err, imag_err, iters):
        self.real_avg = real_avg
        self.imag_avg = imag_avg
        self.real_err = real_err
        self.imag_err = imag_err
        self.iters = iters

    @staticmethod
    def from_live_estimates(real_live_estimate, imag_live_estimate):
        real_avg, real_err, *_ = real_live_estimate
        imag_avg, imag_err, *_ = imag_live_estimate

        iters = real_live_estimate[5]

        return ComplexIntegrationResult(
            real_avg=real_avg,
            imag_avg=imag_avg,
            real_err=real_err,
            imag_err=imag_err,
            iters=iters,
        )

    def complex_result(self):
        return self.real_avg + self.imag_avg * 1j

    def abs_err(self):
        return np.sqrt(self.real_err ** 2 + self.imag_err ** 2)

    def relative_abs_err(self):
        return self.abs_err() / abs(self.complex_result())

    def nr_iters(self):
        return self.iters

    def cv(self):
        return self.relative_abs_err() * np.sqrt(self.nr_iters())

    def __repr__(self):
        rae = self.relative_abs_err() * 100
        conv = self.cv()
        iters = self.nr_iters()

        return (
            f"ComplexIntegratorResult:\n"
            f"value = ({self.real_avg:.6f}±{self.real_err:.6f}) + i({self.imag_avg:.6f}±{self.imag_err:.6f})\n"
            f"relative absolute error = {rae:.2g}% , convergence={conv:.3f}, iters={iters}"
        )
