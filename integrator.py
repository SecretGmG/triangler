from typing import Callable
import symbolica
import numpy as np
from tqdm import tqdm


class ComplexIntegrator:

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
    ):

        for _ in range(n_epochs):
            samples = self.sampler.sample(
                samples_per_epoch, rng=symbolica.RandomNumberGenerator(0, 0)
            )
            xs = ComplexIntegrator.samples_to_np(samples)

            points, jacs = parametrization(xs)

            values = integrand(points) * jacs
            
            # create boolean mask as a NumPy array
            mask = np.isfinite(values.real) & np.isfinite(values.imag)

            # ensure samples is a NumPy array
            samples_np = np.array(samples)

            # filter using the mask
            samples_f = samples_np[mask]
            vals_f = values[mask]
            
            if not np.all(mask):
                print(xs[mask])
                print(values[mask])

            self.real_integral.add_training_samples(samples_f, vals_f.real)
            self.imag_integral.add_training_samples(samples_f, vals_f.imag)
            self.sampler.add_training_samples(samples_f, np.abs(vals_f))

            self.sampler.update(1.5, 1.5)
        return ComplexIntegratorResult.from_live_estimates(
            real_live_estimate=self.real_integral.get_live_estimate(),
            imag_live_estimate=self.imag_integral.get_live_estimate(),
        )


class ComplexIntegratorResult:
    real_avg = 0
    imag_avg = 0
    real_err = 0
    imag_err = 0
    iters = 0
    imag_live_estimate: tuple


class ComplexIntegratorResult:
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

        return ComplexIntegratorResult(
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

    def convergence(self):
        return self.relative_abs_err() * np.sqrt(self.nr_iters())

    def __repr__(self):
        rae = self.relative_abs_err() * 100
        conv = self.convergence()
        iters = self.nr_iters()

        return (
            f"ComplexIntegratorResult:\n"
            f"value = ({self.real_avg:.6f}±{self.real_err:.6f}) + i({self.imag_avg:.6f}±{self.imag_err:.6f})\n"
            f"relative absolute error = {rae:.2g}% , convergence={conv:.3f}, iters={iters}"
        )

    def __add__(self, other):
        if not isinstance(other, ComplexIntegratorResult):
            return NotImplemented

        real_avg = self.real_avg + other.real_avg
        imag_avg = self.imag_avg + other.imag_avg

        real_err = np.sqrt(self.real_err**2 + other.real_err**2)
        imag_err = np.sqrt(self.imag_err**2 + other.imag_err**2)

        iters = self.iters + other.iters

        return ComplexIntegratorResult(
            real_avg=real_avg,
            imag_avg=imag_avg,
            real_err=real_err,
            imag_err=imag_err,
            iters=iters,
        )
    def __sub__(self, other):
        if not isinstance(other, ComplexIntegratorResult):
            return NotImplemented

        real_avg = self.real_avg - other.real_avg
        imag_avg = self.imag_avg - other.imag_avg

        real_err = np.sqrt(self.real_err**2 + other.real_err**2)
        imag_err = np.sqrt(self.imag_err**2 + other.imag_err**2)

        iters = self.iters + other.iters

        return ComplexIntegratorResult(
            real_avg=real_avg,
            imag_avg=imag_avg,
            real_err=real_err,
            imag_err=imag_err,
            iters=iters,
        )
