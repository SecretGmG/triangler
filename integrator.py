from typing import Callable
import symbolica
import numpy as np
from tqdm import tqdm


class ComplexIntegrator:
    
    
    def __init__(self, n_dims = 3):
        self.sampler = symbolica.NumericalIntegrator.continuous(n_dims=n_dims)
        self.real_integral = symbolica.NumericalIntegrator.continuous(n_dims=n_dims)
        self.imag_integral = symbolica.NumericalIntegrator.continuous(n_dims=n_dims)

    
    @staticmethod
    def samples_to_np(samples: list[symbolica.Sample]):
        return np.array(list(map(lambda s: s.c,samples)))
    @staticmethod
    def sample_weights(samples: list[symbolica.Sample]):
        return np.array(list(map(lambda s: s.weights)))


    def integrate(self, integrand : Callable, parametrization : Callable, n_epochs : int = 10, samples_per_epoch : int = 1_000):

        for _ in tqdm(range(n_epochs)):
            samples = self.sampler.sample(samples_per_epoch, rng = symbolica.RandomNumberGenerator(12,0))
            xs = self.samples_to_np(samples)
            
            points, jacs = parametrization(xs)
            
            values = integrand(points) * jacs
            
            self.real_integral.add_training_samples(samples, np.nan_to_num(values.real))
            self.imag_integral.add_training_samples(samples, np.nan_to_num(values.imag))
            self.sampler.add_training_samples(samples, np.nan_to_num(abs(values)))
            self.sampler.update(1.5, 1.5)
            #print('real: ', self.real_integral.get_live_estimate())
            #print('imag: ', self.imag_integral.get_live_estimate())
        return ComplexIntegratorResult(real_live_estimate=self.real_integral.get_live_estimate(), imag_live_estimate=self.imag_integral.get_live_estimate())

    
class ComplexIntegratorResult:
    real_live_estimate : tuple
    imag_live_estimate : tuple
    
    def __init__(self, real_live_estimate, imag_live_estimate):
        self.real_live_estimate = real_live_estimate
        self.imag_live_estimate = imag_live_estimate
    
    def result(self):
        return self.real_live_estimate[0] + self.imag_live_estimate[0]*1j
    
    def real_err(self):
        return self.real_live_estimate[1]
    
    def imag_err(self):
        return self.imag_live_estimate[1]
    
    def abs_err(self):
        return np.sqrt(self.real_err()**2 + self.imag_err()**2)
    
    def relative_abs_err(self):
        return self.abs_err() / abs(self.result())
    
    def nr_iters(self):
        return self.real_live_estimate[5]
    
    def convergence(self):
        return self.relative_abs_err() * np.sqrt(self.nr_iters())
    
    def __repr__(self):
        re, im = self.real_live_estimate[0], self.imag_live_estimate[0]
        re_err, im_err = self.real_err(), self.imag_err()
        rae = self.relative_abs_err()*100
        conv = self.convergence()
        iters = self.nr_iters()
    
        return (f"ComplexIntegratorResult:\n"
                f"value = ({re:.6f}±{re_err:.6f}) + i({im:.6f}±{im_err:.6f})\n"
                f"relative absolute error = {rae:.2g}% , convergence={conv:.3f}, iters={iters}")