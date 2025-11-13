from typing import Callable
import symbolica
import numpy as np


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

        for i in range(n_epochs):
            samples = self.sampler.sample(samples_per_epoch, rng = symbolica.RandomNumberGenerator(12,0))
            xs = self.samples_to_np(samples)
            
            points, jacs = parametrization(xs)
            
            values = integrand(points)[:,0] * jacs
            
            self.real_integral.add_training_samples(samples, np.nan_to_num(values.real))
            self.imag_integral.add_training_samples(samples, np.nan_to_num(values.imag))
            self.sampler.add_training_samples(samples, np.nan_to_num(abs(values)))
            self.sampler.update(1.5, 1.5)
            print('real: ', self.real_integral.get_live_estimate())
            print('imag: ', self.imag_integral.get_live_estimate())

    
