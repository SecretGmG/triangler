from symbolica_vectors import SymbolicaLorenzVec, SymbolicaVec
from symbolica import S, N, Expression, Evaluator
import numpy as np
from numpy.typing import NDArray


class Theta:
    @staticmethod
    def theta(self, x):
        return 1 if x[0][0] > 0 else 0
    
    symbol = S('theta')
    external_functions_dict = {(S('theta'), 'Theta') : theta}
    
from typing import Literal, Protocol
from numpy.typing import NDArray

class WrappedEvaluator(Protocol):
    def __call__(self, m: complex, loop_momenta: list[NDArray], k: NDArray) -> complex:
        pass

class IntegrandBuilder:
    p1: SymbolicaLorenzVec = SymbolicaLorenzVec.from_name('p1')
    p2: SymbolicaLorenzVec = SymbolicaLorenzVec.from_name('p2')
    m: Expression = S('m')
    integrand : SymbolicaVec = SymbolicaVec.from_name('k')

    part_indices = [
        (0,1,0,2),
        (1,0,2,0),
        (1,2,1,0),
        (2,1,0,1),
        (2,0,2,1),
        (0,2,1,2)
    ]

    def loop_momenta(self):
        return [SymbolicaLorenzVec.zero(), -self.p1, self.p2]
    
    def ose(self, k: SymbolicaVec, i: int):
        temp = k - self.loop_momenta()[i].spacial()
        return (temp*temp + self.m * self.m)**(N(1)/2)
    
    def eta(self, k: SymbolicaVec, i: int, j: int):
        return self.ose(k, i) + self.ose(k, j) + self.loop_momenta()[i].t() - self.loop_momenta()[j].t()
    
    def improved_ltd_prefactor(self, k: SymbolicaVec):
        return (4*Expression.PI)**N(-3)/(self.ose(k,0)*self.ose(k,1)*self.ose(k,2))
    
    def improved_ltd_expression(self) -> Expression:
        expr = N(0)
        for (i,j,k,l) in self.part_indices:
            part_builder = IntegrandPartBuilder(self, i, j, k, l)
            expr += part_builder.improved_ltd(self.integrand)
        return expr
    
    def evaluator_args(self):
        return [Expression.PI, self.m] + [sym for m in [self.p1, self.p2] for sym in m.symbols] + self.integrand.symbols

    def get_evaluator(self, e: Expression) -> Evaluator:
        return e.evaluator({},{}, self.evaluator_args())
    
    def get_wrapped_evaluator(self, e: Expression) -> WrappedEvaluator:
        evaluator = self.get_evaluator(e)
        def wrapped(m: complex, p1: NDArray, p2: NDArray, k: NDArray):
            return evaluator.evaluate_complex([[np.pi, m] + list(p1) + list(p2) + list(k)])
        return wrapped
    
    def evaluate_improved_ltd(self, m,p1, p2, k):
        self.get_wrapped_evaluator(self.improved_ltd_expression())(m, p1,p2, k)
class IntegrandPartBuilder:
    integrand_builder = None

    def __init__(self, integrand_builder : IntegrandBuilder, i, j, k, l):
        self.integrand_builder = integrand_builder
        self.i = i
        self.j = j
        self.k = k
        self.l = l
    
    def improved_ltd(self, k):
        return self.integrand_builder.improved_ltd_prefactor(k)/(self.integrand_builder.eta(k, self.i,self.j)*self.integrand_builder.eta(k, self.k, self.l))