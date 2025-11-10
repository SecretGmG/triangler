from symbolica_vectors import SymbolicaLorenzVec, SymbolicaVec
from symbolica import S, N, Expression
from evaluator_builder import THETA
import numpy as np

HALF = N(1) / 2


class IntegrandBuilder:
    p1: SymbolicaLorenzVec = SymbolicaLorenzVec.from_name("p1")
    p2: SymbolicaLorenzVec = SymbolicaLorenzVec.from_name("p2")
    m: Expression = S("m")
    integrand: SymbolicaVec = SymbolicaVec.from_name("k")

    part_indices = [
        (0, 1, 0, 2),
        (1, 0, 2, 0),
        (1, 2, 1, 0),
        (2, 1, 0, 1),
        (2, 0, 2, 1),
        (0, 2, 1, 2),
    ]

    def loop_momenta(self):
        return [SymbolicaLorenzVec.zero(), -self.p1, self.p2]

    def ose(self, k: SymbolicaVec, i: int):
        temp = k - self.loop_momenta()[i].spacial()
        return (temp * temp + self.m * self.m) ** HALF

    def improved_ltd_prefactor(self, k: SymbolicaVec):
        return (4 * Expression.PI) ** N(-3) / (
            self.ose(k, 0) * self.ose(k, 1) * self.ose(k, 2)
        )

    def improved_ltd_expression(self) -> Expression:
        expr = N(0)
        for i, j, k, l in self.part_indices:
            eta_ij = IntegrandEtaBuilder(self, i, j)
            eta_kl = IntegrandEtaBuilder(self, k, l)
            part_builder = IntegrandPartBuilder(self, (eta_ij, eta_kl))
            expr += part_builder.improved_ltd(self.integrand)
        return expr

    def counter_term(self):
        ct = N(0)
        for i, j, k, l in self.part_indices:
            eta_ij = IntegrandEtaBuilder(self, i, j)
            eta_kl = IntegrandEtaBuilder(self, k, l)
            part_builder = IntegrandPartBuilder(self, (eta_ij, eta_kl))

            ct += part_builder.counter_term(self.integrand)

        return ct

LAMBDA = 1

class IntegrandEtaBuilder:
    def __init__(self, integrand_builder: IntegrandBuilder, i, j, center=None):
        self.integrand_builder = integrand_builder
        self.i = i
        self.j = j
        self.qi = self.integrand_builder.loop_momenta()[self.i]
        self.qj = self.integrand_builder.loop_momenta()[self.j]
        if center is None:
            center = SymbolicaVec.zero()
        self.center = center

    def eta(self, k):
        return (
            self.integrand_builder.ose(k, self.i)
            + self.integrand_builder.ose(k, self.j)
            + self.qi.t()
            - self.qj.t()
        )

    def q(self) -> SymbolicaLorenzVec:
        return (self.qi - self.qj) * HALF

    def q_avg(self) -> SymbolicaLorenzVec:
        return (self.qi + self.qj) * HALF

    def center_prime(self):
        return self.center - self.q_avg().spacial()

    def k_hat(self, k):
        k_hat = (k - self.center).norm()
        return k_hat

    def radius_poles(self, k: SymbolicaVec):
        q = self.q()
        k_hat = (k - self.center).norm()
        v = q.spacial() * (N(1) / q.t())

        k_0_p = self.center_prime()

        a = N(1) - (k_hat * v) ** N(2)
        b = N(2) * (k_hat * k_0_p) - N(2) * (k_hat * v) * (k_0_p * v)
        c = (
            k_0_p.squared()
            - (k_0_p * v) ** N(2)
            - q.squared()
            + self.integrand_builder.m ** N(2)
        )
        d = b ** N(2) - N(4) * a * c
        return [
            (-b + d**HALF) / (N(2) * a),
            (-b - d**HALF) / (N(2) * a),
        ]

    def _derivative(self, k, k_hat):
        d1 = k_hat * (k - self.qi.spacial()) / self.integrand_builder.ose(k, self.i)
        d2 = k_hat * (k - self.qj.spacial()) / self.integrand_builder.ose(k, self.j)
        return d1 + d2

    def counter_term(self, k: SymbolicaVec):
        poles = self.radius_poles(k)
        k_hat = self.k_hat(k)
        r = (k - self.center).squared()**HALF
        
        out = []
        for r_star in poles:
            k_star = self.center + r_star * k_hat
            
            d_r : Expression = (r - r_star)
            
            cutoff = THETA(N(LAMBDA) - d_r * d_r.conjugate() ) #TODO: DONT HARDCODE THIS
                        
            ct = cutoff * THETA(-self.q().symbols[0]) / (d_r * self._derivative(k_star, k_hat))
            out.append((ct, k_star))
        return out
        
        

COUNTER_JACOBIAN = False


class IntegrandPartBuilder:
    etas: tuple[IntegrandEtaBuilder, IntegrandEtaBuilder] = None
    integrand_builder: IntegrandBuilder = None

    def __init__(
        self,
        integrand_builder: IntegrandBuilder,
        etas: tuple[IntegrandEtaBuilder, IntegrandBuilder],
    ):
        self.integrand_builder = integrand_builder
        self.etas = etas

        self.center = SymbolicaVec.zero()
        # self.center = SymbolicaVec.from_name(
        #    f"k{eta_ij.i}{eta_ij.j}{eta_kl.i}{eta_kl.j}"
        # )

    def improved_ltd(self, k: SymbolicaVec):
        return self.integrand_builder.improved_ltd_prefactor(k) / (
            self.etas[0].eta(k) * self.etas[1].eta(k)
        )

    def counter_term(self, k: SymbolicaVec):
        r = (k - self.center).squared() ** HALF

        ct = N(0)

        for eta1, eta2 in [self.etas, reversed(self.etas)]:
            for ct_, k_ in eta1.counter_term(k):
                counter_jac = N(1)
                if COUNTER_JACOBIAN:
                    counter_jac = ((k_ - self.center).squared() ** HALF / r) ** N(2)

                ct += (
                    ct_
                    * self.integrand_builder.improved_ltd_prefactor(k_)
                    / eta2.eta(k_)
                    * counter_jac
                )
        return ct

    def radially_integrated_counter_term(self, k_hat):
        pass
