from integrator import ComplexIntegrator
from symbolica_vectors import SymbolicaLorenzVec, SymbolicaVec
from symbolica import S, N, Expression
from wrapped_eval import THETA, WrappedEvaluator
import numpy as np
import matplotlib.pyplot as plt
import oneloop_bridge


HALF = N(1) / 2


class IntegrandBuilder:
    p1: SymbolicaLorenzVec = SymbolicaLorenzVec.from_name("p1")
    p2: SymbolicaLorenzVec = SymbolicaLorenzVec.from_name("p2")
    k: SymbolicaVec = SymbolicaVec.from_name("k")
    r = k.squared() ** HALF
    k_hat: SymbolicaVec = k.norm()
    m: Expression = S("m")
    thresh: Expression = S("lambda")

    qs: list[SymbolicaLorenzVec] = None

    def __init__(self):
        self.qs = [SymbolicaLorenzVec.zero(), -self.p1, self.p2]
        pass

    eta_indices = [
        (0, 1),
        (1, 2),
        (2, 0),
        (0, 2),
        (1, 0),
        (2, 1),
    ]

    part_indices = [
        (0, 1, 0, 2),
        (1, 0, 2, 0),
        (1, 2, 1, 0),
        (2, 1, 0, 1),
        (2, 0, 2, 1),
        (0, 2, 1, 2),
    ]

    def ose(self, i: int, k: SymbolicaVec):
        temp = k - self.qs[i].spacial()
        return (temp * temp + self.m * self.m) ** HALF

    def prefactor(self, k: SymbolicaVec):
        return (4 * Expression.PI) ** N(-3) / (
            self.ose(0, k) * self.ose(1, k) * self.ose(2, k)
        )

    def eta(self, i, j, k):
        return self.ose(i, k) + self.ose(j, k) + self.qs[i].t() - self.qs[j].t()

    def cff_integrand(self):
        integrand = N(0)
        for i, j, k, l in self.part_indices:
            integrand += 1 / (self.eta(i, j, self.k) * self.eta(k, l, self.k))
        return integrand * self.prefactor(self.k)

    def eta_radius_roots(self, i, j):
        q: SymbolicaLorenzVec = (self.qs[i] - self.qs[j]) * HALF
        v: SymbolicaVec = q.spacial() * (N(1) / q.t())
        q_c: SymbolicaLorenzVec = (self.qs[i] + self.qs[j]) * HALF
        k_0_p = -q_c.spacial()

        a = N(1) - (self.k_hat * v) ** N(2)
        b = N(2) * (self.k_hat * k_0_p) - N(2) * (self.k_hat * v) * (k_0_p * v)
        c = k_0_p.squared() - (k_0_p * v) ** N(2) - q.squared() + self.m ** N(2)
        d = b ** N(2) - N(4) * a * c
        return [
            (-b + d**HALF) / (N(2) * a),
            (-b - d**HALF) / (N(2) * a),
        ]

    def ddk_eta(self, i, j, r):
        k = self.k_hat * r
        d1 = self.k_hat * (k - self.qs[i].spacial()) / self.ose(i, k)
        d2 = self.k_hat * (k - self.qs[j].spacial()) / self.ose(j, k)
        return d1 + d2

    def eta_ct(self, i, j) -> list[(Expression, SymbolicaVec)]:
        selector = THETA(self.qs[j].t() - self.qs[i].t())
        poles = self.eta_radius_roots(i, j)

        out = []

        for r_star in poles:
            k_star = r_star * self.k_hat
            ct = (
                selector
                * self.collect_other_etas(i, j, k_star)
                * self.prefactor(k_star)
                / self.ddk_eta(i, j, r_star)
            )
            out.append((ct, r_star))

        return out

    def collect_other_indices(self, i, j):
        order = j - i
        return [(i, (i - order) % 3), ((j + order) % 3, j)]

    def collect_other_etas(self, i, j, k):
        other_etas = N(0)
        for i, j in self.collect_other_indices(i, j):
            other_etas += 1 / (self.eta(i, j, k))
        return other_etas

    def counter_term(self):
        ct = N(0)
        for i, j in self.eta_indices:
            for factor, r_star in self.eta_ct(i, j):
                selector = THETA(self.thresh - self.r)
                ct += selector * factor / (self.r - r_star)
        return ct

    def integrated_counter_term(self):
        ct = N(0)
        for i, j in self.eta_indices:
            for factor, r_star in self.eta_ct(i, j):
                ct += -factor * Expression.LOG(
                    (-self.thresh - r_star) / (self.thresh - r_star)
                )


class CompiledIntegrand:
    def __init__(self):
        self.integrand_builder = IntegrandBuilder()
        self.integrand = self.integrand_builder.cff_integrand()
        self.counter_term = self.integrand_builder.counter_term()

        self.constant_arguments = self.get_default_constant_args()

        self.compiled_integrand = WrappedEvaluator(
            self.integrand,
            self.constant_arguments,
            [self.integrand_builder.k],
            "integrand",
        )
        self.compiled_counter_term = WrappedEvaluator(
            self.counter_term,
            self.constant_arguments,
            [self.integrand_builder.k],
            "counter_term",
        )
        self.compiled_subtracted = WrappedEvaluator(
            self.integrand - self.counter_term,
            self.constant_arguments,
            [self.integrand_builder.k],
            "subtracted",
        )

    def get_default_constant_args(self):
        return {
            Expression.PI: complex(np.pi),
            self.integrand_builder.thresh: 1.0,
            self.integrand_builder.p1: np.array([-0.005, 0, 0, 0.005]),
            self.integrand_builder.p2: np.array([0.005, 0, 0, 0.005]),
            self.integrand_builder.m: 0.02,
        }

    def clean_args(self):
        """Boosts the arguments such that p1 + p2 = 0 in spatial components"""

        p1 = np.array(self.p1, dtype=complex)
        p2 = np.array(self.p2, dtype=complex)

        # 4-momentum components: [E, px, py, pz]
        E1, p1_vec = p1[0], p1[1:]
        E2, p2_vec = p2[0], p2[1:]

        # total momentum and energy
        P_vec = p1_vec + p2_vec
        E_tot = E1 + E2
        P_mag2 = np.dot(P_vec, P_vec)
        P_mag = np.sqrt(P_mag2.real)

        # If already in COM frame, skip boost
        if P_mag < 1e-12:
            pass
        else:
            beta = P_vec / E_tot  # boost velocity vector
            beta2 = np.dot(beta, beta)
            gamma = 1.0 / np.sqrt(1.0 - beta2)

            def boost(p):
                E, p_vec = p[0], p[1:]
                bp = np.dot(beta, p_vec)
                E_prime = gamma * (E - bp)
                p_prime = p_vec + ((gamma - 1) * bp / beta2 - gamma * E) * beta
                return np.array([E_prime, *p_prime], dtype=complex)

            p1 = boost(p1)
            p2 = boost(p2)

        # Set boosted values
        self.p1 = p1
        self.p2 = p2

    @property
    def thresh(self):
        return self.constant_arguments[self.integrand_builder.thresh]

    @thresh.setter
    def thresh(self, value):
        self.constant_arguments[self.integrand_builder.thresh] = float(value)

    @property
    def p1(self):
        return self.constant_arguments[self.integrand_builder.p1]

    @p1.setter
    def p1(self, value):
        self.constant_arguments[self.integrand_builder.p1] = np.array(
            value, dtype=float
        )

    @property
    def p2(self):
        return self.constant_arguments[self.integrand_builder.p2]

    @p2.setter
    def p2(self, value):
        self.constant_arguments[self.integrand_builder.p2] = np.array(
            value, dtype=float
        )

    @property
    def m(self):
        return self.constant_arguments[self.integrand_builder.m]

    @m.setter
    def m(self, value):
        self.constant_arguments[self.integrand_builder.m] = complex(value)

    def eval_integrand(self, k):
        return self.compiled_integrand.evaluate(np.asarray([k]))

    def eval_counterterm(self, k):
        return self.compiled_counter_term.evaluate(np.asarray([k]))

    def eval_subtracted(self, k):
        return self.compiled_subtracted.evaluate(np.asarray([k]))

    def compile(self):
        self.compiled_integrand.compile()
        self.compiled_counter_term.compile()
        self.compiled_subtracted.compile()
    

    def spherical(self, xs):
        """
        xs: (N,3) array
        scale: float
        returns (v, jac)
        v: (N,3)
        jac: (N,)
        """
        
        scale = 1
        
        x = xs[:, 0]
        y = xs[:, 1]
        z = xs[:, 2]

        r = x / (1.0 - x)
        r_jac = 1.0 / (1.0 - x)**2

        th = y * 2.0 * np.pi
        th_jac = 2.0 * np.pi

        phi = z * np.pi
        phi_jac = np.pi

        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)

        v = np.empty_like(xs)
        v[:, 0] = scale * r * sin_phi * np.cos(th)
        v[:, 1] = scale * r * sin_phi * np.sin(th)
        v[:, 2] = scale * r * cos_phi

        jac = r_jac * (r**2) * th_jac * phi_jac * sin_phi * scale**3

        return v, jac

    
    
    def integrate_naive(self, epochs, samples_per_epoch):
        integrator = ComplexIntegrator()
        integrator.integrate(self.eval_integrand, self.spherical, epochs, samples_per_epoch)
    
        
    
    def get_reference(self) -> complex:
        def norm(lvec):
            return lvec[0]**2 - (lvec[1:]**2).sum()
        
        return oneloop_bridge.three_point(norm(self.p1), norm(self.p2), norm(self.p1+self.p2), self.m, self.m, self.m).epsilon_0 * oneloop_bridge.TO_FEYNMAN