from plot_util import plot_complex, plot_complex_plane
from symbolica_vectors import SymbolicaLorenzVec, SymbolicaVec
from symbolica import S, N, Expression
from wrapped_eval import THETA, WrappedEvaluator
import numpy as np
import matplotlib.pyplot as plt


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

    def get_args(self):
        return [Expression.PI, self.thresh, self.m, self.p1, self.p2, self.k]

    def __init__(self):
        self.qs = [SymbolicaLorenzVec.zero(), -self.p1, self.p2]
        pass

    eta_indices = [
        #(0, 1),
        #(2, 1),
        #(2, 0),
        (1, 2),
        (0, 2),
        (1, 0),
    ]

    part_indices = [
        (1, 0, 2, 0),
        (0, 1, 0, 2),
        (0, 1, 2, 1),
        (1, 0, 1, 2),
        (0, 2, 1, 2),
        (2, 0, 2, 1),
    ]

    def ose(self, i: int, k: SymbolicaVec):
        temp : Expression = k - self.qs[i].spacial()
        return (temp.squared() + self.m**N(2))**HALF

    def prefactor(self, k: SymbolicaVec):
        return (4 * Expression.PI) ** N(-3) / (
            self.ose(0, k) * self.ose(1, k) * self.ose(2, k)
        )

    def eta(self, i, j, k):
        return self.ose(i, k) + self.ose(j, k) + self.qs[i].t() - self.qs[j].t()

    def cff(self):
        integrand = N(0)
        for i, j, k, l in self.part_indices:
            integrand += 1 / (self.eta(i, j, self.k) * self.eta(k, l, self.k))
        return integrand * self.prefactor(self.k)

    def eta_radius_roots(self, i, j, center):
        q: SymbolicaLorenzVec = (self.qs[i] - self.qs[j]) * HALF
        v: SymbolicaVec = q.spacial() * (N(1) / q.t())
        q_c: SymbolicaLorenzVec = (self.qs[i] + self.qs[j]) * HALF
        k_0_p = center-q_c.spacial()
        
        k_hat = (self.k-center).norm()

        a = N(1) - (k_hat * v) ** N(2)
        b = N(2) * (k_hat * k_0_p) - N(2) * (k_hat * v) * (k_0_p * v)
        c = k_0_p.squared() - (k_0_p * v) ** N(2) - q.squared() + self.m ** N(2)
        
        d = b ** N(2) - N(4) * a * c
        
        
        return [
            (-b + d**HALF) / (N(2) * a),
            (-b - d**HALF) / (N(2) * a)
        ], k_hat

    def ddk_eta(self, i, j, k_hat, k):
        d1 = k_hat * (k - self.qs[i].spacial()) / self.ose(i, k)
        d2 = k_hat * (k - self.qs[j].spacial()) / self.ose(j, k)
        return d1 + d2

    def eta_ct(self, i, j) -> list[(Expression, SymbolicaVec)]:
        q: SymbolicaLorenzVec = (self.qs[i] - self.qs[j]) * HALF
        #center: SymbolicaLorenzVec = (self.qs[i] + self.qs[j]).spacial() * HALF
        center = SymbolicaVec.zero()
        
        
        selector = N(1)
        # this selector is not necessary if we presuppose q_j > q_i for the uncommented eta_indices
        #selector *= THETA(-q.t())
        selector *= THETA(q.squared()-self.m**N(2))
        poles, k_hat = self.eta_radius_roots(i, j, center)

        out = []

        for r_star in poles:
            k_star = r_star * k_hat + center
            factor = (
                self.collect_other_etas(i, j, k_star)
                * selector
                * self.prefactor(k_star)
                / self.ddk_eta(i, j, k_hat, k_star)
            )
            out.append((factor, r_star))

        return out, center

    def collect_other_indices(self, i, j):
        order = j - i
        return [(i, (i - order) % 3), ((j + order) % 3, j)]

    def collect_other_etas(self, i, j, k):
        other_etas = N(0)
        for i, j in self.collect_other_indices(i, j):
            other_etas += 1 / (self.eta(i, j, k))
        return other_etas

    @staticmethod
    def real(e: Expression):
        return (e + e.conjugate())*HALF
    
    @staticmethod
    def imag(e: Expression):
        return (e - e.conjugate())*HALF
    
    @staticmethod
    def c_abs(e: Expression):
        return (e*e.conjugate())**HALF
        
    
    def ct(self):
        ct = N(0)
        for i, j in self.eta_indices:
            values, center = self.eta_ct(i, j)
            r = (self.k-center).squared()**HALF
            for factor, r_star in values:
                #offset = self.c_abs(self.real(r_star))*HALF
                #selector = THETA(r_star + offset - r)*THETA(r - (offset - r_star))
                selector = THETA(self.thresh - self.r)
                ct += selector * factor * (self.real(r_star) / r) ** N(2) / (r - r_star)
        return ct

    def ct_int(self):
        ct = N(0)
        for i, j in self.eta_indices:
            values, _ = self.eta_ct(i, j)
            for factor, r_star in values:
                #offset = self.c_abs(self.real(r_star))*HALF
                #upper = self.real(r_star) + offset
                #lower = self.real(r_star) - offset
                upper = self.thresh
                lower = -self.thresh
                ct += (
                    factor
                    * self.real(r_star) ** N(2)
                    * (Expression.LOG((upper - r_star) / (lower - r_star)))
                )
        return ct

import numpy as np

def hemispherical(xs):
    """
    Uniformly sample points on the unit hemisphere (z >= 0).
    xs: (N,2) array with values in [0,1)
    Returns:
        v: (N,3) Cartesian coordinates on unit hemisphere
        jac: (N,) Jacobian (area element) for Monte Carlo integration
    """
    xs = np.asarray(xs)
    if xs.ndim != 2 or xs.shape[1] != 2:
        raise ValueError("xs must have shape (N,2)")
    N = xs.shape[0]

    theta = 2 * np.pi * xs[:, 0]
    cos_phi = xs[:, 1]
    # numeric safety: enforce range and avoid tiny negative under sqrt due to fp error
    cos_phi = np.clip(cos_phi, 0.0, 1.0)
    sin_phi = np.sqrt(np.clip(1.0 - cos_phi**2, 0.0, None))

    v = np.empty((N, 3), dtype=float)
    v[:, 0] = sin_phi * np.cos(theta)
    v[:, 1] = sin_phi * np.sin(theta)
    v[:, 2] = cos_phi

    jac = np.full(N, 2.0 * np.pi)
    return v, jac


def spherical(xs):
    """
    Sample points in R^3 with radial transform r = u/(1-u) (u in [0,1) -> r in [0,inf)).
    xs: (N,3) array with values in [0,1)
    Returns:
        v: (N,3) Cartesian coordinates
        jac: (N,) Jacobian for Monte Carlo integration (dV / d(u1,u2,u3))
    """
    xs = np.asarray(xs)
    if xs.ndim != 2 or xs.shape[1] != 3:
        raise ValueError("xs must have shape (N,3)")
    u = xs[:, 0]
    if np.any(u >= 1.0):
        raise ValueError("xs[:,0] must be < 1.0 (u < 1)")

    # radial transform and its derivative
    r = u / (1.0 - u)
    r_jac = 1.0 / (1.0 - u) ** 2

    theta = 2.0 * np.pi * xs[:, 1]
    cos_phi = 1.0 - 2.0 * xs[:, 2]
    cos_phi = np.clip(cos_phi, -1.0, 1.0)
    sin_phi = np.sqrt(np.clip(1.0 - cos_phi**2, 0.0, None))

    v = np.empty_like(xs, dtype=float)
    v[:, 0] = r * sin_phi * np.cos(theta)
    v[:, 1] = r * sin_phi * np.sin(theta)
    v[:, 2] = r * cos_phi

    jac = r_jac * r**2 * 4.0 * np.pi
    return v, jac


def line_segment(k_hat, thresh):
    k_hat = np.asarray(k_hat)

    def temp(xs):
        return ((xs[:, 0] * 2 - 1) * thresh)[:, None] * k_hat[None, :], (2 * thresh)

    return temp


class ContextManager:

    p1 = np.array([3, 1, 0, 1])
    p2 = np.array([4, 0, 2, -1])
    m = 0.02
    threshold = 5

    def __init__(self, force_rebuild=True):
        ib = IntegrandBuilder()
        self.cff = WrappedEvaluator(ib.cff(), ib.get_args(), "cff", force_rebuild)
        self.ct = WrappedEvaluator(ib.ct(), ib.get_args(), "ct", force_rebuild)
        self.sub = WrappedEvaluator(ib.cff() - ib.ct(),ib.get_args(), "sub", force_rebuild)
        self.ct_int = WrappedEvaluator(ib.ct_int(), ib.get_args(),"ct_int", force_rebuild)

    def get_context_args(self):
        return [np.pi, self.threshold, self.m, self.p1, self.p2]

    def eval(self, compiled: WrappedEvaluator, k):
        return compiled.evaluate(self.get_context_args() + [k])
    
    def get_reference(self) -> complex:
        from oneloop_bridge import three_point, TO_FEYNMAN

        def norm(v):
            return v[0] ** 2 - (v[1:] ** 2).sum()
        
        res = three_point(
                norm(self.p1),
                norm(self.p2),
                norm(self.p1 + self.p2),
                self.m**2,
                self.m**2,
                self.m**2,
            )
        assert res.epsilon_minus_1 == 0
        assert res.epsilon_minus_2 == 0

        return (
            res.epsilon_0
            * TO_FEYNMAN
        )

    def plot_threshold_subtraction(
        self, x_lim, y_lim, x_axis=None, y_axis=None, res=300
    ):
        if x_axis is None:
            x_axis = np.array([1, 0, 0])
        if y_axis is None:
            y_axis = np.array([0, 1, 0])

        x = np.linspace(x_lim[0], x_lim[1], res)
        y = np.linspace(y_lim[0], y_lim[1], res)
        X, Y = np.meshgrid(x, y)

        xs_plane = X + Y * 1j
        ks_plane = (X[..., None] * x_axis + Y[..., None] * y_axis).reshape(-1, 3)
        ks_line = (x_axis[:, None] * x).T

        ks_plane_jac = np.sum(ks_plane**2, axis=1)
        ks_line_jac = np.sum(ks_line**2, axis=1)

        plt.figure(figsize=(20, 10))
        plt.subplot(2, 3, 1)
        integrand = (self.eval(self.cff, ks_plane) * ks_plane_jac).reshape(res, res)
        plot_complex_plane(xs_plane, integrand)
        plt.subplot(2, 3, 2)
        counter_term = (self.eval(self.ct, ks_plane) * ks_plane_jac).reshape(res, res)
        plot_complex_plane(xs_plane, counter_term)
        plt.subplot(2, 3, 3)
        subtracted = (self.eval(self.sub, ks_plane) * ks_plane_jac).reshape(res, res)
        plot_complex_plane(xs_plane, subtracted)
        plt.subplot(2, 3, 4)
        plot_complex(x, self.eval(self.cff, ks_line) * ks_line_jac)
        plt.subplot(2, 3, 5)
        plot_complex(x, self.eval(self.ct, ks_line) * ks_line_jac)
        plt.subplot(2, 3, 6)
        plot_complex(x, self.eval(self.sub, ks_line) * ks_line_jac)
    
    def normalize_args(self):
        """
        Apply a Lorentz boost such that the spatial sum of p1+p2 is zero.
        Updates self.p1 and self.p2 in place.
        
        swaps p1 and p2 if p1.0 > p2.0
        """
        p1, p2 = self.p1.copy(), self.p2.copy()
        
        # Spatial sum
        p_sum = p1[1:] + p2[1:]
        norm_p = np.linalg.norm(p_sum)
        if norm_p == 0:
            return  # Already normalized
    
        # Boost along the direction of p_sum
        E_sum = p1[0] + p2[0]
        beta = p_sum / E_sum
        gamma = 1 / np.sqrt(1 - np.dot(beta, beta))
    
        def boost(p):
            E = p[0]
            p_vec = p[1:]
            factor = (gamma - 1) * np.dot(p_vec, beta) / np.dot(beta, beta) - gamma * E
            p_new = p_vec + factor * beta
            E_new = gamma * (E - np.dot(beta, p_vec))
            return np.array([E_new, *p_new])
    
        self.p1 = boost(p1)
        self.p2 = boost(p2)
        if self.p1[0]>self.p2[0]:
            self.p1, self.p2 = self.p2, self.p1