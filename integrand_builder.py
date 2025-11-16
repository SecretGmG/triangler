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

    def cff(self):
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
            factor = (
                selector
                * self.collect_other_etas(i, j, k_star)
                * self.prefactor(k_star)
                / self.ddk_eta(i, j, r_star)
            )
            out.append((factor, r_star))

        return out

    def collect_other_indices(self, i, j):
        order = j - i
        return [(i, (i - order) % 3), ((j + order) % 3, j)]

    def collect_other_etas(self, i, j, k):
        other_etas = N(0)
        for i, j in self.collect_other_indices(i, j):
            other_etas += 1 / (self.eta(i, j, k))
        return other_etas

    def ct(self):
        ct = N(0)
        for i, j in self.eta_indices:
            for factor, r_star in self.eta_ct(i, j):
                selector = THETA(self.thresh - self.r)
                ct += selector * factor * (r_star / self.r) ** N(2) / (self.r - r_star)
        return ct

    def ct_int(self):
        ct = N(0)
        for i, j in self.eta_indices:
            for factor, r_star in self.eta_ct(i, j):
                ct += (
                    r_star ** N(2)
                    * factor
                    * (Expression.LOG((self.thresh - r_star) / (-self.thresh - r_star)))
                )
        return ct

def hemispherical(xs):
    """
    Uniformly samples points on the unit hemisphere oriented to positive z
    xs: (N,2) array with entries in [0,1)
    Returns:
        v: (N,3) Cartesian coordinates on unit hemisphere
        jac: (N,) Jacobian for Monte Carlo integration
    """
    theta = 2 * np.pi * xs[:, 0]
    cos_phi = xs[:, 1]
    sin_phi = np.sqrt(1 - cos_phi**2)
    v = np.empty([xs.shape[0], 3])
    v[:, 0] = sin_phi * np.cos(theta)
    v[:, 1] = sin_phi * np.sin(theta)
    v[:, 2] = cos_phi
    jac = 2 * np.pi
    return v, jac


def spherical(xs):
    """
    Uniformly samples points in 3D with a radial transform r = u / (1-u).
    xs: (N,3) array with entries in [0,1)
    Returns:
        v: (N,3) Cartesian coordinates
        jac: (N,) Jacobian for Monte Carlo integration
    """
    r = xs[:, 0] / (1.0 - xs[:, 0])
    r_jac = 1.0 / (1.0 - xs[:, 0]) ** 2
    theta = 2 * np.pi * xs[:, 1]
    cos_phi = 1 - 2 * xs[:, 2]
    sin_phi = np.sqrt(1 - cos_phi**2)
    v = np.empty_like(xs)
    v[:, 0] = r * sin_phi * np.cos(theta)
    v[:, 1] = r * sin_phi * np.sin(theta)
    v[:, 2] = r * cos_phi
    jac = r_jac * r**2 * 4 * np.pi
    return v, jac


def line_segment(k_hat, thresh):
    k_hat = np.asarray(k_hat)

    def temp(xs):
        return ((xs[:, 0] * 2 - 1) * thresh)[:, None] * k_hat[None, :], (2 * thresh)

    return temp


class ContextManager:

    p1 = np.array([4, 1, 1, 1])
    p2 = np.array([3, -1, 0, 1])
    m = 1 - 0.01j
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

        return (
            three_point(
                norm(self.p1),
                norm(self.p2),
                norm(self.p1 + self.p2),
                self.m,
                self.m,
                self.m,
            ).epsilon_0
            * TO_FEYNMAN
        )

    def normalize_args(self):
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

        self.p1 = p1
        self.p2 = p2

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
