from .plot_util import plot_complex, plot_complex_plane
from .symbolica_vectors import SymbolicaLorenzVec, SymbolicaVec
from symbolica import S, N, Expression
from .wrapped_eval import THETA, WrappedEvaluator
import numpy as np
import matplotlib.pyplot as plt


HALF = N(1) / 2
EPS = N(np.finfo(np.float64).eps)
I_EPS = Expression.I * EPS


def sqrt(e: Expression):
    """
    returns e**HALF
    """
    return e**HALF


def real(e: Expression):
    """
    returns the real part of e
    """
    return (e + e.conjugate()) * HALF


def imag(e: Expression):
    """
    returns the imaginary part of e
    """
    return (e - e.conjugate()) * HALF


def c_abs(e: Expression):
    """
    returns the absolute value of e
    """
    return sqrt(e * e.conjugate())


class IntegrandBuilder:
    """
    provides functions for building the integrand, the counterterm of the integrand and the radially integrated counterterm
    """

    qs: list[SymbolicaLorenzVec] = [
        SymbolicaLorenzVec.from_name(f"q{i}") for i in range(3)
    ]
    k: SymbolicaVec = SymbolicaVec.from_name("k")
    r = k.squared() ** HALF
    k_hat: SymbolicaVec = k.normalized()
    masses = [S(f"m{i}") for i in range(3)]
    thresh: Expression = S("lambda")

    a = S("a")
    b = S("b")
    c = S("c")

    # indices for the counterterm
    # this only works by assuming that q0_0 < q1_0 < q2_0
    eta_indices = [
        (0, 1),
        (1, 2),
        (0, 2),
    ]
    part_indices = [
        (1, 0, 2, 0),
        (0, 1, 0, 2),
        (0, 1, 2, 1),
        (1, 0, 1, 2),
        (0, 2, 1, 2),
        (2, 0, 2, 1),
    ]

    def get_args(self):
        """
        returns all the arguments of the integrand as a list
        [pi, threshold, a, b, c, m0, m1, m2, [q0,4], [q1,4], [q2,4], [k,3]]
        where:
         - threshold is the radius of the threshold subtraction region
         - a, b, c are the factors of the integral, the counterterm and the radially integrated counterterm
         - m0, m1, m2 are the internal masses
         - q0, q1, q2 are the internal 4-momenta offsets
         - k is the argument of the integrand (3-momentum)
        """
        return (
            [Expression.PI, self.thresh, self.a, self.b, self.c]
            + self.masses
            + self.qs
            + [self.k]
        )

    def ose(self, i: int, k: SymbolicaVec):
        """
        returns  the onshell energy of particle i
        """
        temp: Expression = k - self.qs[i].spacial()
        return sqrt(temp.squared() + self.masses[i] ** N(2))

    def prefactor(self, k: SymbolicaVec):
        """
        returns the prefactor of the integrand
        """
        return (4 * Expression.PI) ** N(-3) / (
            self.ose(0, k) * self.ose(1, k) * self.ose(2, k)
        )

    def eta(self, i, j, k):
        """
        returns the e-surface (i,j) value at k
        """
        return self.ose(i, k) + self.ose(j, k) + self.qs[i].t() - self.qs[j].t()

    def unsubtracted(self):
        """
        returns the unsubtracted integrand at self.k
        """
        integrand = N(0)
        for i, j, k, l in self.part_indices:
            integrand += 1 / (self.eta(i, j, self.k) * self.eta(k, l, self.k))
        return integrand * self.prefactor(self.k)

    def eta_radius_roots(self, i, j):
        """
        returns the location of the roots of the e-surface (i,j) when parameterized hemispherically with the direction vector self.k_hat
        """
        q: SymbolicaLorenzVec = (self.qs[i] - self.qs[j]) * HALF
        v: SymbolicaVec = q.spacial() * (N(1) / q.t())
        k_0_p = -(self.qs[i] + self.qs[j]).spacial() * HALF

        k_hat_v = self.k_hat * v
        delta = (self.masses[i] ** N(2) - self.masses[j] ** N(2)) / (N(4) * q.t())
        m_2_avg = (self.masses[i] ** N(2) + self.masses[j] ** N(2)) / N(2)

        a = N(1) - k_hat_v ** N(2)

        b = N(2) * ((self.k_hat * k_0_p) - k_hat_v * (k_0_p * v) + delta * k_hat_v)

        c = (
            k_0_p.squared()
            - (k_0_p * v) ** N(2)
            + 2 * delta * (k_0_p * v)
            - q.squared()
            + m_2_avg
            - delta**2
        ) - I_EPS

        d = b ** N(2) - N(4) * a * c
        
        #selector = THETA((m_2_avg - q.squared() + delta**2 * (1-2*v*v)/(1-v*v)))
        #selector = THETA(-self.eta(i, j, SymbolicaVec.zero()))
        #selector = N(1)
        #selector = THETA(d)
        selector = THETA(N(2)*q.t()**N(2) - q.spacial().squared() - m_2_avg)
        
        
        return [
            (-b + sqrt(d)) / (N(2) * a),
            (-b - sqrt(d)) / (N(2) * a),
        ], selector

    def ddk_eta(self, i, j, k):
        """
        returns the partial derivative of the e-surface (i,j) with respect to k along the direction k_hat
        """
        d1 = self.k_hat * (k - self.qs[i].spacial()) / self.ose(i, k)
        d2 = self.k_hat * (k - self.qs[j].spacial()) / self.ose(j, k)
        return d1 + d2

    def eta_ct(self, i, j) -> list[(Expression, SymbolicaVec)]:
        """
        returns the counterterm for the e-surface (i,j) when parameterized around the origin
        """
        poles, selector = self.eta_radius_roots(i, j)

        out = []

        for r_star in poles:
            k_star = r_star * self.k_hat
            
            selector = THETA(N(0.00001)-c_abs(self.eta(i, j, k_star)))
            
            factor = (
                self.collect_other_etas(i, j, k_star)
                * selector
                * self.prefactor(k_star)
                / self.ddk_eta(i, j, k_star)
            )
            out.append((factor, r_star))

        return out

    def collect_other_indices(self, i, j):
        """
        returns the indices of the e-surfaces that appear as factors of the e-surface (i,j)
        """
        order = j - i
        return [(i, (i - order) % 3), ((j + order) % 3, j)]

    def collect_other_etas(self, i, j, k):
        """
        collects the factors of the e-surface (i,j)
        """
        other_etas = N(0)
        for i, j in self.collect_other_indices(i, j):
            other_etas += 1 / (self.eta(i, j, k))
        return other_etas

    def ct(self):
        """
        returns the counterterm of the integrand at self.k
        """
        ct = N(0)
        for i, j in self.eta_indices:
            values = self.eta_ct(i, j)
            for factor, r_star in values:
                selector = THETA(self.thresh-self.r)
                ct += (
                    selector
                    * factor
                    * (real(r_star) / self.r) ** N(2)
                    / (self.r - r_star)
                )
        return ct

    def integrated_counterterm(self):
        """
        returns the radially integrated counterterm along the direction self.k_hat
        """
        ct = N(0)
        for i, j in self.eta_indices:
            values = self.eta_ct(i, j)
            for factor, r_star in values:

                upper = self.thresh
                lower = -self.thresh

                integrated = Expression.LOG((upper - r_star) / (lower - r_star))

                ct += factor * real(r_star) ** N(2) * integrated
        return ct

    def combined_result(self):
        """
        returns the combined result
            a * unsubtracted() - b * ct() + c * integrated_counterterm() * factor
            where factor = THETA(thresh - r) / (2/3 * thresh**3)

        integrating this over R^3 gives the result of the integral
        """
        integrated_ct_factor = THETA(self.thresh - self.r) / (
            N(2) / N(3) * (self.thresh**N(3))
        )
        return (
            self.unsubtracted() * self.a
            - self.ct() * self.b
            + self.integrated_counterterm() * self.c * integrated_ct_factor
        )


import numpy as np


def norm(v):
    return v[0] ** 2 - (v[1:] ** 2).sum()


def hemispherical(xs):
    """
    transforms sample points from [0,1)^2 to the unit hemisphere (z >= 0).
    xs: (N,2) array with values in [0,1)
    Returns:
        v: (N,3) Cartesian coordinates on unit hemisphere
        jac: (N,) Jacobian (area element) for Monte Carlo integration
    """
    xs = np.asarray(xs)
    N = xs.shape[0]

    theta = 2 * np.pi * xs[:, 0]
    cos_phi = xs[:, 1]
    sin_phi = np.sqrt(1.0 - cos_phi**2)

    v = np.empty((N, 3), dtype=float)
    v[:, 0] = sin_phi * np.cos(theta)
    v[:, 1] = sin_phi * np.sin(theta)
    v[:, 2] = cos_phi

    jac = np.full(N, 2.0 * np.pi)
    return v, jac


def spherical(xs):
    """
    transform sample points from [0,1)^3 to R^3 parameterized hemispherically
    xs: (N,3) array with values in [0,1)
    Returns:
        v: (N,3) Cartesian coordinates
        jac: (N,) Jacobian for Monte Carlo integration (dV / d(u1,u2,u3))
    """
    xs = np.asarray(xs)
    w = 2 * xs[:, 2] - 1  # (-1,1)
    r = w - 1 / w
    r_jac = (1 + 1 / w**2) * 2

    v, h_jac = hemispherical(xs[:, 0:2])

    jac = r_jac * r**2 * h_jac
    return r[:, None] * v, jac


def line_segment(k_hat, thresh):
    """ 
    returns a function that is a parametrization from 0,1 
    to a line segment from - k_hat * thresh to k_hat * thresh
    """
    k_hat = np.asarray(k_hat)

    def temp(xs):
        return ((xs[:, 0] * 2 - 1) * thresh)[:, None] * k_hat[None, :], (2 * thresh)

    return temp


class ContextManager:
    """
    manages the parameters of the integrand like external momenta and internal masses.
    provides functions to extract the correct arguments for the integrand builder
    and evaluate the integrand
    
    provides functions to integrate and plot the integrand
    """
    origin = np.array([0,0,0,0])
    
    p1 = np.array([2, 1, 0, 0])
    p2 = np.array([2, -1, 0, 0])
    masses = [1, 1, -0.01j, 1 - 0.1j]
    threshold = 5
    a = 1
    b = 1
    c = 1
    
    def only_integrated_counterterm(self):
        self.a = 0
        self.b = 0
        self.c = 1
    
    def only_unsubtracted(self):
        self.a = 1
        self.b = 0
        self.c = 0
    
    def only_counterterm(self):
        self.a = 0
        self.b = -1
        self.c = 0
    
    def combined_integrand(self):
        self.a = 1
        self.b = 1
        self.c = 1
    
    def combined_integrand_without_integrated_counterterm(self):
        self.a = 1
        self.b = 1
        self.c = 0

    def __init__(self, force_rebuild=True):
        ib = IntegrandBuilder()
        self.ib = ib

        self.evaluator = WrappedEvaluator(
            ib.combined_result(), ib.get_args(), "combined_result", force_rebuild
        )

    def get_qs(self):
        """
        returns the qs arguments of the integrand as a list
        ordered such that q0_0 < q1_0 < q2_0
        """
        qs = [
            self.origin + self.p1,
            self.origin + np.zeros_like(self.p1),
            self.origin - self.p2,
        ]
        
        return qs


    def get_context_args(self):
        """
        returns the arguments of the integrand as a list except for the k argument
        """
        
        ordered_qs, ordered_masses = zip(*sorted(zip(self.get_qs(),self.masses), key=lambda q: q[0][0]))
        
        return (
            [np.pi, self.threshold, self.a, self.b, self.c]
            + list(ordered_masses)
            + list(ordered_qs)
        )

    def eval(self, k):
        """
        evaluates the integrand at k
        """
        return self.evaluator.evaluate(self.get_context_args() + [k])

    def get_reference(self) -> complex:
        """
        returns the reference value of the integration using oneloop_bridge
        """
        from oneloop_bridge import three_point, TO_FEYNMAN

        res = three_point(
            norm(self.p1),
            norm(self.p2),
            norm(self.p1 + self.p2),
            self.masses[0] ** 2,
            self.masses[1] ** 2,
            self.masses[2] ** 2,
        )
        assert res.epsilon_minus_1 == 0
        assert res.epsilon_minus_2 == 0

        return res.epsilon_0 * TO_FEYNMAN

    def plot_threshold_subtraction(
        self, x_lim, y_lim, x_axis=None, y_axis=None, res=200
    ):
        """
        Visualizes the threshold subtraction procedure
        """
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

        self.only_unsubtracted()
        plt.figure(figsize=(20, 10))
        plt.subplot(2, 3, 1)
        integrand = (self.eval(ks_plane) * ks_plane_jac).reshape(res, res)
        plot_complex_plane(xs_plane, integrand)
        plt.subplot(2, 3, 4)
        plot_complex(x, self.eval(ks_line) * ks_line_jac)

        self.only_counterterm()
        plt.subplot(2, 3, 2)
        counter_term = (self.eval(ks_plane) * ks_plane_jac).reshape(res, res)
        plot_complex_plane(xs_plane, counter_term)
        plt.subplot(2, 3, 5)
        plot_complex(x, self.eval(ks_line) * ks_line_jac)

        self.combined_integrand_without_integrated_counterterm()
        plt.subplot(2, 3, 3)
        subtracted = (self.eval(ks_plane) * ks_plane_jac).reshape(res, res)
        plot_complex_plane(xs_plane, subtracted)
        plt.subplot(2, 3, 6)
        plot_complex(x, self.eval(ks_line) * ks_line_jac)

    def plot_planes(self, x_lim, y_lim, res=100):
        """
        Visualizes the integrand in the three planes
        """
        plt.figure(figsize=(20, 7))
        for i, (x_, y_) in enumerate([(0, 1), (1, 2), (2, 0)]):
            x_axis = np.zeros(3)
            y_axis = np.zeros(3)
            x_axis[x_] = 1
            y_axis[y_] = 1

            x = np.linspace(x_lim[0], x_lim[1], res)
            y = np.linspace(y_lim[0], y_lim[1], res)
            X, Y = np.meshgrid(x, y)
            xs_plane = X + Y * 1j
            ks_plane = (X[..., None] * x_axis + Y[..., None] * y_axis).reshape(-1, 3)
            ks_plane_jac = np.sum(ks_plane**2, axis=1)

            plt.subplot(1, 3, i + 1)
            integrand = (self.eval(ks_plane) * ks_plane_jac).reshape(res, res)
            plot_complex_plane(xs_plane, integrand)
            plt.xlabel(["x", "y", "z"][x_])
            plt.ylabel(["x", "y", "z"][y_])

    def plot_integrated_counterterm(self, res=200):
        """
        plots the integrated counterterm along the unit hemisphere
        """
        u = np.linspace(0, 1, res)
        v = np.linspace(0, 1, res)

        U, V = np.meshgrid(u, v)
        xs = np.stack([U, V], axis=-1).reshape(-1, 2)
        ks, _ = hemispherical(xs)
        self.only_integrated_counterterm()
        plot_complex_plane(U + V * 1j, self.eval(ks).reshape(res, res))
        plt.xlabel(r"$\theta/2\pi$")
        plt.ylabel(r"$\cos(\phi)$")

    def set_external_momenta(self, p12, p22, p32):
        """
        Set p1 and p2 in the center-of-momentum frame.
        p1^2 = p12, p2^2 = p22, (p1 + p2)^2 = p32.
        """
    
        s = p32
        sqrt_s = np.sqrt(s)
    
        # energies in the COM frame
        E1 = (s + p12 - p22) / (2.0 * sqrt_s)
        E2 = (s + p22 - p12) / (2.0 * sqrt_s)
    
        # common three-momentum magnitude
        p_abs_sq = E1**2 - p12
        if p_abs_sq < 0:
            raise ValueError("Kinematic point is not physically allowed (negative momentum^2).")
    
        p_abs = np.sqrt(p_abs_sq)
    
        # pick them back-to-back along z
        self.p1 = np.array([E1, 0.0, 0.0,  p_abs])
        self.p2 = np.array([E2, 0.0, 0.0, -p_abs])
        
            
    