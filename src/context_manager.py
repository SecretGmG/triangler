import numpy as np
import matplotlib.pyplot as plt

from .integrand_builder import IntegrandBuilder
from .plot_util import plot_complex, plot_complex_plane
from .wrapped_eval import WrappedEvaluator

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
        
            
    