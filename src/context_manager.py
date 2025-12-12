import numpy as np
import matplotlib.pyplot as plt

from integrand_builder import IntegrandBuilder
from plot_util import plot_complex, plot_complex_plane, get_contour
from wrapped_eval import WrappedEvaluator

def norm(v):
    """
    computes the minkowski norm
    """
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


class TriangleIntegrandContext:
    """
    Manages the context of the triangle integrand
    provides functions for setting up the parameters,
    evaluating and plotting the integrand
    """
    
    origin = np.array([0,0,0,0])

    p1 = np.array([2, 1, 0, 0])
    p2 = np.array([2, -1, 0, 0])

    masses = [1, 1, 1]
    threshold = 5
    
    max_imag_root_part = 1e15
    max_eta_min = 0
    
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
    
    def get_qs(self):
        """
        returns the qs arguments of the integrand as a list
        """
        qs = [
            self.origin + self.p1,
            self.origin + np.zeros_like(self.p1),
            self.origin - self.p2,
        ]
        
        return qs
    
    def get_ordered_qs_and_masses(self):
        """
        returns masses and qs ordered by q0
        """
        
        return zip(*sorted(zip(self.get_qs(),self.masses), key=lambda q: q[0][0]))
    
    def get_args(self):
        """
        returns the arguments of the integrand as a list except for the k argument
        """
        ordered_qs, ordered_masses = self.get_ordered_qs_and_masses()
        
        return (
            [np.pi, self.threshold, self.a, self.b, self.c, self.max_imag_root_part, self.max_eta_min]
            + list(ordered_masses)
            + list(ordered_qs)
        )
    
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
    
    def set_external_momenta(self, p12, p22, p32):
        """
        Set p1 and p2 in the center-of-momentum frame such that.
        p1^2 = p12, p2^2 = p22, (p1 + p2)^2 = p32
        and
        p1.x = p2.x = p1.y = p2.y = 0
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
    
    def set_anomalous_configuration(self, p2 =203): # TODO: make this exact
        s = 350**2 # GeV
        p12 = 120**2 # GeV
        top_mass = 172.76 # GeV
        bottom_mass = 4.18 # GeV
        self.masses = [top_mass, bottom_mass, top_mass]
        self.origin = np.array([0,0,0,-37.9]) # TODO: make this exact
        self.set_external_momenta(p12,p2**2,s)

class TriangleIntegrandEvaluator:
    """
    provides functions for evaluating and plotting compiled expressions
    with a triangle integrand context
    """
    
    def __init__(self, evaluator = None, context = None):
        if evaluator is None:
            ib = IntegrandBuilder()
            evaluator = WrappedEvaluator(
                ib.combined_result(), ib.get_args(), "combined_result", True
            )
        self.evaluator = evaluator

        if context is None:
            context = TriangleIntegrandContext()
        self.context = context

    def eval(self, k):
        """
        evaluates the integrand at k
        """
        shape = k.shape[:-1]
        return self.evaluator.evaluate(self.context.get_args() + [k.reshape(-1, 3)]).reshape(shape)


    def plot_threshold_subtraction(
        self, x_lim, y_lim, x_axis: int=0, y_axis: int=1, res=200
    ):
        """
        Visualizes the threshold subtraction procedure
        """
        axis_labels = ["x", "y", "z"]
        
        x_dir = np.zeros(3)
        y_dir = np.zeros(3)
        x_dir[x_axis] = 1
        y_dir[y_axis] = 1

        x = np.linspace(x_lim[0], x_lim[1], res)
        y = np.linspace(y_lim[0], y_lim[1], res)
        X, Y = np.meshgrid(x, y)

        xs_plane = X + Y * 1j
        ks_plane = (X[..., None] * x_dir + Y[..., None] * y_dir).reshape(-1, 3)
        ks_line = (x_dir[:, None] * x).T

        ks_plane_jac = np.sum(ks_plane**2, axis=1)
        ks_line_jac = np.sum(ks_line**2, axis=1)

        self.context.only_unsubtracted()
        plt.figure(figsize=(20, 10))
        plt.subplot(2, 3, 1)
        plt.title('Unsubtracted integrand')
        integrand = (self.eval(ks_plane) * ks_plane_jac).reshape(res, res)
        plot_complex_plane(xs_plane, integrand, cmap_factor=20)
        plt.xlabel(axis_labels[x_axis])
        plt.ylabel(axis_labels[y_axis])
        plt.subplot(2, 3, 4)
        plot_complex(x, self.eval(ks_line) * ks_line_jac)
        plt.xlabel(axis_labels[x_axis])
        plt.ylabel('Integrand')

        self.context.only_counterterm()
        plt.subplot(2, 3, 2)
        plt.title('Counterterm')
        counter_term = (self.eval(ks_plane) * ks_plane_jac).reshape(res, res)
        plot_complex_plane(xs_plane, counter_term, cmap_factor=20)
        plt.xlabel(axis_labels[x_axis])
        plt.ylabel(axis_labels[y_axis])
        plt.subplot(2, 3, 5)
        plot_complex(x, self.eval(ks_line) * ks_line_jac)
        plt.xlabel(axis_labels[x_axis])
        plt.ylabel('Integrand')

        self.context.combined_integrand_without_integrated_counterterm()
        plt.subplot(2, 3, 3)
        plt.title('Subtracted integrand')
        subtracted = (self.eval(ks_plane) * ks_plane_jac).reshape(res, res)
        plot_complex_plane(xs_plane, subtracted)
        plt.xlabel(axis_labels[x_axis])
        plt.ylabel(axis_labels[y_axis])
        plt.subplot(2, 3, 6)
        plot_complex(x, self.eval(ks_line) * ks_line_jac)
        plt.xlabel(axis_labels[x_axis])
        plt.ylabel('Integrand')

    def plot_planes(self, x_lim, y_lim, res=300, divide_jacobian=False):
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
            ks_plane = (X[..., None] * x_axis + Y[..., None] * y_axis)

            plt.subplot(1, 3, i + 1)
            integrand = self.eval(ks_plane)
            if divide_jacobian:
                integrand *= np.sum(ks_plane**2, axis = -1)
            plot_complex_plane(xs_plane, integrand)
            plt.xlabel(["x", "y", "z"][x_])
            plt.ylabel(["x", "y", "z"][y_])
    
    def plot_unit_sphere(self, res=200):
        """
        Visualizes the integrand along the unit hemisphere
        """
        u = np.linspace(-1, 1, res)
        v = np.linspace(0, 1, res)

        U, V = np.meshgrid(u, v)
        xs = np.stack([U, V], axis=-1).reshape(-1, 2)
        ks, _ = hemispherical(xs)
        plot_complex_plane(U + V * 1j, self.eval(ks).reshape(res, res))
        plt.xlabel(r"$\theta/2\pi$")
        plt.ylabel(r"$\cos(\phi)$")

    def plot_complex_line(self, k_hat, re_lim, im_lim, res, ax = None):
        if ax is None:
            ax = plt.gca()
        x = np.linspace(re_lim[0], re_lim[1], res)
        y = np.linspace(im_lim[0], im_lim[1], res)
        X, Y = np.meshgrid(x, y)

        xs = X + 1j*Y
        val = self.eval(xs[..., None] * k_hat)
        plot_complex_plane(xs, val, ax)


    

            
    