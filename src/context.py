from typing import Callable
import numpy as np
import matplotlib.pyplot as plt

from integrand_builder import IntegrandBuilder
from plot_util import plot_complex, plot_complex_plane
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


class TriangleIntegrandContext:
    """
    Manages the constant (hyper-)parameters of the triangle integrand
    provides functions for setting up the parameters,
    evaluating and plotting the integrand
    """

    origin = np.array([0, 0, 0, 0])

    p1 = np.array([2, 1, 0, 0])
    p2 = np.array([2, -1, 0, 0])

    masses = [1.0, 1.0, 1.0]

    subtraction_width = 5.0
    max_imag_root_part = 1e15
    max_eta_min = 0.0
    mask_width = 5.0

    a = 1.0
    b = 1.0
    c = 1.0

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
        returns the loop momenta offset arguments of the integrand as a list
        """
        qs = [
            -self.origin + self.p1,
            -self.origin + np.zeros_like(self.p1),
            -self.origin - self.p2,
        ]

        return qs

    def get_ordered_qs_and_masses(self):
        """
        returns masses and qs ordered by q0
        to conform to the expected ordering of the integrand
        """

        return zip(*sorted(zip(self.get_qs(), self.masses), key=lambda q: q[0][0]))

    def get_args(self):
        """
        returns the arguments of the integrand (except for the k argument) as a list
        """
        ordered_qs, ordered_masses = self.get_ordered_qs_and_masses()

        return (
            [
                np.pi,
                self.a,
                self.b,
                self.c,
                self.subtraction_width,
                self.max_imag_root_part,
                self.max_eta_min,
                self.mask_width,
            ]
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

    def set_external_momenta(self, p1_sq, p2_sq, p3_sq):
        """
        Set p1 and p2 in the center-of-momentum frame such that.
        p1^2 = p1_sq, p2^2 = p2_sq, (p1 + p2)^2 = p3_sq
        and
        p1.x = p2.x = p1.y = p2.y = 0
        and
        p1.z >= 0, p2.z <= 0
        """

        s = p3_sq
        sqrt_s = np.sqrt(s)

        # energies in the COM frame
        E1 = (s + p1_sq - p2_sq) / (2.0 * sqrt_s)
        E2 = (s + p2_sq - p1_sq) / (2.0 * sqrt_s)

        p_abs_sq = E1**2 - p1_sq
        if p_abs_sq < 0:
            raise ValueError(
                "Kinematic point is not physically allowed (negative momentum^2)."
            )

        p_abs = np.sqrt(p_abs_sq)

        self.p1 = np.array([E1, 0.0, 0.0, p_abs])
        self.p2 = np.array([E2, 0.0, 0.0, -p_abs])




class TriangleIntegrandEvaluator:
    """
    provides functions for evaluating and plotting compiled expressions
    with a triangle integrand context
    """

    def __init__(self, evaluator=None, context=None):
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
        return self.evaluator.evaluate(
            self.context.get_args() + [k.reshape(-1, 3)]
        ).reshape(shape)

    def plot_threshold_subtraction(
        self, x_lim, y_lim, x_axis: int = 0, y_axis: int = 1, res=200
    ,clip = 100):
        """
        Visualizes the threshold subtraction procedure to successfully visualize the thresholds a clip value may be provided, which clips the function at the percentile of the given clip value
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
        plt.subplots(2, 3, height_ratios=[3,2], figsize=(20,10))
        plt.subplot(2, 3, 1)
        plt.title("Unsubtracted integrand")
        integrand = (self.eval(ks_plane) * ks_plane_jac).reshape(res, res)
        plot_complex_plane(xs_plane, clip_abs(integrand, clip))
        plt.xlabel(axis_labels[x_axis])
        plt.ylabel(axis_labels[y_axis])
        plt.subplot(2, 3, 4)
        plot_complex(x, self.eval(ks_line) * ks_line_jac)
        plt.legend()
        plt.xlabel(axis_labels[x_axis])
        plt.ylabel("Integrand")

        self.context.only_counterterm()
        plt.subplot(2, 3, 2)
        plt.title("Counterterm")
        counter_term = (self.eval(ks_plane) * ks_plane_jac).reshape(res, res)
        plot_complex_plane(xs_plane, clip_abs(counter_term, clip))
        plt.xlabel(axis_labels[x_axis])
        plt.ylabel(axis_labels[y_axis])
        plt.subplot(2, 3, 5)
        plot_complex(x, self.eval(ks_line) * ks_line_jac)
        plt.legend()
        plt.xlabel(axis_labels[x_axis])
        plt.ylabel("Integrand")

        self.context.combined_integrand_without_integrated_counterterm()
        plt.subplot(2, 3, 3)
        plt.title("Subtracted integrand")
        subtracted = (self.eval(ks_plane) * ks_plane_jac).reshape(res, res)
        plot_complex_plane(xs_plane, subtracted)
        plt.xlabel(axis_labels[x_axis])
        plt.ylabel(axis_labels[y_axis])
        plt.subplot(2, 3, 6)
        plot_complex(x, self.eval(ks_line) * ks_line_jac)
        plt.legend()
        plt.xlabel(axis_labels[x_axis])
        plt.ylabel("Integrand")

    def plot_planes(self, x_lim, y_lim, res=300, divide_jacobian=False):
        """
        Visualizes the integrand in the three planes xy, xz, yz
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
            ks_plane = X[..., None] * x_axis + Y[..., None] * y_axis

            plt.subplot(1, 3, i + 1)
            integrand = self.eval(ks_plane)
            if divide_jacobian:
                integrand *= np.sum(ks_plane**2, axis=-1)
            plot_complex_plane(xs_plane, integrand)
            plt.xlabel(["x", "y", "z"][x_])
            plt.ylabel(["x", "y", "z"][y_])

    def plot_unit_sphere(self, res=200):
        """
        Visualizes the integrand along the unit sphere
        """
        u = np.linspace(-1, 1, res)
        v = np.linspace(0, 1, res)

        U, V = np.meshgrid(u, v)
        xs = np.stack([U, V], axis=-1).reshape(-1, 2)
        ks, _ = hemispherical(xs)
        plot_complex_plane(U + V * 1j, self.eval(ks).reshape(res, res))
        plt.xlabel(r"$\theta/2\pi$")
        plt.ylabel(r"$\cos(\phi)$")

    def plot_complex_line(self, k_hat, re_lim, im_lim, res, ax=None):
        """
        plots the complex -> complex function integrand(x*k_hat) where x is in re_lim x im_lim*j
        """
        if ax is None:
            ax = plt.gca()
        x = np.linspace(re_lim[0], re_lim[1], res)
        y = np.linspace(im_lim[0], im_lim[1], res)
        X, Y = np.meshgrid(x, y)

        xs = X + 1j * Y
        val = self.eval(xs[..., None] * k_hat)
        plot_complex_plane(xs, val, ax)


def clip_abs(x, clip):
    return x*np.clip(np.abs(x), None, np.percentile(np.abs(x), clip))/(np.abs(x) + 1e-10)




class ThresholdFinder:
    """
    Initialize the ThresholdFinder.

    Builds integrand evaluators for two selected eta-radius roots and binds
    them to the provided evaluation context.

    Parameters
    ----------
    ctx
        Evaluation context whose state is modified during threshold search.
    eta_a : tuple[int, int], optional
        Indices passed to IntegrandBuilder.eta_radius_roots for surface A.
    eta_b : tuple[int, int], optional
        Indices passed to IntegrandBuilder.eta_radius_roots for surface B.
    root_a : int, optional
        Which root to select from the eta_a root list.
    root_b : int, optional
        Which root to select from the eta_b root list.
    """
    def __init__(self, ctx, eta_a = (0,1), eta_b = (0,2), root_a = 0, root_b = 1):
        ib = IntegrandBuilder()
        a = ib.eta_radius_roots(*eta_a)[root_a]
        b = ib.eta_radius_roots(*eta_b)[root_b]
        self.a_eval = TriangleIntegrandEvaluator(WrappedEvaluator(a, ib.get_args(), "a"), ctx)
        self.b_eval = TriangleIntegrandEvaluator(WrappedEvaluator(b, ib.get_args(), "b"), ctx)
        self.ctx = ctx
    
    def get_root_a(self) -> float:
        return self.a_eval.eval(np.array([0,0,1])).real # type: ignore
    def get_root_b(self) -> float:
        return self.b_eval.eval(np.array([0,0,1])).real # type: ignore
    
    def find_threshold(
    self,
    parameter: Callable[[float], None],
    lower_bound: float,
    upper_bound: float,
    max_iter: int = 1000,
    tol: float = 1e-10,
) -> float | None:
        """
        Find a parameter value where the two roots coincide using bisection.

        Performs a binary search over the given parameter interval to find a
        value for which get_root_a() == get_root_b(). The supplied callable is
        assumed to mutate the internal context accordingly.

        Monotonicity of both roots is assumed, but their direction is not; the
        method automatically determines whether the solution is bracketed.

        Parameters
        ----------
        parameter : Callable[[float], None]
            Function that applies a parameter value to the context.
        lower_bound : float
            Lower end of the search interval.
        upper_bound : float
            Upper end of the search interval.
        max_iter : int, optional
            Maximum number of bisection iterations.
        tol : float, optional
            Absolute tolerance for convergence in root difference or interval size.

        Returns
        -------
        float or None
            Parameter value at which the two roots coincide, or None if no
            bracketing solution exists or convergence fails.
        """
        def eval_diff(x: float) -> float:
            _ = parameter(x)
            return self.get_root_a() - self.get_root_b()

        f_low = eval_diff(lower_bound)
        f_up = eval_diff(upper_bound)

        # Check if a solution is bracketed
        if f_low == 0:
            return lower_bound
        if f_up == 0:
            return upper_bound
        if f_low * f_up > 0:
            print(f"Bounds do not bracket a solution: f(lower)={f_low}, f(upper)={f_up}")
            return None

        lo, hi = lower_bound, upper_bound
        f_lo = f_low
        mid = 0.5 * (lo + hi)
        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            f_mid = eval_diff(mid)

            if abs(f_mid) < tol or abs(hi - lo) < tol:
                return mid

            # Standard bisection logic, sign-agnostic
            if f_lo * f_mid <= 0:
                hi = mid
            else:
                lo = mid
                f_lo = f_mid
        print("Maximum iterations reached without convergence")
        return mid