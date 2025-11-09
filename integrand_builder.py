from symbolica_vectors import SymbolicaLorenzVec, SymbolicaVec
from symbolica import S, N, Expression, Evaluator, CompiledComplexEvaluator
import numpy as np
from numpy.typing import NDArray
from numpy.typing import NDArray
import matplotlib.pyplot as plt

HALF = N(1) / 2


class WrappedEvaluator:
    """
    Wraps a Symbolica Expression into a compiled evaluator for fast numerical evaluation.
    Handles vector flattening, broadcasting, and mapping inputs to compiled C++ code.
    """
    evaluator: CompiledComplexEvaluator = None
    contant_args: dict[Expression | SymbolicaLorenzVec | SymbolicaVec, NDArray] = None
    args: list[Expression | SymbolicaLorenzVec | SymbolicaVec]

    @staticmethod
    def _flatten_keys(args: list[Expression | SymbolicaLorenzVec | SymbolicaVec]):
        """Flatten a list of expressions and vector symbols into a single list of Expression objects."""
        args = list(args)
        flat_args = []
        for arg in args:
            if isinstance(arg, SymbolicaLorenzVec) or isinstance(arg, SymbolicaVec):
                flat_args += arg.symbols
            else:
                flat_args += [arg]
        return flat_args

    @staticmethod
    def _flatten_vectors(
        
        args: dict[Expression | SymbolicaLorenzVec | SymbolicaVec, float | NDArray],
    ) -> list[NDArray]:
        """
        Flatten values of scalars and vectors into a 2D array suitable for the evaluator.
    
        Scalars remain as single-element arrays, vector components are flattened along rows,
        and broadcasting is applied to ensure consistent shapes.
        """
        flat_values = []
        for key, value in args.items():
            if isinstance(key, SymbolicaLorenzVec):
                l = list(value.T)
                assert len(l) == 4
                flat_values += l
            elif isinstance(key, SymbolicaVec):
                l = list(value.T)
                assert len(l) == 3
                flat_values += l
            else:
                flat_values += [value]
                
        return np.atleast_2d(np.column_stack(list(np.broadcast_arrays(*flat_values))))

    def __init__(
        self,
        e: Expression,
        contant_args: dict[Expression | SymbolicaLorenzVec | SymbolicaVec, NDArray],
        args: list[Expression | SymbolicaLorenzVec | SymbolicaVec],
        name: str,
    ):
        self.evaluator: CompiledComplexEvaluator = e.evaluator(
            {}, {}, WrappedEvaluator._flatten_keys(list(contant_args.keys()) + args)
        ).compile(
            name,
            f"evaluators/{name}.cpp",
            f"evaluators/{name}.so",
            number_type="complex",
            custom_header=None,
        )
        self.args = args
        self.constant_args = contant_args

    
    def evaluate(self, args: list[NDArray]) -> NDArray:
        """Evaluate the compiled expression on given input arrays.

        Args:
            args: list of arrays corresponding to each argument (scalars or vectors). 
                  Each array should be shaped [N, D], with D=1 for scalars, D=3 for vectors and 4 for lorentz vectors.

        Returns:
            A NumPy array of evaluated results with shape [N, ...].
        """
        args_dict = self.constant_args | dict(zip(self.args, args))
        values = WrappedEvaluator._flatten_vectors(args_dict)
        
        return np.array(self.evaluator.evaluate(values))


def plot_slice(
    e: WrappedEvaluator, x_dim: int = 0, y_dim: int = 1, lims=(-1, 1, -1, 1), res=300
):
    """Plot a 2D slice of a 3D vector field using HSV color encoding for phase and magnitude.
    
    Args:
        e: WrappedEvaluator instance to compute values.
        x_dim, y_dim: indices of axes to plot in the 2D plane.
        lims: (xmin, xmax, ymin, ymax) bounds for the plot.
        res: resolution of the grid along each axis.
    """

    x = np.linspace(lims[0], lims[1], res)
    y = np.linspace(lims[2], lims[3], res)
    X, Y = np.meshgrid(x, y)

    x_hat = np.zeros(3)
    y_hat = np.zeros(3)
    x_hat[x_dim] = 1
    y_hat[y_dim] = 1

    xs = X[..., None] * x_hat + Y[..., None] * y_hat
    ys = e.evaluate([xs.reshape(-1,3)]).reshape(res,res)

    axes = ["x", "y", "z"]

    # Convert to HSV: hue = phase, value = normalized magnitude
    phase = np.angle(ys)
    mag = np.abs(ys)
    #mag = 1 - 1/(mag**0.5+1)
    mag /= np.max(mag) if np.max(mag) != 0 else 1

    # Map phase [-π, π] → hue [0, 1]
    hue = (phase + np.pi) / (2 * np.pi)
    value = mag

    rgb = plt.cm.hsv(hue)  # hue → color
    rgb[..., :3] *= value[..., None]  # scale brightness by magnitude

    plt.imshow(rgb, extent=lims, origin="lower", interpolation="nearest")
    plt.xlabel(axes[x_dim])
    plt.ylabel(axes[y_dim])


def plot_line(e: WrappedEvaluator, x_0: NDArray, x_hat: NDArray, lims=None, res=300):
    """Plot a complex-valued line along a specified direction in 3D space.
    
    Args:
        e: WrappedEvaluator instance.
        x_0: starting point in 3D space.
        x_hat: direction vector.
        lims: (t_min, t_max) range of parameter along the line.
        res: number of points to evaluate.
    """
    if lims is None:
        lims = (-1,1)
    
    ts = np.linspace(lims[0], lims[1], res)

    xs = x_0[:, None] + x_hat[:, None] * ts
    ys = np.zeros(res, dtype=np.complex128)
    ys = e.evaluate([xs.T]).reshape(res)

    plt.plot(ts, ys.real, label="re")
    plt.plot(ts, ys.imag, label="im")


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
            part_builder = IntegrandPartBuilder(eta_ij, eta_kl)
            expr += part_builder.improved_ltd(self.integrand)
        return expr

    def counter_term(self, p1, p2):
        qs = [np.zeros_like(p1), -p1, p2]

        expr = N(0)
        for i, j, k, l in self.part_indices:
            eta_ij = IntegrandEtaBuilder(self, i, j)
            eta_kl = IntegrandEtaBuilder(self, k, l)
            part_builder = IntegrandPartBuilder(eta_ij, eta_kl)
            ct_ij, ct_kl = part_builder.counter_term(self.integrand)

            q = (qs[i] - qs[j]) / 2
            if q[0] < 0:
                expr += ct_ij

            q = (qs[k] - qs[l]) / 2
            if q[0] < 0:
                expr += ct_kl
        return expr


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
        k1_r, k2_r = self.radius_poles(k)
        k_hat = self.k_hat(k)
        k1 = self.center + k1_r * k_hat
        k2 = self.center + k2_r * k_hat
        r = (k - self.center).squared() ** HALF

        # TODO: Think about the complex roots and if everything will be evaluated on the correct sheet

        return (1 / ((r - k1_r) * self._derivative(k1, k_hat)), k1), (
            1 / ((r - k2_r) * self._derivative(k2, k_hat)),
            k2,
        )


COUNTER_JACOBIAN = False


class IntegrandPartBuilder:
    eta_ij: IntegrandEtaBuilder = None
    eta_kl: IntegrandEtaBuilder = None

    def __init__(self, eta_ij: IntegrandEtaBuilder, eta_kl: IntegrandEtaBuilder):
        self.eta_ij = eta_ij
        self.eta_kl = eta_kl

        self.center = SymbolicaVec.zero()
        #self.center = SymbolicaVec.from_name(
        #    f"k{eta_ij.i}{eta_ij.j}{eta_kl.i}{eta_kl.j}"
        #)

    def improved_ltd(self, k):
        return self.eta_ij.integrand_builder.improved_ltd_prefactor(k) / (
            self.eta_ij.eta(k) * self.eta_kl.eta(k)
        )

    def counter_term(self, k: SymbolicaVec):
        
        r = (k-self.center).squared()**HALF       
         
        ct_ij = N(0)
        for ct, k_ in self.eta_ij.counter_term(k):
            
            r_star = (k_-self.center).squared()**HALF
            
            ct_ij += (
                ct
                * self.eta_ij.integrand_builder.improved_ltd_prefactor(k_)
                / self.eta_kl.eta(k_) * ((r_star/r)**N(2) if COUNTER_JACOBIAN else N(1))
            )
        ct_kl = N(0)
        for ct, k_ in self.eta_kl.counter_term(k):
            
            r_star = (k_-self.center).squared()**HALF
            
            ct_kl += (
                ct
                * self.eta_kl.integrand_builder.improved_ltd_prefactor(k_)
                / self.eta_ij.eta(k_) * ((r_star/r)**N(2) if COUNTER_JACOBIAN else N(1))
            )
        return ct_ij, ct_kl

    def radially_integrated_counter_term(self, k_hat):
        pass
