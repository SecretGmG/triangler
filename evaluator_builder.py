from symbolica import Expression, CompiledComplexEvaluator, S, PrintMode, AtomType
from symbolica_vectors import *
from numpy.typing import NDArray
import numpy as np
import matplotlib.pyplot as plt



def print_theta(theta: Expression, mode: PrintMode, **kwargs) -> str | None:
    if mode == PrintMode.Latex:
        if theta.get_type() == AtomType.Fn:
            return "\mu_{" + ",".join(a.format() for a in theta) + "}"
        else:
            return "\mu"

THETA = S(
    "Theta",
    custom_print=print_theta,
)

EXTERNAL_FUNCTIONS = {
    (THETA, "theta"): lambda args: 1.0 * (args[0] > 0),
}
CUSTOM_HEADER = (
    "template<typename T> T theta(T x) { return x.real() > 0 ? T(1) : T(0); }"
)


class WrappedEvaluator:
    """
    Wraps a Symbolica Expression into a compiled evaluator for fast numerical evaluation.
    Handles vector flattening, broadcasting, and mapping inputs to compiled C++ code.
    """

    constant_args: dict[Expression | SymbolicaLorenzVec | SymbolicaVec, NDArray] = None
    args: list[Expression | SymbolicaLorenzVec | SymbolicaVec]
    expression: Expression = None
    evaluator: CompiledComplexEvaluator = None

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
        expression: Expression,
        constant_args: dict[Expression | SymbolicaLorenzVec | SymbolicaVec, NDArray],
        args: list[Expression | SymbolicaLorenzVec | SymbolicaVec],
        name: str,
    ):
        self.expression = expression
        self.constant_args = constant_args
        self.args = args
        self.name = name

    def flat_all_args(self):
        """Flatten the list of possibly vector valued arguments into a single list of Expression objects."""
        all_args = list(list(self.constant_args.keys()) + self.args)
        flat_args = []
        for arg in all_args:
            if isinstance(arg, SymbolicaLorenzVec) or isinstance(arg, SymbolicaVec):
                flat_args += arg.symbols
            else:
                flat_args += [arg]
        return flat_args

    def compile(self):
        print(f'Compiling evaluator: "{self.name}"')
        self.evaluator: CompiledComplexEvaluator = self.expression.evaluator(
            {}, {}, self.flat_all_args(), external_functions=EXTERNAL_FUNCTIONS
        ).compile(
            self.name,
            f"evaluators/{self.name}.cpp",
            f"evaluators/{self.name}.so",
            number_type="complex",
            custom_header=CUSTOM_HEADER,
        )
        print('Done!')

    def evaluate(self, args: list[NDArray]) -> NDArray:
        """Evaluate the compiled expression on given input arrays.

        Args:
            args: list of arrays corresponding to each argument (scalars or vectors).
                  Each array should be shaped [N, D], with D=1 for scalars, D=3 for vectors and D=4 for lorentz vectors.

        Returns:
            A NumPy array of evaluated results with shape [N, ...].
        """
        if self.evaluator is None:
            self.compile()
        
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
    ys = e.evaluate([xs.reshape(-1, 3)]).reshape(res, res)

    axes = ["x", "y", "z"]

    # Convert to HSV: hue = phase, value = normalized magnitude
    phase = np.angle(ys)
    mag = np.abs(ys)
    # mag = 1 - 1/(mag**0.5+1)
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
        lims = (-1, 1)

    ts = np.linspace(lims[0], lims[1], res)

    xs = x_0[:, None] + x_hat[:, None] * ts
    ys = np.zeros(res, dtype=np.complex128)
    ys = e.evaluate([xs.T]).reshape(res)

    plt.plot(ts, ys.real, label="re")
    plt.plot(ts, ys.imag, label="im")
