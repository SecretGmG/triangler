from symbolica import Expression, CompiledComplexEvaluator, S, PrintMode, AtomType
from symbolica_vectors import *
from numpy.typing import NDArray
import numpy as np
import matplotlib.pyplot as plt



def print_theta(theta: Expression, mode: PrintMode, **kwargs) -> str | None:
    if mode == PrintMode.Latex:
        if theta.get_type() == AtomType.Fn:
            return "\\theta_{" + ",".join(a.format() for a in theta) + "}"
        else:
            return "\\theta"

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

