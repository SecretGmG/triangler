from symbolica import Expression, CompiledComplexEvaluator, S, PrintMode, AtomType
from symbolica_vectors import *
from numpy.typing import NDArray
import numpy as np


# Heaviside theta function
THETA = S("Theta", is_real=True, is_positive=True, is_scalar=True)
# Extracts the real part of a complex number
REAL = S("Real", is_antisymmetric=True, is_linear=True, is_scalar=True, is_real=True)
# Extracts the imaginary part of a complex number
IMAG = S("Imag", is_antisymmetric=True, is_linear=True, is_scalar=True, is_real=True)


EXTERNAL_FUNCTIONS = {
    (THETA, "theta"): lambda args: 1.0 * (args[0] > 0),
    (REAL, "real"): lambda args: np.real(args[0]),
    (IMAG, "imag"): lambda args: np.imag(args[0]),
}
CUSTOM_HEADER = """
template<typename T> T theta(T x) { return x.real() > 0 ? T(1) : T(0); }
template<typename T> T real(T x) { return T(x.real()); }
template<typename T> T imag(T x) { return T(x.imag()); }
"""

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
    def flatten_vectors(
        args: dict[Expression | SymbolicaLorenzVec | SymbolicaVec, float | NDArray],
    ) -> NDArray:
        """
        Flatten values of scalars and vectors into a 2D array suitable for the evaluator.

        Scalars remain as single-element arrays, vector components are flattened along rows,
        and broadcasting is applied to ensure consistent shapes.
        """
        flat_values = []
        for key, value in args.items():
            if isinstance(key, SymbolicaLorenzVec):
                l = list(np.asarray(value).T)
                assert len(l) == 4
                flat_values += l
            elif isinstance(key, SymbolicaVec):
                l = list(np.asarray(value).T)
                assert len(l) == 3
                flat_values += l
            else:
                flat_values += [np.asarray(value)]

        return np.atleast_2d(np.column_stack(list(np.broadcast_arrays(*flat_values)))).astype(complex)

    def __init__(
        self,
        expression: Expression,
        args: list[Expression | SymbolicaLorenzVec | SymbolicaVec],
        name: str,
        force_rebuild: bool = True
    ):
        self.expression = expression
        self.args = args
        self.name = name
        self.ensure_evaluator(force_rebuild)

    def flat_args(self):
        """Flatten the list of possibly vector valued arguments into a single list of Expression objects."""
        flat_args = []
        for arg in self.args:
            if isinstance(arg, SymbolicaLorenzVec) or isinstance(arg, SymbolicaVec):
                flat_args += arg.values
            else:
                flat_args += [arg]
        return flat_args

    def ensure_evaluator(self, force_rebuild):
        if not force_rebuild:
            path = f'../evaluators/{self.name}.so'
            try:
                self.evaluator = CompiledComplexEvaluator.load(path, self.name, len(self.flat_args()), 1)
                print(f'loaded "{path}"')
                return
            except Exception as e:
                print(f'could not load {path} due to {e}')
        
        print(f'Compiling evaluator: "{self.name}"')
        
        self.evaluator: CompiledComplexEvaluator = self.expression.evaluator(
            {}, {}, self.flat_args(), external_functions=EXTERNAL_FUNCTIONS
        ).compile(
            self.name,
            f"../evaluators/{self.name}.cpp",
            f"../evaluators/{self.name}.so",
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
            A NumPy array of evaluated results with shape [N].
        """
        
        values = WrappedEvaluator.flatten_vectors(dict(zip(self.args, args)))
        return self.evaluator.evaluate(values)[:,0]

