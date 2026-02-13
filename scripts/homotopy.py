import numpy as np
import symbolica as sb
import itertools
import time

class HomotopySystem:
    def __init__(self, exprs, vars_list, degrees):
        self.exprs = exprs
        self.vars_list = vars_list
        self.degrees = degrees
        self.jac_exprs = jacobian(exprs, vars_list)

        self.f_func = sb.Expression.evaluator_multiple(exprs, {}, {}, vars_list)
        self.j_func = sb.Expression.evaluator_multiple(
            list(self.jac_exprs.ravel()), {}, {}, vars_list
        )
    
    @staticmethod
    def from_expr_with_sqrts(expr : sb.Expression, var: sb.Expression, degree = 1):
        
        sqrt_pattern = sb.S('x_').sqrt()
        
        i = 0 
        
        terms = []
        
        
        while expr.matches(sqrt_pattern):
            term_under_sqrt = expr.match(sqrt_pattern).__next__().popitem()[1]
            i += 1
            new_symbol = sb.S(f'P_{i}')
            expr = expr.replace(term_under_sqrt.sqrt(), new_symbol)
            terms.append(new_symbol**sb.N(2) - term_under_sqrt)
        
        return HomotopySystem([expr] + terms, [sb.S(f'P_{j+1}') for j in range(i)] + [var], [2] * i + [degree])        
    
    def evaluate_F(self, x):
        return np.array(self.f_func.evaluate_complex(x), dtype=np.complex128).ravel()
    
    def evaluate_J_F(self, x):
        return np.array(self.j_func.evaluate_complex(x), dtype=np.complex128).reshape(self.jac_exprs.shape)
    
    def evaluate_G(self, x):
        return np.array(
            [x[i] ** self.degrees[i] - 1 for i in range(len(x))],
            dtype=np.complex128,
        )

    def evaluate_J_G(self, x):
        J = np.zeros((len(x), len(x)), dtype=np.complex128)
        for i, d in enumerate(self.degrees):
            J[i, i] = 1 if d == 1 else d * x[i] ** (d - 1)
        return J
    
    def solve_system(self, settings):
        roots = [np.exp(2j * np.pi * np.arange(d) / d) for d in self.degrees]
        start_solutions = list(itertools.product(*roots))
        print(len(start_solutions))

        gamma = np.exp(1j * 1.2345)
        
        results = []
        for s in start_solutions:
            results.append(track_path(s, gamma, settings, self))

        sols = [r for r in results if r is not None]
        return sols


def jacobian(cvec : list[sb.Expression], vars : list[sb.Expression]):
    """
    Creates the Jacobian matrix from a column vector of expressions and a list of variables.
    """
    jac_entries = []
    for i in range(len(cvec)):
        row = []
        for var in vars:
            row.append(cvec[i].derivative(var))
        jac_entries.append(row)
    return np.asarray(jac_entries)


def track_path(x_start, gamma, settings, homotopy_system):
    x = np.array(x_start, dtype=np.complex128)
    t = 0.0
    x_prev, t_prev = None, 0.0

    max_steps = settings["max_steps"]
    step_size = settings["initial_step"]

    for _ in range(max_steps):
        if np.max(np.abs(x)) > settings["divergence_limit"]:
            return None

        if t >= 1.0:
            break

        h = min(step_size, 1.0 - t)

        x_pred, success = _predict_step(
            x, t, h, x_prev, t_prev,
            gamma, settings, homotopy_system
        )

        if not success:
            step_size *= 0.5
            continue

        x_new, success = _correct_step(
            x_pred, t + h,
            gamma, settings, homotopy_system
        )

        if not success:
            step_size *= 0.5
            if step_size < settings["min_step"]:
                return None
            continue

        x_prev, t_prev = x, t
        x, t = x_new, t + h
        step_size = min(step_size * 1.1, 0.1)

    return _final_newton_refinement(x, homotopy_system)


def _predict_step(x, t, h, x_prev, t_prev, gamma, settings, homotopy_system):
    use_secant = settings["predictor"] == "secant"

    if use_secant and x_prev is not None and abs(t - t_prev) > 1e-14:
        dx_dt = (x - x_prev) / (t - t_prev)
        return x + dx_dt * h, True

    try:
        dH_dt = (
            -gamma * homotopy_system.evaluate_G(x)
            + homotopy_system.evaluate_F(x)
        )

        JH = (
            gamma * (1 - t) * homotopy_system.evaluate_J_G(x)
            + t * homotopy_system.evaluate_J_F(x)
        )

        x_pred = x - np.linalg.solve(JH, dH_dt) * h
        return x_pred, True

    except np.linalg.LinAlgError:
        return None, False


def _correct_step(x_pred, t_next, gamma, settings, homotopy_system):
    tol = settings["tolerance"]
    x_new = x_pred

    for _ in range(5):
        try:
            H = (
                gamma * (1 - t_next) * homotopy_system.evaluate_G(x_new)
                + t_next * homotopy_system.evaluate_F(x_new)
            )

            if np.max(np.abs(H)) < tol:
                return x_new, True

            JH = (
                gamma * (1 - t_next) * homotopy_system.evaluate_J_G(x_new)
                + t_next * homotopy_system.evaluate_J_F(x_new)
            )

            x_new -= np.linalg.solve(JH, H)

        except np.linalg.LinAlgError:
            return None, False

    return None, False


def _final_newton_refinement(x, homotopy_system):
    for _ in range(15):
        try:
            F = homotopy_system.evaluate_F(x)

            if np.max(np.abs(F)) < 1e-11:
                return x

            x -= np.linalg.solve(
                homotopy_system.evaluate_J_F(x),
                F
            )

        except np.linalg.LinAlgError:
            break

    return x if np.max(np.abs(homotopy_system.evaluate_F(x))) < 1e-6 else None


def check_sqrt_system(k):
    return (
        np.sqrt(np.sum((k_dir * k) ** 2) + m_1**2)
        + np.sqrt(np.sum((l_dir * k) ** 2) + m_2**2)
        + np.sqrt(
            np.sum((k_dir * k + l_dir * k + p_1[1:] + p_2[1:]) ** 2) + m_3**2
        )
        - p_1[0]
        - p_2[0]
    )
    
def deduplicate_solutions(sols, tol=1e-3):
    unique = []
    for s in sols:
        if not any(np.linalg.norm(s - u) < tol for u in unique):
            unique.append(s)
    return unique


k_dir = np.array([1, 1, 1], dtype=np.float64)
l_dir = np.array([1, 1, 1], dtype=np.float64)

m_1 = 1.0  # GeV
m_2 = 0.7  # GeV
m_3 = 1.0  # GeV
p = 10  # GeV
p_1 = np.array([p, 1, 1, 1], dtype=np.float64)  # GeV
p_2 = np.array([p, 1, 1, 1], dtype=np.float64)  # GeV

norm = np.sum(k_dir) + np.sum(l_dir)
k_dir /= norm
l_dir /= norm


def get_double_box_configuration(K: sb.Expression) -> sb.Expression:
    return (sum((k * K) ** 2 for k in k_dir) + m_1**2).sqrt() + (sum((l * K) ** 2 for l in l_dir) + m_2**2).sqrt() + (sum((k * K + l * K + p_1_ + p_2_) ** 2 for k, l, p_1_, p_2_ in zip(k_dir, l_dir, p_1[1:], p_2[1:])) + m_3**2).sqrt() - (p_1[0] + p_2[0])
    

settings = {
    "max_steps": 5000,
    "initial_step": 0.02,
    "min_step": 1e-8,
    "tolerance": 1e-9,
    "divergence_limit": 1e7,
    "predictor": "secant",
}




def print_configuration_latex():
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\begin{tabular}{l c}")
    print(r"\hline")
    print(r"Quantity & Value \\")
    print(r"\hline")
    print(rf"$\vec{{k}}$ & $({k_dir[0]:.5f}, {k_dir[1]:.5f}, {k_dir[2]:.5f})$ \\")
    print(rf"$\vec{{l}}$ & $({l_dir[0]:.5f}, {l_dir[1]:.5f}, {l_dir[2]:.5f})$ \\")
    print(rf"$m_1$ & ${m_1:.2f}\,\si{{GeV}}$ \\")
    print(rf"$m_2$ & ${m_2:.2f}\,\si{{GeV}}$ \\")
    print(rf"$m_3$ & ${m_3:.2f}\,\si{{GeV}}$ \\")
    print(
        rf"$p_1^\mu$ & $({p_1[0]:.1f}, {p_1[1]:.1f}, {p_1[2]:.1f}, {p_1[3]:.1f})\,\si{{GeV}}$ \\"
    )
    print(
        rf"$p_2^\mu$ & $({p_2[0]:.1f}, {p_2[1]:.1f}, {p_2[2]:.1f}, {p_2[3]:.1f})\,\si{{GeV}}$ \\"
    )
    print(r"\hline")
    print(r"\end{tabular}")
    print(r"\caption{Input configuration for the square-root system.}")
    print(r"\end{table}")
    print()
def print_all_solutions_latex(sols, names):
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\setlength{\tabcolsep}{5pt}")
    print(r"\renewcommand{\arraystretch}{1.25}")

    # One column per variable, plus solution index and residual
    print(r"\begin{tabular}{c " + " ".join(["c"] * len(names)) + r" c}")
    print(r"\hline")

    header = (
        ["Solution"]
        + [rf"${{{n}}} \,[\mathrm{{GeV}}]$" for n in names]
        + [r"Residual $|\eta|$"]
    )
    print(" & ".join(header) + r" \\")
    print(r"\hline")

    for i, sol in enumerate(sols, start=1):
        row = [str(i)]
        for v in sol:
            # Format as a + b i
            row.append(f"${v.real:.2f} {'+' if v.imag >= 0 else '-'} {abs(v.imag):.2f}i$")
        res = check_sqrt_system(sol[-1])
        row.append(f"${abs(res):.2e}$")
        print(" & ".join(row) + r" \\")

    print(r"\hline")
    print(r"\end{tabular}")
    print(r"\caption{All solutions of the square-root system in $a + b i$ notation. Residuals are given as $|\eta|$.}")
    print(r"\label{tab:solutions}")
    print(r"\end{table}")
    print()



if __name__ == "__main__":
    k = sb.S("K")
    double_box =  get_double_box_configuration(k)
    
    print_configuration_latex()
    homotopy_system = HomotopySystem.from_expr_with_sqrts(double_box, k, 2)
    
    t_0 = time.time()
    sols = homotopy_system.solve_system(settings)
    unique = deduplicate_solutions(sols)
    
    print(f"Done in {time.time() - t_0:.2f}s. Found {len(unique)} solutions.")
    names =  [str(v) for v in homotopy_system.vars_list]
    
    print_all_solutions_latex(unique, names)
        