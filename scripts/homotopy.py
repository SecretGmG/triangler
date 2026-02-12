import numpy as np
from pyparsing import Callable
import sympy as sp
import itertools
import multiprocessing
import time
import warnings


# ==========================================
# Double Box System Definition
# ==========================================

def generate_double_box_system():

    OSE1, OSE2, OSE3, K = sp.symbols("P_1 P_2 P_3 k")
    vars_list = [OSE1, OSE2, OSE3, K]

    F_exprs = [
        OSE1 + OSE2 + OSE3 - (p_1[0] + p_2[0]),
        OSE1**2 - (sum((k_dir * K) ** 2) + m_1**2),
        OSE2**2 - (sum((l_dir * K) ** 2) + m_2**2),
        OSE3**2 - (sum((k_dir * K + l_dir * K + p_1[1:] + p_2[1:]) ** 2) + m_3**2)
    ]


    degrees = np.array([1, 2, 2, 2], dtype=int)
    return compile_system(F_exprs, vars_list, degrees)


def compile_system(F_exprs, vars_list, degrees):
    F_mat = sp.Matrix(F_exprs)
    J_mat = F_mat.jacobian(vars_list)

    f_func = sp.lambdify([vars_list], F_mat, modules="numpy", cse=True)
    j_func = sp.lambdify([vars_list], J_mat, modules="numpy", cse=True)

    def F(x):
        return np.array(f_func(x), dtype=np.complex128).ravel()

    def J(x):
        return np.array(j_func(x), dtype=np.complex128)

    return F, J, degrees, [str(v) for v in vars_list]


# ==========================================
# Globals (worker-local)
# ==========================================

GLOBAL_F: Callable = None # type: ignore
GLOBAL_J: Callable = None # type: ignore
GLOBAL_DEGREES : np.ndarray = None # type: ignore


def init_worker():
    global GLOBAL_F, GLOBAL_J, GLOBAL_DEGREES
    warnings.filterwarnings("ignore")

    GLOBAL_F, GLOBAL_J, GLOBAL_DEGREES, _ = generate_double_box_system()


def evaluate_F(x):
    return GLOBAL_F(x)


def evaluate_J_F(x):
    return GLOBAL_J(x)


def evaluate_G(x):
    return np.array(
        [x[i] ** GLOBAL_DEGREES[i] - 1 for i in range(len(x))],
        dtype=np.complex128,
    )


def evaluate_J_G(x):
    J = np.zeros((len(x), len(x)), dtype=np.complex128)
    for i, d in enumerate(GLOBAL_DEGREES):
        J[i, i] = 1 if d == 1 else d * x[i] ** (d - 1)
    return J


# ==========================================
# Path Tracker
# ==========================================


def track_path_worker(args):
    x_start, gamma, settings = args

    max_steps = settings["max_steps"]
    step_size = settings["initial_step"]
    min_step = settings["min_step"]
    tol = settings["tolerance"]
    divergence_limit = settings["divergence_limit"]
    use_secant = settings["predictor"] == "secant"

    x = np.array(x_start, dtype=np.complex128)
    t = 0.0

    x_prev = None
    t_prev = 0.0

    for _ in range(max_steps):
        if np.max(np.abs(x)) > divergence_limit:
            return None

        if t >= 1.0:
            break

        h = min(step_size, 1.0 - t)

        # Predictor
        if use_secant and x_prev is not None and abs(t - t_prev) > 1e-14:
            dx_dt = (x - x_prev) / (t - t_prev)
            x_pred = x + dx_dt * h
        else:
            try:
                dH_dt = -gamma * evaluate_G(x) + evaluate_F(x)
                JH = gamma * (1 - t) * evaluate_J_G(x) + t * evaluate_J_F(x)
                x_pred = x - np.linalg.solve(JH, dH_dt) * h
            except np.linalg.LinAlgError:
                step_size *= 0.5
                continue

        # Corrector
        t_next = t + h
        x_new = x_pred

        for _ in range(5):
            try:
                H = gamma * (1 - t_next) * evaluate_G(x_new) + t_next * evaluate_F(
                    x_new
                )
                if np.max(np.abs(H)) < tol:
                    break
                JH = gamma * (1 - t_next) * evaluate_J_G(x_new) + t_next * evaluate_J_F(
                    x_new
                )
                x_new -= np.linalg.solve(JH, H)
            except np.linalg.LinAlgError:
                step_size *= 0.5
                break
        else:
            step_size *= 0.5
            if step_size < min_step:
                return None
            continue

        x_prev, t_prev = x, t
        x, t = x_new, t_next
        step_size = min(step_size * 1.1, 0.1)

    # Final Newton refinement
    for _ in range(15):
        try:
            F = evaluate_F(x)
            if np.max(np.abs(F)) < 1e-11:
                return x
            x -= np.linalg.solve(evaluate_J_F(x), F)
        except np.linalg.LinAlgError:
            break

    return x if np.max(np.abs(evaluate_F(x))) < 1e-6 else None


# ==========================================
# Solver
# ==========================================


def solve_double_box():
    degrees = [2, 2, 2 ,2]
    var_names = ["P_1", "P_2", "P_3", "k"]

    settings = {
        "max_steps": 5000,
        "initial_step": 0.02,
        "min_step": 1e-8,
        "tolerance": 1e-9,
        "divergence_limit": 1e7,
        "predictor": "secant",
    }

    roots = [np.exp(2j * np.pi * np.arange(d) / d) for d in degrees]
    start_solutions = list(itertools.product(*roots))
    print(len(start_solutions))

    gamma = np.exp(1j * 1.2345)
    tasks = [(s, gamma, settings) for s in start_solutions]

    t0 = time.time()
    with multiprocessing.Pool(initializer=init_worker) as pool:
        results = pool.map(track_path_worker, tasks)

    sols = [r for r in results if r is not None]

    # Deduplicate
    unique = []
    for s in sols:
        if not any(np.linalg.norm(s - u) < 1e-3 for u in unique):
            unique.append(s)

    print(f"Done in {time.time() - t0:.2f}s. Found {len(unique)} solutions.")
    return unique, var_names


# ==========================================
# Run
# ==========================================

k_dir = np.array([1, 1, 1], dtype=np.float64)
l_dir = np.array([1, 1, 1], dtype=np.float64)

m_1 = 1.0  # GeV
m_2 = 1.0  # GeV
m_3 = 1.0  # GeV
p = 10  # GeV
p_1 = np.array([p, 1, 1, 1], dtype=np.float64)  # GeV
p_2 = np.array([p, 1, 1, 1], dtype=np.float64)  # GeV

norm = np.sum(k_dir) + np.sum(l_dir)
k_dir /= norm
l_dir /= norm


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
    multiprocessing.freeze_support()

    print_configuration_latex()

    sols, names = solve_double_box()
    print_all_solutions_latex(sols, names)
        