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
    return e**HALF

def real(e: Expression):
    return (e + e.conjugate())*HALF

def imag(e: Expression):
    return (e - e.conjugate())*HALF

def c_abs(e: Expression):
    return IntegrandBuilder.sqrt(e*e.conjugate())
        

class IntegrandBuilder:
    qs: list[SymbolicaLorenzVec] = [SymbolicaLorenzVec.from_name(f"q{i}") for i in range(3)]
    k: SymbolicaVec = SymbolicaVec.from_name("k")
    r = k.squared() ** HALF
    k_hat: SymbolicaVec = k.norm()
    masses = [S(f'm{i}') for i in range(3)]
    thresh: Expression = S("lambda")
    
    a = S('a')
    b = S('b')
    c = S('c')

    eta_indices = [
        (0, 1),
        (1, 2),
        (0, 2),
        #(0, 1),
        #(2, 1),
        #(2, 0),
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
        return [Expression.PI,self.thresh, self.a, self.b, self.c] + self.masses + self.qs + [self.k]

    def ose(self, i: int, k: SymbolicaVec):
        temp : Expression = k - self.qs[i].spacial()
        return sqrt(temp.squared() + self.masses[i]**N(2))

    def prefactor(self, k: SymbolicaVec):
        return (4 * Expression.PI) ** N(-3) / (
            self.ose(0, k) * self.ose(1, k) * self.ose(2, k)
        )
    
    def eta(self, i, j, k):
        return self.ose(i, k) + self.ose(j, k) + self.qs[i].t() - self.qs[j].t()

    def unsubtracted(self):
        integrand = N(0)
        for i, j, k, l in self.part_indices:
            integrand += 1 / (self.eta(i, j, self.k) * self.eta(k, l, self.k))
        return integrand * self.prefactor(self.k)

    def eta_radius_roots(self, i, j, center):
        q: SymbolicaLorenzVec = (self.qs[i] - self.qs[j]) * HALF
        v: SymbolicaVec = q.spacial() * (N(1) / q.t())
        q_c: SymbolicaLorenzVec = (self.qs[i] + self.qs[j]) * HALF
        k_0_p = center-q_c.spacial()
        
        k_hat = (self.k-center).norm()

        a = N(1) - (k_hat * v) ** N(2)
        b = N(2) * (k_hat * k_0_p) - N(2) * (k_hat * v) * (k_0_p * v)
        
        m = self.masses[0] # TEMPORARY THIS NEEDS TO BE SWAPPED TO THE IMPROVED FORMULA
        c = k_0_p.squared() - (k_0_p * v) ** N(2) - q.squared() + m ** N(2)
        
        d = b ** N(2) - N(4) * a * c
        
        
        return [
            (-b + sqrt(d + I_EPS)) / (N(2) * a),
            (-b - sqrt(d + I_EPS)) / (N(2) * a)
        ], k_hat

    def ddk_eta(self, i, j, k_hat, k):
        d1 = k_hat * (k - self.qs[i].spacial()) / self.ose(i, k)
        d2 = k_hat * (k - self.qs[j].spacial()) / self.ose(j, k)
        return d1 + d2

    def eta_ct(self, i, j) -> list[(Expression, SymbolicaVec)]:
        q: SymbolicaLorenzVec = (self.qs[i] - self.qs[j]) * HALF
        #center: SymbolicaLorenzVec = (self.qs[i] + self.qs[j]).spacial() * HALF
        center = SymbolicaVec.zero()
        
        
        selector = N(1)
        # this selector could be optimized away by asserting e.g. qs[0].t() < qs[1].t() < qs[2].t()
        #selector *= THETA(-q.t())
        
        # this selector is not necessary, the subtraction will still work for 'non-existent' threshold singularities
        #selector *= THETA(q.squared()-self.m**N(2))
        
        poles, k_hat = self.eta_radius_roots(i, j, center)

        out = []

        for r_star in poles:
            k_star = r_star * k_hat + center
            factor = (
                self.collect_other_etas(i, j, k_star)
                * selector
                * self.prefactor(k_star)
                / self.ddk_eta(i, j, k_hat, k_star)
            )
            out.append((factor, r_star))

        return out, center

    def collect_other_indices(self, i, j):
        order = j - i
        return [(i, (i - order) % 3), ((j + order) % 3, j)]

    def collect_other_etas(self, i, j, k):
        other_etas = N(0)
        for i, j in self.collect_other_indices(i, j):
            other_etas += 1 / (self.eta(i, j, k))
        return other_etas

    def ct(self):
        ct = N(0)
        for i, j in self.eta_indices:
            values, center = self.eta_ct(i, j)
            r = sqrt((self.k-center).squared())
            for factor, r_star in values:
                #offset = c_abs(real(r_star))*HALF
                #selector = THETA(r_star + offset - r)*THETA(r - (offset - r_star))
                selector = THETA(self.thresh - self.r)
                ct += selector * factor * (real(r_star) / r) ** N(2) / (r - r_star)
        return ct

    def integrated_counterterm(self):        
        ct = N(0)
        for i, j in self.eta_indices:
            values, _ = self.eta_ct(i, j)
            for factor, r_star in values:
                #offset = c_abs(self.real(r_star))*HALF
                #upper = real(r_star) + offset
                #lower = real(r_star) - offset
                upper = self.thresh
                lower = -self.thresh
                
                corrected_log = Expression.LOG((upper-r_star)/(lower-r_star))
                
                ct += (
                    factor
                    * real(r_star) ** N(2)
                    * corrected_log
                )
        return ct
    
    def combined_result(self):
        integrated_ct_factor = THETA(self.thresh - self.r) / (N(2)/N(3)*self.thresh**N(3))
        return self.unsubtracted()*self.a - self.ct()*self.b + self.integrated_counterterm()*self.c*integrated_ct_factor
        

import numpy as np

def norm(v):
    return v[0] ** 2 - (v[1:] ** 2).sum()

def hemispherical(xs):
    """
    Uniformly sample points on the unit hemisphere (z >= 0).
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
    Sample points in R^3 with radial transform r = u/(1-u) (u in [0,1) -> r in [0,inf)).
    xs: (N,3) array with values in [0,1)
    Returns:
        v: (N,3) Cartesian coordinates
        jac: (N,) Jacobian for Monte Carlo integration (dV / d(u1,u2,u3))
    """
    xs = np.asarray(xs)
    w = 2*xs[:, 2]-1 # (-1,1)
    r = w-1/w    
    r_jac = (1+1/w**2)*2

    v, h_jac = hemispherical(xs[:,0:2])

    jac = r_jac * r**2 * h_jac
    return r[:,None]*v, jac


def line_segment(k_hat, thresh):
    k_hat = np.asarray(k_hat)

    def temp(xs):
        return ((xs[:, 0] * 2 - 1) * thresh)[:, None] * k_hat[None, :], (2 * thresh)

    return temp


class ContextManager:

    origin = np.array([0, 0, 0, 0])
    
    p1 = np.array([2, 1, 0, 0])
    p2 = np.array([2, 0, 1, 0])
    masses = [0.5,0.5,0.5]
    threshold = 5
    a = 1
    b = 1
    c = 1
    

    def __init__(self, force_rebuild=True):
        ib = IntegrandBuilder()
        self.ib = ib
        
        self.evaluator = WrappedEvaluator(ib.combined_result(), ib.get_args(), 'combined_result', force_rebuild)

    def get_threshold_report(self):
        qs = self.get_qs()
        
        def ose_at_origin(i):
            return np.sqrt(np.sum(qs[i][1:]**2) + self.masses[i]**2)
        
        for (i,j) in self.ib.eta_indices:
            q = 0.5*(qs[i]-qs[j])
            q2 = norm(q)
            
            print(f'q = {q}')
            print(f'q² = {q2}')
            
            eta_at_origin = ose_at_origin(i) + ose_at_origin(j) + qs[i][0] - qs[j][0]
            
            print(f'E-surface value at origin = {eta_at_origin}')
            
    def get_qs(self):
        qs = [self.origin + np.zeros_like(self.p1), self.origin - self.p1, self.origin + self.p2]
        return sorted(qs, key = lambda q: q[0])
        
    
    def get_context_args(self):
        return [np.pi,self.threshold, self.a, self.b, self.c] + self.masses + self.get_qs()

    def eval(self, k):
        return self.evaluator.evaluate(self.get_context_args() + [k])
    
    def get_reference(self) -> complex:
        from oneloop_bridge import three_point, TO_FEYNMAN


        
        res = three_point(
                norm(self.p1),
                norm(self.p2),
                norm(self.p1 + self.p2),
                self.masses[0]**2,
                self.masses[1]**2,
                self.masses[2]**2,
            )
        assert res.epsilon_minus_1 == 0
        assert res.epsilon_minus_2 == 0

        return (
            res.epsilon_0
            * TO_FEYNMAN
        )

    def plot_threshold_subtraction(
        self, x_lim, y_lim, x_axis=None, y_axis=None, res=300
    ):
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

        self.a = 1
        self.b = 0
        self.c = 0
        plt.figure(figsize=(20, 10))
        plt.subplot(2, 3, 1)
        integrand = (self.eval(ks_plane) * ks_plane_jac).reshape(res, res)
        plot_complex_plane(xs_plane, integrand)
        plt.subplot(2, 3, 4)
        plot_complex(x, self.eval(ks_line) * ks_line_jac)
        
        self.a = 0
        self.b = 1
        self.c = 0
        plt.subplot(2, 3, 2)
        counter_term = (self.eval(ks_plane) * ks_plane_jac).reshape(res, res)
        plot_complex_plane(xs_plane, counter_term)
        plt.subplot(2, 3, 5)
        plot_complex(x, self.eval(ks_line) * ks_line_jac)
        
        self.a = 1
        self.b = 1
        self.c = 0
        plt.subplot(2, 3, 3)
        subtracted = (self.eval(ks_plane) * ks_plane_jac).reshape(res, res)
        plot_complex_plane(xs_plane, subtracted)
        plt.subplot(2, 3, 6)
        plot_complex(x, self.eval(ks_line) * ks_line_jac)
    
    def plot_planes(self,x_lim, y_lim, res = 300):
        plt.figure(figsize=(20, 7))
        for i,(x_,y_) in enumerate([(0,1),(1,2),(2,0)]):
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

            plt.subplot(1, 3, i+1)
            integrand = (self.eval(ks_plane) * ks_plane_jac).reshape(res, res)
            plot_complex_plane(xs_plane, integrand)
            plt.xlabel(['x', 'y', 'z'][x_])
            plt.ylabel(['x', 'y', 'z'][y_])
    
    def plot_integrated_counterterm(self, res = 300):
        u = np.linspace(0,1, res)
        v = np.linspace(0,1, res)
        
        U, V = np.meshgrid(u, v)
        xs = np.stack([U,V], axis=-1).reshape(-1,2)
        ks, _ = hemispherical(xs)
        self.a = 0
        self.b = 0
        self.c = 1
        plot_complex_plane(U + V*1j, self.eval(ks).reshape(res,res))
        plt.xlabel(r'$\theta/2\pi$')
        plt.ylabel(r'$\cos(\phi)$')