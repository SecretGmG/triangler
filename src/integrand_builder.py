from symbolica_vectors import SymbolicaLorenzVec, SymbolicaVec
from symbolica import S, N, Expression
from wrapped_eval import THETA
import numpy as np


HALF = N(1) / 2
EPS = N(np.finfo(np.float64).eps)
I_EPS = Expression.I * EPS


def sqrt(e: Expression):
    """
    returns e**HALF
    """
    return e**HALF


def real(e: Expression):
    """
    returns the real part of e
    """
    return (e + e.conjugate()) * HALF


def imag(e: Expression):
    """
    returns the imaginary part of e
    """
    return (e - e.conjugate()) * HALF


def c_abs(e: Expression):
    """
    returns the absolute value of e
    """
    return sqrt(e * e.conjugate())


class IntegrandBuilder:
    """
    provides functions for building the integrand, the counterterm of the integrand and the radially integrated counterterm
    """

    check_singularities = False
    subtract_only_real_roots = False
    subtract_only_existing_surfaces = True

    qs: list[SymbolicaLorenzVec] = [
        SymbolicaLorenzVec.from_name(f"q{i}") for i in range(3)
    ]
    k: SymbolicaVec = SymbolicaVec.from_name("k")
    r = k.squared() ** HALF
    k_hat: SymbolicaVec = k.normalized()
    masses = [S(f"m{i}") for i in range(3)]
    thresh: Expression = S("lambda")

    a = S("a")
    b = S("b")
    c = S("c")

    # indices for the counterterm
    # this only works by assuming that q0_0 < q1_0 < q2_0
    eta_indices = [
        (0, 1),
        (1, 2),
        (0, 2),
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
        """
        returns all the arguments of the integrand as a list
        [pi, threshold, a, b, c, m0, m1, m2, [q0,4], [q1,4], [q2,4], [k,3]]
        where:
         - threshold is the radius of the threshold subtraction region
         - a, b, c are the factors of the integral, the counterterm and the radially integrated counterterm
         - m0, m1, m2 are the internal masses
         - q0, q1, q2 are the internal 4-momenta offsets
         - k is the argument of the integrand (3-momentum)
        """
        return (
            [Expression.PI, self.thresh, self.a, self.b, self.c]
            + self.masses
            + self.qs
            + [self.k]
        )

    def ose(self, i: int, k: SymbolicaVec):
        """
        returns  the onshell energy of particle i
        """
        temp: Expression = k - self.qs[i].spatial()
        return sqrt(temp.squared() + self.masses[i] ** N(2))

    def prefactor(self, k: SymbolicaVec):
        """
        returns the prefactor of the integrand
        """
        return (4 * Expression.PI) ** N(-3) / (
            self.ose(0, k) * self.ose(1, k) * self.ose(2, k)
        )

    def eta(self, i, j, k):
        """
        returns the e-surface (i,j) value at k
        """
        return self.ose(i, k) + self.ose(j, k) + self.qs[i].t() - self.qs[j].t()

    def k_p_min(self, i, j):
        return

    def unsubtracted(self):
        """
        returns the unsubtracted integrand at self.k
        """
        integrand = N(0)
        for i, j, k, l in self.part_indices:
            integrand += 1 / (self.eta(i, j, self.k) * self.eta(k, l, self.k))
        return integrand * self.prefactor(self.k)

    def eta_radius_roots(self, i, j):
        """
        returns the location of the roots of the e-surface (i,j) when parameterized hemispherically with the direction vector self.k_hat
        """
        q: SymbolicaLorenzVec = (self.qs[i] - self.qs[j]) * HALF
        v: SymbolicaVec = q.spatial() * (N(1) / q.t())
        k_0_p = -(self.qs[i] + self.qs[j]).spatial() * HALF

        k_hat_v = self.k_hat * v
        delta = (self.masses[i] ** N(2) - self.masses[j] ** N(2)) / (N(4) * q.t())
        m_2_avg = (self.masses[i] ** N(2) + self.masses[j] ** N(2)) / N(2)

        a = N(1) - k_hat_v ** N(2)

        b = N(2) * ((self.k_hat * k_0_p) - k_hat_v * (k_0_p * v) + delta * k_hat_v)

        c = (
            k_0_p.squared()
            - (k_0_p * v) ** N(2)
            + 2 * delta * (k_0_p * v)
            - q.squared()
            + m_2_avg
            - delta**2
        )

        d = b ** N(2) - N(4) * a * c

        return [
            (-b + sqrt(d)) / (N(2) * a)
            + I_EPS,  # necessary for the log to select the correct branch
            (-b - sqrt(d)) / (N(2) * a)
            - I_EPS,  # necessary for the log to select the correct branch
        ]

    def ddk_eta(self, i, j, k):
        """
        returns the partial derivative of the e-surface (i,j) with respect to k along the direction k_hat
        """
        d1 = self.k_hat * (k - self.qs[i].spatial()) / self.ose(i, k)
        d2 = self.k_hat * (k - self.qs[j].spatial()) / self.ose(j, k)
        return d1 + d2

    def eta_min(self, i, j):
        q: SymbolicaLorenzVec = (self.qs[i] - self.qs[j]) * HALF
        q_min = (
            (self.masses[j] - self.masses[i])
            / (self.masses[i] + self.masses[j])
            * q.spatial()
        )
        return (
            sqrt((q_min-q.spatial()).squared() + self.masses[i] ** N(2))
            + sqrt((q_min+q.spatial()).squared() + self.masses[j] ** N(2))
            + N(2) * q.t()
        )

    def eta_ct(self, i, j) -> list[(Expression, SymbolicaVec)]:
        """
        returns the counterterm for the e-surface (i,j) when parameterized around the origin
        """
        poles = self.eta_radius_roots(i, j)

        out = []

        for r_star in poles:
            k_star = r_star * self.k_hat

            selector = N(1)
            if self.check_singularities:
                selector *= THETA(N(1e-10) - c_abs(self.eta(i, j, k_star)))

            if self.subtract_only_real_roots:
                selector *= THETA(N(1e-10) - c_abs(imag(r_star)))

            if self.subtract_only_existing_surfaces:
                selector *= THETA(-self.eta_min(i, j))

            factor = (
                self.collect_other_etas(i, j, k_star)
                * selector
                * self.prefactor(k_star)
                / self.ddk_eta(i, j, k_star)
            )
            out.append((factor, r_star))

        return out

    def collect_other_indices(self, i, j):
        """
        returns the indices of the e-surfaces that appear as factors of the e-surface (i,j)
        """
        order = j - i
        return [(i, (i - order) % 3), ((j + order) % 3, j)]

    def collect_other_etas(self, i, j, k):
        """
        collects the factors of the e-surface (i,j)
        """
        other_etas = N(0)
        for i, j in self.collect_other_indices(i, j):
            other_etas += 1 / (self.eta(i, j, k))
        return other_etas

    def counterterm(self):
        """
        returns the counterterm of the integrand at self.k
        """
        ct = N(0)
        for i, j in self.eta_indices:
            values = self.eta_ct(i, j)
            for factor, r_star in values:
                selector = THETA(self.thresh - self.r)
                ct += (
                    selector
                    * factor
                    * (r_star / self.r) ** N(2)
                    / (self.r - r_star)
                )
        return ct

    def integrated_counterterm(self):
        """
        returns the radially integrated counterterm along the direction self.k_hat
        """
        ct = N(0)
        for i, j in self.eta_indices:
            values = self.eta_ct(i, j)
            for factor, r_star in values:

                upper = self.thresh
                lower = -self.thresh

                integrated = Expression.LOG((upper - r_star) / (lower - r_star))

                ct += factor * real(r_star) ** N(2) * integrated
        return ct

    def combined_result(self):
        """
        returns the combined result
            a * unsubtracted() - b * ct() + c * integrated_counterterm() * factor
            where factor = THETA(thresh - r) / (2/3 * thresh**3)

        integrating this over R^3 gives the result of the integral
        """
        integrated_ct_factor = THETA(self.thresh - self.r) / (
            N(2) / N(3) * (self.thresh ** N(3))
        )
        return (
            self.unsubtracted() * self.a
            - self.counterterm() * self.b
            + self.integrated_counterterm() * self.c * integrated_ct_factor
        )
