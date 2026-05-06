# -*- coding: utf-8 -*-
"""
Estimate cost of solving LWE using Gröbner bases (Arora–GB).
Pure Python version – no Sage required.
"""

import math
from itertools import accumulate
from scipy.special import erfc

from .cost import Cost
from .lwe_parameters import LWEParameters
from .io import Logging


# ---------- auxiliary functions ----------
def _binomial(n, k):
    """Return C(n,k) for n >= 0, return 0 for k<0 or k>n."""
    if k < 0 or k > n:
        return 0
    return math.comb(n, k)


def gb_cost(n, D, omega=2, prec=None):
    """
    Estimate the complexity of computing a Gröbner basis.

    :param n: Number of variables n > 0.
    :param D: Tuple of `(d,m)` pairs where `m` is number polynomials and `d` is a degree.
    :param omega: Linear algebra exponent, i.e. matrix-multiplication costs O(n^ω) operations.
    :param prec: Compute power series up to this precision (default: `2n`).

    EXAMPLE (from the original Sage code)::

        >>> from estimator.gb import gb_cost
        >>> gb_cost(128, [(2, 256)])
        rop: ≈2^144.6, dreg: 17, mem: ≈2^144.6

    """
    if prec is None:
        prec = 2 * n

    # Compute coefficients of (1 - z)^{-n} up to degree prec-1.
    # (1 - z)^{-n} = sum_{k >= 0} C(n + k - 1, k) z^k
    s = [_binomial(n + k - 1, k) for k in range(prec)]

    # Multiply by each (1 - z^d)^m
    for d, m in D:
        # (1 - z^d)^m = sum_{j=0}^{m} (-1)^j * C(m, j) * z^{d*j}
        poly_len = d * m + 1
        poly = [0] * poly_len
        for j in range(m + 1):
            coeff = _binomial(m, j)
            if j % 2 == 1:
                coeff = -coeff
            poly[d * j] = coeff

        # Convolution: s = s * poly, keep only first `prec` terms.
        new_s = [0] * prec
        for i in range(prec):
            if s[i] == 0:
                continue
            # j loop up to min(poly_len, prec - i)
            max_j = min(poly_len, prec - i)
            for j in range(max_j):
                if poly[j] != 0:
                    new_s[i + j] += s[i] * poly[j]
        s = new_s

    # Find the first index where the coefficient is negative (degree of regularity)
    retval = Cost(rop=float('inf'), dreg=float('inf'))
    retval.register_impermanent({"rop": True, "dreg": False, "mem": False})
    for dreg in range(prec):
        if s[dreg] < 0:
            retval["dreg"] = dreg
            rop = _binomial(n + dreg, dreg) ** omega
            retval["rop"] = rop
            retval["mem"] = _binomial(n + dreg, dreg) ** 2
            break
    return retval


class AroraGB:
    @staticmethod
    def ps_single(C):
        """
        Probability that a Gaussian (absolute value) is within C standard deviations.
        Uses erfc to compute tail probability.
        """
        # P(|X| >= C*sigma) = erfc(C / sqrt(2))
        return 1.0 - erfc(C / math.sqrt(2.0))

    @classmethod
    def cost_bounded(cls, params, success_probability=0.99, omega=2, log_level=1, **kwds):
        """
        Estimate cost using absolute bounds for secrets and noise.

        :param params: LWE parameters.
        :param success_probability: target success probability
        :param omega: linear algebra constant.
        """
        Xe = params.Xe
        # Number of possible error values
        d = Xe.bounds[1] - Xe.bounds[0] + 1
        dn = cls.equations_for_secret(params)

        m = min(params.m, params.n ** d)
        cost = gb_cost(params.n, [(d, m)] + dn, omega)
        cost["t"] = (d - 1) // 2
        if cost["dreg"] < float('inf') and _binomial(params.n + cost["dreg"], cost["dreg"]) < params.m:
            cost["m"] = _binomial(params.n + cost["dreg"], cost["dreg"])
        else:
            cost["m"] = m
        cost.register_impermanent(t=False, m=True)
        return cost

    @classmethod
    def cost_Gaussian_like(cls, params, success_probability=0.99, omega=2, log_level=1, **kwds):
        """
        Estimate cost using absolute bounds for secrets and Gaussian tail bounds for noise.

        :param params: LWE parameters.
        :param success_probability: target success probability
        :param omega: linear algebra constant.
        """
        dn = cls.equations_for_secret(params)

        best, stuck = None, 0
        Xe_stddev = params.Xe.stddev

        # Evaluate possible t (half the degree)
        for t in range(math.ceil(Xe_stddev), params.n):
            C = t / Xe_stddev
            # Probability that a single error sample is within t (absolute value)
            single_prob = AroraGB.ps_single(C)

            if single_prob <= 0 or single_prob >= 1:
                # avoid log overflow
                m_can = float('inf')
            else:
                # m such that single_prob^m >= overall success probability
                # m = log(success_probability) / log(single_prob)
                if single_prob == 1.0:
                    m_can = 2**31  # arbitrary large number
                else:
                    try:
                        m_can = math.floor(math.log(success_probability) / math.log(single_prob))
                    except (ValueError, ZeroDivisionError):
                        m_can = float('inf')

            if m_can > params.m:
                break

            d = 2 * t + 1
            current = gb_cost(params.n, [(d, int(m_can))] + dn, omega)

            if current["dreg"] == float('inf'):
                continue

            current["t"] = t
            current["m"] = int(m_can)
            current.register_impermanent(t=False, m=True)
            current = current.reorder("rop", "m", "dreg", "t")

            Logging.log("repeat", log_level + 1, repr(current))

            if best is None:
                best = current
            elif best > current:
                best = current
                stuck = 0
            else:
                stuck += 1
                if stuck >= 5:
                    break

        if best is None:
            return Cost(rop=float('inf'), dreg=float('inf'))
        return best

    @classmethod
    def equations_for_secret(cls, params):
        """
        Return ``(d,n)`` tuple to encode that `n` equations of degree `d` are available from the LWE secret.

        :param params: LWE parameters.
        """
        if params.Xs > params.Xe:
            return []

        a, b = params.Xs.bounds
        if (b - a) < float('inf'):
            d = b - a + 1
        elif getattr(params.Xs, 'is_Gaussian_like', False):
            d = 2 * math.ceil(3 * params.Xs.stddev) + 1
        else:
            raise NotImplementedError(f"Do not know how to handle {params.Xs}.")
        return [(int(d), int(params.n))]

    def __call__(
        self, params: LWEParameters, success_probability=0.99, omega=2, log_level=1, **kwds
    ):
        """
        Arora-GB as described in [ICALP:AroGe11]_, [EPRINT:ACFP14]_.

        :param params: LWE parameters.
        :param success_probability: targeted success probability < 1.
        :param omega: linear algebra constant.
        :return: A cost dictionary

        The returned cost dictionary has the following entries:

        - ``rop``: Total number of word operations (≈ CPU cycles).
        - ``m``: Number of samples consumed.
        - ``dreg``: The degree of regularity or "solving degree".
        - ``t``: Polynomials of degree 2t + 1 are considered.
        - ``mem``: Total memory usage.

        EXAMPLE (from original Sage code)::

            >>> from estimator import *
            >>> params = LWE.Parameters(n=64, q=7681, Xs=ND.DiscreteGaussian(3.0), Xe=ND.DiscreteGaussian(3.0), m=2**50)
            >>> LWE.arora_gb(params)
            rop: ≈2^307.1, m: ≈2^46.8, dreg: 99, t: 25, mem: ≈2^307.1, tag: arora-gb

        """
        params = params.normalize()

        best = Cost(rop=float('inf'), dreg=float('inf'))
        best.register_impermanent({"rop": True, "dreg": False, "mem": False})

        if params.Xe.is_bounded:
            cost = self.cost_bounded(
                params,
                success_probability=success_probability,
                omega=omega,
                log_level=log_level,
            )
            Logging.log("gb", log_level, f"b: {cost!r}")
            best = min(best, cost, key=lambda x: x["dreg"] if x["dreg"] is not None else float('inf'))

        if getattr(params.Xe, 'is_Gaussian_like', False):
            cost = self.cost_Gaussian_like(
                params,
                success_probability=success_probability,
                omega=omega,
                log_level=log_level,
            )
            Logging.log("gb", log_level, f"G: {cost!r}")
            best = min(best, cost, key=lambda x: x["dreg"] if x["dreg"] is not None else float('inf'))

        best["tag"] = "arora-gb"
        best["problem"] = params
        return best

    __name__ = "arora_gb"


arora_gb = AroraGB()
