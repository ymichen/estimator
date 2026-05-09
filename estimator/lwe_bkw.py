# -*- coding: utf-8 -*-
"""
Coded-BKW for LWE (pure Python version, no Sage required).
"""

import math
from functools import lru_cache
from scipy.optimize import newton
from scipy.special import erf

from .lwe_parameters import LWEParameters
from .util import local_minimum
from .cost import Cost
from .errors import InsufficientSamplesError
from .prob import amplify_sigma
from .nd import sigmaf
from .io import Logging

cfft = 1  # convolutions mod q


class CodedBKW:
    @staticmethod
    def N(i, sigma_set, b, q):
        """
        Return N_i for the i-th [N_i, b] linear code.
        """
        # Use floating point throughout
        sigma_set = float(sigma_set)
        b = float(b)
        i = float(i)
        q = float(q)
        try:
            denom = 1.0 - 0.5 * math.log(12.0 * sigma_set**2 / (2**i)) / math.log(q)
        except (ValueError, ZeroDivisionError):
            return 0.0
        if denom <= 0:
            return 0.0
        return math.floor(b / denom)

    @staticmethod
    @lru_cache(maxsize=None)
    def ntest(n, ell, t1, t2, b, q):
        """
        Estimate ntest if not provided.
        """
        if t1 * b >= n:
            return 0

        def ntop(ntest):
            ntest = float(ntest)
            if ntest <= 0:
                return float('inf')
            try:
                sigma_set = math.sqrt(q**(2 * (1 - ell / ntest)) / 12.0)
            except (ValueError, OverflowError):
                return float('inf')
            ncod = sum(CodedBKW.N(i, sigma_set, b, q) for i in range(1, t2 + 1))
            return n - ncod - ntest - t1 * b

        # find a start point near the root
        try:
            # newton requires a function returning scalar
            start = newton(lambda x: ntop(x), n - t1 * b, maxiter=200, tol=1e-6)
            start = int(max(2, round(start) - 1))
        except (RuntimeError, ValueError):
            start = 2
        ntest_min = 1
        min_val = abs(ntop(ntest_min))
        for ntest in range(start, max(start, int(n - t1 * b) + 1)):
            val = abs(ntop(ntest))
            if val < min_val:
                min_val = val
                ntest_min = ntest
            else:
                break
        return ntest_min

    @staticmethod
    def t1(params, ell, t2, b, ntest=None):
        n = params.n
        if ntest is None:
            ntest = CodedBKW.ntest(n, ell, 0, t2, b, params.q)  # t1=0 to compute ntest
        sigma_set = math.sqrt(params.q**(2 * (1 - ell / ntest)) / 12.0)
        Ni = [CodedBKW.N(i, sigma_set, b, params.q) for i in range(1, t2 + 1)]
        t1 = sum(1 for e in Ni if e <= b)
        if b * t1 > n:
            t1 = n // b
        return t1

    @staticmethod
    def cost(t2, b, ntest, params, success_probability=0.99, log_level=1):
        cost = Cost()
        cost["b"] = b
        ell = b - 1
        cost["ell"] = ell
        q = params.q
        n = params.n

        secret_bounds = params.Xs.bounds
        if hasattr(params.Xs, 'is_Gaussian_like') and params.Xs.is_Gaussian_like and params.Xs.mean == 0:
            secret_bounds = (max(secret_bounds[0], -3 * params.Xs.stddev),
                             min(secret_bounds[1], 3 * params.Xs.stddev))
        zeta = secret_bounds[1] - secret_bounds[0] + 1

        t1 = CodedBKW.t1(params, ell, t2, b, ntest)
        t2_eff = t2 - t1
        cost["t1"] = t1
        cost["t2"] = t2_eff
        cost.register_impermanent(t1=False, t2=False)

        if ntest is None:
            ntest = CodedBKW.ntest(n, ell, t1, t2_eff, b, q)
        ntest = int(ntest)

        if ntest > 0:
            sigma_set = math.sqrt(q**(2 * (1 - ell / ntest)) / 12.0)
            ncod = sum(CodedBKW.N(i, sigma_set, b, q) for i in range(1, t2_eff + 1))
        else:
            sigma_set = 0.0
            ncod = 0

        ntot = ncod + ntest
        ntop = max(n - ncod - ntest - t1 * b, 0)
        cost["#cod"] = ncod
        cost["#top"] = ntop
        cost["#test"] = ntest
        cost.register_impermanent({"#cod": False, "#top": False, "#test": False})

        # Noise variance
        coding_variance = (params.Xs.stddev**2) * (sigma_set**2) * (ntot)
        sigma_final = float(math.sqrt(
            2**(t1 + t2_eff) * params.Xe.stddev**2 + coding_variance
        ))

        M = amplify_sigma(success_probability, sigmaf(sigma_final), q)
        if M == float('inf'):
            cost["rop"] = float('inf')
            cost["m"] = float('inf')
            return cost

        m = (t1 + t2_eff) * (q**b - 1) / 2.0 + M
        cost["m"] = float(m)
        cost.register_impermanent(m=True)

        # Preprocessing C0
        if not params.Xs <= params.Xe:
            C0 = (m - n) * (n + 1) * math.ceil(n / (b - 1))
            assert C0 >= 0
        else:
            C0 = 0.0

        # Plain BKW cost C1
        C1 = sum(
            (n + 1 - j * b) * (m - j * (q**b - 1) / 2.0)
            for j in range(1, t1 + 1)
        )

        # Coded BKW cost C2
        C2 = 0.0
        for i in range(1, t2_eff + 1):
            Ni = CodedBKW.N(i, sigma_set, b, q)
            C2 += 4 * (M + i * (q**b - 1) / 2.0) * Ni
            C2 += (ntop + ntest + sum(CodedBKW.N(j, sigma_set, b, q) for j in range(1, i + 1))) * \
                  (M + (i - 1) * (q**b - 1) / 2.0)

        # Guessing C3
        C3 = M * ntop * ((2 * zeta + 1) ** ntop) if ntop > 0 else 0.0

        # Hypothesis test / FFT C4
        C4 = 4 * M * ntest
        if ntop > 0:
            C4 += ((2 * zeta + 1) ** ntop) * (
                cfft * (q**(ell + 1)) * (ell + 1) * math.log(q, 2) + q**(ell + 1)
            )

        total = (C0 + C1 + C2 + C3 + C4) / (erf(zeta / (math.sqrt(2) * params.Xe.stddev)) ** ntop)
        cost["rop"] = total
        cost["mem"] = (t1 + t2_eff) * q**b

        cost = cost.reorder("rop", "m", "mem", "b", "t1", "t2")
        cost["tag"] = "coded-bkw"
        cost["problem"] = params
        Logging.log("bkw", log_level + 1, f"{cost!r}")
        return cost

    @classmethod
    def b(cls, params, ntest=None, log_level=1):
        def sf(x, best):
            return (x["rop"] <= best["rop"]) and not (best["m"] <= params.m < x["m"])

        b_max = 3 * math.ceil(math.log(params.q, 2))
        with local_minimum(2, b_max, smallerf=sf) as it_b:
            for b in it_b:
                t2_max = max(3, params.n // b)
                with local_minimum(2, t2_max, smallerf=sf) as it_t2:
                    for t2 in it_t2:
                        y = cls.cost(t2=t2, b=b, ntest=ntest, params=params)
                        it_t2.update(y)
                    it_b.update(it_t2.y)
            best = it_b.y

        if best["m"] > params.m:
            raise InsufficientSamplesError(
                f"Got m≈2^{float(math.log2(params.m)):.1f} samples, "
                f"but require ≈2^{float(math.log2(best['m'])):.1f}.",
                best["m"],
            )
        return best

    def __call__(self, params, ntest=None, log_level=1):
        params = LWEParameters.normalize(params)
        params_ = params
        while True:
            try:
                return self.b(params_, ntest=ntest, log_level=log_level)
            except InsufficientSamplesError as e:
                m = e.args[1]
                params_ = params.amplify_m(m)


coded_bkw = CodedBKW()
