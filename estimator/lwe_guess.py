# -*- coding: utf-8 -*-
"""
Generic multiplicative composition of guessing some components of the LWE secret
and some LWE solving algorithm. Pure Python, no Sage required.
"""

import math
from functools import lru_cache

from .conf import mitm_opt
from .cost import Cost
from .errors import InsufficientSamplesError, OutOfBoundsError
from .lwe_parameters import LWEParameters
from .prob import amplify as prob_amplify, drop as prob_drop, amplify_sigma
from .util import local_minimum, log2
from .nd import sigmaf, SparseTernary

OO = float('inf')


class guess_composition:
    def __init__(self, f):
        self.f = f
        self.__name__ = f"{f.__name__}+guessing"

    @classmethod
    def dense_solve(cls, f, params, log_level=5, **kwds):
        base = params.Xs.bounds[1] - params.Xs.bounds[0] + 1
        baseline_cost = f(params, **kwds)
        if baseline_cost["rop"] == OO:
            return baseline_cost
        max_zeta = min(math.floor(math.log(baseline_cost["rop"], base)), params.n)
        with local_minimum(0, max_zeta, log_level=log_level) as it:
            for zeta in it:
                search_space = base ** zeta
                cost = f(params.updated(n=params.n - zeta), log_level=log_level + 1, **kwds)
                if cost["rop"] == OO:
                    return Cost(rop=OO)
                repeated_cost = cost.repeat(search_space)
                repeated_cost["zeta"] = zeta
                it.update(repeated_cost)
            return it.y if it.y else Cost(rop=OO)

    @classmethod
    def gammaf(cls, n, h, zeta, base, g=lambda x: x):
        if h == 0 or not zeta:
            return 1, 0, 0, 1.0
        search_space = 0
        gamma = 0
        probability = 0
        best = (None, None, None, None)
        while gamma < min(h, zeta):
            probability += prob_drop(n, h, zeta, fail=gamma)
            search_space += math.comb(zeta, gamma) * base ** gamma
            repeat = prob_amplify(0.99, probability) * g(search_space)
            if best[0] is not None and repeat >= best[0]:
                break
            best = (repeat, gamma, search_space, probability)
            gamma += 1
        return best

    @classmethod
    def sparse_solve(cls, f, params, log_level=5, **kwds):
        base = params.Xs.bounds[1] - params.Xs.bounds[0]  # exclude zero
        h = params.Xs.hamming_weight
        with local_minimum(0, params.n - 40, log_level=log_level) as it:
            for zeta in it:
                single_cost = f(params.updated(n=params.n - zeta), log_level=log_level + 1, **kwds)
                if single_cost["rop"] == OO:
                    return Cost(rop=OO)
                repeat, gamma, search_space, probability = cls.gammaf(params.n, h, zeta, base)
                cost = single_cost.repeat(repeat)
                cost["zeta"] = zeta
                cost["|S|"] = search_space
                cost["prop"] = probability
                it.update(cost)
            return it.y

    def __call__(self, params, log_level=5, **kwds):
        params = LWEParameters.normalize(params)
        solve = self.sparse_solve if params.Xs.is_sparse else self.dense_solve
        return solve(self.f, params, log_level, **kwds)


class ExhaustiveSearch:
    def __call__(self, params: LWEParameters, success_probability=0.99, quantum: bool = False):
        params = LWEParameters.normalize(params)
        probability = math.sqrt(success_probability)
        try:
            size = params.Xs.support_size(probability)
        except NotImplementedError:
            return Cost(rop=OO, mem=OO, m=1)

        if quantum:
            size = math.isqrt(int(size))  # sqrt of integer approximation

        sigma = params.Xe.stddev / params.q
        m_required = 8 * math.exp(4 * math.pi * math.pi * sigma * sigma) * \
                     (math.log(size) - math.log(math.log(1 / probability)))
        if params.m < m_required:
            raise InsufficientSamplesError(
                f"Exhaustive search: Need {m_required} samples but only {params.m} available."
            )
        cost = 2 * size * m_required
        ret = Cost(rop=cost, mem=cost / 2, m=m_required)
        return ret.sanity_check()

    __name__ = "exhaustive_search"


exhaustive_search = ExhaustiveSearch()


class MITM:
    locality = 0.05

    def X_range(self, nd):
        if nd.is_bounded:
            a, b = nd.bounds
            return b - a + 1, 1.0
        else:
            rng = nd.resize(1).support_size(0.0)
            return rng, nd.gaussian_tail_prob

    def local_range(self, center):
        return math.floor((1 - self.locality) * center), math.ceil((1 + self.locality) * center)

    def mitm_analytical(self, params: LWEParameters, success_probability=0.99):
        nd_rng, nd_p = self.X_range(params.Xe)
        delta = nd_rng / params.q
        sd_rng, sd_p = self.X_range(params.Xs)
        n = params.n
        k = round(n / (2 - delta))
        if params.Xs.is_sparse:
            h = params.Xs.hamming_weight
            if isinstance(params.Xs, SparseTernary):
                success_probability_ = params.Xs.split_probability(k)
            else:
                split_h = round(h * k / n)
                success_probability_ = math.comb(k, split_h) * math.comb(n - k, h - split_h) / math.comb(n, h)
            logT = h * (log2(n) - log2(h) + log2(sd_rng - 1) + log2(math.e)) / (2 - delta)
            logT -= log2(h) / 2
            logT -= h * h * log2(math.e) / (2 * n * (2 - delta) ** 2)
        else:
            success_probability_ = 1.0
            logT = k * math.log2(sd_rng)
        m_required = max(1, round(logT + math.log2(logT)))
        if params.m < m_required:
            raise InsufficientSamplesError(
                f"MITM: Need {m_required} samples but only {params.m} available."
            )
        ret = Cost(rop=2**m_required, mem=2**logT * m_required, m=m_required, k=k)
        repeat = prob_amplify(success_probability, sd_p ** n * nd_p ** m_required * success_probability_)
        return ret.repeat(times=repeat)

    def cost(self, params: LWEParameters, k: int, success_probability=0.99):
        nd_rng, nd_p = self.X_range(params.Xe)
        delta = nd_rng / params.q
        sd_rng, sd_p = self.X_range(params.Xs)
        n = params.n
        if params.Xs.is_sparse:
            h = params.Xs.hamming_weight
            if isinstance(params.Xs, SparseTernary):
                sec_tab, sec_sea = params.Xs.split_balanced(k)
                size_tab = sec_tab.support_size()
                size_sea = sec_sea.support_size()
            else:
                split_h = round(h * k / n)
                size_tab = (sd_rng - 1) ** split_h * math.comb(k, split_h)
                size_sea = (sd_rng - 1) ** (h - split_h) * math.comb(n - k, h - split_h)
            success_probability_ = size_tab * size_sea / params.Xs.support_size()
        else:
            size_tab = sd_rng ** k
            size_sea = sd_rng ** (n - k)
            success_probability_ = 1

        m_ = max(math.ceil(math.log2(size_tab) + math.log2(math.log2(size_tab))), 1)
        a, b = self.local_range(m_)
        with local_minimum(a, b, smallerf=lambda x, best: x[1] <= best[1]) as it:
            for m in it:
                cost = (m, size_sea * (2 * m + 2 ** (delta * m) * (1 + size_tab * m / 2**m)))
                it.update(cost)
            m, cost_search = it.y
        m = min(m, params.m)
        cost_table = size_tab * 2 * m
        ret = Cost(rop=cost_table + cost_search, m=m, k=k)
        ret["mem"] = size_tab * (k + m) + size_sea * (n - k + m)
        repeat = prob_amplify(success_probability, sd_p ** n * nd_p ** m * success_probability_)
        return ret.repeat(times=repeat)

    def __call__(self, params: LWEParameters, success_probability=0.99, optimization=mitm_opt):
        Cost.register_impermanent(rop=True, mem=False, m=True, k=False)
        params = LWEParameters.normalize(params)
        nd_rng = self.X_range(params.Xe)[0]
        if nd_rng >= params.q:
            return Cost(rop=OO, mem=OO, m=0, k=0)
        if "analytical" in optimization:
            return self.mitm_analytical(params=params, success_probability=success_probability)
        elif "numerical" in optimization:
            with local_minimum(1, params.n - 1) as it:
                for k in it:
                    cost = self.cost(k=k, params=params, success_probability=success_probability)
                    it.update(cost)
                ret = it.y
                ret1 = self.cost(k=1, params=params, success_probability=success_probability)
                return min(ret, ret1)
        else:
            raise ValueError("Unknown optimization method for MITM.")

    __name__ = "mitm"


mitm = MITM()


class Distinguisher:
    def __call__(self, params: LWEParameters, success_probability=0.99):
        if params.n > 0:
            raise OutOfBoundsError(
                "Secret dimension should be 0 for distinguishing. Try exhaustive search for n > 0."
            )
        m = amplify_sigma(success_probability, sigmaf(params.Xe.stddev), params.q)
        if m > params.m:
            raise InsufficientSamplesError("Not enough samples to distinguish with target advantage.")
        return Cost(rop=m, mem=m, m=m).sanity_check()

    __name__ = "distinguish"


distinguish = Distinguisher()
