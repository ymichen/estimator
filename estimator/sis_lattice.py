# -*- coding: utf-8 -*-
"""
Estimate cost of solving SIS using lattice reduction attacks. Pure Python, no Sage.
"""

from functools import partial, lru_cache
import math

from .reduction import beta as betaf
from .reduction import cost as costf
from .util import local_minimum
from .cost import Cost
from .sis_parameters import SISParameters
from .simulator import normalize as simulator_normalize
from .prob import gaussian_cdf
from .prob import amplify as prob_amplify
from .io import Logging
from .conf import red_cost_model as red_cost_model_default
from .conf import red_shape_model as red_shape_model_default
from .conf import red_simulator as red_simulator_default

OO = float('inf')


class SISLattice:
    @staticmethod
    def _solve_for_delta_euclidean(params, d):
        root_volume = (params.n / d) * math.log2(params.q)
        log_delta = (1 / (d - 1)) * (math.log2(params.length_bound) - root_volume)
        return 2 ** log_delta

    @staticmethod
    def _opt_sis_d(params):
        log_delta = math.log2(params.length_bound) ** 2 / (4 * params.n * math.log2(params.q))
        d = math.sqrt(params.n * math.log2(params.q) / log_delta)
        return d

    @staticmethod
    @lru_cache(maxsize=None)
    def cost_euclidean(params, d=None, red_cost_model=red_cost_model_default, log_level=None, **kwds):
        if params.length_bound >= (params.q - 1) / 2:
            raise ValueError("SIS trivially easy. Please set norm bound < (q-1)/2.")
        if d is None:
            d = min(math.floor(SISLattice._opt_sis_d(params)), params.m)
        delta = SISLattice._solve_for_delta_euclidean(params, d)
        if delta >= 1 and betaf(delta) <= d:
            beta = betaf(delta)
            reduction_possible = True
        else:
            beta = d
            reduction_possible = False
        lb = min(math.sqrt(params.n * math.log2(params.q)), math.sqrt(d) * params.q ** (params.n / d))
        return costf(red_cost_model, beta, d, predicate=params.length_bound > lb and reduction_possible)

    @staticmethod
    @lru_cache(maxsize=None)
    def cost_infinity(beta, params, simulator, zeta=0, success_probability=0.99, d=None,
                      red_cost_model=red_cost_model_default, log_level=None, **kwds):
        if params.length_bound >= (params.q - 1) / 2:
            raise ValueError("SIS trivially easy. Please set norm bound < (q-1)/2.")
        if d is None:
            d = params.m
        d_ = d - zeta
        if d_ < beta:
            return Cost(rop=OO, mem=OO)
        r = simulator(d=d_, n=d_ - params.n, q=params.q, beta=beta, xi=1, tau=False)
        rho, cost_red, N, sieve_dim = red_cost_model.short_vectors(beta, d_)
        bkz_cost = costf(red_cost_model, beta, d_)
        if math.sqrt(d) * params.length_bound <= params.q:
            vector_length = rho * math.sqrt(r[0])
            sigma = vector_length / math.sqrt(d_)
            log_trial_prob = d_ * math.log2(1 - 2 * gaussian_cdf(0, sigma, -params.length_bound))
        else:
            if abs(math.sqrt(r[0]) - params.q) < 1e-8:
                idx_start = next(i for i, r_ in enumerate(r) if r_ < r[0])
            else:
                idx_start = 0
            if abs(r[-1] - 1) < 1e-8:
                idx_end = next((i - 1 for i, r_ in enumerate(r) if math.sqrt(r_) <= 1 + 1e-8), d_ - 1)
            else:
                idx_end = d_ - 1
            vector_length = math.sqrt(r[idx_start])
            gaussian_coords = max(idx_end - idx_start + 1, sieve_dim)
            sigma = vector_length / math.sqrt(gaussian_coords)
            log_trial_prob = gaussian_coords * math.log2(1 - 2 * gaussian_cdf(0, sigma, -params.length_bound))
            log_trial_prob += idx_start * math.log2((2 * params.length_bound + 1) / params.q)
        probability = min(1.0, 2 ** (log_trial_prob + math.log2(N)))
        ret = Cost()
        ret["rop"] = cost_red
        ret["red"] = bkz_cost["rop"]
        ret["sieve"] = max(cost_red - bkz_cost["rop"], 1e-100)
        ret["beta"] = beta
        ret["eta"] = sieve_dim
        ret["zeta"] = zeta
        ret["d"] = d_
        ret["prob"] = probability
        ret.register_impermanent(rop=True, red=True, sieve=True, eta=False, zeta=False, prob=False)
        if probability and not math.isnan(probability):
            ret = ret.repeat(prob_amplify(success_probability, probability))
        else:
            return Cost(rop=OO)
        return ret

    @classmethod
    def cost_zeta(cls, zeta, params, ignore_qary=False, red_shape_model=red_simulator_default,
                  red_cost_model=red_cost_model_default, d=None, log_level=5, **kwds):
        params_baseline = params.updated(norm=2, length_bound=2 if params.length_bound == 1 else params.length_bound)
        baseline_cost = lattice(params_baseline, ignore_qary=ignore_qary, red_shape_model=red_shape_model,
                                red_cost_model=red_cost_model, log_level=log_level + 1, **kwds)
        f = partial(cls.cost_infinity, params=params, zeta=zeta, simulator=red_shape_model,
                    red_cost_model=red_cost_model, d=d, **kwds)
        with local_minimum(40, baseline_cost["beta"] + 1, precision=2, log_level=log_level + 1) as it:
            for beta in it:
                it.update(f(beta))
            for beta in it.neighborhood:
                it.update(f(beta))
            cost = it.y
        return cost if cost else Cost(rop=OO)

    def __call__(self, params, zeta=None, red_shape_model=red_shape_model_default,
                 red_cost_model=red_cost_model_default, log_level=1, **kwds):
        if params.norm == 2:
            tag = "euclidean"
        elif params.norm == OO:
            tag = "infinity"
        else:
            raise NotImplementedError("Only euclidean and infinity norms supported")
        if tag == "infinity":
            red_shape_model = simulator_normalize(red_shape_model)
            f = partial(self.cost_zeta, params=params, red_shape_model=red_shape_model,
                        red_cost_model=red_cost_model, log_level=log_level + 1)
            if zeta is None:
                with local_minimum(0, params.m, log_level=log_level) as it:
                    for zeta in it:
                        it.update(f(zeta=zeta, **kwds))
                    cost = min(it.y, f(zeta=0, **kwds))
            else:
                cost = f(zeta=zeta)
        else:
            cost = self.cost_euclidean(params=params, red_cost_model=red_cost_model, log_level=log_level + 1)
        cost["tag"] = tag
        cost["problem"] = params
        if tag == "euclidean":
            for k in ("sieve", "prob", "repetitions", "zeta"):
                cost.pop(k, None)
        return cost.sanity_check()

    __name__ = "lattice"


lattice = SISLattice()
