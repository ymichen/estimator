# -*- coding: utf-8 -*-
"""
Estimate cost of solving LWE using primal attacks.
Pure Python version – no Sage required.
"""

from functools import partial, lru_cache
import math
from scipy.optimize import minimize_scalar

from .reduction import delta as deltaf
from .reduction import cost as costf
from .util import local_minimum
from .cost import Cost
from .lwe_parameters import LWEParameters
from .simulator import normalize as simulator_normalize
from .prob import guessing_set_and_hit_probability, amplify as prob_amplify
from .prob import babai as prob_babai, mitm_babai_probability
from .io import Logging
from .conf import red_cost_model as red_cost_model_default
from .conf import red_shape_model as red_shape_model_default
from .conf import max_beta as max_beta_global

OO = float('inf')


class PrimalUSVP:
    """Estimate cost of solving LWE via uSVP reduction."""

    @staticmethod
    def _xi_factor(Xs, Xe):
        xi = 1.0
        if Xs < Xe:
            xi = Xe.stddev / Xs.stddev
        return xi

    @staticmethod
    def _solve_for_d(params, m, beta, tau, xi):
        """Find smallest d ∈ [n,m] to satisfy uSVP condition."""
        delta = deltaf(beta)
        a = -math.log(delta)
        assert a < 0, "Root-Hermite factor delta must be > 1, so a must be negative"

        if tau is None or not tau:
            C = math.log(params.Xe.stddev**2 * (beta - 1)) / 2.0
            c = params.n * math.log(xi) - (params.n + 1) * math.log(params.q)
        else:
            C = math.log(params.Xe.stddev**2 * (beta - 1) + tau**2) / 2.0
            c = math.log(tau) + params.n * math.log(xi) - (params.n + 1) * math.log(params.q)
        b = math.log(delta) * (2 * beta - 1) + math.log(params.q) - C
        nb = max(params.n, beta)

        if a * nb * nb + b * nb + c >= -1e-9:
            return nb
        disc = b * b - 4 * a * c
        if disc < 0:
            if disc >= -1e-9:
                disc = 0.0
            else:
                return OO
        d1 = (-b + math.sqrt(disc)) / (2 * a)

        if nb <= d1 + 1e-6:
            return min(m, math.ceil(d1))
        return OO

    @staticmethod
    @lru_cache(maxsize=None)
    def cost_gsa(beta, params, m=OO, tau=None, d=None,
                 red_cost_model=red_cost_model_default, log_level=None):
        delta = deltaf(beta)
        xi = PrimalUSVP._xi_factor(params.Xs, params.Xe)
        if tau is None:
            tau = params.Xe.stddev
        if params._homogeneous:
            tau = False
        if d is None:
            d = PrimalUSVP._solve_for_d(params, m, beta, tau, xi)
        if d == OO:
            return Cost(rop=OO)
        if d < beta:
            d = beta
        if d == beta and d < m:
            d += 1
        assert d <= m + 1

        if tau is False:
            lhs = math.log(math.sqrt(params.Xe.stddev**2 * (beta - 1)))
            rhs = (math.log(delta) * (2 * beta - d - 1)
                   + (math.log(xi) * params.n + math.log(params.q) * (d - params.n - 1)) / d)
        else:
            lhs = math.log(math.sqrt(params.Xe.stddev**2 * (beta - 1) + tau**2))
            rhs = (math.log(delta) * (2 * beta - d - 1)
                   + (math.log(tau) + math.log(xi) * params.n + math.log(params.q) * (d - params.n - 1)) / d)
        return costf(red_cost_model, beta, d, predicate=lhs <= rhs)

    @staticmethod
    @lru_cache(maxsize=None)
    def cost_simulator(beta, params, simulator, m=OO, tau=None, d=None,
                       red_cost_model=red_cost_model_default, log_level=None):
        delta = deltaf(beta)
        if d is None:
            d = min(math.ceil(math.sqrt(params.n * math.log(params.q) / math.log(delta))), m) + 1
            d = max(d, beta)
        xi = PrimalUSVP._xi_factor(params.Xs, params.Xe)
        if tau is None:
            tau = params.Xe.stddev
        if params._homogeneous:
            tau = False
            d -= 1
        r = simulator(d=d, n=params.n, q=params.q, beta=beta, xi=xi, tau=tau)
        if tau is False:
            lhs = params.Xe.stddev**2 * (beta - 1)
        else:
            lhs = params.Xe.stddev**2 * (beta - 1) + tau**2
        predicate = r[d - beta] > lhs
        return costf(red_cost_model, beta, d, predicate=predicate)

    def __call__(self, params, red_cost_model=red_cost_model_default,
                 red_shape_model=red_shape_model_default, optimize_d=True,
                 log_level=1, **kwds):
        params = LWEParameters.normalize(params)
        if params.Xs <= params.Xe:
            m = params.m + params.n
        else:
            m = params.m

        if red_shape_model == "gsa":
            precision = 5
            max_beta = max(min(max_beta_global, m), 40 + precision)
            with local_minimum(40, max_beta, precision=precision) as it:
                for beta in it:
                    cost = self.cost_gsa(beta=beta, params=params, m=m,
                                         red_cost_model=red_cost_model, **kwds)
                    it.update(cost)
                for beta in it.neighborhood:
                    cost = self.cost_gsa(beta=beta, params=params, m=m,
                                         red_cost_model=red_cost_model, **kwds)
                    it.update(cost)
                cost = it.y
            cost["tag"] = "usvp"
            cost["problem"] = params
            return cost.sanity_check()

        try:
            red_shape_model = simulator_normalize(red_shape_model)
        except ValueError:
            pass

        cost_gsa = self(params, red_cost_model=red_cost_model,
                        red_shape_model="gsa", log_level=log_level+1, **kwds)
        Logging.log("usvp", log_level + 1, f"GSA: {repr(cost_gsa)}")

        f = partial(self.cost_simulator, simulator=red_shape_model,
                    red_cost_model=red_cost_model, m=m, params=params)
        with local_minimum(max(cost_gsa["beta"] - math.ceil(0.10 * cost_gsa["beta"]), 40),
                           max(cost_gsa["beta"] + math.ceil(0.20 * cost_gsa["beta"]), 40)) as it:
            for beta in it:
                it.update(f(beta=beta, **kwds))
            cost = it.y
        Logging.log("usvp", log_level, f"Opt-β: {repr(cost)}")

        if cost and optimize_d:
            with local_minimum(max(params.n, cost["beta"]), stop=cost["d"] + 1) as it:
                for d in it:
                    it.update(f(d=d, beta=cost["beta"], **kwds))
                cost = it.y
            Logging.log("usvp", log_level + 1, f"Opt-d: {repr(cost)}")

        cost["tag"] = "usvp"
        cost["problem"] = params
        return cost.sanity_check()

    __name__ = "primal_usvp"


primal_usvp = PrimalUSVP()


class PrimalHybrid:
    @classmethod
    def babai_cost(cls, d):
        return Cost(rop=max(d, 1)**2)

    @classmethod
    def svp_dimension(cls, r, D, is_homogeneous=False):
        from math import lgamma, log, pi
        def ball_log_vol(n):
            return (n / 2.0) * log(pi) - lgamma(n / 2.0 + 1)
        def svp_gaussian_heuristic_log_input(r, tau):
            if tau is None:
                n = len(r)
                log_vol = sum(r)
            else:
                n = len(r) + 1
                log_vol = sum(r) + 2 * log(tau)
            log_gh = 1.0 / n * (log_vol - 2 * ball_log_vol(n))
            return log_gh

        d = len(r)
        r = [log(x) for x in r]
        min_i = 0 if d <= 4096 else d - 1754
        if is_homogeneous:
            tau = None
            for i in range(min_i, d):
                if svp_gaussian_heuristic_log_input(r[i:], tau) < log(D.stddev**2 * (d - i)):
                    return d - (i - 1)
            return 2
        else:
            tau = D.stddev
            for i in range(min_i, d):
                if svp_gaussian_heuristic_log_input(r[i:], tau) < log(D.stddev**2 * (d - i) + tau**2):
                    return d - (i - 1) + 1
            return 2

    @classmethod
    def svp_dimension_gsa(cls, d, log_total_vol, log_delta, D, is_homogeneous=False):
        from math import lgamma, log, pi
        def log_projected_vol(i):
            return (d - i) / d * log_total_vol - i * (d - i) * log_delta
        def ball_log_vol(n):
            return (n / 2.0) * log(pi) - lgamma(n / 2.0 + 1)
        def svp_gaussian_heuristic_gsa(i, tau):
            if tau is None:
                n = d - i
                log_vol = 2 * log_projected_vol(i)
            else:
                n = d - i + 1
                log_vol = 2 * log_projected_vol(i) + 2 * log(tau)
            log_gh = 1.0 / n * (log_vol - 2 * ball_log_vol(n))
            return log_gh

        min_i = 0 if d <= 4096 else d - 1754
        if is_homogeneous:
            tau = None
            for i in range(min_i, d):
                if svp_gaussian_heuristic_gsa(i, tau) < log(D.stddev**2 * (d - i)):
                    return d - (i - 1)
            return 2
        else:
            tau = D.stddev
            for i in range(min_i, d):
                if svp_gaussian_heuristic_gsa(i, tau) < log(D.stddev**2 * (d - i) + tau**2):
                    return d - (i - 1) + 1
            return 2

    @classmethod
    @lru_cache(maxsize=None)
    def beta_params(cls, beta, params, zeta=0, babai=False, mitm=False,
                    m=OO, d=None, red_shape_model=red_shape_model_default,
                    red_cost_model=red_cost_model_default, log_level=5):
        if m - zeta < beta:
            return {"bkz_cost": Cost(rop=OO)}
        if d is not None and d < beta:
            return {"bkz_cost": Cost(rop=OO)}

        simulator = simulator_normalize(red_shape_model)
        xi = PrimalUSVP._xi_factor(params.Xs, params.Xe)

        if d is None:
            delta = deltaf(beta)
            d = max(beta, min(math.ceil(math.sqrt((params.n - zeta) * math.log(params.q / xi) / math.log(delta))),
                              m - zeta))

        r = simulator(d, params.n - zeta, params.q, beta, xi=xi, tau=False, dual=True)
        bkz_cost = costf(red_cost_model, beta, d)

        if babai:
            eta = 2
            svp_cost = cls.babai_cost(d)
        else:
            if red_shape_model == "gsa":
                log_vol = ((d - (params.n - zeta)) * math.log(params.q) + (params.n - zeta) * math.log(xi))
                log_delta = math.log(deltaf(beta))
                svp_dim = cls.svp_dimension_gsa(d, log_vol, log_delta, params.Xe, params._homogeneous)
            else:
                svp_dim = cls.svp_dimension(r, params.Xe, is_homogeneous=params._homogeneous)
            eta = svp_dim if params._homogeneous else svp_dim - 1
            if eta > d:
                return {"svp_cost": Cost(rop=OO)}
            svp_cost = costf(red_cost_model, svp_dim, svp_dim)
            svp_cost["rop"] += cls.babai_cost(d - eta)["rop"]

        if babai:
            babai_probability = prob_babai(r, math.sqrt(d) * params.Xe.stddev)
        else:
            babai_probability = prob_babai(r[:d - eta], math.sqrt(d - eta) * params.Xe.stddev)

        if mitm and zeta > 0:
            if babai:
                mitm_probability = mitm_babai_probability(r, params.Xe.stddev)
            else:
                mitm_probability = 1.0
        else:
            mitm_probability = 1.0

        return {"bkz_cost": bkz_cost, "d": d, "svp_cost": svp_cost, "eta": eta,
                "babai_probability": babai_probability, "mitm_probability": mitm_probability}

    @staticmethod
    @lru_cache(maxsize=None)
    def cost(beta, params, zeta=0, babai=False, mitm=False, m=OO, d=None,
             red_shape_model=red_shape_model_default, red_cost_model=red_cost_model_default,
             search_space=None, hit_probability=None, log_level=5):
        beta_params = PrimalHybrid.beta_params(beta=beta, params=params, zeta=zeta, babai=babai,
                                               mitm=mitm, m=m, d=d, red_shape_model=red_shape_model,
                                               red_cost_model=red_cost_model)
        if len(beta_params) == 1:
            return Cost(rop=OO)

        def ssf(x):
            return math.sqrt(x) if mitm else x

        if search_space is None or hit_probability is None:
            f = partial(PrimalHybrid.cost, beta=beta, params=params, zeta=zeta, babai=babai,
                        mitm=mitm, m=m, d=beta_params["d"], red_shape_model=red_shape_model,
                        red_cost_model=red_cost_model)
            min_hw = max(0, zeta - params.n + params.Xs.hamming_weight)
            max_hw = min(params.Xs.hamming_weight, zeta)
            cost = Cost(rop=OO)
            for hw in range(min_hw, max_hw + 1):
                search_space, hit_probability = guessing_set_and_hit_probability(zeta, params.Xs, hw)
                new_cost = f(search_space=search_space, hit_probability=hit_probability)
                if new_cost["rop"] > cost["rop"]:
                    return cost
                cost = new_cost
            return cost
        else:
            svp_cost = beta_params["svp_cost"].repeat(ssf(search_space))
            probability = hit_probability

        probability *= beta_params["babai_probability"]
        probability *= beta_params["mitm_probability"]

        bkz_cost = beta_params["bkz_cost"]
        ret = Cost()
        ret["rop"] = bkz_cost["rop"] + svp_cost["rop"]
        ret["red"] = bkz_cost["rop"]
        ret["svp"] = svp_cost["rop"]
        ret["beta"] = beta
        ret["eta"] = beta_params["eta"]
        ret["zeta"] = zeta
        ret["|S|"] = search_space
        ret["d"] = beta_params["d"]
        ret["prob"] = probability

        ret.register_impermanent({"|S|": False}, rop=True, red=True, svp=True,
                                 eta=False, zeta=False, prob=False)

        if probability and not math.isnan(probability):
            ret = ret.repeat(prob_amplify(0.99, probability))
        else:
            return Cost(rop=OO)
        return ret

    @classmethod
    def cost_zeta(cls, zeta, params, red_shape_model=red_shape_model_default,
                  red_cost_model=red_cost_model_default, m=OO, babai=True, mitm=True,
                  optimize_d=True, log_level=5, **kwds):
        baseline_cost = primal_usvp(params, red_shape_model=red_shape_model,
                                    red_cost_model=red_cost_model, optimize_d=False,
                                    log_level=log_level+1, **kwds)
        Logging.log("bdd", log_level, f"H0: {repr(baseline_cost)}")

        f = partial(cls.cost, params=params, zeta=zeta, babai=babai, mitm=mitm,
                    red_shape_model=red_shape_model, red_cost_model=red_cost_model, m=m, **kwds)

        if baseline_cost["rop"] == OO:
            max_beta = max_beta_global
        else:
            max_beta = baseline_cost["beta"]

        min_hw = max(0, zeta - params.Xs.n + params.Xs.hamming_weight)
        max_hw = min(zeta, params.Xs.hamming_weight)
        cost = Cost(rop=OO)
        for hw in range(min_hw, max_hw + 1):
            search_space, hit_probability = guessing_set_and_hit_probability(zeta, params.Xs, hw)
            precision = 2
            with local_minimum(40, max_beta + precision, precision=precision, log_level=log_level+1) as it:
                for beta in it:
                    it.update(f(beta, search_space=search_space, hit_probability=hit_probability))
                for beta in it.neighborhood:
                    it.update(f(beta, search_space=search_space, hit_probability=hit_probability))
                new_cost = it.y
            if new_cost["rop"] > cost["rop"]:
                break
            else:
                cost = new_cost
        Logging.log("bdd", log_level, f"H1: {cost!r}")

        if cost and cost.get("tag", "XXX") != "usvp" and optimize_d:
            with local_minimum(params.n - zeta, cost["d"] + 1, log_level=log_level+1) as it:
                for d in it:
                    it.update(f(beta=cost["beta"], d=d))
                cost = it.y
            Logging.log("bdd", log_level, f"H2: {cost!r}")

        if cost is None:
            return Cost(rop=OO)
        return cost

    def __call__(self, params, babai=True, zeta=None, mitm=True,
                 red_shape_model=red_shape_model_default,
                 red_cost_model=red_cost_model_default, log_level=1, **kwds):
        if zeta == 0:
            tag = "bdd"
        else:
            tag = "hybrid"

        params = LWEParameters.normalize(params)
        m = params.m + params.n if params.Xs <= params.Xe else params.m

        f = partial(self.cost_zeta, params=params, red_shape_model=red_shape_model,
                    red_cost_model=red_cost_model, babai=babai, mitm=mitm, m=m,
                    log_level=log_level+1)

        if zeta is None:
            cost_min_zeta = f(zeta=0, optimize_d=False, **kwds)
            if cost_min_zeta["rop"] < OO:
                min_zeta = 0
            else:
                min_zeta = 0
                for candidate in range(1, params.n):
                    if f(candidate, optimize_d=False, **kwds)["rop"] < OO:
                        min_zeta = candidate
                        break
                else:
                    min_zeta = params.n - 1

            cost_max_zeta = f(zeta=params.n-1, optimize_d=False, **kwds)
            if cost_max_zeta["rop"] < OO:
                max_zeta = params.n - 1
            else:
                max_zeta = params.n - 1
                for candidate in range(params.n-2, -1, -1):
                    if f(candidate, optimize_d=False, **kwds)["rop"] < OO:
                        max_zeta = candidate
                        break
                else:
                    max_zeta = 0

            ret = minimize_scalar(lambda x: math.log(f(zeta=int(round(x)), optimize_d=False, **kwds)["rop"]),
                                  bounds=(min_zeta, max_zeta), method="bounded")
            zeta = int(ret.x)
            cost = f(zeta=zeta, optimize_d=False, **kwds)
            precision = 3
            for z in range(max(0, zeta - precision), min(params.n, zeta + precision) + 1):
                cost = min(cost, f(zeta=z, optimize_d=False, **kwds))
            cost = min(cost, cost_min_zeta, cost_max_zeta)
        else:
            cost = f(zeta=zeta)

        cost["tag"] = tag
        cost["problem"] = params

        if tag == "bdd":
            for k in ("|S|", "prob", "repetitions", "zeta"):
                cost.pop(k, None)

        return cost.sanity_check()

    __name__ = "primal_hybrid"


primal_hybrid = PrimalHybrid()


def primal_bdd(params, red_shape_model=red_shape_model_default,
               red_cost_model=red_cost_model_default, log_level=1, **kwds):
    return primal_hybrid(params, zeta=0, mitm=False, babai=False,
                         red_shape_model=red_shape_model, red_cost_model=red_cost_model,
                         log_level=log_level, **kwds)
