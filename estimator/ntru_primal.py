# -*- coding: utf-8 -*-
"""
Estimate cost of solving NTRU using primal attacks. Pure Python, no Sage.
"""

from functools import lru_cache
import math
from scipy.special import digamma

from .reduction import cost as costf
from .util import zeta_precomputed, zeta_prime_precomputed, gh_constant
from .lwe_primal import PrimalUSVP, PrimalHybrid
from .ntru_parameters import NTRUParameters
from .simulator import normalize as simulator_normalize
from .prob import conditional_chi_squared, chisquared_table
from .io import Logging
from .conf import red_cost_model as red_cost_model_default
from .conf import red_shape_model as red_shape_model_default
from .conf import max_n_cache

OO = float('inf')
EULER_GAMMA = 0.57721566490153286060651209008240243104215933593992  # Euler–Mascheroni constant

class PrimalDSD:
    """Estimate cost of solving (overstretched) NTRU via dense sublattice discovery"""

    @staticmethod
    @lru_cache(maxsize=None)
    def ball_log_vol(n):
        return (n / 2.0) * math.log(math.pi) - math.lgamma(n / 2.0 + 1)

    @staticmethod
    def log_gh(d, logvol=0):
        if d < 49:
            return gh_constant[d] + logvol / d
        return (1.0 / d) * (logvol - PrimalDSD.ball_log_vol(d))

    @staticmethod
    def DSL_logvol_matrix(n, sigmasq):
        total = n * (math.log(sigmasq) + math.log(2.0) + digamma(n)) / 2.0
        proj_loss = sum((digamma((2 * n - i) / 2.0) - digamma(n)) for i in range(n)) / 2.0
        return total + proj_loss

    @staticmethod
    def DSL_logvol_circulant(n, sigmasq):
        lambda0 = (math.log(2) - EULER_GAMMA + math.log(n) + math.log(sigmasq)) / 2.0
        lambdai = (n - 1) * (1 - EULER_GAMMA + math.log(n) + math.log(sigmasq)) / 2.0
        return lambda0 + lambdai

    @staticmethod
    def DSL_logvol_circulant_fixed(n, R):
        lambda0 = (-EULER_GAMMA + math.log(R)) / 2.0
        lambdai = (n - 1) * (1 - EULER_GAMMA + math.log(R) - math.log(2)) / 2.0
        return lambda0 + lambdai

    @staticmethod
    @lru_cache(maxsize=None)
    def DSL_logvol(n, sigmasq, ntru="circulant"):
        if ntru == "matrix":
            return PrimalDSD.DSL_logvol_matrix(n, sigmasq)
        if ntru == "circulant":
            return PrimalDSD.DSL_logvol_circulant(n, sigmasq)
        if ntru == "fixed":
            return PrimalDSD.DSL_logvol_circulant_fixed(n, sigmasq)
        raise ValueError(f"NTRU type: {ntru} is not supported.")

    @staticmethod
    @lru_cache(maxsize=None)
    def proj_logloss(d, k):
        return (digamma((d - k) / 2.0) - digamma(d / 2.0)) / 2.0

    @staticmethod
    def DSLI_vols(dsl_logvol, FL_shape):
        n = len(FL_shape) // 2
        vols = (2 * n + 1) * [None]
        dsl_dim = n
        vols[2 * n] = dsl_logvol
        for s in range(2 * n - 1, n, -1):
            x = -FL_shape[s]
            x += PrimalDSD.proj_logloss(s + 1, n)
            x += zeta_prime_precomputed[dsl_dim] / zeta_precomputed[dsl_dim]
            dsl_logvol += x
            vols[s] = dsl_logvol
            dsl_dim -= 1
        assert dsl_dim == 1
        assert s == n + 1
        return vols

    @staticmethod
    @lru_cache(maxsize=None)
    def prob_dsd(beta, params, simulator, m=OO, tau=None, d=None, dsl_logvol=None,
                 red_cost_model=red_cost_model_default, log_level=None):
        if d is None:
            d = m
        xi = PrimalUSVP._xi_factor(params.Xs, params.Xe)
        if dsl_logvol is None:
            dsl_logvol = PrimalDSD.DSL_logvol(params.n, params.Xs.stddev**2, ntru=params.ntru_type)

        B_shape = [math.log(r_) / 2 for r_ in simulator(d, params.n, params.q, beta, xi=xi, tau=tau)]
        dsli_vols = PrimalDSD.DSLI_vols(dsl_logvol, B_shape)
        prob_all_not = 1.0
        prob_pos = (2 * params.n) * [0.0]
        for i in range(1, params.n + 1):
            s = params.n + i
            dslv_len = PrimalDSD.log_gh(i, dsli_vols[s])
            sigma_sq = math.exp(2 * dslv_len) / s
            if sigma_sq > 1e10:
                prob_pos[s - beta] = 0.0
                continue
            norm_threshold = math.exp(2 * (B_shape[s - beta])) / sigma_sq
            proba_one = chisquared_table[beta].cdf(norm_threshold)
            if proba_one <= 1e-7:
                continue
            if beta <= 20:
                for j in range(2, int(s / beta + 1)):
                    if proba_one < 1e-6:
                        proba_one = 0.0
                        break
                    ind = s - j * (beta - 1) - 1
                    norm_bt = math.exp(2 * B_shape[ind]) / sigma_sq
                    norm_b2 = math.exp(2 * B_shape[ind + beta - 1]) / sigma_sq
                    proba_one *= conditional_chi_squared(beta - 1, s - ind - (beta - 1), norm_bt, norm_b2)
            prob_pos[s - beta] = proba_one
            prob_all_not *= max(1.0 - proba_one, 0.0)
            Logging.log("dsd", log_level + 1, f"Pr[dsd, {beta}] = {prob_all_not}")
        return 1.0 - prob_all_not, prob_pos

    def __call__(self, params, red_cost_model=red_cost_model_default,
                 red_shape_model=red_shape_model_default, log_level=1, **kwds):
        if params.Xs.stddev != params.Xe.stddev:
            raise NotImplementedError("Dense sublattice attack not supported for Xs != Xe")
        params = NTRUParameters.normalize(params)
        m = params.m + params.n if params.Xs <= params.Xe else params.m
        try:
            red_shape_model = simulator_normalize(red_shape_model)
        except ValueError:
            pass
        if params.n > max_n_cache:
            raise ValueError("Please increase max_n_cache to run predictor for such large n")
        remaining_proba = 1.0
        average_beta = 0.0
        total_DSD_prob = 0.0
        DSD_prob = 0.0
        prob_pos_total = (2 * params.n) * [0.0]

        for beta in range(2, params.n):
            tours = math.floor(params.n**2 / beta**2) + 3
            DSD_prob, DSD_prob_pos = self.prob_dsd(
                beta, params, red_shape_model, m=m,
                red_cost_model=red_cost_model, log_level=log_level)
            if DSD_prob > 1e-7:
                for t in range(tours):
                    for i in range(2 * params.n):
                        prob_pos = DSD_prob_pos[i]
                        average_beta += beta * remaining_proba * prob_pos
                        prob_pos_total[i] += remaining_proba * prob_pos
                        total_DSD_prob += remaining_proba * prob_pos
                        remaining_proba *= 1.0 - prob_pos
                Logging.log("dsd", log_level + 1,
                            f"β= {beta},\t pr={DSD_prob:.4e}, \t rem-pr={remaining_proba:.4e}")
            if remaining_proba < 0.001:
                average_beta += beta * remaining_proba
                break

        if not average_beta:
            average_beta = m
            predicate = False
        else:
            predicate = True

        cost = costf(red_cost_model, average_beta, m, predicate=predicate)
        cost["tag"] = "dsd"
        cost["problem"] = params
        return cost.sanity_check()

    __name__ = "primal_dsd"


primal_dsd = PrimalDSD()


class NTRUPrimalUSVP(PrimalUSVP):
    def __call__(self, params, red_cost_model=red_cost_model_default,
                 red_shape_model=red_shape_model_default, optimize_d=True, log_level=1, **kwds):
        return super().__call__(
            params,
            red_cost_model=red_cost_model,
            red_shape_model=red_shape_model,
            optimize_d=optimize_d,
            log_level=log_level,
            **kwds,
        )


primal_usvp = NTRUPrimalUSVP()


class NTRUPrimalHybrid(PrimalHybrid):
    def __call__(self, params, babai=True, zeta=None, mitm=True,
                 red_shape_model=red_shape_model_default,
                 red_cost_model=red_cost_model_default, log_level=1, **kwds):
        return super().__call__(
            params,
            babai=babai,
            zeta=zeta,
            mitm=mitm,
            red_shape_model=red_shape_model,
            red_cost_model=red_cost_model,
            log_level=log_level,
            **kwds,
        )


primal_hybrid = NTRUPrimalHybrid()


def primal_bdd(params, red_shape_model=red_shape_model_default,
               red_cost_model=red_cost_model_default, log_level=1, **kwds):
    return primal_hybrid(
        params,
        zeta=0,
        mitm=False,
        babai=False,
        red_shape_model=red_shape_model,
        red_cost_model=red_cost_model,
        log_level=log_level,
        **kwds,
    )
