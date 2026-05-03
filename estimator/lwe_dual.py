# -*- coding: utf-8 -*-
"""
Estimate cost of solving LWE using dual attacks.
Pure Python version – no Sage required.
"""

from functools import partial, lru_cache
import math
from scipy.optimize import minimize_scalar

from .reduction import delta as deltaf
from .util import local_minimum, early_abort_range
from .cost import Cost
from .lwe_parameters import LWEParameters
from .prob import drop as prob_drop, amplify as prob_amplify
from .io import Logging
from .conf import red_cost_model as red_cost_model_default, mitm_opt as mitm_opt_default, max_beta as max_beta_global
from .errors import OutOfBoundsError, InsufficientSamplesError
from .nd import DiscreteGaussian, SparseTernary
from .lwe_guess import exhaustive_search, mitm, distinguish

OO = float('inf')


class DualHybrid:
    """Estimate cost of solving LWE using dual attacks."""

    @staticmethod
    @lru_cache(maxsize=None)
    def dual_reduce(delta, params, zeta=0, h1=0, rho=1.0, t=0, log_level=None):
        if not 0 <= zeta <= params.n:
            raise OutOfBoundsError(f"Splitting dimension {zeta} must be between 0 and n={params.n}.")

        if params.Xs.is_sparse:
            h = params.Xs.hamming_weight
            if not 0 <= h1 <= h:
                raise OutOfBoundsError(f"Splitting weight {h1} must be between 0 and h={h}.")
            if isinstance(params.Xs, SparseTernary):
                slv_Xs, red_Xs = params.Xs.split_balanced(zeta, h1)
            else:
                raise NotImplementedError(f"Unknown how to exploit sparsity of {params.Xs}")
            if h1 == h:
                return params.updated(Xs=slv_Xs, m=OO), 1
        else:
            red_Xs = params.Xs.resize(params.n - zeta)
            slv_Xs = params.Xs.resize(zeta)

        c = red_Xs.stddev * params.q / params.Xe.stddev

        m_ = max(1, math.ceil(math.sqrt(red_Xs.n * math.log(c) / math.log(delta))) - red_Xs.n)
        m_ = min(params.m, m_)

        d = m_ + red_Xs.n
        rho /= 2 ** (t / d)

        sigma_ = rho * red_Xs.stddev * delta**d / c ** (m_ / d)
        slv_Xe = DiscreteGaussian(params.q * sigma_)

        slv_params = LWEParameters(n=zeta, q=params.q, Xs=slv_Xs, Xe=slv_Xe)
        return slv_params, m_

    @staticmethod
    @lru_cache(maxsize=None)
    def cost(solver, params, beta, zeta=0, h1=0, t=0, success_probability=0.99,
             red_cost_model=red_cost_model_default, log_level=None):
        Logging.log("dual", log_level, f"β={beta}, ζ={zeta}, h1={h1}")

        delta = deltaf(beta)
        rho = red_cost_model.short_vectors(beta=beta, d=2*beta)[0]

        params_slv, m_ = DualHybrid.dual_reduce(delta, params, zeta, h1, rho, t, log_level=log_level+1)
        Logging.log("dual", log_level + 1, f"red LWE instance: {repr(params_slv)}")

        if t:
            cost = DualHybrid.fft_solver(params_slv, success_probability, t)
        else:
            cost = solver(params_slv, success_probability)
        cost["beta"] = beta

        if cost["rop"] == OO or cost["m"] == OO:
            return cost

        d = m_ + params.n - zeta
        _, cost_red, N, sieve_dim = red_cost_model.short_vectors(beta, d, cost["m"])
        Logging.log("dual", log_level + 2, f"red: {Cost(rop=cost_red)!r}")

        cost["rop"] += cost_red
        cost["mem"] += sieve_dim * N
        cost["m"] = m_

        if d < params.n - zeta:
            raise RuntimeError(f"{d} < {params.n - zeta}")
        cost["d"] = d

        Logging.log("dual", log_level, f"{repr(cost)}")

        rep = 1
        if params.Xs.is_sparse:
            h = params.Xs.hamming_weight
            probability = prob_drop(params.n, h, zeta, h1)
            rep = prob_amplify(success_probability, probability)
        return cost.repeat(times=rep, select={"m": False})

    @staticmethod
    def fft_solver(params, success_probability, t=0):
        probability = math.sqrt(success_probability)
        try:
            size = params.Xs.support_size(probability)
            size_fft = 2**t
        except NotImplementedError:
            return Cost(rop=OO, mem=OO, m=1)

        sigma = params.Xe.stddev / params.q
        m_required = 4 * math.exp(4 * math.pi**2 * sigma**2) * (math.log(size_fft * size) - math.log(math.log(1/probability)))

        if params.m < m_required:
            raise InsufficientSamplesError(f"Need {m_required} samples but only {params.m} available.")

        runtime_cost = size * (t * size_fft)
        runtime_cost += size * (4 * m_required)

        memory_cost = size_fft
        return Cost(rop=runtime_cost, mem=memory_cost, m=m_required, t=t)

    @staticmethod
    def optimize_blocksize(solver, params, zeta=0, h1=0, success_probability=0.99,
                           red_cost_model=red_cost_model_default, log_level=5, opt_step=8, fft=False):
        f_t = partial(DualHybrid.cost, solver=solver, params=params, zeta=zeta, h1=h1,
                      success_probability=success_probability, red_cost_model=red_cost_model,
                      log_level=log_level)

        if fft:
            def f(beta):
                with local_minimum(0, params.n - zeta) as it:
                    for t in it:
                        it.update(f_t(beta=beta, t=t))
                    return it.y
        else:
            f = f_t

        max_beta = max(min(params.m - zeta, max_beta_global), 40 + opt_step)
        with local_minimum(40, max_beta, opt_step) as it:
            for beta in it:
                cost = f(beta=beta)
                it.update(cost)
            for beta in it.neighborhood:
                it.update(f(beta=beta))
            cost = it.y
        beta = cost["beta"]

        cost["zeta"] = zeta
        if params.Xs.is_sparse:
            cost["h1"] = h1
        return cost

    def __call__(self, solver, params, success_probability=0.99, red_cost_model=red_cost_model_default,
                 opt_step=8, log_level=1, fft=False):
        Cost.register_impermanent(rop=True, mem=False, red=True, beta=False, delta=False,
                                  m=True, d=False, zeta=False, t=False)
        Logging.log("dual", log_level, f"costing LWE instance: {repr(params)}")

        params = params.normalize()

        if params.Xs.is_sparse:
            Cost.register_impermanent(h1=False)

            def _optimize_blocksize(solver, params, zeta=0, success_probability=0.99,
                                    red_cost_model=red_cost_model_default, log_level=None, fft=False):
                h = params.Xs.hamming_weight
                h1_min = max(0, h - (params.n - zeta))
                h1_max = min(zeta, h)
                if h1_min == h1_max:
                    h1_max = h1_min + 1
                Logging.log("dual", log_level, f"h1 ∈ [{h1_min},{h1_max}] (zeta={zeta})")
                with local_minimum(h1_min, h1_max, log_level=log_level+1) as it:
                    for h1 in it:
                        cost = self.optimize_blocksize(
                            h1=h1, solver=solver, params=params, zeta=zeta,
                            success_probability=success_probability,
                            red_cost_model=red_cost_model, log_level=log_level+2)
                        it.update(cost)
                    return it.y
        else:
            _optimize_blocksize = self.optimize_blocksize

        f = partial(_optimize_blocksize, solver=solver, params=params,
                    success_probability=success_probability, red_cost_model=red_cost_model,
                    log_level=log_level+1, fft=fft)
        with local_minimum(0, params.n, opt_step) as it:
            for zeta in it:
                it.update(f(zeta=zeta))
            for zeta in it.neighborhood:
                it.update(f(zeta=zeta))
            cost = it.y

        cost["problem"] = params
        return cost.sanity_check()


DH = DualHybrid()


class MATZOV:
    C_mul = 32**2
    C_add = 5 * 32

    @classmethod
    def T_fftf(cls, k, p):
        return cls.C_mul * k * p ** (k + 1)

    @classmethod
    def T_tablef(cls, D):
        return 4 * cls.C_add * D

    @classmethod
    def Nf(cls, params, m, beta_bkz, beta_sieve, k_enum, k_fft, p):
        mu = 0.5
        k_lat = params.n - k_fft - k_enum
        lsigma_s = (params.Xe.stddev ** (m / (m + k_lat))
                    * (params.Xs.stddev * params.q) ** (k_lat / (m + k_lat))
                    * math.sqrt(4/3.0)
                    * math.sqrt(beta_sieve / (2 * math.pi * math.e))
                    * deltaf(beta_bkz) ** (m + k_lat - beta_sieve))
        N = (math.exp(4 * (lsigma_s * math.pi / params.q) ** 2)
             * math.exp(k_fft / 3.0 * (params.Xs.stddev * math.pi / p) ** 2)
             * (k_enum * cls.Hf(params.Xs) + k_fft * math.log(p) + math.log(1 / mu)))
        return N

    @staticmethod
    def Hf(Xs):
        return (0.5 + math.log(math.sqrt(2 * math.pi) * Xs.stddev)
                + math.log(math.cosh(math.pi**2 * Xs.stddev**2))) / math.log(2)

    @classmethod
    def cost(cls, beta, params, m=None, p=2, k_enum=0, k_fft=0, beta_sieve=None,
             red_cost_model=red_cost_model_default):
        if m is None:
            m = params.n
        k_lat = params.n - k_fft - k_enum

        N = cls.Nf(params, m, beta, beta_sieve if beta_sieve else beta, k_enum, k_fft, p)

        rho, T_sample, _, beta_sieve = red_cost_model.short_vectors(
            beta, N=N, d=k_lat + m, sieve_dim=beta_sieve)

        H = cls.Hf(params.Xs)
        coeff = 1 / (1 - math.exp(-1 / (2 * params.Xs.stddev**2)))
        tmp_alpha = math.pi**2 * params.Xs.stddev**2
        tmp_a = math.exp(8 * tmp_alpha * math.exp(-2 * tmp_alpha) * math.tanh(tmp_alpha))
        T_guess = coeff * (((2 * tmp_a / math.e) ** k_enum)
                           * (2 ** (k_enum * H))
                           * (cls.T_fftf(k_fft, p) + cls.T_tablef(N)))

        cost = Cost(rop=T_sample + T_guess, problem=params)
        cost["red"] = T_sample
        cost["guess"] = T_guess
        cost["beta"] = beta
        cost["p"] = p
        cost["zeta"] = k_enum
        cost["t"] = k_fft
        cost["beta_"] = beta_sieve
        cost["N"] = N
        cost["m"] = m

        cost.register_impermanent({"β'": False, "ζ": False, "t": False}, rop=True, p=False, N=False)
        return cost

    def __call__(self, params, red_cost_model=red_cost_model_default, log_level=1):
        params = params.normalize()

        for p in early_abort_range(2, params.q):
            for k_enum in early_abort_range(0, params.n, 10):
                for k_fft in early_abort_range(0, params.n - k_enum[0], 10):
                    precision = 1
                    max_beta = max(min(params.m - k_enum[0] - k_fft[0], max_beta_global), 40 + precision)
                    with local_minimum(40, max_beta, precision=precision, log_level=log_level+4) as it:
                        for beta in it:
                            cost = self.cost(beta, params, p=p[0], k_enum=k_enum[0], k_fft=k_fft[0],
                                             red_cost_model=red_cost_model)
                            it.update(cost)
                        Logging.log("dual", log_level+3, f"t: {k_fft[0]}, {repr(it.y)}")
                        k_fft[1].update(it.y)
                Logging.log("dual", log_level+2, f"ζ: {k_enum[0]}, {repr(k_fft[1].y)}")
                k_enum[1].update(k_fft[1].y)
            Logging.log("dual", log_level+1, f"p:{p[0]}, {repr(k_enum[1].y)}")
            p[1].update(k_enum[1].y)
            if p[1].y["t"] == 0 and p[0] > 2:
                break
        Logging.log("dual", log_level, f"{repr(p[1].y)}")
        return p[1].y


matzov = MATZOV()


def dual(params, success_probability=0.99, red_cost_model=red_cost_model_default):
    Cost.register_impermanent(rop=True, mem=False, red=True, beta=False, delta=False,
                              m=True, d=False)
    ret = DH.optimize_blocksize(solver=distinguish, params=params, zeta=0, h1=0,
                                success_probability=success_probability,
                                red_cost_model=red_cost_model, log_level=1)
    ret.pop("zeta", None)
    ret.pop("h1", None)
    ret["tag"] = "dual"
    return ret


def dual_hybrid(params, success_probability=0.99, red_cost_model=red_cost_model_default,
                mitm_optimization=False, opt_step=8, fft=False):
    if mitm_optimization is True:
        mitm_optimization = mitm_opt_default
    if mitm_optimization:
        solver = partial(mitm, optimization=mitm_optimization)
    else:
        solver = exhaustive_search

    ret = DH(solver=solver, params=params, success_probability=success_probability,
             red_cost_model=red_cost_model, opt_step=opt_step, fft=fft)
    ret["tag"] = "dual_mitm_hybrid" if mitm_optimization else "dual_hybrid"
    return ret
