# -*- coding: utf-8 -*-
"""
High-level LWE interface. Pure Python, no Sage required.
"""

from functools import partial

from .lwe_primal import primal_usvp, primal_bdd, primal_hybrid
from .lwe_bkw import coded_bkw   # uncomment if you have lwe_bkw.py
from .lwe_guess import exhaustive_search, mitm, distinguish, guess_composition
from .lwe_dual import dual
from .lwe_dual import matzov as dual_hybrid
# from .gb import arora_gb         # uncomment if you have gb.py
from .lwe_parameters import LWEParameters as Parameters  # noqa
from .conf import (
    red_cost_model as red_cost_model_default,
    red_shape_model as red_shape_model_default,
)
from .util import batch_estimate, f_name
from .reduction import RC
from .io import Logging

OO = float('inf')


class Estimate:

    def rough(self, params, jobs=1, catch_exceptions=True, quiet=False):
        params = params.normalize()
        algorithms = {}
        algorithms["usvp"] = partial(primal_usvp, red_cost_model=RC.ADPS16, red_shape_model="gsa")
        algorithms["dual_hybrid"] = partial(dual_hybrid, red_cost_model=RC.ADPS16)
        # if arora_gb available, uncomment:
        # if params.m > params.n**2 and params.Xe.is_bounded:
        #     algorithms["arora-gb"] = guess_composition(arora_gb.cost_bounded) if params.Xs.is_sparse else arora_gb.cost_bounded

        res_raw = batch_estimate(params, algorithms.values(), log_level=1, jobs=jobs, catch_exceptions=catch_exceptions)
        res_raw = res_raw[params]
        res = {algo: v for algo, attack in algorithms.items() for k, v in res_raw.items() if f_name(attack) == k}
        for algo in algorithms:
            if algo not in res:
                continue
            result = res[algo]
            if result["rop"] != OO:
                Logging.print("estimator", int(quiet), f"{algo:20s} :: {result!r}")
        return res

    def __call__(self, params, red_cost_model=red_cost_model_default,
                 red_shape_model=red_shape_model_default, deny_list=tuple(), add_list=tuple(),
                 jobs=1, catch_exceptions=True, quiet=False):
        params = params.normalize()
        algorithms = {}

        # algorithms["arora-gb"] = guess_composition(arora_gb)   # if available
        algorithms["bkw"] = coded_bkw                           
        algorithms["usvp"] = partial(primal_usvp, red_cost_model=red_cost_model, red_shape_model=red_shape_model)
        algorithms["bdd"] = partial(primal_bdd, red_cost_model=red_cost_model, red_shape_model=red_shape_model)
        algorithms["bdd_hybrid"] = partial(primal_hybrid, mitm=False, babai=False,
                                           red_cost_model=red_cost_model, red_shape_model=red_shape_model)
        algorithms["bdd_mitm_hybrid"] = partial(primal_hybrid, mitm=True, babai=True,
                                                red_cost_model=red_cost_model, red_shape_model=red_shape_model)
        algorithms["dual"] = partial(dual, red_cost_model=red_cost_model)
        algorithms["dual_hybrid"] = partial(dual_hybrid, red_cost_model=red_cost_model)

        algorithms = {k: v for k, v in algorithms.items() if k not in deny_list}
        algorithms.update(add_list)

        res_raw = batch_estimate(params, algorithms.values(), log_level=1, jobs=jobs, catch_exceptions=catch_exceptions)
        res_raw = res_raw[params]
        res = {algo: v for algo, attack in algorithms.items() for k, v in res_raw.items() if f_name(attack) == k}

        for algo in algorithms:
            if algo not in res:
                continue
            result = res[algo]
            if result["rop"] == OO:
                continue
            if algo == "bdd_hybrid" and res["bdd"]["rop"] <= result["rop"]:
                continue
            if algo == "bdd_mitm_hybrid" and res["bdd_hybrid"]["rop"] <= result["rop"]:
                continue
            if algo == "dual_mitm_hybrid" and res["dual_hybrid"]["rop"] < result["rop"]:
                continue
            Logging.print("estimator", int(quiet), f"{algo:20s} :: {result!r}")
        return res


estimate = Estimate()
