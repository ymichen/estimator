# -*- coding: utf-8 -*-
"""
High-level NTRU interface. Pure Python, no Sage.
"""

from functools import partial

from .ntru_primal import primal_dsd, primal_usvp, primal_bdd, primal_hybrid
# from .lwe_bkw import coded_bkw          # 可选，暂未转换
from .lwe_guess import exhaustive_search, mitm, distinguish, guess_composition  # noqa
from .lwe_dual import dual, dual_hybrid    # noqa
# from .gb import arora_gb                 # 可选，暂未转换
from .ntru_parameters import NTRUParameters as Parameters  # noqa
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

        algorithms["usvp"] = partial(primal_usvp, red_cost_model=RC.ADPS16, red_shape_model="zgsa")

        if params.possibly_overstretched:
            algorithms["dsd"] = partial(
                primal_dsd, red_cost_model=RC.ADPS16, red_shape_model="zgsa"
            )

        if params.Xs.is_sparse:
            algorithms["bdd_hybrid"] = partial(
                primal_hybrid,
                mitm=False,
                babai=False,
                red_cost_model=RC.ADPS16,
                red_shape_model="ZGSA",
            )

        res_raw = batch_estimate(
            params, algorithms.values(), log_level=1, jobs=jobs, catch_exceptions=catch_exceptions
        )
        res_raw = res_raw[params]
        res = {
            algorithm: v
            for algorithm, attack in algorithms.items()
            for k, v in res_raw.items()
            if f_name(attack) == k
        }

        for algorithm in algorithms:
            if algorithm not in res:
                continue
            result = res[algorithm]
            if result["rop"] != OO:
                Logging.print("estimator", int(quiet), f"{algorithm:20s} :: {result!r}")

        return res

    def __call__(
        self,
        params,
        red_cost_model=red_cost_model_default,
        red_shape_model=red_shape_model_default,
        deny_list=tuple(),
        add_list=tuple(),
        jobs=1,
        catch_exceptions=True,
        quiet=False,
    ):
        params = params.normalize()
        algorithms = {}

        algorithms["usvp"] = partial(
            primal_usvp, red_cost_model=red_cost_model, red_shape_model=red_shape_model
        )
        algorithms["dsd"] = partial(
            primal_dsd, red_cost_model=red_cost_model, red_shape_model=red_shape_model
        )

        algorithms["bdd"] = partial(
            primal_bdd, red_cost_model=red_cost_model, red_shape_model=red_shape_model
        )
        algorithms["bdd_hybrid"] = partial(
            primal_hybrid,
            mitm=False,
            babai=False,
            red_cost_model=red_cost_model,
            red_shape_model=red_shape_model,
        )
        algorithms["bdd_mitm_hybrid"] = partial(
            primal_hybrid,
            mitm=True,
            babai=True,
            red_cost_model=red_cost_model,
            red_shape_model=red_shape_model,
        )

        algorithms = {k: v for k, v in algorithms.items() if k not in deny_list}
        algorithms.update(add_list)

        res_raw = batch_estimate(
            params, algorithms.values(), log_level=1, jobs=jobs, catch_exceptions=catch_exceptions
        )
        res_raw = res_raw[params]
        res = {
            algorithm: v
            for algorithm, attack in algorithms.items()
            for k, v in res_raw.items()
            if f_name(attack) == k
        }
        for algorithm in algorithms:
            if algorithm not in res:
                continue
            result = res[algorithm]
            if result["rop"] == OO:
                continue
            if algorithm == "hybrid" and res["bdd"]["rop"] < result["rop"]:
                continue
            if algorithm == "dsd" and res["usvp"]["rop"] < result["rop"]:
                continue
            Logging.print("estimator", int(quiet), f"{algorithm:20s} :: {result!r}")

        return res


estimate = Estimate()
