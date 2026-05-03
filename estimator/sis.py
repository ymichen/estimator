# -*- coding: utf-8 -*-
"""
High-level SIS interface. Pure Python, no Sage.
"""

from functools import partial

from .sis_lattice import lattice
from .sis_parameters import SISParameters as Parameters  # noqa
from .conf import red_cost_model as red_cost_model_default, red_shape_model as red_shape_model_default
from .util import batch_estimate, f_name
from .reduction import RC
from .io import Logging

OO = float('inf')


class Estimate:
    def rough(self, params, jobs=1, catch_exceptions=True, quiet=False):
        algorithms = {}
        algorithms["lattice"] = partial(lattice, red_cost_model=RC.ADPS16, red_shape_model="lgsa")
        res_raw = batch_estimate(params, algorithms.values(), log_level=1, jobs=jobs, catch_exceptions=catch_exceptions)
        res_raw = res_raw[params]
        res = {algo: v for algo, attack in algorithms.items() for k, v in res_raw.items() if f_name(attack) == k}
        for algo in algorithms:
            if algo not in res:
                continue
            result = res[algo]
            if result["rop"] != OO:
                Logging.print("estimator", int(quiet), f"{algo:8s} :: {result!r}")
        return res

    def __call__(self, params, red_cost_model=red_cost_model_default, red_shape_model=red_shape_model_default,
                 deny_list=tuple(), add_list=tuple(), jobs=1, catch_exceptions=True, quiet=False):
        algorithms = {}
        algorithms["lattice"] = partial(lattice, red_cost_model=red_cost_model, red_shape_model=red_shape_model)
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
            Logging.print("estimator", int(quiet), f"{algo:8s} :: {result!r}")
        return res


estimate = Estimate()
