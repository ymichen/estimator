# msis/evaluate_dilithium.py
"""Dilithium 专用评估：UF‑CMA 无压缩 / 带压缩 / SUF‑CMA"""

import math
from .estimator import find_optimal_b_raw

def evaluate_params(params):
    """
    输入：一个 DilithiumParams 对象
    输出：一个字典，包含三个场景的评估结果。
    """
    q = params.q
    n = params.n
    k = params.k
    l = params.l
    m = params.m

    results = {}

    # 1. UF‑CMA 无压缩
    zeta_unc = params.zeta_unc
    w_max_unc = params.w_max_unc
    b_unc, w_unc, cost_unc = find_optimal_b_raw(m, params.logq, zeta_unc, w_max_unc)
    results["UF-CMA_uncompressed"] = {
        "beta": b_unc,
        "w": w_unc,
        "log2_cost": cost_unc,
        "zeta": zeta_unc,
        "w_max": w_max_unc,
    }

    # 2. UF‑CMA 带压缩
    zeta_comp = params.zeta_comp
    w_max_comp = params.w_max_comp
    b_comp, w_comp, cost_comp = find_optimal_b_raw(m, params.logq, zeta_comp, w_max_comp)
    results["UF-CMA_compressed"] = {
        "beta": b_comp,
        "w": w_comp,
        "log2_cost": cost_comp,
        "zeta": zeta_comp,
        "w_max": w_max_comp,
    }

    # 3. SUF‑CMA
    zeta_suf = params.zeta_suf
    w_max_suf = params.w_max_suf
    b_suf, w_suf, cost_suf = find_optimal_b_raw(m, params.logq, zeta_suf, w_max_suf)
    results["SUF-CMA"] = {
        "beta": b_suf,
        "w": w_suf,
        "log2_cost": cost_suf,
        "zeta": zeta_suf,
        "w_max": w_max_suf,
    }

    return results



def evaluate_scenarios(params):
    """
    params: DilithiumParams 实例
    返回一个字典，包含三个场景的评估结果 (beta, log2_cost)
    """
    m = params.m
    logq = params.logq

    results = {}

    # 场景 1：UF‑CMA 无压缩
    b, w, cost = find_optimal_b_raw(m, logq, params.zeta_unc, params.w_max_unc)
    results["UF‑CMA uncompr."] = {"beta": b, "log2_cost": cost}

    # 场景 2：UF‑CMA 带压缩
    b, w, cost = find_optimal_b_raw(m, logq, params.zeta_comp, params.w_max_comp)
    results["UF‑CMA compr."]   = {"beta": b, "log2_cost": cost}

    # 场景 3：SUF‑CMA
    b, w, cost = find_optimal_b_raw(m, logq, params.zeta_suf, params.w_max_suf)
    results["SUF‑CMA"]        = {"beta": b, "log2_cost": cost}

    return results
