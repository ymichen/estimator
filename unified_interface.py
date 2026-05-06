#!/usr/bin/env python3
"""统一评估接口：根据输入类型分发到不同评估模块，返回标准化 JSON（带友好错误与 max_beta 警告）"""

import math
import json
from functools import partial

from estimator import ND, RC
from estimator.lwe_parameters import LWEParameters
from estimator.lwe_primal import primal_usvp, primal_bdd, primal_hybrid
from estimator.lwe_bkw import coded_bkw
from estimator.conf import max_beta as MAX_BETA

from dual.attack import optimize_attack as dual_optimize
from lwr_to_lwe import initialize_from_LWR_instance
from msis.estimator import find_optimal_b_raw as msis_evaluate

COST_MODELS = {
    "C0": RC.ADPS16,
    "CN": RC.Kyber,
}

MSIS_C0_CONSTANT = 0.292
MSIS_CN_CONSTANT = 0.292


# ---------- 通用参数校验函数 ----------
def _require_positive_ints(case, *keys):
    """
    检查 case 中所有指定的键存在且为正整数。
    若任一不合法，抛出 ValueError 并指明具体参数。
    """
    for k in keys:
        v = case.get(k)
        if not isinstance(v, int) or v <= 0:
            raise ValueError(f"parameter '{k}' must be a positive integer, got {v}")


def _require_dist_type(case, key, allowed=("centered_binomial", "uniform")):
    """检查分布类型是否在允许列表中"""
    v = case.get(key, "centered_binomial")
    if v not in allowed:
        raise ValueError(f"parameter '{key}' must be one of {allowed}, got {v}")


# ---------- 原有辅助函数 ----------
def _make_dist(chi, dist_type, n):
    """根据分布类型和参数创建 NoiseDistribution，带参数检查"""
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if dist_type == "centered_binomial":
        return ND.CenteredBinomial(chi, n=n)
    elif dist_type == "uniform":
        return ND.Uniform(-chi, chi, n=n)
    else:
        raise ValueError(f"Unknown distribution type: {dist_type}")


def _run_estimator_attack(attack_fn, params, cost_model):
    """运行单个 estimator 攻击，返回 (beta, log2_rop) 或 (None, None)"""
    try:
        res = attack_fn(params, red_cost_model=cost_model)
        beta = res["beta"]
        rop = res["rop"]
        if rop <= 0 or rop == float('inf'):
            return beta, float('inf')
        return beta, float(math.log2(rop))
    except Exception:
        return None, None


# ---------- 评估主函数 ----------
def evaluate_mlwe(params):
    """评估 MLWE 参数（已标准化为 LWE 参数），返回 JSON 兼容的字典"""
    results = {"type": "mlwe", "attacks": {}}

    for model_label, cost_model in COST_MODELS.items():
        results["attacks"][model_label] = {}
        attacks = {
            "usvp": partial(primal_usvp, red_cost_model=cost_model),
            "bdd": partial(primal_bdd, red_cost_model=cost_model),
            "bkw": partial(coded_bkw, red_cost_model=cost_model),
            "bdd_hybrid": partial(primal_hybrid, mitm=False, babai=False, red_cost_model=cost_model),
            "bdd_mitm_hybrid": partial(primal_hybrid, mitm=True, babai=True, red_cost_model=cost_model),
        }
        for atk_name, atk_fn in attacks.items():
            beta, log2cost = _run_estimator_attack(atk_fn, params, cost_model)
            results["attacks"][model_label][atk_name] = {
                "beta": beta,
                "log2_rop": log2cost,
            }

        # Dual 攻击（当前为 MATZOV 改进版）
        beta, log2cost = dual_optimize(params, model_label)
        dual_data = {"beta": beta, "log2_rop": log2cost}
        if beta is not None and beta >= MAX_BETA - 5:
            dual_data["warning"] = f"β 达到最大搜索限制 (max_beta={MAX_BETA})，结果可能不准确"
        results["attacks"][model_label]["dual"] = dual_data

    # 计算最优攻击
    best = {"model": None, "attack": None, "beta": None, "log2_rop": float('inf')}
    for model_label, attacks in results["attacks"].items():
        for atk_name, data in attacks.items():
            if data["log2_rop"] is not None and data["log2_rop"] < best["log2_rop"]:
                best["model"] = model_label
                best["attack"] = atk_name
                best["beta"] = data["beta"]
                best["log2_rop"] = data["log2_rop"]
    results["best"] = best if best["log2_rop"] != float('inf') else None
    return results


def evaluate_mlwe_wrap(case):
    """
    处理 MLWE 输入参数：校验、展开模块维度、构造 LWEParameters，
    然后调用 evaluate_mlwe 返回评估结果。
    """
    # 校验必需参数
    _require_positive_ints(case, "q", "n", "chi_s", "chi_e")
    # 模块维度默认值：若未提供则视为标准 LWE (k=1, l=1)
    k = case.get("k", 1)
    l = case.get("l", 1)
    if not isinstance(k, int) or k <= 0:
        raise ValueError(f"parameter 'k' must be a positive integer, got {k}")
    if not isinstance(l, int) or l <= 0:
        raise ValueError(f"parameter 'l' must be a positive integer, got {l}")

    # 分布类型校验
    _require_dist_type(case, "secret_type")
    _require_dist_type(case, "error_type")

    ring_n = case["n"]
    total_n = ring_n * k
    m = case.get("m", ring_n * l)   # 默认每密文贡献 ring_n 个标量样本

    # m 若显式提供也需校验
    if not isinstance(m, int) or m <= 0:
        raise ValueError(f"parameter 'm' must be a positive integer, got {m}")

    secret_dist = _make_dist(case["chi_s"], case.get("secret_type", "centered_binomial"), total_n)
    error_dist  = _make_dist(case["chi_e"], case.get("error_type", "centered_binomial"), total_n)

    params = LWEParameters(
        n=total_n,
        q=case["q"],
        Xs=secret_dist,
        Xe=error_dist,
        m=m,
        tag=case.get("name", "mlwe")
    )
    return evaluate_mlwe(params)


def evaluate_mlwr(case):
    """评估 MLWR 参数，先转换为 LWE 再评估，带参数校验"""
    if not isinstance(case.get("p"), int) or case["p"] <= 0:
        return {"error": f"Invalid modulus p = {case.get('p')}. Must be a positive integer."}
    try:
        q_lwe, n_ring, k_lwe, l_lwe, secret_dist, error_dist = initialize_from_LWR_instance(
            case["q"], case["p"], case["n"], case["k"], case["l"], case["mu"]
        )
    except Exception as e:
        return {"error": f"LWR to LWE conversion failed: {e}"}

    n_lwe = n_ring * k_lwe
    secret_dist = secret_dist.resize(n_lwe)
    error_dist = error_dist.resize(n_lwe)

    lwe_params = LWEParameters(
        n=n_lwe, q=q_lwe,
        Xs=secret_dist, Xe=error_dist,
        m=n_lwe, tag=case.get("name", "mlwr")
    )
    return evaluate_mlwe(lwe_params)


def evaluate_msis(case):
    """评估 MSIS 参数，分别计算 C0 和 CN 两种模型，带参数校验和 max_beta 警告"""
    if not isinstance(case.get("q"), int) or case["q"] <= 0:
        return {"error": f"Invalid modulus q = {case.get('q')}. Must be a positive integer."}
    if case.get("n", 0) <= 0:
        return {"error": f"Invalid n = {case.get('n')}. Must be positive."}
    try:
        m = case["n"] * case["k"]
        logq = math.log2(case["q"])
        w_max = case["n"] * (case["l"] + 1)

        beta_c0, w_c0, cost_c0 = msis_evaluate(m, logq, case["zeta"], w_max, c_C=MSIS_C0_CONSTANT)
        beta_cn, w_cn, cost_cn = msis_evaluate(m, logq, case["zeta"], w_max, c_C=MSIS_CN_CONSTANT)

        msis_data = {
            "type": "msis",
            "attacks": {
                "C0": {"msis_lattice": {"beta": beta_c0, "log2_rop": cost_c0}},
                "CN": {"msis_lattice": {"beta": beta_cn, "log2_rop": cost_cn}},
            },
        }

        # 警告信息
        if beta_c0 is not None and beta_c0 >= MAX_BETA - 5:
            msis_data["attacks"]["C0"]["msis_lattice"]["warning"] = f"β 达到最大搜索限制 (max_beta={MAX_BETA})"
        if beta_cn is not None and beta_cn >= MAX_BETA - 5:
            msis_data["attacks"]["CN"]["msis_lattice"]["warning"] = f"β 达到最大搜索限制 (max_beta={MAX_BETA})"

        # 最优
        best = None
        costs = []
        if cost_c0 is not None: costs.append(("C0", beta_c0, cost_c0))
        if cost_cn is not None: costs.append(("CN", beta_cn, cost_cn))
        if costs:
            best_model, best_beta, best_cost = min(costs, key=lambda x: x[2])
            best = {
                "model": best_model,
                "attack": "msis_lattice",
                "beta": best_beta,
                "log2_rop": best_cost,
            }
        msis_data["best"] = best
        return msis_data
    except Exception as e:
        return {"error": f"MSIS evaluation failed: {e}"}


# ---------- 分发入口 ----------
def evaluate_single(case):
    """根据 case['type'] 分发到对应的评估函数（纯分发，不含构造逻辑）"""
    case_type = case.get("type", "").lower()
    try:
        if case_type == "mlwe":
            return evaluate_mlwe_wrap(case)

        elif case_type == "mlwr":
            return evaluate_mlwr(case)

        elif case_type == "msis":
            return evaluate_msis(case)

        else:
            return {"error": f"Unknown type: {case_type}"}

    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Unexpected error: {e}"}


def evaluate_all(test_cases):
    """批量评估，返回结果列表"""
    return [{"name": case.get("name", "unnamed"), **evaluate_single(case)} for case in test_cases]
