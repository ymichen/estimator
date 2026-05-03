import math
from .params import DilithiumParams

def gsa_slope(b: int) -> float:
    if b <= 1:
        return float('inf')
    return (1.0 / (b - 1)) * math.log2(
        (b / (2 * math.pi * math.e)) * (math.pi * b) ** (1.0 / b)
    )

def log2_success_forgotten(b: int, w: int, m: float, logq: float, zeta: float) -> float:
    """遗忘 q-向量模型（纯三角形）的 log₂ 攻击成功概率"""
    s = gsa_slope(b)
    if s <= 0:
        return -float('inf')
    inside = 2.0 * s * m * logq
    if inside <= 0:
        return -float('inf')
    ell1 = math.sqrt(inside)
    w0 = ell1 / s

    # 有效维度与 sigma
    if w0 <= w:
        eff_dim = w0
        sigma = (2 ** ell1) / math.sqrt(eff_dim) if eff_dim > 0 else float('inf')
    else:
        eff_dim = w
        sigma = (2 ** ell1) / math.sqrt(w)

    # 单坐标通过概率
    p_coord = math.erf(zeta / (sigma * math.sqrt(2.0)))
    if p_coord <= 0.0:
        return -float('inf')

    # 对数概率计算
    log_p_one = eff_dim * math.log2(p_coord)
    log_N = (b / 2.0) * math.log2(4.0 / 3.0)
    log_avg = log_N + log_p_one

    # 数值稳定的成功率返回
    if log_avg < -50:
        return log_avg
    avg = 2.0 ** log_avg
    if avg < 1e-7:
        return log_avg
    if avg > 500:
        return 0.0

    p_succ = -math.expm1(-avg)
    # 确保 p_succ > 0，防止 math.log2(p_succ) 出错
    if p_succ <= 0.0:
        return -float('inf')
    return math.log2(p_succ)

def find_optimal_b(params: DilithiumParams, scenario: str, c_C: float = 0.292):
    if scenario == 'unc':
        zeta, w_max = params.zeta_unc, params.w_max_unc
    elif scenario == 'comp':
        zeta, w_max = params.zeta_comp, params.w_max_comp
    elif scenario == 'suf':
        zeta, w_max = params.zeta_suf, params.w_max_suf
    else:
        raise ValueError("Unknown scenario: choose from 'unc', 'comp', 'suf'")

    m = params.m
    logq = params.logq
    best_cost = float('inf')
    best_b = None
    best_w = None

    for w in range(400, w_max + 1, 5):
        for b in range(20, 1000, 1):
            lp = log2_success_forgotten(b, w, m, logq, zeta)
            # 跳过无效或未定义的返回值
            if lp is None or lp == -float('inf'):
                continue
            cost = c_C * b - lp
            if cost < best_cost:
                best_cost = cost
                best_b = b
                best_w = w
    return best_b, best_w, best_cost

def evaluate(params: DilithiumParams, scenarios=None):
    if scenarios is None:
        scenarios = ['unc', 'comp', 'suf']
    results = {}
    for sc in scenarios:
        b, w, cost = find_optimal_b(params, sc)
        results[sc] = {'b': b, 'w': w, 'log2_cost': cost}
    return results


def find_optimal_b_raw(m: int, logq: float, zeta: float, w_max: int,
                       c_C: float = 0.292):
    """
    直接使用 SIS 底层参数搜索最优 BKZ 分块大小。
    参数:
        m:      方程数 (256 * k)
        logq:   log2(q)
        zeta:   无穷范数界
        w_max:  最大可选择列数
        c_C:    经典 SVP 成本常数
    返回: (best_b, best_w, best_cost_log2)
    """
    best_cost = float('inf')
    best_b = None
    best_w = None

    for w in range(400, w_max + 1, 5):
        for b in range(20, 2500, 1):
            lp = log2_success_forgotten(b, w, m, logq, zeta)
            if lp is None or lp == -float('inf'):
                continue
            cost = c_C * b - lp
            if cost < best_cost:
                best_cost = cost
                best_b = b
                best_w = w
    return best_b, best_w, best_cost


def evaluate_raw(m: int, logq: float, zeta: float, w_max: int,
                 c_C: float = 0.292) -> dict:
    """原始评估的便捷函数，返回字典。"""
    b, w, cost = find_optimal_b_raw(m, logq, zeta, w_max, c_C)
    return {'b': b, 'w': w, 'log2_cost': cost}
