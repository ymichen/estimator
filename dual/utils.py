"""数学工具函数（纯 Python，无 Sage）"""

import math
from math import log, exp, sqrt, pi, e, gamma, erf
from functools import lru_cache

def root_Hermite(b):
    """计算 BKZ-β 的根 Hermite 因子 δ"""
    b = float(b)
    return ((b/(2*pi*e)) * ((pi*b)**(1/b))) ** (1/(2*(b-1)))

def compute_p0(alpha):
    """返回中心二项分布中零的概率 (对应 Kyber 的 alpha=2 或 3)"""
    if alpha == 1:
        return 1.0 / 2.0
    elif alpha == 2:
        return 6.0 / 16.0
    elif alpha == 3:
        return 20.0 / 64.0
    else:
        raise ValueError("alpha must be 1, 2, or 3")

@lru_cache(maxsize=None)
def binom(n, k):
    """计算组合数"""
    if k < 0 or k > n:
        return 0
    k = min(k, n - k)
    result = 1
    for i in range(1, k + 1):
        result = result * (n - i + 1) // i
    return result

def compute_eta_max(alpha, nenu, nfft):
    """计算最小 η 上限（当 R → ∞ 时）"""
    p0 = compute_p0(alpha)
    eta = 1.0
    binom_val = 1.0
    prob = (1.0 - p0) ** (nenu + nfft)
    for t in range(nenu):
        eta -= binom_val * prob
        binom_val = binom_val * (nenu + nfft - t) // (t + 1)
        prob *= p0 / (1.0 - p0)
    return eta

def compute_eta(R, alpha, nenu, nfft):
    """计算实际 η(R) —— 已修正，与原 Sage 代码完全一致"""
    p0 = compute_p0(alpha)
    eta = 1.0
    binom_val = 1.0
    prob = (1.0 - p0) ** (nenu + nfft)
    for t in range(nenu):
        eta -= binom_val * prob
        binom_val = binom_val * (nenu + nfft - t) // (t + 1)
        prob *= p0 / (1.0 - p0)

    binom_bis = 1.0 / binom(nenu + nfft, nenu)
    for t in range(nenu, nenu + nfft + 1):
        # 使用 float 计算概率幂
        prob_term = (1.0 - binom_bis) ** R if binom_bis < 1.0 else 0.0
        eta -= prob_term * binom_val * prob
        binom_bis = binom_bis * (t + 1) / (t + 1 - nenu)
        if binom_bis > 1.0:
            binom_bis = 1.0
        binom_val = binom_val * (nenu + nfft - t) // (t + 1)
        prob *= p0 / (1.0 - p0)
    return float(eta)
