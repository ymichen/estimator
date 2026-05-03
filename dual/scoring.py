"""评分函数：阈值 T、Pgood、Pwrong 估计，使用 scipy 替代 Sage"""

import math
from math import log, exp, sqrt, pi, erf
import scipy.special as sp
import scipy.integrate as integrate

def survival_normal(T, N):
    """正态尾概率的对数（以2为底）"""
    if N <= 0:
        return -float('inf')
    z = T / sqrt(N)
    p = 0.5 * math.erfc(z / sqrt(2.0))
    if p <= 0:
        return -float('inf')
    return math.log2(p)

def find_relative_threshold(alpha, q, m, nenu, nlat, nfft, kfft, beta0, beta1, dlat, avg_dlsc, sdv_dlsc):
    """正确猜测时的期望分数（相对值）"""
    # 第一个积分 (dlat 部分)
    def integrand1(t):
        return (t ** (beta1 - 1)) * math.exp(-alpha * (pi * t * dlat / q) ** 2)
    I1, _ = integrate.quad(integrand1, 0, 1)
    res = beta1 * I1
    # 高斯衰减部分
    num = math.exp(-alpha * (pi * avg_dlsc / q) ** 2 / (1 + 2 * alpha * (pi * sdv_dlsc / q) ** 2))
    den = math.sqrt(1 + 2 * alpha * (pi * sdv_dlsc / q) ** 2)
    res *= num / den
    return float(res)

def find_relative_threshold_tmp(alpha, q, m, nenu, nlat, nfft, kfft, beta0, beta1, dlat, dlsc):
    """没有实验极码时使用的阈值（直接用 dlsc 常数）"""
    def integrand1(t):
        return (t ** (beta1 - 1)) * math.exp(-alpha * (pi * t * dlat / q) ** 2)
    I1, _ = integrate.quad(integrand1, 0, 1)
    res = beta1 * I1
    def integrand2(t):
        return (t ** (nfft - 1)) * math.exp(-alpha * (pi * t * dlsc / q) ** 2)
    I2, _ = integrate.quad(integrand2, 0, 1)
    res *= nfft * I2
    return float(res)

def BesselY(alpha, x):
    """Bessel 相关函数 Y(alpha, x) = Γ(α+1) * J_α(x) / (x/2)^α"""
    if x < 1e-100:
        return 1.0
    return float(sp.gamma(alpha + 1) * sp.jv(alpha, x) / ((x / 2) ** alpha))

from .cost import cost_sample       # 避免循环导入，在函数内导入
from .utils import root_Hermite, compute_p0
from estimator.reduction import RC, MATZOV

def sv_D(T, N, q, m, alpha, nenu, nlat, nfft, kfft, beta1, beta0, dlat, avg_dlsc, sdv_dlsc, red_cost_model):
    """
    Pwrong 的主要贡献（由 D 分布导致），已用 scipy 实现。
    与原始 Sage 代码功能一致。
    """
    T, N = float(T), float(N)
    q = float(q)
    beta1, beta0 = int(beta1), int(beta0)
    dlat, avg_dlsc, sdv_dlsc = float(dlat), float(avg_dlsc), float(sdv_dlsc)

    # 计算 bi (i 的上限)
    bi_inf = 0.0
    bi_sup = sqrt(beta1) * q / 2
    bi_cur = (bi_inf + bi_sup) / 2
    while bi_sup - bi_inf > 1e-8:
        if BesselY(beta1 / 2, (2 * pi * dlat * bi_cur) / q) >= T / N:
            bi_inf = bi_cur
        else:
            bi_sup = bi_cur
        bi_cur = (bi_inf + bi_sup) / 2
    bi = bi_sup

    # 计算 VolLat
    log_vol_lat = -beta1 * m / (m + nlat) * math.log2(q) + beta1 * (m + nlat - beta1) * math.log2(root_Hermite(beta0))
    log_vol_lsc = -kfft * math.log2(q)
    log_vol_total = log_vol_lat + log_vol_lsc

    def integrand_outer(dlsc):
        # 计算 sv_D_Sphere(dlsc)
        def func_j(ii):
            Bi = BesselY(beta1 / 2, (2 * pi * dlat * ii) / q)
            Thres = T / N / Bi
            j_inf = 0.0
            j_sup = sqrt(nfft) * q / 2
            j_cur = (j_inf + j_sup) / 2
            while j_sup - j_inf > 1e-8:
                if BesselY(nfft / 2 - 1, (2 * pi * dlsc * j_cur) / q) >= Thres:
                    j_inf = j_cur
                else:
                    j_sup = j_cur
                j_cur = (j_inf + j_sup) / 2
            return j_sup

        # 积分 i 从 0 到 bi
        res_val = 0.0
        step = bi / 48.0
        ii = step / 2.0
        while ii < bi:
            jj = func_j(ii)
            # VolSphere_i
            log_vol_ci = math.log2(sp.gamma(beta1 / 2)) if beta1 > 0 else 0
            vol_sphere_i_log = 1 + (beta1 / 2) * math.log2(pi) + (beta1 - 1) * math.log2(ii) - log_vol_ci
            # VolBall_j
            vol_ball_j_log = (nfft / 2) * math.log2(pi) + nfft * math.log2(jj) - math.log2(sp.gamma(nfft / 2 + 1))
            vol_wrong_log = vol_sphere_i_log + vol_ball_j_log
            res_val += step * (2 ** vol_wrong_log)
            ii += step
        return min(0.0, log_vol_total + math.log2(res_val))

    # 外层积分 over dlsc
    norm = 0.0
    res = 0.0
    step = (3.0 * sdv_dlsc) / 25.0
    dlsc = avg_dlsc - 3.0 * sdv_dlsc + step / 2.0
    while dlsc < avg_dlsc + 3.0 * sdv_dlsc:
        prob = math.exp(-0.5 * ((dlsc - avg_dlsc) / sdv_dlsc) ** 2) / (sdv_dlsc * sqrt(2 * pi))
        norm += prob * step
        res += step * prob * (2 ** integrand_outer(dlsc))
        dlsc += step
    return math.log2(res / norm)
