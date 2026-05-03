"""Dual attack 主流程：参数优化与复杂度计算（完全实时）"""

import math
from .cost import cost_sample, cost_FFT, cost_decode, get_reduction_cost_model
from .scoring import (
    find_relative_threshold, find_relative_threshold_tmp,
    survival_normal, sv_D
)
from .utils import compute_eta, compute_eta_max, root_Hermite, compute_p0

def complexity(alpha, q, m, nenu, nlat, nfft, kfft, red_cost_model,
               target_proba_false_candidate_global=0.05,
               target_proba_senu=0.6,
               option_dlsc={'ratio_GV': 1.0}):
    """
    计算一组参数的攻击总代价（log2 complexity）。
    option_dlsc 可含 'experimental_Clsc': True/False 等。
    """
    q = int(q); m = int(m)
    nenu, nlat, nfft, kfft = int(nenu), int(nlat), int(nfft), int(kfft)

    if option_dlsc.get('experimental_Clsc', False):
        avg_dlsc = option_dlsc['code']['mean_decoding_norm']
        sdv_dlsc = option_dlsc['code']['sigma_decoding_norm']
    else:
        ratio = option_dlsc.get('ratio_GV', 1.0)
        avg_dlsc = ratio * ((q ** (1 - kfft / nfft)) *
                            (math.gamma(nfft / 2 + 1) ** (1 / nfft)) / math.sqrt(math.pi))
        sdv_dlsc = None
        dlsc = avg_dlsc * (nfft + 1) / nfft

    p0 = compute_p0(alpha)

    # 计算所需的 R（迭代次数）和 η
    R_min = max(1, int(2 * p0**(-nenu)))
    R_max = max(R_min, 2**100)
    eta_max = compute_eta_max(alpha, nenu, nfft)
    if eta_max < target_proba_senu:
        return math.inf, 0, 0

    # 二分搜索 R
    R_low, R_high = R_min, R_max
    R = (R_low + R_high) // 2
    while R_high - R_low > 1:
        eta = compute_eta(R, alpha, nenu, nfft)
        if eta < target_proba_senu:
            R_low = R
        else:
            R_high = R
        R = (R_low + R_high) // 2
    R = R_high
    eta = compute_eta(R, alpha, nenu, nfft)

    target_Pwrong = target_proba_false_candidate_global / (R * q ** kfft)

    # 二分搜索 beta0
    beta0_min, beta0_max = 200, 2200
    beta0 = (beta0_min + beta0_max) // 2
    while beta0_max - beta0_min > 1:
        rho, _, beta1 = cost_sample(m + nlat, beta0, None, None, red_cost_model)
        avg_dlat = rho * (q ** (nlat / (m + nlat))) * (root_Hermite(beta0) ** (m + nlat - 1))
        dlat = avg_dlat * (beta1 + 1) / beta1

        if option_dlsc.get('experimental_Clsc', False):
            thresh = find_relative_threshold(alpha, q, m, nenu, nlat, nfft, kfft, beta0, beta1, dlat, avg_dlsc, sdv_dlsc)
        else:
            thresh = find_relative_threshold_tmp(alpha, q, m, nenu, nlat, nfft, kfft, beta0, beta1, dlat, dlsc)

        N = (math.sqrt(4/3)) ** beta1
        if survival_normal(N * thresh, N) > math.log2(target_Pwrong):
            beta0_min = beta0
        else:
            beta0_max = beta0
        beta0 = (beta0_min + beta0_max) // 2
    beta0 = beta0_max

    # 最终计算
    rho, T_sample, beta1 = cost_sample(m + nlat, beta0, None, None, red_cost_model)
    avg_dlat = rho * (q ** (nlat / (m + nlat))) * (root_Hermite(beta0) ** (m + nlat - 1))
    dlat = avg_dlat * (beta1 + 1) / beta1
    if option_dlsc.get('experimental_Clsc', False):
        thresh = find_relative_threshold(alpha, q, m, nenu, nlat, nfft, kfft, beta0, beta1, dlat, avg_dlsc, sdv_dlsc)
    else:
        thresh = find_relative_threshold_tmp(alpha, q, m, nenu, nlat, nfft, kfft, beta0, beta1, dlat, dlsc)
    N = (math.sqrt(4/3)) ** beta1
    T_FFT = cost_FFT(q, kfft)
    T_decode = N * cost_decode(q, nfft)
    T_total = T_sample + R * (T_decode + T_FFT)
    return float(math.log2(T_total)), int(beta0), int(beta1)

def optimize_attack(params, cost_model_label="CC"):
    """对任意 LWE 参数进行 dual 攻击参数搜索（失败返回 None）"""
    n = params.n
    # 自适应 alpha：对中心二项分布 β_μ，alpha 就是 μ 的值
    # 这里简单地从 Xs 的标准差反推（CenteredBinomial(eta) 的 stddev = sqrt(eta/2)）
    try:
        if hasattr(params.Xs, 'stddev'):
            alpha = int(round(2 * params.Xs.stddev ** 2))
        else:
            alpha = 2  # 安全默认值
    except:
        alpha = 2

    red_cost_model = get_reduction_cost_model(cost_model_label)
    q = params.q
    m_end = min(n + 100, params.m if params.m != float('inf') else n + 100)
    best_log_cost = float('inf')
    best_beta = None
    option_dlsc = {'experimental_Clsc': False, 'ratio_GV': 1.0}

    # 网格步长随 n 自适应增大
    step_nenu = max(5, n // 100)
    step_nfft = max(10, n // 50)
    step_kfft = max(3, n // 100)

    for m in range(max(1, n - 50), m_end, 20):
        for nenu in range(5, min(50, n - 10), step_nenu):
            for nfft in range(20, min(150, n - nenu), step_nfft):
                nlat = n - nenu - nfft
                if nlat <= 0:
                    continue
                for kfft in range(1, nfft // 2 + 1, step_kfft):
                    try:
                        log_cost, beta0, _ = complexity(
                            alpha, q, m, nenu, nlat, nfft, kfft,
                            red_cost_model,
                            target_proba_false_candidate_global=0.05,
                            target_proba_senu=0.6,
                            option_dlsc=option_dlsc
                        )
                        if log_cost < best_log_cost:
                            best_log_cost = log_cost
                            best_beta = beta0
                    except (ValueError, ArithmeticError):
                        continue
    if best_beta is None:
        return None, None
    return best_beta, best_log_cost



