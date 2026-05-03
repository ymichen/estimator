# dual/cost.py
"""成本计算（短向量采样、FFT、极码解码）"""

import math
from estimator.reduction import RC, MATZOV

C_mul = 1024.0
C_add = 160.0

# FFT 常数（针对 q=3329）
N_FFT_mul = 115928
N_FFT_add = 240500

def get_reduction_cost_model(nn):
    matzov_nns = {
        "CN": "list_decoding-naive_classical",
        "CC": "list_decoding-classical",
    }
    if nn in matzov_nns:
        return MATZOV(nn=matzov_nns[nn])
    elif nn == "C0":
        return RC.ADPS16
    else:
        raise ValueError(f"Unknown cost model: {nn}")

def cost_sample(m, beta0, beta1, N, red_cost_model):
    """返回 (rho, T_sample, beta1_out)，T_sample 是原始运算量"""
    rho, T, _, beta1_out = red_cost_model.short_vectors(
        beta=beta0, N=N, d=m, sieve_dim=beta1
    )
    return float(rho), float(T), int(beta1_out)

def cost_FFT(q, kfft):
    """FFT 阶段原始运算量（非 log2）"""
    if q == 3329:
        ops_per_fft = C_mul * N_FFT_mul + C_add * N_FFT_add
    else:
        # 通用估计：用 Cooley-Tukey 复杂度
        log_q = math.log2(q)
        n_add = 4 * q * log_q
        n_mul = 2 * q * log_q
        ops_per_fft = C_mul * n_mul + C_add * n_add
    total_ops = ops_per_fft * kfft * (q ** (kfft - 1))
    return total_ops

def cost_decode(q, nfft, L_size=1):
    """极码解码原始运算量（非 log2）"""
    # 下一个2的幂
    nfft_pow2 = 1 << (nfft.bit_length())   # 等价于 2^ceil(log2(nfft))
    node_cost = cost_FFT(q, 1)             # 一次一维 FFT 的原始运算量
    total_ops = 3 * L_size * node_cost * nfft_pow2 * math.log2(nfft_pow2)
    return total_ops
