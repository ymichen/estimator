# lwr_to_lwe.py
import math
from math import comb

from estimator.nd import (
    NoiseDistribution,
    DiscreteGaussian,
    CenteredBinomial,
    Uniform,
    UniformMod,
    SparseTernary,
    SparseBinary,
    Binary,
    Ternary,
)


# ---------------------- 基础函数 ----------------------
def mod_switch(x: int, q: int, p: int) -> int:
    """从模 q 转换到模 p 并取整"""
    return round(p * x / q)


def mod_centered(x: int, q: int) -> int:
    """将整数 x 中心化到 [-q/2, q/2] 区间内"""
    x = x % q
    if x > q // 2:
        x -= q
    return x


def build_mod_switching_error_law(q: int, p: int) -> dict:
    """
    构建模数切换 (round-trip) 引入的舍入误差分布。
    假设输入 x 在 [0, q-1] 均匀分布。
    返回 {error_value: probability} 字典。
    """
    D = {}
    for x in range(q):
        y = mod_switch(x, q, p)      # q -> p
        z = mod_switch(y, p, q)      # p -> q
        d = mod_centered(x - z, q)   # 舍入误差
        D[d] = D.get(d, 0) + 1.0 / q
    return D


def law_mean_var(D: dict) -> tuple:
    """计算分布的均值与方差"""
    mean = sum(v * prob for v, prob in D.items())
    var = sum((v - mean) ** 2 * prob for v, prob in D.items())
    return mean, var


def law_convolution(D1: dict, D2: dict) -> dict:
    """两个离散分布卷积，用于合并额外噪声（假设额外噪声也是离散字典）"""
    D = {}
    for v1, p1 in D1.items():
        for v2, p2 in D2.items():
            v = v1 + v2
            D[v] = D.get(v, 0) + p1 * p2
    return D


# ---------------------- 主要转换函数 ----------------------
def initialize_from_LWR_instance(
    q: int,
    p: int,
    n: int,
    k: int,
    l: int,
    mu_or_dist,  # 可以是整数 μ（中心二项分布）或 NoiseDistribution 对象
    extra_noise: NoiseDistribution = None,
):
    """
    将 LWR 实例转换为等效 LWE 实例。

    参数:
        q, p       : 模数
        n          : 环维度 (Saber 固定 256)
        k, l       : 矩阵维度 (通常 k = l)
        mu_or_dist : 秘密系数的分布，可以是整数 μ（CenteredBinomial(μ)）或
                     NoiseDistribution 的实例
        extra_noise: 如果方案中还有额外的显式 LWE 噪声，可传入其分布（默认 None）

    返回:
        q, n, k, l, secret_dist, error_dist
        其中:
          - n 是环维度 (256)
          - secret_dist 是 CenteredBinomial 对象
          - error_dist 是一个 DiscreteGaussian 对象，标准差等于等效误差的标准差
    """
    # 1. 解析秘密分布
    if isinstance(mu_or_dist, int):
        secret_dist = CenteredBinomial(mu_or_dist)
    elif isinstance(mu_or_dist, NoiseDistribution):
        secret_dist = mu_or_dist
    else:
        raise TypeError(f"mu_or_dist must be int or NoiseDistribution, got {type(mu_or_dist)}")

    # 2. 计算舍入误差分布及其方差
    rounding_law = build_mod_switching_error_law(q, p)
    var_round = law_mean_var(rounding_law)[1]

    # 3. 如果有额外噪声，则总方差 = 舍入误差方差 + 额外噪声方差
    if extra_noise is not None:
        var_extra = extra_noise.stddev ** 2
        total_var = var_round + var_extra
    else:
        total_var = var_round

    # 4. 构造等效的误差分布：离散高斯，标准差 = sqrt(total_var)
    error_std = math.sqrt(total_var)
    error_dist = DiscreteGaussian(error_std)

    return q, n, k, l, secret_dist, error_dist


# 测试代码
if __name__ == "__main__":
    q, p = 2**13, 2**10

    # 测试整數输入
    for name, k_val, l_val, mu_val in [
        ("LightSaber", 2, 2, 10),
        ("Saber",      3, 3, 8),
        ("FireSaber",  4, 4, 6),
    ]:
        q_lwe, n_lwe, k_lwe, l_lwe, s_dist, e_dist = initialize_from_LWR_instance(
            q, p, 256, k_val, l_val, mu_val
        )
        print(f"{name}:")
        print(f"  秘密分布: {s_dist}, 标准差={s_dist.stddev:.3f}")
        print(f"  误差分布: {e_dist}, 标准差={e_dist.stddev:.3f}\n")

    # 测试分布对象输入
    from estimator.nd import CenteredBinomial
    cb = CenteredBinomial(8)
    q_lwe2, n_lwe2, k_lwe2, l_lwe2, s_dist2, e_dist2 = initialize_from_LWR_instance(
        q, p, 256, 3, 3, cb
    )
    print(f"分布对象输入: secret={s_dist2}, error={e_dist2}")

