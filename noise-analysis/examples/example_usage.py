# examples/example_usage.py
from noise.tools import create_distribution, theoretical_Y_stats
from math import sqrt

def equivalent_sigma_from_mean(dist, n):
    """返回匹配 E[Y] 的等效高斯标准差 σ_eq = sqrt(E[Y] / n)"""
    EY, _, _, _ = theoretical_Y_stats(dist, n)
    return sqrt(EY / n)

def main():
    n = 256
    sqrt_n = sqrt(n)      # 约 16.0
    quantiles = [0.05, 0.01, 0.001]

    distributions = {
        "离散高斯 σ=3.0": create_distribution('discrete_gaussian', sigma=3.0),
        "中心二项 η=8": create_distribution('centered_binomial', eta=8),
        "均匀 U(-3,3)": create_distribution('uniform', a=-3, b=3),
        "三元": create_distribution('ternary'),
        "二元": create_distribution('binary'),
        "TUniform b=2": create_distribution('tuniform', b=2),
    }

    print(f"向量维度 n = {n}\n")
    for name, dist in distributions.items():
        print("=" * 60)
        print(f"分布: {name}")
        print(f"  分量均值 = {dist.mean:.4f},  分量标准差 = {dist.stddev:.4f}")

        # 理论统计量
        EY, _, _, quant_edge = theoretical_Y_stats(dist, n)

        # 期望 2‑范数
        expected_norm = sqrt(EY)
        sigma_mean = equivalent_sigma_from_mean(dist, n)
        print(f"  期望 2-范数 (E[‖x‖]) ≈ sqrt(E[Y]) = {expected_norm:.4f}")
        print(f"  基于期望的等效高斯标准差 σ_eq(mean) = {sigma_mean:.4f}")

        # 分位数 2‑范数及其对应的等效高斯标准差（使高斯期望等于该分位数）
        print("  原分布 2-范数分位数 (Edgeworth 修正) 及 等效高斯标准差:")
        for q in quantiles:
            y_q = quant_edge[q]
            norm_q = sqrt(y_q)
            # 等效高斯：sigma = norm_q / sqrt(n)
            sigma_eq = norm_q / sqrt_n
            print(f"    q={q:.3f} : 2-范数={norm_q:.4f}  等效高斯 σ = {sigma_eq:.4f}")
        print()

if __name__ == "__main__":
    main()
