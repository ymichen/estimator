#!/usr/bin/env python3
"""评估用户自定义的7个SIS测试参数"""

import sys
from pathlib import Path
# 确保项目根目录在路径中
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import math
from src.estimator import find_optimal_b_raw

# 参数定义：(名称, q, n, k, l, zeta)
CUSTOM_PARAMS = [
    ("Param1", 8380417, 256, 4, 9, 380930),
    ("Param2", 8380417, 256, 6, 12, 1048576),
    ("Param3", 8380417, 256, 8, 16, 1048576),
    ("Param4", "sa", 256, 6, 12, 1048576),      # 无效 q
    ("Param5", 8380417, 256, 3, 7, 1048576),
    ("Param6", 8380417, 1024, 4, 9, 1048576),
    ("Param7", 3870721, 1024, 4, 8, 380930),
]

def evaluate_param(name, q, n, k, l, zeta):
    """对一组SIS参数进行评估并打印结果"""
    try:
        # 参数有效性检查
        if not isinstance(q, int) or q <= 0:
            raise ValueError(f"Invalid modulus q = {q}")

        m = n * k
        logq = math.log2(q)
        w_max = n * (l + 1)          # 无压缩场景，矩阵 [A | t]

        print(f"\n=== {name} ===")
        print(f"  q = {q}, n = {n}, k = {k}, l = {l}, zeta = {zeta}")
        print(f"  m = {m}, log2(q) = {logq:.2f}, w_max = {w_max}")

        b, w, cost = find_optimal_b_raw(m, logq, zeta, w_max)

        if b is None:
            print("  No feasible attack found (all probabilities too low).")
        else:
            print(f"  Optimal β* = {b}, w* = {w}, log2 cost = {cost:.2f} (≈2^{cost:.1f})")

    except Exception as e:
        print(f"\n=== {name} === (ERROR)")
        print(f"  Could not evaluate: {e}")

if __name__ == "__main__":
    for params in CUSTOM_PARAMS:
        evaluate_param(*params)
