#!/ What you requested: a script that reads test_parameters.json,
# evaluates only MSIS entries for three security scenarios,
# and prints the results without modifying unified_interface.

import json
import math
import sys
from pathlib import Path

# 确保项目根目录在搜索路径中
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from msis.params import DilithiumParams, PARAMS_LEVEL2, PARAMS_LEVEL3, PARAMS_LEVEL5
from msis.estimator import find_optimal_b_raw

def load_test_data(json_path):
    with open(json_path) as f:
        return json.load(f)

def get_dilithium_params(case):
    """根据测试用例名称推断对应的 Dilithium 参数集。
       如果无法推断，返回 None。
    """
    name = case.get("name", "")
    # 简单关键词匹配
    if "Dilithium2" in name:
        return PARAMS_LEVEL2
    elif "Dilithium3" in name:
        return PARAMS_LEVEL3
    elif "Dilithium5" in name:
        return PARAMS_LEVEL5
    # 其他自定义测试用例无法自动推断
    return None

def evaluate_msis_three_scenes(case):
    """
    对单个 MSIS 测试用例执行三场景评估。
    如果无法获取完整的 Dilithium 参数，则只做基于 case['zeta'] 的无压缩评估。
    """
    try:
        # 尝试获取标准 Dilithium 参数
        params = get_dilithium_params(case)
        if params is not None:
            from msis.evaluate_dilithium import evaluate_scenarios
            return evaluate_scenarios(params)

        # 否则，使用 case 中给定的值进行无压缩评估
        q = case["q"]
        n = case["n"]
        k = case["k"]
        l = case["l"]
        zeta = case["zeta"]
        m = n * k
        logq = math.log2(q)
        w_max = n * (l + 1)          # 无压缩场景
        b, w, cost = find_optimal_b_raw(m, logq, zeta, w_max)
        return {"UF‑CMA uncompr.": {"beta": b, "log2_cost": cost}}

    except Exception as e:
        return {"error": str(e)}

def print_scenario_results(case_name, scenarios):
    print(f"\n=== {case_name} ===")
    if "error" in scenarios:
        print(f"  错误: {scenarios['error']}")
        return
    for scene, data in scenarios.items():
        beta = data.get("beta")
        cost = data.get("log2_cost")
        if beta is not None and cost is not None:
            print(f"  {scene:20s}: β*={beta:4d}, log₂(cost)={cost:.2f} (≈2^{cost:.1f})")
        else:
            print(f"  {scene:20s}: 攻击失败")

def main():
    json_path = ROOT / "test_parameters.json"
    cases = load_test_data(json_path)

    print("=" * 60)
    print("MSIS 三场景安全评估（UF‑CMA 无压缩 / 带压缩 / SUF‑CMA）")
    print("=" * 60)

    for case in cases:
        if case.get("type", "").lower() != "msis":
            continue
        name = case.get("name", "unknown")
        scenarios = evaluate_msis_three_scenes(case)
        print_scenario_results(name, scenarios)

if __name__ == "__main__":
    main()
