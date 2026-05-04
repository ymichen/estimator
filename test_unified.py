#!/usr/bin/env python3
"""统一测试脚本（带警告输出）"""

import json
from pathlib import Path
from unified_interface import evaluate_single

def main():
    json_path = Path(__file__).resolve().parent / "test_parameters.json"
    with open(json_path) as f:
        test_cases = json.load(f)

    for tcase in test_cases:
        r = evaluate_single(tcase)
        res = {"name": tcase.get("name", "unnamed"), **r}
        name = res["name"]
        print(f"\n{'='*60}")
        print(name)
        if "error" in res:
            print(f"  错误: {res['error']}")
            continue

        if res.get("type") == "msis":
            for model_label, attacks in res.get("attacks", {}).items():
                print(f"  {model_label} 结果:")
                for atk_name, data in attacks.items():
                    beta = data["beta"]
                    log2 = data["log2_rop"]
                    warning = data.get("warning")
                    msg = ""
                    if beta is not None and log2 is not None:
                        msg = f"    {atk_name:12s}: β={beta:4d}, log₂(rop)={log2:7.2f}"
                    else:
                        msg = f"    {atk_name:12s}: 攻击失败"
                    if warning:
                        msg += f"  ⚠️ {warning}"
                    print(msg)
            best = res.get("best")
            if best:
                print(f"  最优攻击: {best['attack']} ({best['model']}), β={best['beta']}, log₂(rop)={best['log2_rop']:.2f}")
            else:
                print("  所有攻击均失败")
            continue

        for model_label, attacks in res.get("attacks", {}).items():
            print(f"  {model_label} 结果:")
            for atk_name, data in attacks.items():
                beta = data["beta"]
                log2 = data["log2_rop"]
                warning = data.get("warning")
                msg = ""
                if beta is not None and log2 is not None:
                    msg = f"    {atk_name:12s}: β={beta:4d}, log₂(rop)={log2:7.2f}"
                else:
                    msg = f"    {atk_name:12s}: 攻击失败"
                if warning:
                    msg += f"  ⚠️ {warning}"
                print(msg)
        best = res.get("best")
        if best:
            print(f"  最优攻击: {best['attack']} ({best['model']}), β={best['beta']}, log₂(rop)={best['log2_rop']:.2f}")
        else:
            print("  所有攻击均失败")

if __name__ == "__main__":
    main()
