# src/cli.py
import argparse
import sys
import importlib.util
import json
from src.params import ALL_BUILTIN, DilithiumParams
from src.estimator import evaluate

def load_params_from_module(filepath: str):
    """从 Python 文件加载 'params' 变量（DilithiumParams 实例或列表）"""
    spec = importlib.util.spec_from_file_location("custom_params", filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if hasattr(module, 'params'):
        return module.params
    raise ValueError("Custom module must define a 'params' variable (DilithiumParams or list of them)")

def load_params_from_json(filepath: str):
    """从 JSON 文件加载参数（单个对象或数组）"""
    with open(filepath) as f:
        data = json.load(f)
    if isinstance(data, list):
        return [DilithiumParams.from_dict(item) for item in data]
    return DilithiumParams.from_dict(data)

def print_results(params: DilithiumParams):
    print(f"\n=== {params.name} ===")
    res = evaluate(params)
    for sc, data in res.items():
        if data['b'] is not None:
            print(f"  {sc}: β*={data['b']}, w*={data['w']}, log2 cost = {data['log2_cost']:.2f} (≈2^{data['log2_cost']:.1f})")
        else:
            print(f"  {sc}: no feasible attack found")

def main():
    parser = argparse.ArgumentParser(description="Dilithium MSIS Security Estimator")
    parser.add_argument('--level', type=str, help='内置级别 (2,3,5,1--,1-,5+,5++)')
    parser.add_argument('--all', action='store_true', help='评估所有内置级别')
    parser.add_argument('--params-file', type=str, help='自定义参数 Python 文件')
    parser.add_argument('--json-file', type=str, help='自定义参数 JSON 文件')
    args = parser.parse_args()

    params_list = []

    # 1. 自定义 Python 文件优先
    if args.params_file:
        p = load_params_from_module(args.params_file)
        params_list = p if isinstance(p, list) else [p]
    # 2. JSON 文件
    elif args.json_file:
        p = load_params_from_json(args.json_file)
        params_list = p if isinstance(p, list) else [p]
    # 3. 所有内置
    elif args.all:
        params_list = list(ALL_BUILTIN.values())
    # 4. 单个内置级别
    elif args.level:
        if args.level not in ALL_BUILTIN:
            print(f"Unknown level: {args.level}. Available: {list(ALL_BUILTIN.keys())}")
            sys.exit(1)
        params_list = [ALL_BUILTIN[args.level]]
    # 5. 默认 Level 2
    else:
        params_list = [ALL_BUILTIN['2']]

    for params in params_list:
        print_results(params)

if __name__ == "__main__":
    main()
