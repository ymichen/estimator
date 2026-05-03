# noise-equiv-gaussian

将各种 LWE 噪声分布（离散高斯、中心二项、均匀分布、三元等）的平方范数尾部特性转换为**等价高斯分布**。
等价高斯是指：一个均值为 0、标准差为 σ_eq 的连续高斯分布，其欧氏范数平方 Σ X_i² 的左尾分位数与原始分布相同。
该工具基于 Edgeworth (Cornish-Fisher) 展开，提供了比纯正态近似更准确的尾部估计。

## 安装

1. 建议在虚拟环境中操作：
   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/Mac
   venv\Scripts\activate      # Windows
   ```

2. 安装依赖

``` pip install -r requirements.txt
```

## 使用示例
```from noise.tools import create_distribution, equivalent_gaussian
```

# 创建一个中心二项分布（η=8）
```dist = create_distribution('centered_binomial', eta=8)
```

# 计算 n=256 时，在分位 0.001, 0.01, 0.05 下的等价高斯标准差
```
eq_sigmas = equivalent_gaussian(dist, n=256, quantiles=[0.001, 0.01, 0.05])
print(eq_sigmas)  # {0.001: 2.xx, 0.01: 2.xx, 0.05: 2.xx}
```

更多示例运行 examples/example_usage.py
```
python examples/example_usage.py
```

支持分布
分布名称	参数	说明
discrete_gaussian	sigma, mean	离散高斯（截断至 ±6σ）
continuous_gaussian	stddev, mean	连续高斯
centered_binomial	eta	中心二项分布
uniform	a, b	离散均匀分布 U(a,b)
ternary	无	{-1,0,1} 等概率
binary	无	{0,1} 等概率
tuniform	b	TUniform，范围 [-2^b,2^b]
