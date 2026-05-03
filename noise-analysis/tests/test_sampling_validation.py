import numpy as np
from scipy.stats import norm
from math import sqrt, pi, exp, comb
from collections import namedtuple
from pathlib import Path


# ========== 基础辅助：计算 W = X^2 的矩 ==========
def _compute_w_moments_from_pmf(xs, ps):
    """给定 x 的取值和概率，返回 W = x^2 的均值、方差、偏度、超额峰度"""
    w = xs**2
    mu = np.sum(w * ps)
    var = np.sum((w - mu)**2 * ps)
    mu3 = np.sum((w - mu)**3 * ps)
    mu4 = np.sum((w - mu)**4 * ps)
    skew = mu3 / var**1.5 if var > 0 else 0.0
    exkurt = mu4 / var**2 - 3.0 if var > 0 else 0.0
    return mu, var, skew, exkurt

def _moments_discrete_gaussian_w(sigma, mean=0.0, tail=6):
    """离散高斯：枚举 x 到 ± tail*σ"""
    low = int(mean - tail * sigma)
    high = int(mean + tail * sigma)
    xs = np.arange(low, high + 1, dtype=float)
    logps = -((xs - mean) ** 2) / (2 * sigma ** 2)
    ps = np.exp(logps)
    ps /= ps.sum()
    return _compute_w_moments_from_pmf(xs, ps)

def _moments_centered_binomial_w(eta):
    """中心二项分布"""
    xs = np.arange(-eta, eta + 1, dtype=float)
    binom_pmf = np.array([comb(eta, k) * 0.5**eta for k in range(eta + 1)])
    ps = np.zeros_like(xs)
    for idx, x in enumerate(xs):
        k_min = max(0, int(x))
        k_max = min(eta, eta + int(x))
        if k_min > k_max:
            continue
        k_vals = np.arange(k_min, k_max + 1)
        ps[idx] = np.sum(binom_pmf[k_vals] * binom_pmf[k_vals - int(x)])
    return _compute_w_moments_from_pmf(xs, ps)

def _moments_uniform_w(a, b):
    """离散均匀 U(a, b)"""
    xs = np.arange(a, b + 1, dtype=float)
    ps = np.full(len(xs), 1.0 / len(xs))
    return _compute_w_moments_from_pmf(xs, ps)

def _moments_ternary_w():
    """三元 {-1, 0, 1}，等概率"""
    xs = np.array([-1., 0., 1.])
    ps = np.array([1./3, 1./3, 1./3])
    return _compute_w_moments_from_pmf(xs, ps)

def _moments_binary_w():
    """二元 {0, 1}，等概率"""
    xs = np.array([0., 1.])
    ps = np.array([0.5, 0.5])
    return _compute_w_moments_from_pmf(xs, ps)

def _moments_tuniform_w(b):
    """TUniform: -2^b .. 2^b"""
    high = 2**b
    xs = np.arange(-high, high + 1, dtype=float)
    ps = np.full(len(xs), 1.0 / 2**(b+1))
    ps[0] = 1.0 / 2**(b+2)      # 左端点
    ps[-1] = 1.0 / 2**(b+2)     # 右端点
    ps /= ps.sum()
    return _compute_w_moments_from_pmf(xs, ps)

# ========== 分布对象：加入 W 的矩 ==========
Distribution = namedtuple('Distribution',
    ['name', 'params', 'mean', 'stddev', 'ex2', 'ex4',
     'w_mean', 'w_var', 'w_skew', 'w_exkurt'])

def create_distribution(name, **params):
    name = name.lower()
    if name == 'discrete_gaussian':
        sigma = params.get('sigma', 1.0)
        mean_val = params.get('mean', 0.0)
        ex2, ex4 = (0,0)  # 这里先不用，后面用 W 的矩
        # 计算 W 的矩
        w_mean, w_var, w_skew, w_exkurt = _moments_discrete_gaussian_w(sigma, mean_val)
        # 标准差和均值（原始变量 X）
        stddev = sigma
        return Distribution(name, params, mean_val, stddev, w_mean, w_var,
                            w_mean, w_var, w_skew, w_exkurt)

    elif name == 'continuous_gaussian':
        stddev = params.get('stddev', 1.0)
        mean_val = params.get('mean', 0.0)
        # X ~ N(mu, sigma^2)
        mu, s = mean_val, stddev
        # W ~ Gamma(shape=1/2, scale=2*sigma^2) 偏移? 对于 X~N(0,sigma^2), W = sigma^2 * chi2_1
        # 均值 = sigma^2，方差 = 2*sigma^4，偏度 = sqrt(8)，超额峰度 = 12
        w_mean = s**2
        w_var = 2 * s**4
        w_skew = sqrt(8.0)
        w_exkurt = 12.0
        # 原始 X 的标准差和均值
        return Distribution(name, params, mu, s, w_mean, w_var,
                            w_mean, w_var, w_skew, w_exkurt)

    elif name == 'centered_binomial':
        eta = params['eta']
        w_mean, w_var, w_skew, w_exkurt = _moments_centered_binomial_w(eta)
        stddev = sqrt(eta / 2.0)
        return Distribution(name, params, 0.0, stddev, w_mean, w_var,
                            w_mean, w_var, w_skew, w_exkurt)

    elif name == 'uniform':
        a = params['a']; b = params['b']
        w_mean, w_var, w_skew, w_exkurt = _moments_uniform_w(a, b)
        mean_val = (a + b) / 2.0
        var_val = ((b - a + 1)**2 - 1) / 12.0
        stddev = sqrt(var_val)
        return Distribution(name, params, mean_val, stddev, w_mean, w_var,
                            w_mean, w_var, w_skew, w_exkurt)

    elif name == 'ternary':
        w_mean, w_var, w_skew, w_exkurt = _moments_ternary_w()
        return Distribution(name, {}, 0.0, sqrt(2./3), w_mean, w_var,
                            w_mean, w_var, w_skew, w_exkurt)

    elif name == 'binary':
        w_mean, w_var, w_skew, w_exkurt = _moments_binary_w()
        return Distribution(name, {}, 0.5, 0.5, w_mean, w_var,
                            w_mean, w_var, w_skew, w_exkurt)

    elif name == 'tuniform':
        b = params['b']
        w_mean, w_var, w_skew, w_exkurt = _moments_tuniform_w(b)
        s2 = (2**(2*b+1) + 1) / 6.0
        stddev = sqrt(s2)
        return Distribution(name, params, 0.0, stddev, w_mean, w_var,
                            w_mean, w_var, w_skew, w_exkurt)
    else:
        raise ValueError(f"Unsupported distribution: {name}")

# ========== Y 的理论统计（含 Edgeworth 修正） ==========
def theoretical_Y_stats(dist, n):
    """
    返回:
      EY, VarY: 均值和方差
      quantiles_norm: 正态近似分位数
      quantiles_edge: Edgeworth 修正分位数
    """
    mu_w = dist.w_mean          # E[W]
    var_w = dist.w_var          # Var(W)
    EY = n * mu_w
    VarY = n * var_w
    stdY = sqrt(VarY)

    qs = [0.001, 0.01, 0.05]
    quant_norm = {}
    quant_edge = {}

    for q in qs:
        w = norm.ppf(q)  # 标准正态分位点
        # 正态近似
        y_norm = EY + stdY * w
        quant_norm[q] = y_norm

        # Cornish-Fisher 展开（对称分布 skew=0 自动简化）
        skew = dist.w_skew
        exkurt = dist.w_exkurt
        # 到 O(1/n) 的展开
        z_edge = w + (skew/6)*(w**2 - 1)/sqrt(n) + \
                     (exkurt/24)*(w**3 - 3*w)/n + \
                     (skew**2/72)*(w**5 - 10*w**3 + 15*w)/n
        # 由于 skew 可能很小，但仍保留完整公式
        y_edge = EY + stdY * z_edge
        quant_edge[q] = y_edge

    return EY, VarY, quant_norm, quant_edge

# ========== 采样模拟 ==========
def sample_Y(dist, n, num_samples=200000):
    """根据分布采样，返回 Y 样本"""
    name = dist.name
    params = dist.params
    if name in ('discrete_gaussian', 'continuous_gaussian'):
        if name == 'discrete_gaussian':
            sigma = params['sigma']
            mean_val = params.get('mean', 0.0)
            samples = np.random.normal(mean_val, sigma, (num_samples, n))
            samples = np.round(samples)
        else:
            stddev = params['stddev']
            mean_val = params.get('mean', 0.0)
            samples = np.random.normal(mean_val, stddev, (num_samples, n))
    elif name == 'centered_binomial':
        eta = params['eta']
        a = np.random.binomial(1, 0.5, (num_samples, n, eta))
        b = np.random.binomial(1, 0.5, (num_samples, n, eta))
        samples = a.sum(axis=2) - b.sum(axis=2)
    elif name == 'uniform':
        a, b = params['a'], params['b']
        samples = np.random.randint(a, b + 1, (num_samples, n))
    elif name == 'ternary':
        samples = np.random.choice([-1, 0, 1], size=(num_samples, n))
    elif name == 'binary':
        samples = np.random.choice([0, 1], size=(num_samples, n))
    elif name == 'tuniform':
        b = params['b']
        high = 2**b
        choices = np.arange(-high, high + 1)
        ps = np.full(len(choices), 1.0 / 2**(b+1))
        ps[0] = 1.0 / 2**(b+2)
        ps[-1] = 1.0 / 2**(b+2)
        ps /= ps.sum()
        samples = np.random.choice(choices, size=(num_samples, n), p=ps)
    else:
        raise ValueError(f"Sampling not implemented for {name}")

    Y = np.sum(samples.astype(np.float64)**2, axis=1)
    return Y

def empirical_quantiles(Y, qs=[0.001, 0.01, 0.05]):
    return {q: np.quantile(Y, q) for q in qs}

# ========== 主研究函数（对比输出） ==========
def study_distribution(dist_name, dist_params, n, num_samples=200000):
    dist = create_distribution(dist_name, **dist_params)
    EY, VarY, theory_norm, theory_edge = theoretical_Y_stats(dist, n)

    print(f"=== {dist_name} (params={dist_params}), n={n} ===")
    print(f"单个分量: E[X]={dist.mean:.4f}, σ={dist.stddev:.4f}")
    print(f"W = X^2 的矩: E[W]={dist.w_mean:.4f}, Var(W)={dist.w_var:.4f}, "
          f"偏度={dist.w_skew:.4f}, 超额峰度={dist.w_exkurt:.4f}")
    print(f"Y = Σ x_i^2 理论值:")
    print(f"  E[Y] = {EY:.4f},  Var[Y] = {VarY:.4f} (std={sqrt(VarY):.4f})")

    # 采样
    Y_samples = sample_Y(dist, n, num_samples=num_samples)
    emp_q = empirical_quantiles(Y_samples)

    # 对比表格
    print(f"\n分位数对比 (左尾):")
    print(f"{'分位':<10} {'正态近似':<15} {'Edgeworth':<15} {'经验模拟':<15} {'正态偏差%':<12} {'Edge偏差%':<12}")
    for q in [0.001, 0.01, 0.05]:
        t_n = theory_norm[q]
        t_e = theory_edge[q]
        e = emp_q[q]
        dev_n = (e - t_n) / abs(t_n) * 100 if t_n != 0 else 0.0
        dev_e = (e - t_e) / abs(t_e) * 100 if t_e != 0 else 0.0
        print(f"{q:<10.3f} {t_n:<15.4f} {t_e:<15.4f} {e:<15.4f} {dev_n:<12.2f} {dev_e:<12.2f}")

    print(f"模拟 E[Y] = {np.mean(Y_samples):.4f}, 模拟 Var[Y] = {np.var(Y_samples):.4f}\n")

# 示例运行（与之前相同）
if __name__ == "__main__":
    study_distribution('centered_binomial', {'eta': 8}, n=256, num_samples=200000)
    study_distribution('discrete_gaussian', {'sigma': 3.0}, n=256, num_samples=200000)
    study_distribution('uniform', {'a': -3, 'b': 3}, n=256, num_samples=200000)
    study_distribution('ternary', {}, n=256, num_samples=200000)
