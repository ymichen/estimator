# noise/tools.py
import numpy as np
from scipy.stats import norm, chi2
from math import sqrt, comb
from collections import namedtuple

# ---------- 基础矩计算（W = X^2） ----------
def _compute_w_moments_from_pmf(xs, ps):
    w = xs**2
    mu = np.sum(w * ps)
    var = np.sum((w - mu)**2 * ps)
    mu3 = np.sum((w - mu)**3 * ps)
    mu4 = np.sum((w - mu)**4 * ps)
    skew = mu3 / var**1.5 if var > 0 else 0.0
    exkurt = mu4 / var**2 - 3.0 if var > 0 else 0.0
    return mu, var, skew, exkurt

def _moments_discrete_gaussian_w(sigma, mean=0.0, tail=6):
    low = int(mean - tail * sigma)
    high = int(mean + tail * sigma)
    xs = np.arange(low, high + 1, dtype=float)
    logps = -((xs - mean) ** 2) / (2 * sigma ** 2)
    ps = np.exp(logps)
    ps /= ps.sum()
    return _compute_w_moments_from_pmf(xs, ps)

def _moments_centered_binomial_w(eta):
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
    xs = np.arange(a, b + 1, dtype=float)
    ps = np.full(len(xs), 1.0 / len(xs))
    return _compute_w_moments_from_pmf(xs, ps)

def _moments_ternary_w():
    xs = np.array([-1., 0., 1.])
    ps = np.array([1./3, 1./3, 1./3])
    return _compute_w_moments_from_pmf(xs, ps)

def _moments_binary_w():
    xs = np.array([0., 1.])
    ps = np.array([0.5, 0.5])
    return _compute_w_moments_from_pmf(xs, ps)

def _moments_tuniform_w(b):
    high = 2**b
    xs = np.arange(-high, high + 1, dtype=float)
    ps = np.full(len(xs), 1.0 / 2**(b+1))
    ps[0] = 1.0 / 2**(b+2)      # 左端点
    ps[-1] = 1.0 / 2**(b+2)     # 右端点
    ps /= ps.sum()
    return _compute_w_moments_from_pmf(xs, ps)

# ---------- 分布对象定义 ----------
Distribution = namedtuple('Distribution',
    ['name', 'params', 'mean', 'stddev',
     'w_mean', 'w_var', 'w_skew', 'w_exkurt'])

def create_distribution(name, **params):
    """根据名称和参数创建分布对象。
    
    支持的分布：
    - 'discrete_gaussian' : 离散高斯，参数 sigma (标准差), mean (均值，默认0)
    - 'continuous_gaussian' : 连续高斯，参数 stddev, mean (默认0)
    - 'centered_binomial' : 中心二项分布，参数 eta
    - 'uniform' : 离散均匀分布，参数 a, b (包含端点)
    - 'ternary' : 三元分布 {-1,0,1}，等概率
    - 'binary' : 二元分布 {0,1}，等概率
    - 'tuniform' : TUniform 分布，参数 b (范围 [-2^b, 2^b])
    """
    name = name.lower()
    if name == 'discrete_gaussian':
        sigma = params.get('sigma', 1.0)
        mean_val = params.get('mean', 0.0)
        w_mean, w_var, w_skew, w_exkurt = _moments_discrete_gaussian_w(sigma, mean_val)
        return Distribution(name, params, mean_val, sigma, w_mean, w_var, w_skew, w_exkurt)

    elif name == 'continuous_gaussian':
        stddev = params.get('stddev', 1.0)
        mean_val = params.get('mean', 0.0)
        s = stddev
        w_mean = s**2
        w_var = 2 * s**4
        w_skew = sqrt(8.0)
        w_exkurt = 12.0
        return Distribution(name, params, mean_val, stddev, w_mean, w_var, w_skew, w_exkurt)

    elif name == 'centered_binomial':
        eta = params['eta']
        w_mean, w_var, w_skew, w_exkurt = _moments_centered_binomial_w(eta)
        stddev = sqrt(eta / 2.0)
        return Distribution(name, params, 0.0, stddev, w_mean, w_var, w_skew, w_exkurt)

    elif name == 'uniform':
        a = params['a']; b = params['b']
        w_mean, w_var, w_skew, w_exkurt = _moments_uniform_w(a, b)
        mean_val = (a + b) / 2.0
        var_val = ((b - a + 1)**2 - 1) / 12.0
        stddev = sqrt(var_val)
        return Distribution(name, params, mean_val, stddev, w_mean, w_var, w_skew, w_exkurt)

    elif name == 'ternary':
        w_mean, w_var, w_skew, w_exkurt = _moments_ternary_w()
        return Distribution(name, {}, 0.0, sqrt(2./3), w_mean, w_var, w_skew, w_exkurt)

    elif name == 'binary':
        w_mean, w_var, w_skew, w_exkurt = _moments_binary_w()
        return Distribution(name, {}, 0.5, 0.5, w_mean, w_var, w_skew, w_exkurt)

    elif name == 'tuniform':
        b = params['b']
        w_mean, w_var, w_skew, w_exkurt = _moments_tuniform_w(b)
        s2 = (2**(2*b+1) + 1) / 6.0
        stddev = sqrt(s2)
        return Distribution(name, params, 0.0, stddev, w_mean, w_var, w_skew, w_exkurt)

    else:
        raise ValueError(f"Unsupported distribution: {name}")

# ---------- Y 统计与 Edgeworth 修正 ----------
def theoretical_Y_stats(dist, n):
    """计算 Y = Σ X_i^2 的理论统计量。
    
    返回：
        EY, VarY, quant_norm, quant_edge
        其中 quant_norm 和 quant_edge 是字典，键为分位点 (0.001, 0.01, 0.05)。
    """
    mu_w = dist.w_mean
    var_w = dist.w_var
    EY = n * mu_w
    VarY = n * var_w
    stdY = sqrt(VarY)

    qs = [0.001, 0.01, 0.05]
    quant_norm = {}
    quant_edge = {}
    for q in qs:
        w = norm.ppf(q)
        y_norm = EY + stdY * w
        quant_norm[q] = y_norm

        # Cornish-Fisher 展开
        skew = dist.w_skew
        exkurt = dist.w_exkurt
        z_edge = (w + (skew/6)*(w**2 - 1)/sqrt(n) +
                   (exkurt/24)*(w**3 - 3*w)/n +
                   (skew**2/72)*(w**5 - 10*w**3 + 15*w)/n)
        y_edge = EY + stdY * z_edge
        quant_edge[q] = y_edge

    return EY, VarY, quant_norm, quant_edge

# ---------- 等价高斯计算 ----------
def equivalent_gaussian(dist, n, quantiles=(0.001, 0.01, 0.05), use_edgeworth=True):
    """计算等价高斯分布的标准差。
    
    等价高斯：均值为 0，标准差为 sigma_eq 的连续高斯分布，使得该分布的
    Y = Σ X_i^2 在指定分位数 q 下的值与给定分布 dist 相同。
    
    参数
    ----
    dist : Distribution
        通过 create_distribution 创建的分布对象。
    n : int
        向量维度。
    quantiles : 可迭代对象，默认 (0.001, 0.01, 0.05)
        需要匹配的分位数概率。
    use_edgeworth : bool, 默认 True
        是否使用 Edgeworth 修正后的分位数作为目标；如果为 False 则使用正态近似。
    
    返回
    ----
    dict : {q: sigma_eq}，键为分位数值，值为等价高斯标准差。
    """
    _, _, _, quant_edge = theoretical_Y_stats(dist, n)
    source_quant = quant_edge if use_edgeworth else theoretical_Y_stats(dist, n)[2]

    result = {}
    for q in quantiles:
        y_q = source_quant[q]
        # 对于连续高斯 X ~ N(0, sigma^2)：Y = sigma^2 * Chi2(n)
        # 因此 sigma = sqrt( y_q / chi2.ppf(q, n) )
        chi2_val = chi2.ppf(q, n)
        sigma_eq = sqrt(y_q / chi2_val)
        result[q] = sigma_eq
    return result
