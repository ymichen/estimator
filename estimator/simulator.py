# simulator_py.py
import math
from functools import lru_cache

def GSA(d, n, q, beta, xi=1, tau=1, dual=False):
    from .reduction import delta as deltaf
    delta = deltaf(beta)
    if tau is None or not tau:
        log_vol = math.log2(q) * (d - n) + math.log2(xi) * n
    else:
        log_vol = math.log2(q) * (d - n - 1) + math.log2(xi) * n + math.log2(tau)
    log_delta = math.log2(delta)
    r_log = [(d - 1 - 2*i) * log_delta + log_vol/d for i in range(d)]
    return [2**(2*r_) for r_ in r_log]

def LGSA(d, n, q, beta, xi=1, tau=1, dual=False):
    from .reduction import delta as deltaf
    if tau is None or not tau:
        log_vol = math.log2(q) * (d - n) + math.log2(xi) * n
        r_log = [math.log2(xi)] * d
    else:
        log_vol = math.log2(q) * (d - n - 1) + math.log2(xi) * n + math.log2(tau)
        r_log = [math.log2(xi)] * (d - 1) + [math.log2(tau)]
    slope = -2 * math.log2(deltaf(beta))
    log_vec_len = 0
    profile_log_vol = sum(r_log)
    for i in range(d - 1, -1, -1):
        log_vec_len -= slope
        profile_log_vol += log_vec_len
        r_log[i] += log_vec_len
        if profile_log_vol > log_vol:
            break
    num_gsa_vec = d - i
    r_log = sorted(r_log, reverse=True)
    profile_log_vol = sum(r_log)
    diff = profile_log_vol - log_vol
    for i in range(num_gsa_vec):
        r_log[i] -= diff / num_gsa_vec
    # 建议增加 volume 检查
    return [2**(2*r_) for r_ in r_log]

def ZGSA(d, n, q, beta, xi=1, tau=1, dual=False):
    # 需要 gh_constant, small_slope_t8 从 util.py 导入
    from .util import gh_constant, small_slope_t8
    from scipy.special import gammaln
    @lru_cache(maxsize=None)
    def ball_log_vol(n):
        return (n/2.0) * math.log(math.pi) - gammaln(n/2.0 + 1)
    # 余下实现见原文件，将 RR 换成 float，log 换成 math.log
    # 略...
    pass

def normalize(name):
    # 映射字符串到函数
    mapping = {'GSA': GSA, 'LGSA': LGSA, 'ZGSA': ZGSA}
    return mapping.get(str(name).upper(), name)
