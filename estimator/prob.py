# -*- coding: utf-8 -*-
import math
from functools import lru_cache
from scipy.stats import chi2, beta as beta_dist
from .conf import max_n_cache
from .nd import NoiseDistribution

OO = float('inf')

# Precompute chi-squared distributions for all needed degrees of freedom
chisquared_table = {i: chi2(i) for i in range(2 * max_n_cache + 1)}


def conditional_chi_squared(d1, d2, lt, l2):
    D1 = chisquared_table[d1].cdf
    D2 = chisquared_table[d2].cdf
    PE2 = D2(l2)
    if PE2 == 0:
        raise ValueError("Numerical underflow in conditional_chi_squared")
    steps = 5 * (d1 + d2)
    proba = 0.0
    for i in range(steps - 1, -1, -1):
        l2_min = i * l2 / steps
        l2_mid = (i + 0.5) * l2 / steps
        l2_max = (i + 1) * l2 / steps
        PC2 = (D2(l2_max) - D2(l2_min)) / PE2
        PE1 = D1(lt - l2_mid)
        proba += PC2 * PE1
    return proba


def gaussian_cdf(mu, sigma, t):
    return 0.5 * (1 + math.erf((t - mu) / (math.sqrt(2) * sigma)))


def mitm_babai_probability(r, stddev, fast=False):
    if fast:
        return 1.0
    xs = [math.sqrt(0.5 * ri) / stddev for ri in r]
    p = math.prod(math.erf(x) - (1 - math.exp(-x**2)) / (x * math.sqrt(math.pi)) for x in xs)
    return p


def babai(r, norm):
    denom = (2 * norm) ** 2
    n = len(r)
    T = beta_dist((n - 1) / 2, 0.5)
    probs = [1 - T.cdf(1 - r_ / denom) for r_ in r]
    return math.prod(probs)


def drop(n, h, k, fail=0, rotations=False):
    N = n
    K = n - h
    n_draw = k
    k_success = n_draw - fail
    prob_drop = math.comb(K, k_success) * math.comb(N - K, n_draw - k_success) / math.comb(N, n_draw)
    if rotations:
        return 1 - (1 - prob_drop) ** N
    return prob_drop


def amplify(target_success_probability, success_probability, majority=False):
    if target_success_probability < success_probability:
        return 1
    if success_probability == 0.0:
        return OO

    if majority:
        eps = success_probability / 2
        return math.ceil(2 * math.log(2 - 2 * target_success_probability) / math.log(1 - 4 * eps**2))
    else:
        return math.ceil(math.log(1 - target_success_probability) / math.log(1 - success_probability))


def amplify_sigma(target_advantage, sigma, q):
    if isinstance(sigma, (list, tuple)):
        sigma = math.sqrt(sum(s**2 for s in sigma))
    if sigma > 16 * q:
        return OO
    advantage = math.exp(-math.pi * (sigma / q) ** 2)
    return amplify(target_advantage, advantage, majority=True)


@lru_cache(maxsize=None)
def guessing_set_and_hit_probability(zeta, dist, hw):
    if zeta > dist.n:
        raise ValueError(f"Trying to guess {zeta} coordinates of a vector of length {dist.n}")

    if zeta == 0:
        return 1, 1.0

    h = dist.hamming_weight
    base = dist.bounds[1] - dist.bounds[0]
    min_hw = max(0, zeta - dist.n + h)
    max_hw = min(zeta, h)

    if hw < min_hw or hw > max_hw:
        raise ValueError(f"hw={hw} not in feasible range [{min_hw},{max_hw}]")

    if hw == 0:
        search_space = math.comb(zeta, hw)
    elif base == OO:
        search_space = OO
    else:
        search_space = math.comb(zeta, hw) * base**hw

    probability = drop(dist.n, h, zeta, fail=hw)

    if hw > min_hw:
        prev_search_space, prev_prob = guessing_set_and_hit_probability(zeta, dist, hw - 1)
        search_space += prev_search_space
        probability += prev_prob

    return search_space, probability
