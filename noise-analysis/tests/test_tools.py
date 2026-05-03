# tests/test_tools.py
import unittest
from noise.tools import create_distribution, equivalent_gaussian

class TestEquivalentGaussian(unittest.TestCase):

    def test_continuous_gaussian_self_equivalent(self):
        """连续高斯分布应该等价于自身（在 Edgeworth 近似误差范围内）。"""
        dist = create_distribution('continuous_gaussian', stddev=3.0)
        sigmas = equivalent_gaussian(dist, n=256, quantiles=[0.05, 0.01, 0.001])
        for q, s_eq in sigmas.items():
            self.assertAlmostEqual(s_eq, 3.0, delta=0.05,
                msg=f"q={q}: expected ~3.0, got {s_eq}")

    def test_discrete_gaussian_approx(self):
        """离散高斯应产生接近其连续对应标准差的等价高斯。"""
        dist = create_distribution('discrete_gaussian', sigma=3.0)
        sigmas = equivalent_gaussian(dist, n=256, quantiles=[0.05])
        # 预期 ≈ 3.0，允许较大偏差因为离散化和 Edgeworth 修正
        self.assertAlmostEqual(sigmas[0.05], 3.0, delta=0.3)

    def test_centered_binomial(self):
        """中心二项分布 η=8 的等价高斯标准差应接近 2.0。"""
        dist = create_distribution('centered_binomial', eta=8)
        sigmas = equivalent_gaussian(dist, n=256, quantiles=[0.05])
        # 分量标准差 √(8/2)=2.0
        self.assertAlmostEqual(sigmas[0.05], 2.0, delta=0.3)

    def test_ternary(self):
        """三元分布的等价高斯标准差应接近 √(2/3) ≈ 0.8165。"""
        dist = create_distribution('ternary')
        sigmas = equivalent_gaussian(dist, n=256, quantiles=[0.05])
        self.assertAlmostEqual(sigmas[0.05], 0.8165, delta=0.1)

    def test_different_quantiles(self):
        """验证不同分位数返回不同的 sigma_eq。"""
        dist = create_distribution('uniform', a=-3, b=3)
        sigmas = equivalent_gaussian(dist, n=256, quantiles=[0.001, 0.05])
        # 由于左尾不同，分位数差距应反映在 sigma_eq 上
        self.assertTrue(sigmas[0.001] > sigmas[0.05])

if __name__ == '__main__':
    unittest.main()
