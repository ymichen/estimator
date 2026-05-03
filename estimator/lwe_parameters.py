import math
from dataclasses import dataclass
from .nd import NoiseDistribution, DiscreteGaussian
from .errors import InsufficientSamplesError

OO = float('inf')

@dataclass
class LWEParameters:
    n: int
    q: int
    Xs: NoiseDistribution
    Xe: NoiseDistribution
    m: int = OO
    tag: str = None

    def __post_init__(self, **kwds):
        self.Xs = self.Xs.resize(self.n)
        if self.m < OO:
            self.Xe = self.Xe.resize(self.m)

    @property
    def _homogeneous(self):
        return False

    def normalize(self):
        if self.m < 1:
            raise InsufficientSamplesError(f"m={self.m} < 1")
        if self.Xe < self.Xs and self.m >= 2 * self.n:
            return LWEParameters(n=self.n, q=self.q, Xs=self.Xe, Xe=self.Xe,
                                 m=self.m - self.n, tag=self.tag)
        if self.Xe < self.Xs and self.m == self.n:
            return LWEParameters(n=self.n, q=self.q, Xs=self.Xe, Xe=self.Xs,
                                 m=self.n, tag=self.tag)
        return self

    def updated(self, **kwds):
        d = dict(self.__dict__)
        d.update(kwds)
        return LWEParameters(**d)

    def amplify_m(self, m):
        if m <= self.m:
            return self
        if self.m == OO:
            return self
        d = dict(self.__dict__)
        if self.Xe.mean != 0:
            raise NotImplementedError("Amplifying for μ≠0 not implemented.")
        for k in range(math.ceil(math.log2(m))):
            if math.comb(self.m, k) * (2**k) - 1 >= m:
                Xe = DiscreteGaussian(float(math.sqrt(k)) * self.Xe.stddev)
                d["Xe"] = Xe
                d["m"] = math.ceil(m)
                return LWEParameters(**d)
        raise NotImplementedError(f"Cannot amplify to ≈2^{math.log2(m):.1f}")

    def switch_modulus(self):
        h = len(self.Xs) * self.Xs._density
        Xr_stddev = math.sqrt(h/12) * self.Xs.stddev
        p = math.ceil(Xr_stddev * self.q / self.Xe.stddev)
        scale = float(p) / self.q
        if scale > 1/math.sqrt(2):
            return self
        return LWEParameters(self.n, p, Xs=self.Xs,
                             Xe=DiscreteGaussian(math.sqrt(2)*self.Xe.stddev*scale),
                             m=self.m, tag=self.tag)

    def __hash__(self):
        return hash((self.n, self.q, self.Xs, self.Xe, self.m, self.tag))
