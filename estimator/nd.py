# nd.py  (Pure Python, no Sage required)
import math
import fractions
from copy import copy
from dataclasses import dataclass

# Infinity constant
INF = float('inf')

def stddevf(sigma):
    """Gaussian width parameter σ → standard deviation."""
    return sigma / math.sqrt(2 * math.pi)

def sigmaf(stddev):
    """Standard deviation → Gaussian width parameter σ."""
    return math.sqrt(2 * math.pi) * stddev

@dataclass
class NoiseDistribution:
    """
    Base class for all noise distributions.
    """
    n: int = None          # dimension of noise
    mean: float = 0.0      # expectation value
    stddev: float = 0.0    # standard deviation (square root of variance)
    bounds: tuple = (-INF, INF)  # range in which each coefficient is sampled with high probability
    is_Gaussian_like: bool = False
    _density: float = 1.0  # proportion of nonzero coefficients in a sample

    def __lt__(self, other):
        return self.stddev < other.stddev

    def __le__(self, other):
        return self.stddev <= other.stddev

    def __str__(self):
        if self.n:
            return f"D(σ={float(self.stddev):.2f}, μ={float(self.mean):.2f}, n={int(self.n)})"
        else:
            return f"D(σ={float(self.stddev):.2f}, μ={float(self.mean):.2f})"

    def __repr__(self):
        if self.mean == 0.0:
            return f"D(σ={float(self.stddev):.2f})"
        else:
            return f"D(σ={float(self.stddev):.2f}, μ={float(self.mean):.2f})"

    def __hash__(self):
        return hash((self.stddev, self.mean, self.n))

    def __len__(self):
        if self.n is None:
            raise ValueError("Distribution has no length.")
        return self.n

    def resize(self, new_n):
        new_self = copy(self)
        new_self.n = new_n
        return new_self

    @property
    def hamming_weight(self):
        return round(len(self) * float(self._density))

    @property
    def is_bounded(self):
        return (self.bounds[1] - self.bounds[0]) < INF

    @property
    def is_sparse(self):
        return self._density < 0.5

    def support_size(self, fraction=1.0):
        raise NotImplementedError("support_size")


class DiscreteGaussian(NoiseDistribution):
    """
    A discrete Gaussian distribution with standard deviation ``stddev`` per component.
    """
    gaussian_tail_bound: int = 2
    gaussian_tail_prob: float = 1 - 2 * math.exp(-4 * math.pi)

    def __init__(self, stddev, mean=0, n=None):
        stddev, mean = float(stddev), float(mean)
        b_val = INF if n is None else math.ceil(math.log(n, 2) * stddev)
        # density approximation valid for large stddev
        density = max(0.0, 1 - 1 / sigmaf(stddev))
        super().__init__(
            n=n,
            mean=mean,
            stddev=stddev,
            bounds=(-b_val, b_val),
            _density=density,
            is_Gaussian_like=True,
        )

    def support_size(self, fraction=1.0):
        n = len(self)
        t = self.gaussian_tail_bound
        p = self.gaussian_tail_prob

        if p ** n < fraction:
            raise NotImplementedError(
                f"TODO(DiscreteGaussian.support_size): raise t. {p ** n}, {n}, {fraction}"
            )
        b = 2 * t * sigmaf(self.stddev) + 1
        return (2 * b + 1) ** n


def DiscreteGaussianAlpha(alpha, q, mean=0, n=None):
    """Create a discrete Gaussian with standard deviation α·q/√(2π) per component."""
    return DiscreteGaussian(stddevf(alpha * q), mean, n)


class CenteredBinomial(NoiseDistribution):
    """
    Sample a_1,…,a_η, b_1,…,b_η uniformly from {0,1}, return Σ(a_i - b_i).
    """
    def __init__(self, eta, n=None):
        eta = int(eta)
        density = 1 - math.comb(2 * eta, eta) * 2 ** (-2 * eta)
        super().__init__(
            n=n,
            mean=0,
            stddev=math.sqrt(eta / 2.0),
            bounds=(-eta, eta),
            _density=density,
            is_Gaussian_like=True,
        )

    def support_size(self, fraction=1.0):
        a, b = self.bounds
        return math.ceil(fraction * (b - a + 1) ** len(self))


class Uniform(NoiseDistribution):
    """
    Uniform distribution ∈ ZZ ∩ [a, b], endpoints inclusive.
    """
    def __init__(self, a, b, n=None):
        a, b = int(math.ceil(a)), int(math.floor(b))
        if b < a:
            raise ValueError(f"upper limit must be larger than lower limit but got: {b} < {a}")
        m = b - a + 1
        super().__init__(
            n=n,
            mean=(a + b) / 2.0,
            stddev=math.sqrt((m ** 2 - 1) / 12.0),
            bounds=(a, b),
            _density=(1 - 1 / m if a <= 0 and b >= 0 else 1),
        )

    def __hash__(self):
        return hash(("Uniform", self.bounds, self.n))

    def support_size(self, fraction=1.0):
        a, b = self.bounds
        return math.ceil(fraction * (b - a + 1) ** len(self))


def UniformMod(q, n=None):
    """Uniform mod q, balanced representation, i.e. values in ZZ ∩ [-q/2, q/2)."""
    a = -(q // 2)
    b = a + q - 1
    return Uniform(a, b, n=n)


class TUniform(NoiseDistribution):
    """
    TUniform distribution ∈ ZZ ∩ [-2^b, 2^b], endpoints inclusive.
    """
    def __init__(self, b, n=None):
        b = int(math.ceil(b))
        super().__init__(
            n=n,
            mean=0.0,
            stddev=math.sqrt((2 ** (2 * b + 1) + 1) / 6.0),
            bounds=(-(2 ** b), 2 ** b),
            _density=(1 - 1 / 2 ** (b + 1)),
        )

    def __hash__(self):
        return hash(("TUniform", self.bounds, self.n))

    def support_size(self, fraction=1.0):
        a, b = self.bounds
        return math.ceil(fraction * (b - a + 1) ** len(self))


class SparseTernary(NoiseDistribution):
    """
    Distribution of vectors of length n with p entries of 1, m entries of -1, rest 0.
    """
    def __init__(self, p, m=None, n=None):
        p = int(p)
        m = int(p if m is None else m)
        self.p, self.m = p, m
        if n is None:
            n = 0
        n = int(n)
        mean = 0 if n == 0 else (p - m) / n
        density = 0 if n == 0 else (p + m) / n
        stddev = math.sqrt(density - mean ** 2)
        super().__init__(
            n=n,
            mean=mean,
            stddev=stddev,
            bounds=(0 if m == 0 else -1, 0 if p == 0 else 1),
            _density=density,
        )

    def __hash__(self):
        return hash(("SparseTernary", self.n, self.p, self.m))

    def resize(self, new_n):
        return SparseTernary(self.p, self.m, new_n)

    def split_balanced(self, new_n, new_hw=None):
        n, hw = len(self), self.hamming_weight
        if new_hw is None:
            new_hw = int(fractions.Fraction(hw * new_n, n).__floor__())
        new_p = int(fractions.Fraction(new_hw * self.p, hw).__floor__())
        new_m = new_hw - new_p
        return (
            SparseTernary(new_p, new_m, new_n),
            SparseTernary(self.p - new_p, self.m - new_m, n - new_n)
        )

    def split_probability(self, new_n, new_hw=None):
        left, right = self.split_balanced(new_n, new_hw)
        return left.support_size() * right.support_size() / self.support_size()

    @property
    def is_sparse(self):
        return True

    @property
    def hamming_weight(self):
        return self.p + self.m

    def support_size(self, fraction=1.0):
        n, p, m = len(self), self.p, self.m
        if n == 0:
            return 1
        return math.ceil(math.comb(n, p) * math.comb(n - p, m) * fraction)

    def __str__(self):
        if self.n:
            return f"T(p={self.p}, m={self.m}, n={int(self.n)})"
        else:
            return f"T(p={int(self.p)}, m={int(self.m)})"

    def __repr__(self):
        return str(self)


def SparseBinary(hw, n=None):
    """Sparse binary noise distribution having hw coefficients equal to 1, rest 0."""
    return SparseTernary(hw, 0, n)


# Convenience
Binary = Uniform(0, 1)
Ternary = Uniform(-1, 1)
