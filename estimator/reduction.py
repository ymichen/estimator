# reduction_py.py
import math
from scipy.optimize import newton

OO = float('inf')

class ReductionCost:
    @staticmethod
    def _delta(beta):
        small = (
            (2, 1.02190),
            (5, 1.01862),
            (10, 1.01616),
            (15, 1.01485),
            (20, 1.01420),
            (25, 1.01342),
            (28, 1.01331),
            (40, 1.01295),
        )
        if beta <= 2:
            return 1.0219
        elif beta < 40:
            for i in range(1, len(small)):
                if small[i][0] > beta:
                    return small[i-1][1]
        elif beta == 40:
            return small[-1][1]
        else:
            b = float(beta)
            return ((b / (2 * math.pi * math.e)) * (math.pi * b) ** (1/b)) ** (1/(2*(b-1)))

    @staticmethod
    def delta(beta):
        beta = int(round(beta))
        return ReductionCost._delta(beta)

    @staticmethod
    def _beta_secant(delta):
        try:
            beta = newton(lambda b: ReductionCost._delta(b) - delta, 100,
                         tol=1.48e-8, maxiter=500)
            beta = math.ceil(beta)
            if beta < 40:
                raise RuntimeError("β < 40")
            return beta
        except (RuntimeError, TypeError):
            return ReductionCost._beta_simple(delta)

    @staticmethod
    def _beta_find_root(delta):
        beta = 40
        if ReductionCost._delta(beta) < delta:
            return beta
        # simple scan because find_root not easily available without sage
        return ReductionCost._beta_simple(delta)

    @staticmethod
    def _beta_simple(delta):
        beta = 40
        while ReductionCost._delta(2*beta) > delta:
            beta *= 2
        while ReductionCost._delta(beta + 10) > delta:
            beta += 10
        while ReductionCost._delta(beta) >= delta:
            beta += 1
        return beta

    @staticmethod
    def beta(delta):
        return ReductionCost._beta_find_root(delta)

    @staticmethod
    def svp_repeat(beta, d):
        if beta < d:
            return 8 * d
        else:
            return 1

    @staticmethod
    def LLL(d, B=None):
        if B is None:
            return d**3
        else:
            return d**3 * B**2

    def short_vectors(self, beta, d, N=None, B=None, preprocess=True, sieve_dim=None):
        if preprocess:
            cost = self(beta, d, B=B)
        else:
            cost = 0
        if N == 1:
            return 1.0, cost + 1, 1, 2
        elif N is None:
            N = 1000
        return 2.0, cost + N * RC.LLL(d), N, 2

    def short_vectors_simple(self, beta, d, N=None, B=None, preprocess=True):
        if N == 1:
            if preprocess:
                return 1.0, self(beta, d, B=B), 1, beta
            else:
                return 1.0, 1, 1, beta
        elif N is None:
            N = 1000
        return 1.0, N * self(beta, d, B=B), N, beta

    def _short_vectors_sieve(self, beta, d, N=None, B=None, preprocess=True, sieve_dim=None):
        if sieve_dim is None:
            sieve_dim = beta
        if N == 1:
            if preprocess:
                return 1.0, self(beta, d, B=B), 1, sieve_dim
            else:
                return 1.0, 1, 1, sieve_dim
        elif N is None:
            N = int(2 ** (0.2075 * beta))
        c0 = float(N)
        c1 = float(2 ** (0.2075 * beta))
        c = c0 / c1
        rho = math.sqrt(4/3.0) * (self.delta(sieve_dim) ** (sieve_dim-1) *
                                   self.delta(beta) ** (1-sieve_dim))
        if c > 2**1000:
            return (rho, OO, OO, sieve_dim)
        return (rho, math.ceil(c) * self(beta, d),
                math.ceil(c) * int(c1), sieve_dim)


class BDGL16(ReductionCost):
    __name__ = "BDGL16"
    short_vectors = ReductionCost._short_vectors_sieve

    @classmethod
    def _small(cls, beta, d, B=None):
        return cls.LLL(d, B) + 2 ** (0.387 * beta + 16.4 + math.log2(cls.svp_repeat(beta, d)))

    @classmethod
    def _asymptotic(cls, beta, d, B=None):
        return cls.LLL(d, B) + 2 ** (0.292 * beta + 16.4 + math.log2(cls.svp_repeat(beta, d)))

    def __call__(self, beta, d, B=None):
        return self._asymptotic(beta, d, B)


class LaaMosPol14(ReductionCost):
    __name__ = "LaaMosPol14"
    short_vectors = ReductionCost._short_vectors_sieve

    def __call__(self, beta, d, B=None):
        return self.LLL(d, B) + 2 ** (0.265 * beta + 16.4 + math.log2(self.svp_repeat(beta, d)))


class CheNgu12(ReductionCost):
    __name__ = "CheNgu12"

    def __call__(self, beta, d, B=None):
        repeat = self.svp_repeat(beta, d)
        cost = (0.270188776350190 * beta * math.log2(beta)
                - 1.0192050451318417 * beta
                + 16.10253135200765
                + math.log2(100))
        return self.LLL(d, B) + repeat * 2 ** cost


class ABFKSW20(ReductionCost):
    __name__ = "ABFKSW20"

    def __call__(self, beta, d, B=None):
        if 1.5 * beta >= d or beta <= 92:
            cost = 0.1839 * beta * math.log2(beta) - 0.995 * beta + 16.25 + math.log2(64)
        else:
            cost = 0.125 * beta * math.log2(beta) - 0.547 * beta + 10.4 + math.log2(64)
        repeat = self.svp_repeat(beta, d)
        return self.LLL(d, B) + repeat * 2 ** cost


class ABLR21(ReductionCost):
    __name__ = "ABLR21"

    def __call__(self, beta, d, B=None):
        if 1.5 * beta >= d or beta <= 97:
            cost = 0.1839 * beta * math.log2(beta) - 1.077 * beta + 29.12 + math.log2(64)
        else:
            cost = 0.1250 * beta * math.log2(beta) - 0.654 * beta + 25.84 + math.log2(64)
        repeat = self.svp_repeat(beta, d)
        return self.LLL(d, B) + repeat * 2 ** cost


class ADPS16(ReductionCost):
    __name__ = "ADPS16"
    short_vectors = ReductionCost._short_vectors_sieve

    def __init__(self, mode="classical"):
        if mode not in ("classical", "quantum", "paranoid"):
            raise ValueError(f"Mode {mode} not understood.")
        self.mode = mode

    def __call__(self, beta, d, B=None):
        c = {"classical": 0.2920, "quantum": 0.2650, "paranoid": 0.2075}
        return 2 ** (c[self.mode] * beta)


class ChaLoy21(ReductionCost):
    __name__ = "ChaLoy21"
    short_vectors = ReductionCost._short_vectors_sieve

    def __call__(self, beta, d, B=None):
        return 2 ** (0.2570 * beta)


class Kyber(ReductionCost):
    __name__ = "Kyber"
    NN_AGPS = {
        "all_pairs-classical": {"a": 0.4215069316613415, "b": 20.1669683097337},
        "all_pairs-dw": {"a": 0.3171724396445732, "b": 25.29828951733785},
        "all_pairs-g": {"a": 0.3155285835002801, "b": 22.478746811528048},
        "all_pairs-ge19": {"a": 0.3222895263943544, "b": 36.11746438609666},
        "all_pairs-naive_classical": {"a": 0.4186251294633655, "b": 9.899382654377058},
        "all_pairs-naive_quantum": {"a": 0.31401512556555794, "b": 7.694659515948326},
        "all_pairs-t_count": {"a": 0.31553282515234704, "b": 20.878594142502994},
        "list_decoding-classical": {"a": 0.2988026130564745, "b": 26.011121212891872},
        "list_decoding-dw": {"a": 0.26944796385592995, "b": 28.97237346443934},
        "list_decoding-g": {"a": 0.26937450988892553, "b": 26.925140365395972},
        "list_decoding-ge19": {"a": 0.2695210400018704, "b": 35.47132142280775},
        "list_decoding-naive_classical": {"a": 0.2973130399197453, "b": 21.142124058689426},
        "list_decoding-naive_quantum": {"a": 0.2674316807758961, "b": 18.720680589028465},
        "list_decoding-t_count": {"a": 0.26945736714156543, "b": 25.913746774011887},
        "random_buckets-classical": {"a": 0.35586144233444716, "b": 23.082527816636638},
        "random_buckets-dw": {"a": 0.30704199612690264, "b": 25.581968903639485},
        "random_buckets-g": {"a": 0.30610964725102385, "b": 22.928235564044563},
        "random_buckets-ge19": {"a": 0.31089687599538407, "b": 36.02129978813208},
        "random_buckets-naive_classical": {"a": 0.35448283789554513, "b": 15.28878540793908},
        "random_buckets-naive_quantum": {"a": 0.30211421791887644, "b": 11.151745013027089},
        "random_buckets-t_count": {"a": 0.30614770082829745, "b": 21.41830142853265},
    }

    def __init__(self, nn="classical"):
        if nn == "classical":
            nn = "list_decoding-classical"
        elif nn == "all_pairs":
            nn = "all_pairs-classical"
        elif nn == "quantum":
            nn = "list_decoding-dw"
        self.nn = nn

    @staticmethod
    def d4f(beta):
        return max(float(beta * math.log(4/3.0) / math.log(beta/(2*math.pi*math.e))), 0.0)

    def __call__(self, beta, d, B=None):
        if beta < 20:
            return CheNgu12()(beta, d, B)
        a = self.NN_AGPS[self.nn]["a"]
        b = self.NN_AGPS[self.nn]["b"]
        C = 1.0 / (1.0 - 2 ** (-a))
        svp_calls = C * max(d - beta, 1)
        beta_ = beta - self.d4f(beta)
        gate_count = C * 2 ** (a * beta_ + b)
        return self.LLL(d, B=B) + svp_calls * gate_count

    def short_vectors(self, beta, d, N=None, B=None, preprocess=True):
        beta_ = beta - math.floor(self.d4f(beta))
        if N == 1:
            if preprocess:
                return 1.0, self(beta, d, B=B), 1, beta
            else:
                return 1.0, 1, 1, beta
        elif N is None:
            N = math.floor(2 ** (0.2075 * beta_))
        c = N / math.floor(2 ** (0.2075 * beta_))
        return 1.1547, math.ceil(c) * self(beta, d), math.ceil(c) * math.floor(2**(0.2075*beta_)), beta_


class GJ21(Kyber):
    __name__ = "GJ21"

    def short_vectors(self, beta, d, N=None, preprocess=True, B=None, sieve_dim=None):
        C = 1.0 / (1.0 - 2 ** (-self.NN_AGPS[self.nn]["a"]))
        beta_ = beta - math.floor(self.d4f(beta))
        if sieve_dim is None:
            sieve_dim = beta_
            if beta < d:
                sieve_dim = min(d, math.floor(beta_ + math.log2((d - beta) * C) / self.NN_AGPS[self.nn]["a"]))
        rho = math.sqrt(4/3.0) * (self.delta(sieve_dim) ** (sieve_dim-1) * self.delta(beta) ** (1-sieve_dim))
        if N == 1:
            if preprocess:
                return 1.0, self(beta, d, B=B), 1, beta
            else:
                return 1.0, 1, 1, beta
        elif N is None:
            N = math.floor(2 ** (0.2075 * sieve_dim))
        c0 = float(N)
        c1 = float(2 ** (0.2075 * sieve_dim))
        c = c0 / math.floor(c1)
        sieve_cost = C * 2 ** (self.NN_AGPS[self.nn]["a"] * sieve_dim + self.NN_AGPS[self.nn]["b"])
        if c > 2**1000:
            return (rho, OO, OO, sieve_dim)
        return (rho, math.ceil(c) * (self(beta, d) + sieve_cost),
                math.ceil(c) * math.floor(c1), sieve_dim)


class MATZOV(GJ21):
    __name__ = "MATZOV"
    NN_AGPS = {
        "all_pairs-classical": {"a": 0.4215069316732438, "b": 20.166968300536567},
        "all_pairs-dw": {"a": 0.3171724396445733, "b": 25.2982895173379},
        "all_pairs-g": {"a": 0.31552858350028, "b": 22.478746811528104},
        "all_pairs-ge19": {"a": 0.3222895263943547, "b": 36.11746438609664},
        "all_pairs-naive_classical": {"a": 0.41862512941897706, "b": 9.899382685790897},
        "all_pairs-naive_quantum": {"a": 0.31401512571180035, "b": 7.694659414353819},
        "all_pairs-t_count": {"a": 0.31553282513562797, "b": 20.87859415484879},
        "list_decoding-classical": {"a": 0.29613500308205365, "b": 20.387885985467914},
        "list_decoding-dw": {"a": 0.2663676536352464, "b": 25.299541499216627},
        "list_decoding-g": {"a": 0.26600114174341505, "b": 23.440974518186337},
        "list_decoding-ge19": {"a": 0.26799889622667994, "b": 30.839871638418543},
        "list_decoding-naive_classical": {"a": 0.29371310617068064, "b": 15.930690682515422},
        "list_decoding-naive_quantum": {"a": 0.2632557273632713, "b": 15.685687713591548},
        "list_decoding-t_count": {"a": 0.2660264010780807, "b": 22.432158856991474},
        "random_buckets-classical": {"a": 0.3558614423344473, "b": 23.08252781663665},
        "random_buckets-dw": {"a": 0.30704199602260734, "b": 25.58196897625173},
        "random_buckets-g": {"a": 0.30610964725102396, "b": 22.928235564044588},
        "random_buckets-ge19": {"a": 0.31089687605567917, "b": 36.02129974535213},
        "random_buckets-naive_classical": {"a": 0.35448283789554536, "b": 15.28878540793911},
        "random_buckets-naive_quantum": {"a": 0.3021142178390157, "b": 11.151745066682524},
        "random_buckets-t_count": {"a": 0.3061477007403873, "b": 21.418301489775203},
    }


def cost(cost_model, beta, d, B=None, predicate=True, **kwds):
    if isinstance(cost_model, type):
        cost_model = cost_model()
    cost_val = cost_model(beta, d, B)
    delta_ = ReductionCost.delta(beta)
    from .cost import Cost  # 避免循环导入
    c = Cost(rop=cost_val, red=cost_val, delta=delta_, beta=beta, d=d, **kwds)
    c.register_impermanent(rop=True, red=True, delta=False, beta=False, d=False)
    if predicate is False:
        c["red"] = OO
        c["rop"] = OO
    return c


beta = ReductionCost.beta
delta = ReductionCost.delta


class RC:
    beta = ReductionCost.beta
    delta = ReductionCost.delta
    LLL = ReductionCost.LLL
    ABFKSW20 = ABFKSW20()
    ABLR21 = ABLR21()
    ADPS16 = ADPS16()
    BDGL16 = BDGL16()
    CheNgu12 = CheNgu12()
    Kyber = Kyber()
    MATZOV = MATZOV()
    GJ21 = GJ21()
    LaaMosPol14 = LaaMosPol14()
    ChaLoy21 = ChaLoy21()
