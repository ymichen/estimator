# util.py (Pure Python, no Sage)
import itertools as it
from multiprocessing import Pool
from functools import partial, lru_cache
from dataclasses import dataclass, field
from typing import Any, Callable, NamedTuple
import math
from scipy.special import zeta as szeta

from .io import Logging
from .lwe_parameters import LWEParameters
from .sis_parameters import SISParameters
from .conf import max_n_cache

OO = float('inf')

def log2(x):
    return math.log2(x)

@lru_cache(maxsize=None)
def zeta(x):
    if x == 1:
        return OO
    return float(szeta(x))

@lru_cache(maxsize=None)
def zeta_prime(x):
    h = 1e-5
    return (zeta(x + h) - zeta(x - h)) / (2 * h)


gh_constant = {1: 0.00000, 2: -0.50511, 3: -0.46488, 4: -0.39100, 5: -0.29759,
               6: -0.24880, 7: -0.21970, 8: -0.15748, 9: -0.14673, 10: -0.07541,
               11: -0.04870, 12: -0.01045, 13: 0.02298, 14: 0.04212, 15: 0.07014,
               16: 0.09205, 17: 0.12004, 18: 0.14988, 19: 0.17351, 20: 0.18659,
               21: 0.20971, 22: 0.22728, 23: 0.24951, 24: 0.26313, 25: 0.27662,
               26: 0.29430, 27: 0.31399, 28: 0.32494, 29: 0.34796, 30: 0.36118,
               31: 0.37531, 32: 0.39056, 33: 0.39958, 34: 0.41473, 35: 0.42560,
               36: 0.44222, 37: 0.45396, 38: 0.46275, 39: 0.47550, 40: 0.48889,
               41: 0.50009, 42: 0.51312, 43: 0.52463, 44: 0.52903, 45: 0.53930,
               46: 0.55289, 47: 0.56343, 48: 0.57204, 49: 0.58184, 50: 0.58852}

small_slope_t8 = {2: 0.04473, 3: 0.04472, 4: 0.04402, 5: 0.04407, 6: 0.04334,
                  7: 0.04326, 8: 0.04218, 9: 0.04237, 10: 0.04144, 11: 0.04054,
                  12: 0.03961, 13: 0.03862, 14: 0.03745, 15: 0.03673, 16: 0.03585,
                  17: 0.03477, 18: 0.03378, 19: 0.03298, 20: 0.03222, 21: 0.03155,
                  22: 0.03088, 23: 0.03029, 24: 0.02999, 25: 0.02954, 26: 0.02922,
                  27: 0.02891, 28: 0.02878, 29: 0.02850, 30: 0.02827, 31: 0.02801,
                  32: 0.02786, 33: 0.02761, 34: 0.02768, 35: 0.02744, 36: 0.02728,
                  37: 0.02713, 38: 0.02689, 39: 0.02678, 40: 0.02671, 41: 0.02647,
                  42: 0.02634, 43: 0.02614, 44: 0.02595, 45: 0.02583, 46: 0.02559,
                  47: 0.02534, 48: 0.02514, 49: 0.02506, 50: 0.02493, 51: 0.02475,
                  52: 0.02454, 53: 0.02441, 54: 0.02427, 55: 0.02407, 56: 0.02393,
                  57: 0.02371, 58: 0.02366, 59: 0.02341, 60: 0.02332}


@dataclass
class LazyEvaluation:
    f: Callable
    max_n_cache: int
    eval: list = field(default_factory=lambda: [])

    def __getitem__(self, key):
        if not self.eval:
            self.eval = [self.f(i) for i in range(self.max_n_cache + 1)]
        return self.eval[key]


zeta_precomputed = LazyEvaluation(lambda i: float(szeta(i)) if i != 1 else OO, max_n_cache)
zeta_prime_precomputed = LazyEvaluation(zeta_prime, max_n_cache)


class Bounds(NamedTuple):
    low: Any
    high: Any


class local_minimum_base:
    def __init__(self, start, stop, smallerf=lambda x, best: x <= best,
                 suppress_bounds_warning=False, log_level=5):
        if stop < start:
            raise ValueError(f"Incorrect bounds {start} > {stop}.")
        self._suppress_bounds_warning = suppress_bounds_warning
        self._log_level = log_level
        self._start = start
        self._stop = stop - 1
        self._initial_bounds = Bounds(start, stop - 1)
        self._smallerf = smallerf
        self._direction = -1
        self._last_x = None
        self._next_x = self._stop
        self._best = Bounds(None, None)
        self._all_x = set()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        pass

    def __iter__(self):
        return self

    def __next__(self):
        if (self._next_x is not None and self._next_x not in self._all_x
                and self._initial_bounds.low <= self._next_x <= self._initial_bounds.high):
            self._last_x = self._next_x
            self._next_x = None
            return self._last_x

        if self._best.low in self._initial_bounds and not self._suppress_bounds_warning:
            msg = f'warning: "optimal" solution {self._best.low} matches a bound ∈ {self._initial_bounds}.'
            Logging.log("bins", self._log_level, msg)

        raise StopIteration

    @property
    def x(self):
        return self._best.low

    @property
    def y(self):
        return self._best.high

    def update(self, res):
        Logging.log("bins", self._log_level, f"({self._last_x}, {repr(res)})")
        self._all_x.add(self._last_x)

        if self._best.low is None:
            self._best = Bounds(self._last_x, res)

        if res is not False and self._smallerf(res, self._best.high):
            self._best = Bounds(self._last_x, res)
            if abs(self._direction) != 1:
                self._direction = -1
                self._next_x = self._last_x - 1
            elif self._direction == -1:
                self._direction = -2
                self._stop = self._last_x
                self._next_x = math.ceil((self._start + self._stop) / 2)
            elif self._direction == 1:
                self._direction = 2
                self._start = self._last_x
                self._next_x = math.floor((self._start + self._stop) / 2)
        else:
            if self._direction == -1:
                self._direction = 1
                self._next_x = self._last_x + 2
            elif self._direction == 1:
                self._next_x = None
            elif self._direction == -2:
                self._start = self._last_x
                self._next_x = math.ceil((self._start + self._stop) / 2)
            elif self._direction == 2:
                self._stop = self._last_x
                self._next_x = math.floor((self._start + self._stop) / 2)

        if self._next_x == self._last_x:
            self._next_x = None


class local_minimum(local_minimum_base):
    def __init__(self, start, stop, precision=1,
                 smallerf=lambda x, best: x <= best,
                 suppress_bounds_warning=False, log_level=5):
        self._precision = precision
        self._orig_bounds = (start, stop)
        start = math.ceil(start / precision)
        stop = math.floor(stop / precision)
        super().__init__(start, stop, smallerf, suppress_bounds_warning, log_level)

    def __next__(self):
        x = super().__next__()
        return x * self._precision

    @property
    def x(self):
        return self._best.low * self._precision

    @property
    def neighborhood(self):
        start_bound, stop_bound = self._orig_bounds
        start = max(start_bound, self.x - self._precision)
        stop = min(stop_bound, self.x + self._precision)
        return range(start, stop)


class early_abort_range:
    def __init__(self, start, stop=OO, step=1,
                 smallerf=lambda x, best: x <= best,
                 suppress_bounds_warning=False, log_level=5):
        if stop < start:
            raise ValueError(f"Incorrect bounds {start} > {stop}.")
        self._suppress_bounds_warning = suppress_bounds_warning
        self._log_level = log_level
        self._start = start
        self._step = step
        self._stop = stop
        self._smallerf = smallerf
        self._last_x = None
        self._next_x = self._start
        self._best = Bounds(None, None)

    def __iter__(self):
        return self

    def __next__(self):
        if self._next_x is None or self._next_x >= self._stop:
            raise StopIteration
        self._last_x = self._next_x
        self._next_x += self._step
        return self._last_x, self

    @property
    def x(self):
        return self._best.low

    @property
    def y(self):
        return self._best.high

    def update(self, res):
        Logging.log("lins", self._log_level, f"({self._last_x}, {repr(res)})")
        if self._best.low is None:
            self._best = Bounds(self._last_x, res)
            return
        if res is False:
            self._next_x = None
        elif self._smallerf(res, self._best.high):
            self._best = Bounds(self._last_x, res)
        else:
            self._next_x = None


def binary_search(f, start, stop, param, step=1,
                  smallerf=lambda x, best: x <= best, log_level=5, *args, **kwds):
    with local_minimum(start, stop + 1, step, smallerf=smallerf, log_level=log_level) as it:
        for x in it:
            kwds_ = dict(kwds)
            kwds_[param] = x
            it.update(f(*args, **kwds_))
        for x in it.neighborhood:
            kwds_ = dict(kwds)
            kwds_[param] = x
            it.update(f(*args, **kwds_))
        return it.y


def _batch_estimatef(f, x, log_level=0, f_repr=None, catch_exceptions=True):
    try:
        y = f(x)
    except Exception as e:
        if catch_exceptions:
            print(f"Algorithm {f_repr} on {x} failed with {e}")
            return None
        raise
    if f_repr is None:
        f_repr = repr(f)
    Logging.log("batch", log_level, f"f: {f_repr}")
    Logging.log("batch", log_level, f"x: {x}")
    Logging.log("batch", log_level, f"f(x): {y!r}")
    return y


def f_name(f):
    try:
        return f.__name__
    except AttributeError:
        return repr(f)


class Task(NamedTuple):
    f: Callable
    x: LWEParameters
    log_level: int
    f_name: str
    catch_exceptions: bool


@dataclass(frozen=True)
class TaskResults:
    _map: dict

    def __getitem__(self, params):
        return {
            task.f_name: result
            for task, result in self._map.items()
            if task.x == params and result is not None
        }


def batch_estimate(params, algorithm, jobs=1, log_level=0, catch_exceptions=True, **kwds):
    if isinstance(params, (LWEParameters, SISParameters)):
        params = (params,)
    if not hasattr(algorithm, "__iter__"):
        algorithm = (algorithm,)
    tasks = [
        Task(partial(f, **kwds), x, log_level, f_name(f), catch_exceptions)
        for f, x in it.product(algorithm, params)
    ]
    if jobs == 1:
        results = [_batch_estimatef(*task) for task in tasks]
    else:
        with Pool(jobs) as pool:
            results = pool.starmap(_batch_estimatef, tasks)
    return TaskResults(dict(zip(tasks, results)))
