# -*- coding: utf-8 -*-
from dataclasses import dataclass
import math

OO = float('inf')


@dataclass
class SISParameters:
    n: int
    q: int
    length_bound: float
    m: int = None
    norm: int = 2
    tag: str = None

    def __post_init__(self, **kwds):
        if self.m is None:
            if self.norm == OO:
                self.m = 2 * math.ceil(self.n * math.log(self.q, 2 * self.length_bound + 1))
            else:
                self.m = 2 * math.ceil(self.n * math.log2(self.q))

    @property
    def _homogeneous(self):
        return True

    def updated(self, **kwds):
        d = dict(self.__dict__)
        d.update(kwds)
        return SISParameters(**d)

    def __hash__(self):
        return hash((self.n, self.q, self.length_bound, self.norm, self.m, self.tag))
