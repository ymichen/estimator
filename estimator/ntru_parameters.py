# -*- coding: utf-8 -*-
from dataclasses import dataclass
import math
from .conf import ntru_fatigue_lb
from .errors import InsufficientSamplesError
from .lwe_parameters import LWEParameters

OO = float('inf')

@dataclass
class NTRUParameters(LWEParameters):
    ntru_type: str = "matrix"

    def __post_init__(self, **kwds):
        super().__post_init__()
        self.m = self.n

    @property
    def possibly_overstretched(self):
        return self.q >= ntru_fatigue_lb(self.n)

    @property
    def _homogeneous(self):
        return True

    def normalize(self):
        if self.m < 1:
            raise InsufficientSamplesError(f"m={self.m} < 1")

        if self.Xe < self.Xs and self.m < 2 * self.n:
            return NTRUParameters(n=self.n, q=self.q, Xs=self.Xe, Xe=self.Xs, m=self.n,
                                  tag=self.tag, ntru_type=self.ntru_type)
        return self

    def updated(self, **kwds):
        d = dict(self.__dict__)
        d.update(kwds)
        return NTRUParameters(**d)

    def amplify_m(self, m):
        raise NotImplementedError("Rerandomizing NTRU instances is not supported yet.")

    def switch_modulus(self):
        raise NotImplementedError("Modulus Switching for NTRU not supported yet.")

    def __hash__(self):
        return hash((self.n, self.q, self.Xs, self.Xe, self.m, self.tag, self.ntru_type))
