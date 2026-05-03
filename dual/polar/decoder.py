from .interface_polar import *
class Polar_Code:
    def __init__(self, n, k, q, experimental_data=None):
        self.n, self.k, self.q = n, k, q
        self.C_polar = polar_random(q, n, k)
        self.mean_error = polar_mean_error(self.C_polar)
    def decode(self, y):
        return [tuple(polar_decode(self.C_polar, y))]
    def __del__(self):
        polar_free(self.C_polar)
