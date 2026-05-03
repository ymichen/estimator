from dataclasses import dataclass
import math
import json

@dataclass(frozen=True)
class DilithiumParams:
    name: str
    q: int
    n: int = 256
    k: int = 4
    l: int = 4
    tau: int = 39
    d: int = 13
    gamma1: int = 131072
    gamma2: int = 95232
    beta: int = 78
    eta: int = 2          # 用于完整性，评估未用

    @property
    def m(self): return self.n * self.k
    @property
    def logq(self): return math.log2(self.q)
    @property
    def zeta_unc(self): return max(self.gamma1 - self.beta, 2 * self.gamma2 + 1)
    @property
    def zeta_comp(self): return max(self.gamma1 - self.beta, 2 * self.gamma2 + 1 + (1 << (self.d - 1)) * self.tau)
    @property
    def zeta_suf(self): return max(2 * (self.gamma1 - self.beta), 4 * self.gamma2 + 2)
    @property
    def w_max_unc(self): return self.n * (self.l + 1)
    @property
    def w_max_comp(self): return self.n * (self.l + self.k + 1)
    @property
    def w_max_suf(self): return self.n * (self.l + self.k)

    @classmethod
    def from_dict(cls, d: dict) -> "DilithiumParams":
        """从字典构造（例如读取 JSON 或 YAML）"""
        return cls(**d)


# ---------- 预定义参数集 ----------
# NIST Level 2
PARAMS_LEVEL2 = DilithiumParams(
    name="Level2",
    q=8380417, n=256, k=4, l=4,
    tau=39, d=13,
    gamma1=1 << 17,
    gamma2=(8380417 - 1) // 88,
    beta=78, eta=2,
)

# NIST Level 3
PARAMS_LEVEL3 = DilithiumParams(
    name="Level3",
    q=8380417, n=256, k=6, l=5,
    tau=49, d=13,
    gamma1=1 << 19,
    gamma2=(8380417 - 1) // 32,
    beta=196, eta=4,
)

# NIST Level 5
PARAMS_LEVEL5 = DilithiumParams(
    name="Level5",
    q=8380417, n=256, k=8, l=7,
    tau=60, d=13,
    gamma1=1 << 19,
    gamma2=(8380417 - 1) // 32,
    beta=120, eta=2,
)

# 挑战参数 1--
PARAMS_CHALLENGE_1MM = DilithiumParams(
    name="1--",
    q=8380417, n=256, k=2, l=2,
    tau=24, d=10,
    gamma1=1 << 17,
    gamma2=(8380417 - 1) // 128,
    beta=144, eta=6,
)

# 挑战参数 1-
PARAMS_CHALLENGE_1M = DilithiumParams(
    name="1-",
    q=8380417, n=256, k=3, l=3,
    tau=30, d=13,
    gamma1=1 << 17,
    gamma2=(8380417 - 1) // 128,
    beta=90, eta=3,
)

# 挑战参数 5+
PARAMS_CHALLENGE_5P = DilithiumParams(
    name="5+",
    q=8380417, n=256, k=9, l=8,
    tau=60, d=13,
    gamma1=1 << 19,
    gamma2=(8380417 - 1) // 32,
    beta=120, eta=2,
)

# 挑战参数 5++
PARAMS_CHALLENGE_5PP = DilithiumParams(
    name="5++",
    q=8380417, n=256, k=10, l=9,
    tau=60, d=13,
    gamma1=1 << 19,
    gamma2=(8380417 - 1) // 32,
    beta=120, eta=2,
)

# 便捷字典：名称到参数实例的映射
ALL_BUILTIN = {
    "2":    PARAMS_LEVEL2,
    "3":    PARAMS_LEVEL3,
    "5":    PARAMS_LEVEL5,
    "1--":  PARAMS_CHALLENGE_1MM,
    "1-":   PARAMS_CHALLENGE_1M,
    "5+":   PARAMS_CHALLENGE_5P,
    "5++":  PARAMS_CHALLENGE_5PP,
}
