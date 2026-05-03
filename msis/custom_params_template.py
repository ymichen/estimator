from src.params import DilithiumParams

params = DilithiumParams(
    name="MyLevel",
    q=8380417,
    k=5,
    l=5,
    tau=40,
    d=14,
    gamma1=1 << 18,
    gamma2=(8380417 - 1) // 64,
    beta=100,
)
