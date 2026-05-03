# Pure Python, no Sage
from .nd import stddevf, Binary, CenteredBinomial, DiscreteGaussian, SparseTernary, UniformMod
from .lwe_parameters import LWEParameters
from .ntru_parameters import NTRUParameters
from .sis_parameters import SISParameters

OO = float('inf')

#
# Kyber
#

Kyber512 = LWEParameters(
    n=2 * 256,
    q=3329,
    Xs=CenteredBinomial(3),
    Xe=CenteredBinomial(3),
    m=2 * 256,
    tag="Kyber 512",
)

Kyber768 = LWEParameters(
    n=3 * 256,
    q=3329,
    Xs=CenteredBinomial(2),
    Xe=CenteredBinomial(2),
    m=3 * 256,
    tag="Kyber 768",
)

Kyber1024 = LWEParameters(
    n=4 * 256,
    q=3329,
    Xs=CenteredBinomial(2),
    Xe=CenteredBinomial(2),
    m=4 * 256,
    tag="Kyber 1024",
)

#
# Saber
#

LightSaber = LWEParameters(
    n=2 * 256,
    q=8192,
    Xs=CenteredBinomial(5),
    Xe=UniformMod(8),
    m=2 * 256,
    tag="LightSaber",
)

Saber = LWEParameters(
    n=3 * 256,
    q=8192,
    Xs=CenteredBinomial(4),
    Xe=UniformMod(8),
    m=3 * 256,
    tag="Saber",
)

FireSaber = LWEParameters(
    n=4 * 256,
    q=8192,
    Xs=CenteredBinomial(3),
    Xe=UniformMod(8),
    m=4 * 256,
    tag="FireSaber",
)

#
# NTRU
#

NTRUHPS2048509Enc = NTRUParameters(
    n=508,
    q=2048,
    Xe=SparseTernary(2048 // 16 - 1),
    Xs=UniformMod(3),
    m=508,
    tag="NTRUHPS2048509Enc",
)

NTRUHPS2048677Enc = NTRUParameters(
    n=676,
    q=2048,
    Xs=UniformMod(3),
    Xe=SparseTernary(2048 // 16 - 1),
    m=676,
    tag="NTRUHPS2048677Enc",
)

NTRUHPS4096821Enc = NTRUParameters(
    n=820,
    q=4096,
    Xs=UniformMod(3),
    Xe=SparseTernary(4096 // 16 - 1),
    m=820,
    tag="NTRUHPS4096821Enc",
)

NTRUHRSS701Enc = NTRUParameters(
    n=700,
    q=8192,
    Xs=UniformMod(3),
    Xe=UniformMod(3),
    m=700,
    tag="NTRUHRSS701",
)

#
# Dilithium
#

Dilithium2_MSIS_WkUnf = SISParameters(
    n=256*4,
    q=8380417,
    length_bound=350209,
    m=256*9,
    norm=OO,
    tag="Dilithium2_MSIS_WkUnf"
)

Dilithium2_MSIS_StrUnf = SISParameters(
    n=256*4,
    q=8380417,
    length_bound=380929,
    m=256*9,
    norm=OO,
    tag="Dilithium2_MSIS_StrUnf"
)

Dilithium3_MSIS_WkUnf = SISParameters(
    n=256*6,
    q=8380417,
    length_bound=724481,
    m=256*6*2,
    norm=OO,
    tag="Dilithium3_MSIS_WkUnf"
)

Dilithium3_MSIS_StrUnf = SISParameters(
    n=256*6,
    q=8380417,
    length_bound=1048576,
    m=256*6*2,
    norm=OO,
    tag="Dilithium3_MSIS_StrUnf"
)

Dilithium5_MSIS_WkUnf = SISParameters(
    n=256*8,
    q=8380417,
    length_bound=769537,
    m=256*8*2,
    norm=OO,
    tag="Dilithium5_MSIS_WkUnf"
)

Dilithium5_MSIS_StrUnf = SISParameters(
    n=256*8,
    q=8380417,
    length_bound=1048576,
    m=256*8*2,
    norm=OO,
    tag="Dilithium5_MSIS_StrUnf"
)

NISTPQC_R3 = (
    Kyber512,
    Kyber768,
    Kyber1024,
    LightSaber,
    Saber,
    FireSaber,
    NTRUHPS2048509Enc,
    NTRUHPS2048677Enc,
    NTRUHPS4096821Enc,
    NTRUHRSS701Enc,
)

#
# Falcon
#

Falcon512_Unf = SISParameters(
    n=512,
    q=12289,
    length_bound=5833.9072,
    m=1024,
    norm=2,
    tag="Falcon512_Unf"
)

Falcon512_SKR = NTRUParameters(
    n=512,
    q=12289,
    Xs=DiscreteGaussian(4.0532),
    Xe=DiscreteGaussian(4.0532),
    m=512,
    ntru_type='circulant',
    tag="Falcon512_SKR"
)

Falcon1024_Unf = SISParameters(
    n=1024,
    q=12289,
    length_bound=8382.4081,
    m=2048,
    norm=2,
    tag="Falcon1024_Unf"
)

Falcon1024_SKR = NTRUParameters(
    n=1024,
    q=12289,
    Xs=DiscreteGaussian(2.866),
    Xe=DiscreteGaussian(2.866),
    m=1024,
    ntru_type='circulant',
    tag="Falcon1024_SKR"
)

# FrodoKEM
Frodo640 = LWEParameters(
    n=640,
    q=2**15,
    Xs=DiscreteGaussian(2.8),
    Xe=DiscreteGaussian(2.8),
    m=640 + 16,
    tag="Frodo640",
)

Frodo976 = LWEParameters(
    n=976,
    q=2**16,
    Xs=DiscreteGaussian(2.3),
    Xe=DiscreteGaussian(2.3),
    m=976 + 16,
    tag="Frodo976",
)

Frodo1344 = LWEParameters(
    n=1344,
    q=2**16,
    Xs=DiscreteGaussian(1.4),
    Xe=DiscreteGaussian(1.4),
    m=1344 + 16,
    tag="Frodo1344",
)

