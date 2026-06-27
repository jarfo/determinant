import timeit

import numpy as np
from flint import fmpz_mat

from determinant import BCHdet, BIdet, BRdet, CVdet, DPdet, FLdet, KAdet, MPdet, STdet


def benchmark():
    print("numpy.linalg.det(Af)", timeit.repeat("numpy.linalg.det(Af)", setup="import numpy; from __main__ import Af", number=10, repeat=5))
    #: [0.0035009384155273, 0.0033931732177734, 0.0033941268920898, 0.0033800601959229, 0.0033988952636719]
    print("fmpz_mat.det()", timeit.repeat("Afmpz.det()", setup="from __main__ import Afmpz", number=10, repeat=5))
    print("BRdet(A)", timeit.repeat("BRdet(A)", setup="from __main__ import BRdet, A", number=10, repeat=5))
    print("DPdet(A)", timeit.repeat("DPdet(A)", setup="from __main__ import DPdet, A", number=10, repeat=5))
    print("CVdet(A)", timeit.repeat("CVdet(A)", setup="from __main__ import CVdet, A", number=10, repeat=5))
    print("MPdet(A)", timeit.repeat("MPdet(A)", setup="from __main__ import MPdet, A", number=10, repeat=5))
    print("FLdet(A)", timeit.repeat("FLdet(A)", setup="from __main__ import FLdet, A", number=10, repeat=5))
    print("BCHdet(A)", timeit.repeat("BCHdet(A)", setup="from __main__ import BCHdet, A", number=10, repeat=5))
    print("BIdet(A)", timeit.repeat("BIdet(A)", setup="from __main__ import BIdet, A", number=10, repeat=5))
    print("STdet(A)", timeit.repeat("STdet(A)", setup="from __main__ import STdet, A", number=10, repeat=5))
    print("KAdet(A)", timeit.repeat("KAdet(A)", setup="from __main__ import KAdet, A", number=10, repeat=5))


if __name__ == "__main__":
    n = 30
    A = np.random.randint(0, 100, size=(n, n), dtype=int).astype(object)
    Af = A.astype("float64")
    Afmpz = fmpz_mat(A.tolist())
    benchmark()
