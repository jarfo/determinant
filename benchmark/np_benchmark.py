import timeit

import numpy as np

from determinant import BCHdet, BRdet, CVdet, DPdet, FLdet, MPdet

n = 50
A = np.random.randint(0, 100, size=(n, n), dtype=int).astype(object)
Af = A.astype("float64")


def benchmark():
    print(timeit.repeat("numpy.linalg.det(Af)", setup="import numpy; from __main__ import Af", number=10, repeat=5))
    #: [0.0035009384155273, 0.0033931732177734, 0.0033941268920898, 0.0033800601959229, 0.0033988952636719]
    print(timeit.repeat("BRdet(A)", setup="from __main__ import BRdet, A", number=10, repeat=5))
    print(timeit.repeat("DPdet(A)", setup="from __main__ import DPdet, A", number=10, repeat=5))
    print(timeit.repeat("CVdet(A)", setup="from __main__ import CVdet, A", number=10, repeat=5))
    print(timeit.repeat("MPdet(A)", setup="from __main__ import MPdet, A", number=10, repeat=5))
    print(timeit.repeat("FLdet(A)", setup="from __main__ import FLdet, A", number=10, repeat=5))
    print(timeit.repeat("BCHdet(A)", setup="from __main__ import BCHdet, A", number=10, repeat=5))


if __name__ == "__main__":
    benchmark()
