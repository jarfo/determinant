import numpy as np


def get_dtype(A) -> type:
    return A.dtype if hasattr(A, 'dtype') else object

def evector(n, i=0, dtype=int):
    e = np.zeros(n, dtype=dtype)
    e[i] = 1
    return e
