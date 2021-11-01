import math
import numpy as np
import sympy as sp
from sympy.abc import lamda
from .util import get_dtype, evector


# Bareiss algorithm
def BRdet(A):
    # Make a copy since A is modified inplace
    A = np.array(A)
    N, sign, prev = A.shape[0], 1, 1
    for i in range(N-1):
        if A[i, i] == 0: # swap with another row having nonzero i's elem
            swapto = next((j for j in range(i+1,N) if A[j, i] != 0), None)
            if swapto is None:
                return 0 # all A[*][i] are zero => zero determinant
            A[[i, swapto]], sign = A[[swapto, i]], -sign

        A[i+1:, i+1:] = (A[i+1:, i+1:] * A[i, i] - A[i+1:, i:i+1] * A[i:i+1, i+1:]) // prev
        prev = A[i, i]
    return sign * A[-1, -1]


# Faddeev–LeVerrier algorithm
def FLcoefs(A):
    n = A.shape[0]
    dtype = get_dtype(A)
    coefs = np.zeros(n+1, dtype=dtype)
    traces = np.zeros(n, dtype=dtype)

    coefs[n] = 1
    Apow = np.eye(n, dtype=dtype)
    for m in range(1, n+1):
        Apow = Apow @ A
        traces[m - 1] = np.trace(Apow)
        coef = - np.dot(coefs[n-m+1:], traces[:m])
        if isinstance(A[0,0], sp.Symbol) or np.issubdtype(type(A[0,0]), np.floating):
            coefs[n-m] = coef/m
        else:
            coefs[n-m] = coef//m

    return list(reversed(coefs))


# Faddeev–LeVerrier algorithm
def FLdet(A):
    n = A.shape[0]
    coefs = FLcoefs(A)
    det = coefs[-1] * (-1)**n
    return det


# Faddeev–LeVerrier algorithm
def FLcharpoly(A):
    n = A.shape[0]
    coefs = FLcoefs(A)
    p = np.sum([c * lamda**(n-i) for i, c in enumerate(coefs)])
    return sp.PurePoly(p, lamda)


# Clow-based algorithm. Dynamic Programming According to Length
#
# "Determinant: combinatorics, algorithms, complexity", M. Mahajan and V. Vinay
#
# "Determinant: Old Algorithms, New Insights.", M. Mahajan and V. Vinay
# 3.1. From cycle covers to clow sequences.
#
# "Division-Free Algorithms for the Determinant and the Pfaffian: Algebraic and Combinatorial Approaches", Günter Rote
# 3.1 Dynamic Programming According to Length
def DPcoefs(A):
    n = A.shape[0]
    D = np.eye(n, dtype=object)
    diag = np.zeros(n, dtype=object)
    coefs = [1]
    for i in range(n):
        D = np.triu(D @ A)
        cumsum = D.diagonal().cumsum()
        coefs.append(-cumsum[-1])
        diag[1:] = -cumsum[0:-1]
        np.fill_diagonal(D, diag)

    return coefs


# Clow-based algorithm. Dynamic Programming According to Length
def DPdet(A):
    n = A.shape[0]
    coefs = DPcoefs(A)
    det = coefs[-1] * (-1)**n
    return det


# Clow-based algorithm. Dynamic Programming According to Length
def DPcharpoly(A):
    n = A.shape[0]
    coefs = DPcoefs(A)
    p = np.sum([c * lamda**(n-i) for i, c in enumerate(coefs)])
    return sp.PurePoly(p, lamda)


# Clow-based algorithm. Explicit Matrix power method for the Dynamic Programming According to Length
def DPmatrix(A):
    n = A.shape[0]
    m = ((n+1)*n)//2 + 1
    dtype = get_dtype(A)

    r = np.zeros((1, m), dtype=dtype)
    r[0, m-1] = 1

    s = np.zeros((m, 1), dtype=dtype)
    icol = 0
    for i in range(n+1):
        s[icol, 0] = 1
        icol += n - i

    M = np.zeros((m, m), dtype=dtype)
    d = 0
    for i in range(n+1):
        p = 0
        for j in range(i):
            M[d, p:p + n - j] = -A[j, j:]
            p += n - j
        if i < n:
            M[d + 1: d + n - i, d: d + n - i] = A[i + 1:, i:]
        d += n - i

    return M, r, s


# Clow-based algorithm. Explicit Matrix power method for the Dynamic Programming According to Length
def MPcoefs(A):
    n = A.shape[0]
    M, r, s = DPmatrix(A)

    Ms = s
    coefs = [1]
    for _ in range(n):
        Ms = M @ Ms
        coefs.append((r @ Ms)[0, 0])

    return coefs


# Clow-based algorithm. Explicit Matrix power method for the Dynamic Programming According to Length
def MPdet(A):
    n = A.shape[0]
    coefs = MPcoefs(A)
    det = coefs[-1] * (-1)**n
    return det


# Clow-based algorithm. Explicit Matrix power method for the Dynamic Programming According to Length
def MPcharpoly(A):
    n = A.shape[0]
    coefs = MPcoefs(A)
    p = np.sum([c * lamda**(n-i) for i, c in enumerate(coefs)])
    return sp.PurePoly(p, lamda)


# "Determinant: Old Algorithms, New Insights.", M. Mahajan and V. Vinay
# 3.2 Clow sequences with the prefix property: Getting to Samuelson’s method

# "Division-Free Algorithms for the Determinant and the Pfaffian: Algebraic and Combinatorial Approaches", Günter Rote
# 3.2 Adding a Vertex at a Time: Combinatorial Approach
def CVcoefs(A):
    n = A.shape[0]
    sign = -1 if n % 2 else 1

    P = np.array([-1, A[n-1, n-1]], dtype=get_dtype(A))
    for i in range(n-2, -1, -1):
        r = A[i, i+1:]
        s = A[i+1:, i]
        M = A[i+1:, i+1:]
        D = [0]*(n - i - 1) + [-1, A[i, i], r @ s]
        rM = r
        for j in range(n - i - 2):
            rM = rM @ M
            D.append(rM @ s)
        P = np.correlate(D, P[::-1], mode='valid')

    return sign*P


# Clow sequences with the prefix property
def CVdet(A):
    n = A.shape[0]
    coefs = CVcoefs(A)
    det = coefs[-1] * (-1)**n
    return det


# Clow sequences with the prefix property
def CVcharpoly(A):
    n = A.shape[0]
    coefs = CVcoefs(A)
    p = np.sum([c * lamda**(n-i) for i, c in enumerate(coefs)])
    return sp.PurePoly(p, lamda)


# Chistoy's Algorithm
#
# "Determinant: Old Algorithms, New Insights.", M. Mahajan and V. Vinay
# 3.3 From clows to tour sequences tables: Getting to Chistov’s algorithm.
#
# "The Design and Analysis of Algorithms", Dexter C. Kazen
def CHcoefs(A):
    n = A.shape[0]
    dtype = get_dtype(A)
    A = np.array(A, dtype=dtype)

    B = np.ones((n, n+1), dtype=dtype)
    C = [evector(n-i, dtype=dtype) for i in range(n)]
    for i in range(n):
        C = [c @ A[i:,i:] for i, c in enumerate(C)]
        B[:, i+1] = [c[0] for c in C]

    d = B[0]
    for i in range(1, n):
        d = np.convolve(d, B[i])[:n+1]

    e = evector(n+1, dtype=dtype)
    for i in range(1, n+1):
        e[i] = - np.sum(d[1:i+1]*e[i-1::-1])

    return e


# Clow sequences with the prefix property
def CHdet(A):
    n = A.shape[0]
    coefs = CHcoefs(A)
    det = coefs[-1] * (-1)**n
    return det


# Clow sequences with the prefix property
def CHcharpoly(A):
    n = A.shape[0]
    coefs = CHcoefs(A)
    p = np.sum([c * lamda**(n-i) for i, c in enumerate(coefs)])
    return sp.PurePoly(p, lamda)


if __name__ == "__main__":
    import timeit

    for n in range(3, 5):
        M = sp.symarray('a', (n, n))
        assert CHcharpoly(M) == sp.Matrix(M).charpoly()
        assert CVcharpoly(M) == sp.Matrix(M).charpoly()
        assert FLcharpoly(M) == sp.Matrix(M).charpoly()
        assert MPcharpoly(M) == sp.Matrix(M).charpoly()
        assert DPcharpoly(M) == sp.Matrix(M).charpoly()

    for n in [15, 20]:
        A = np.random.randint(0, 100, size=(n, n), dtype=int).astype(object)
        print(BRdet(A))
        print(DPdet(A))
        print(CVdet(A))
        print(MPdet(A))
        print(FLdet(A))

    # print(timeit.repeat("numpy.linalg.det(A)", setup="import numpy; from __main__ import A", number=100, repeat=5))
    #: [0.0035009384155273, 0.0033931732177734, 0.0033941268920898, 0.0033800601959229, 0.0033988952636719]
    print(timeit.repeat("BRdet(A)", setup="from __main__ import BRdet, A", number=10, repeat=5))
    print(timeit.repeat("DPdet(A)", setup="from __main__ import DPdet, A", number=10, repeat=5))
    print(timeit.repeat("CVdet(A)", setup="from __main__ import CVdet, A", number=10, repeat=5))
    print(timeit.repeat("MPdet(A)", setup="from __main__ import MPdet, A", number=10, repeat=5))
    print(timeit.repeat("FLdet(A)", setup="from __main__ import FLdet, A", number=10, repeat=5))
