import numpy as np
import sympy as sp
from sympy.abc import lamda

from .util import evector, get_dtype


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


# Clow-based algorithm. Gradients of every characteristic-polynomial coefficient.
#
# MPcoefs computes coefs[i] = r @ M^i @ s, the coefficient of lambda^{n-i} in
# det(lambda*I - A); MPdet just reads off the last one. Because r and s are
# constant and M = M(A) is linear in A, the SAME forward sweep L_k = r @ M^k and
# backward sweep R_k = M^k @ s yield the gradient of EVERY coefficient at once:
#
#   d coefs[i] / d A[a,b] = sum_{k=0}^{i-1} (r @ M^k) @ (dM/dA[a,b]) @ (M^{i-1-k} @ s)
#                         = <dM/dA[a,b], W_i>,  W_i = sum_{k=0}^{i-1} outer(L_k, R_{i-1-k}).
#
# W_i is the discrete convolution of the two sweeps, so all of them come from a
# single pass. Since each M[u, v] is 0 or +/- A[a, b], W_i[u, v] is routed back
# through DPmatrix's construction (with the matching sign) to form the gradient
# cofactors[i] = d coefs[i] / d A.
#
# These gradients are the coefficient matrices of the resolvent expansion
#   adj(lambda*I - A) = sum_{i=0}^{n} (-cofactors[i].T) * lambda^{n-i},
# the determinant's cofactor matrix being the last one (see MPcofactor).
def MPcofactors(A):
    n = A.shape[0]
    M, r, s = DPmatrix(A)
    m = M.shape[0]
    dtype = get_dtype(A)

    # Forward sweep L[k] = r @ M^k and backward sweep R[k] = M^k @ s, k = 0..n-1.
    L = np.zeros((n, m), dtype=dtype)
    R = np.zeros((n, m), dtype=dtype)
    Mk_r, Mk_s = r, s
    for k in range(n):
        L[k] = Mk_r[0]
        R[k] = Mk_s[:, 0]
        Mk_r = Mk_r @ M
        Mk_s = M @ Mk_s

    # cofactors[i] = d coefs[i] / d A; cofactors[0] = d(1)/dA = 0.
    cofactors = [np.zeros((n, n), dtype=dtype)]
    for i in range(1, n + 1):
        # W[u, v] = sum_{k=0}^{i-1} L[k][u] * R[i-1-k][v]
        W = np.zeros((m, m), dtype=dtype)
        for k in range(i):
            W += np.outer(L[k], R[i - 1 - k])

        # Mirror DPmatrix's construction, routing W back to the gradient with sign.
        G = np.zeros((n, n), dtype=dtype)
        d = 0
        for b in range(n + 1):
            p = 0
            for j in range(b):
                # M[d, p:p + n - j] = -A[j, j:]
                G[j, j:] -= W[d, p:p + n - j]
                p += n - j
            if b < n:
                # M[d + 1:d + n - b, d:d + n - b] = A[b + 1:, b:]
                G[b + 1:, b:] += W[d + 1:d + n - b, d:d + n - b]
            d += n - b
        cofactors.append(G)

    return cofactors


# Clow-based algorithm. Cofactor matrix (entrywise derivative of the determinant).
#
# det(A) = (-1)^n * coefs[-1], so the cofactor matrix d det / d A is the last
# coefficient gradient from MPcofactors, scaled by (-1)^n. The result
# G[i, j] = d det / d A[i, j] is the transpose of the classical adjugate:
# adj(A) = G.T = det(A) * inv(A).
def MPcofactor(A):
    n = A.shape[0]
    cofactors = MPcofactors(A)
    cofactor = cofactors[-1] * (-1)**n
    return cofactor


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
    dtype = get_dtype(A)
    A = np.array(A, dtype=dtype)
    sign = -1 if n % 2 else 1

    P = np.array([-1, A[n-1, n-1]], dtype=dtype)
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


# Bivariate Cayley-Hamilton recursion
#
# "On the gradient of the coefficient of the characteristic polynomial", Christian Ikenmeyer (2025)
# Proposition 2.6, eq. (2.8):
#   chi_{n,d} = chi_{n-1,d} + sum_{i=1}^{d} (-1)^{i+1} chi_{n,d-i} * pow_{n,i}
# where chi_{n,d} is the sum of d-by-d principal minors of the leading n-by-n submatrix
# (the d-th elementary symmetric polynomial of its eigenvalues) and pow_{n,i} = [X_n^i]_{n,n}.
def BCHcoefs(A):
    N = A.shape[0]
    dtype = get_dtype(A)
    A = np.array(A, dtype=dtype)

    chi = [1]
    for n in range(1, N+1):
        Xn = A[:n, :n]
        # pows[i] = [X_n^{i+1}]_{n,n}, computed via the row-vector recurrence
        # r_k = e_n^T X_n^k, so pows[i] = r_{i+1}[n-1]. Avoids materializing full
        # matrix powers and drops per-step work from O(n^3) to O(n^2).
        pows = np.zeros(n, dtype=dtype)
        r = np.zeros(n, dtype=dtype)
        r[n-1] = 1
        for i in range(n):
            r = r @ Xn
            pows[i] = r[n-1]

        new_chi = [1]
        for d in range(1, n+1):
            val = chi[d] if d < n else 0
            for i in range(1, d+1):
                term = new_chi[d-i] * pows[i-1]
                val = val + term if i % 2 == 1 else val - term
            new_chi.append(val)
        chi = new_chi

    return [(-1)**d * chi[d] for d in range(N+1)]


# Bivariate Cayley-Hamilton recursion
def BCHdet(A):
    n = A.shape[0]
    coefs = BCHcoefs(A)
    det = coefs[-1] * (-1)**n
    return det


# Bivariate Cayley-Hamilton recursion
def BCHcharpoly(A):
    n = A.shape[0]
    coefs = BCHcoefs(A)
    p = np.sum([c * lamda**(n-i) for i, c in enumerate(coefs)])
    return sp.PurePoly(p, lamda)


# Bivariate Cayley-Hamilton recursion. Gradients of every coefficient.
#
# "On the gradient of the coefficient of the characteristic polynomial",
# Christian Ikenmeyer (2025), Theorem 5.1 (Bivariate Cayley-Hamilton):
#   (grad chi_{n,d+1})^T = sum_{i=0}^{d} (-1)^i chi_{n,d-i} X_n^i,
# where chi_{n,d} is the sum of d-by-d principal minors of the leading n-by-n
# submatrix and grad has entries d/dA[a,b]. Corollary 2.2 (d = n-1) is the
# adjugate; the paper notes grad(det) is the cofactor matrix, (grad det)^T the
# adjugate.
#
# BCHcoefs returns coefs[d] = (-1)^d chi_{N,d}, so coefs[d]'s gradient is
# cofactors[d] = (-1)^d grad chi_{N,d}. Rewriting Theorem 5.1 in this convention
# turns the closed form into a Horner / Faddeev-LeVerrier recursion on the
# adjugate-expansion matrices S_e = sum_{i=0}^{e-1} coefs[e-1-i] A^i:
#   S_e = coefs[e-1] * I + A @ S_{e-1},   cofactors[e] = -S_e.T   (S_0 = 0).
# cofactors[N] * (-1)^N is the cofactor matrix d det / dA (see BCHcofactor), and
# the S_e are the coefficient matrices of adj(lambda*I - A) = sum_e S_e lambda^{N-e}.
def BCHcofactors(A):
    N = A.shape[0]
    dtype = get_dtype(A)
    A = np.array(A, dtype=dtype)
    coefs = BCHcoefs(A)

    Id = np.eye(N, dtype=dtype)
    cofactors = [np.zeros((N, N), dtype=dtype)]   # d coefs[0]/dA = d(1)/dA = 0
    S = np.zeros((N, N), dtype=dtype)             # S_0 = 0
    for e in range(1, N+1):
        S = coefs[e-1] * Id + A @ S               # S_e = coefs[e-1] I + A S_{e-1}
        cofactors.append(-S.T)

    return cofactors


# Bivariate Cayley-Hamilton recursion. Cofactor matrix (entrywise derivative of
# the determinant): the last coefficient gradient scaled by (-1)^n, exactly as
# BCHdet reads off BCHcoefs[-1].
def BCHcofactor(A):
    n = A.shape[0]
    cofactors = BCHcofactors(A)
    cofactor = cofactors[-1] * (-1)**n
    return cofactor
