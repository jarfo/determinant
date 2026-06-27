import math

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


# Bird's algorithm.
#
# "A simple division-free algorithm for computing determinants", Richard S. Bird,
# Information Processing Letters 111 (2011) 1072-1074.
#
# Define the operator mu on an n-by-n matrix X (upper triangular):
#   mu(X)[i, j] = X[i, j]                 for i < j   (strict upper part copied)
#   mu(X)[i, i] = -sum_{k>i} X[k, k]      (negated trailing diagonal sum)
#   mu(X)[i, j] = 0                       for i > j
# Iterating F_1 = A, F_{p+1} = mu(F_p) @ A collapses to a single nonzero entry:
# F_n[0, 0] = (-1)^{n-1} det(A). Pure ring +/* throughout, so it is division-free
# and works over any commutative ring. Unlike the clow/Cayley-Hamilton methods
# here, this is a *direct* determinant algorithm (no characteristic-polynomial
# coefficients), so BIdet does not route through a BIcoefs sweep.
def _mu(X):
    n = X.shape[0]
    M = np.triu(X).copy()                          # diagonal + strict upper part
    d = X.diagonal()
    rev = np.cumsum(d[::-1])[::-1]                  # rev[i] = sum_{k>=i} d[k]
    trail = np.zeros(n, dtype=X.dtype)
    trail[:-1] = rev[1:]                            # trail[i] = sum_{k>=i+1} d[k]
    np.fill_diagonal(M, -trail)
    return M


# Bird's algorithm
def BIdet(A):
    A = np.array(A)
    n = A.shape[0]
    F = A
    for _ in range(n - 1):
        F = _mu(F) @ A
    return (-1)**(n - 1) * F[0, 0]


# Bird's algorithm, characteristic-polynomial coefficients.
#
# Running Bird's iteration on the matrix pencil lambda*I - A keeps every entry a
# polynomial in lambda; the final corner is F_n[0, 0] = (-1)^{n-1} det(lambda*I - A)
# = (-1)^{n-1} P_A(lambda), so (-1)^{n-1} F_n[0, 0] is the (monic) characteristic
# polynomial. Because lambda*I - A has degree <= 1, the matrix product collapses:
#   mu(F) @ (lambda*I - A) = lambda * mu(F) - mu(F) @ A,
# i.e. one shift in the lambda-degree axis plus an ordinary matrix product over A.
# Entries are carried as ascending-power coefficient vectors (last axis); the
# result is returned descending (leading coefficient first), like the others.
def _mu_poly(F):
    n = F.shape[0]
    M = np.zeros_like(F)
    iu = np.triu_indices(n, 1)
    M[iu] = F[iu]                                  # strict upper part copied
    diag = np.diagonal(F, axis1=0, axis2=1).T      # diag[i] = F[i, i, :]
    rev = np.cumsum(diag[::-1], axis=0)[::-1]       # rev[i] = sum_{k>=i} diag[k]
    for i in range(n - 1):
        M[i, i] = -rev[i + 1]                       # -sum_{k>i} F[k, k, :]
    return M


# Bird's algorithm
def BIcoefs(A):
    n = A.shape[0]
    dtype = get_dtype(A)
    A = np.array(A, dtype=dtype)

    # F[i, j, :] = polynomial in lambda (ascending powers); F = lambda*I - A.
    F = np.zeros((n, n, 2), dtype=dtype)
    F[:, :, 0] = -A
    for i in range(n):
        F[i, i, 1] = 1

    for _ in range(n - 1):
        muF = _mu_poly(F)
        D = muF.shape[2]
        newF = np.zeros((n, n, D + 1), dtype=dtype)
        newF[:, :, 1:] = muF                        # lambda * mu(F)
        newF[:, :, :D] -= np.einsum('ikd,kj->ijd', muF, A)  # - mu(F) @ A
        F = newF

    poly = (-1)**(n - 1) * F[0, 0]                  # ascending charpoly coeffs
    return list(poly[::-1])                         # descending (leading first)


# Bird's algorithm
def BIcharpoly(A):
    n = A.shape[0]
    coefs = BIcoefs(A)
    p = np.sum([c * lamda**(n-i) for i, c in enumerate(coefs)])
    return sp.PurePoly(p, lamda)


# Strassen's avoidance of divisions (power-series elimination).
#
# "Vermeidung von Divisionen", V. Strassen, J. Reine Angew. Math. 264 (1973).
#
# "On computing determinants of matrices without divisions", E. Kaltofen, ISSAC
# 1992 -- applies the same power-series division-avoidance to a baby-step/
# giant-step Krylov characteristic-polynomial method to lower the asymptotic
# exponent. That speedup only helps with fast matrix multiplication, so it is
# not reproduced here; this implements the underlying division-avoidance core.
#
# Key identity: det(I + zA) = sum_{k=0}^{n} E_k z^k, where E_k is the sum of the
# k-by-k principal minors of A (E_0 = 1, E_n = det A), i.e. the characteristic-
# polynomial coefficients. Since I + zA == I (mod z), Gaussian elimination over
# the truncated power-series ring R[z]/(z^{n+1}) needs no pivoting and every
# pivot is a unit (constant term 1); inverting a unit power series uses only
# ring +/-/*, so the whole computation is division-free over any commutative
# ring. A single elimination yields every E_k (hence the determinant and the
# full characteristic polynomial) exactly, because det(I + zA) has degree <= n.
def _ps_mul(a, b, T):
    return np.convolve(a, b)[:T]


def _ps_inv(p, T):
    # Power-series inverse of p with unit constant term p[0] == 1.
    b = np.zeros(T, dtype=p.dtype)
    b[0] = 1
    for k in range(1, T):
        acc = 0
        for j in range(1, k + 1):
            acc = acc + p[j] * b[k - j]
        b[k] = -acc
    return b


# Strassen's avoidance of divisions
def STcoefs(A):
    n = A.shape[0]
    dtype = get_dtype(A)
    A = np.array(A, dtype=dtype)
    T = n + 1

    # M[i, j] = delta_ij + z*A[i, j], ascending coeff vectors mod z^{n+1}.
    M = np.zeros((n, n, T), dtype=dtype)
    M[:, :, 1] = A
    for i in range(n):
        M[i, i, 0] = 1

    det = np.zeros(T, dtype=dtype)              # product of pivots == det(I + zA)
    det[0] = 1
    for i in range(n):
        piv = M[i, i]
        det = _ps_mul(det, piv, T)
        pinv = _ps_inv(piv, T)
        for j in range(i + 1, n):
            factor = _ps_mul(M[j, i], pinv, T)
            for k in range(i, n):
                M[j, k] = M[j, k] - _ps_mul(factor, M[i, k], T)

    # det == [E_0, ..., E_n] ascending; charpoly coefs (descending) are (-1)^k E_k.
    return [(-1)**k * det[k] for k in range(T)]


# Strassen's avoidance of divisions
def STdet(A):
    n = A.shape[0]
    coefs = STcoefs(A)
    det = coefs[-1] * (-1)**n
    return det


# Strassen's avoidance of divisions
def STcharpoly(A):
    n = A.shape[0]
    coefs = STcoefs(A)
    p = np.sum([c * lamda**(n-i) for i, c in enumerate(coefs)])
    return sp.PurePoly(p, lamda)


# Kaltofen's algorithm (baby-step/giant-step Krylov, division-free).
#
# "On computing determinants of matrices without divisions", E. Kaltofen, ISSAC
# 1992 -- a Wiedemann/Krylov characteristic-polynomial computation made division-
# free by Strassen's power-series technique (see STcoefs), with a baby-step/
# giant-step recombination that lowers the asymptotic cost (its speedup needs
# fast matrix multiplication, so it is not realized by this reference port).
#
# The pencil B(z) = S + z(A - S), with S the nilpotent shift (super-diagonal
# ones), interpolates B(0) = S and B(1) = A. S is a single Jordan block, so the
# Krylov sequence a_k(z) = e_1^T B(z)^k e_n has a_k(0) = delta_{k, n-1}: the
# Hankel matrix [a_{i+j}] is the exchange matrix at z=0, hence a unit in
# R[z]/(z^{n+1}). Reversing its rows makes that the identity, so the Hankel
# solve runs by Gauss-Jordan with unit (constant-term-1) pivots -- division-free
# and valid for *every* A (not just generic ones). The solve yields the
# coefficients c_i(z) of the monic characteristic polynomial of B(z); evaluating
# at z = 1 (summing each c_i's coefficients) gives the characteristic polynomial
# of A. _ps_mul / _ps_inv are the power-series helpers shared with STcoefs.
def _ps_matmul(M1, M2, T):
    # (A,B,T) . (B,C,T) -> (A,C,T): the z^d coefficient of the poly-matrix
    # product is sum_{d1+d2=d} M1[:, :, d1] @ M2[:, :, d2].
    A, C = M1.shape[0], M2.shape[1]
    res = np.zeros((A, C, T), dtype=M1.dtype)
    for d1 in range(T):
        for d2 in range(T - d1):
            res[:, :, d1 + d2] = res[:, :, d1 + d2] + M1[:, :, d1] @ M2[:, :, d2]
    return res


def _hankel_solve(a, n, T):
    # Solve H c = b over R[z]/(z^T): H[i,j] = a[i+j], b[i] = -a[n+i]. Row-reverse
    # so the z=0 system is the identity, then Gauss-Jordan with unit pivots.
    H = np.zeros((n, n, T), dtype=a.dtype)
    b = np.zeros((n, T), dtype=a.dtype)
    for i in range(n):
        for j in range(n):
            H[i, j] = a[i + j]
        b[i] = -a[n + i]
    H = H[::-1].copy()
    b = b[::-1].copy()
    for k in range(n):
        inv_piv = _ps_inv(H[k, k], T)
        for j in range(n):
            H[k, j] = _ps_mul(H[k, j], inv_piv, T)
        b[k] = _ps_mul(b[k], inv_piv, T)
        for i in range(n):
            if i != k:
                factor = H[i, k].copy()             # copy: the j-loop overwrites H[i,k]
                for j in range(n):
                    H[i, j] = H[i, j] - _ps_mul(factor, H[k, j], T)
                b[i] = b[i] - _ps_mul(factor, b[k], T)
    return b                                         # b[i] = c_i(z), coeff of x^i


# Kaltofen's algorithm
def KAcoefs(A):
    n = A.shape[0]
    dtype = get_dtype(A)
    A = np.array(A, dtype=dtype)
    T = n + 1

    # B(z) = S + z(A - S), S the nilpotent shift (super-diagonal ones).
    B = np.zeros((n, n, T), dtype=dtype)
    B[:, :, 1] = A
    for i in range(n - 1):
        B[i, i + 1, 0] = 1
        B[i, i + 1, 1] = A[i, i + 1] - 1

    u = np.zeros((1, n, T), dtype=dtype)            # u = e_1^T
    v = np.zeros((n, 1, T), dtype=dtype)            # v = e_n
    u[0, 0, 0] = 1
    v[n - 1, 0, 0] = 1

    limit = 2 * n
    r = math.ceil(math.sqrt(limit))
    s = math.ceil(limit / r)

    # Baby steps v_j = B^j v; giant steps u_k = u^T (B^r)^k.
    v_steps = [v]
    for _ in range(1, r):
        v_steps.append(_ps_matmul(B, v_steps[-1], T))
    Z = B
    for _ in range(r - 1):
        Z = _ps_matmul(Z, B, T)
    u_steps = [u]
    for _ in range(1, s):
        u_steps.append(_ps_matmul(u_steps[-1], Z, T))

    a = np.zeros((limit, T), dtype=dtype)           # a_{rk+j} = u^T B^{rk+j} v
    for k in range(s):
        for j in range(r):
            idx = k * r + j
            if idx < limit:
                a[idx] = _ps_matmul(u_steps[k], v_steps[j], T)[0, 0]

    c = _hankel_solve(a, n, T)
    # charpoly_B(x) = x^n + sum_i c_i(z) x^i; evaluate z=1, return descending.
    return [1] + [np.sum(c[i]) for i in range(n - 1, -1, -1)]


# Kaltofen's algorithm
def KAdet(A):
    n = A.shape[0]
    coefs = KAcoefs(A)
    det = coefs[-1] * (-1)**n
    return det


# Kaltofen's algorithm
def KAcharpoly(A):
    n = A.shape[0]
    coefs = KAcoefs(A)
    p = np.sum([c * lamda**(n-i) for i, c in enumerate(coefs)])
    return sp.PurePoly(p, lamda)
