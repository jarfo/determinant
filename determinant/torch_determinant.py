"""Torch-only port of determinant.py.

Every algorithm from determinant.py that produces numbers (coefficients,
determinant, cofactor/adjugate matrices) is reimplemented with torch tensors
instead of numpy arrays. The symbolic SymPy parts (the ``*charpoly`` wrappers)
are dropped: in a numeric setting the characteristic polynomial *is* the
coefficient tensor returned by the ``*coefs`` functions, ordered so that
``coefs[i]`` is the coefficient of ``lambda**(n-i)`` in ``det(lambda*I - A)``.

Conventions
-----------
- Input ``A`` is a square 2-D ``torch.Tensor``; dtype and device are inherited.
- ``*coefs`` return a 1-D tensor of length ``n+1``.
- ``*det`` return a 0-D (scalar) tensor.
- ``*cofactors`` return a stacked ``(n+1, n, n)`` tensor of coefficient
  gradients; ``*cofactor`` return the ``(n, n)`` cofactor matrix ``d det / dA``.

Caveat: unlike determinant.py (which uses numpy ``object`` dtype for arbitrary
precision), torch integer tensors are fixed-width ``int64``. Exact integer
determinants of large matrices can therefore overflow — use ``float64`` for
large inputs, or keep ``n`` small for exact integer arithmetic. ``BRdet``
(Bareiss) relies on integer division and is exact-integer only, as in numpy.
"""

import torch


def _evector(length, ref, i=0):
    e = ref.new_zeros(length)
    e[i] = 1
    return e


def _convolve(a, v):
    # Full 1-D convolution, matching numpy.convolve(a, v, mode='full').
    La, Lv = a.shape[0], v.shape[0]
    out = a.new_zeros(La + Lv - 1)
    for j in range(Lv):
        out[j:j + La] += v[j] * a
    return out


def _correlate_valid(a, v):
    # Matches numpy.correlate(a, v, mode='valid'): out[k] = sum_j a[k+j] * v[j].
    Lv = v.shape[0]
    Lout = a.shape[0] - Lv + 1
    return torch.stack([(a[k:k + Lv] * v).sum() for k in range(Lout)])


# Bareiss algorithm (exact-integer only)
def BRdet(A):
    A = A.clone()  # modified in place
    N = A.shape[0]
    sign = 1
    prev = A.new_ones(())
    for i in range(N - 1):
        if A[i, i] == 0:  # swap with another row having a nonzero i-th element
            swapto = next((j for j in range(i + 1, N) if A[j, i] != 0), None)
            if swapto is None:
                return A.new_zeros(())  # all A[*][i] are zero => zero determinant
            A[[i, swapto]] = A[[swapto, i]]
            sign = -sign

        num = A[i + 1:, i + 1:] * A[i, i] - A[i + 1:, i:i + 1] * A[i:i + 1, i + 1:]
        A[i + 1:, i + 1:] = torch.div(num, prev, rounding_mode='floor')
        prev = A[i, i].clone()
    return sign * A[-1, -1]


# Faddeev–LeVerrier algorithm
def FLcoefs(A):
    n = A.shape[0]
    exact = not (A.is_floating_point() or A.is_complex())
    coefs = A.new_zeros(n + 1)
    traces = A.new_zeros(n)

    coefs[n] = 1
    Apow = torch.eye(n, dtype=A.dtype, device=A.device)
    for m in range(1, n + 1):
        Apow = Apow @ A
        traces[m - 1] = torch.trace(Apow)
        coef = -(coefs[n - m + 1:] @ traces[:m])
        coefs[n - m] = torch.div(coef, m, rounding_mode='floor') if exact else coef / m

    return torch.flip(coefs, [0])


# Faddeev–LeVerrier algorithm
def FLdet(A):
    n = A.shape[0]
    coefs = FLcoefs(A)
    det = coefs[-1] * (-1)**n
    return det


# Clow-based algorithm. Dynamic Programming According to Length
def DPcoefs(A):
    n = A.shape[0]
    D = torch.eye(n, dtype=A.dtype, device=A.device)
    diag = A.new_zeros(n)
    coefs = [A.new_ones(())]
    for i in range(n):
        D = torch.triu(D @ A)
        cumsum = torch.cumsum(torch.diagonal(D), 0)
        coefs.append(-cumsum[-1])
        diag[1:] = -cumsum[:-1]
        D.diagonal().copy_(diag)

    return torch.stack(coefs)


# Clow-based algorithm. Dynamic Programming According to Length
def DPdet(A):
    n = A.shape[0]
    coefs = DPcoefs(A)
    det = coefs[-1] * (-1)**n
    return det


# Clow-based algorithm. Explicit Matrix power method for the Dynamic Programming According to Length
def DPmatrix(A):
    n = A.shape[0]
    m = ((n + 1) * n) // 2 + 1

    r = A.new_zeros((1, m))
    r[0, m - 1] = 1

    s = A.new_zeros((m, 1))
    icol = 0
    for i in range(n + 1):
        s[icol, 0] = 1
        icol += n - i

    M = A.new_zeros((m, m))
    d = 0
    for i in range(n + 1):
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
    coefs = [A.new_ones(())]
    for _ in range(n):
        Ms = M @ Ms
        coefs.append((r @ Ms)[0, 0])

    return torch.stack(coefs)


# Clow-based algorithm. Explicit Matrix power method for the Dynamic Programming According to Length
def MPdet(A):
    n = A.shape[0]
    coefs = MPcoefs(A)
    det = coefs[-1] * (-1)**n
    return det


# Clow-based algorithm. Gradients of every characteristic-polynomial coefficient.
#
# The forward sweep L_k = r @ M^k and backward sweep R_k = M^k @ s yield the
# gradient of every coefficient at once via W_i = sum_k outer(L_k, R_{i-1-k}),
# routed back through DPmatrix's construction. cofactors[i] = d coefs[i] / dA;
# the determinant's cofactor matrix is the last one (see MPcofactor).
def MPcofactors(A):
    n = A.shape[0]
    M, r, s = DPmatrix(A)
    m = M.shape[0]

    # Forward sweep L[k] = r @ M^k and backward sweep R[k] = M^k @ s, k = 0..n-1.
    L = A.new_zeros((n, m))
    R = A.new_zeros((n, m))
    Mk_r, Mk_s = r, s
    for k in range(n):
        L[k] = Mk_r[0]
        R[k] = Mk_s[:, 0]
        Mk_r = Mk_r @ M
        Mk_s = M @ Mk_s

    # cofactors[i] = d coefs[i] / d A; cofactors[0] = d(1)/dA = 0.
    cofactors = [A.new_zeros((n, n))]
    for i in range(1, n + 1):
        # W[u, v] = sum_{k=0}^{i-1} L[k][u] * R[i-1-k][v]
        W = A.new_zeros((m, m))
        for k in range(i):
            W += torch.outer(L[k], R[i - 1 - k])

        # Mirror DPmatrix's construction, routing W back to the gradient with sign.
        G = A.new_zeros((n, n))
        d = 0
        for b in range(n + 1):
            p = 0
            for j in range(b):
                G[j, j:] -= W[d, p:p + n - j]
                p += n - j
            if b < n:
                G[b + 1:, b:] += W[d + 1:d + n - b, d:d + n - b]
            d += n - b
        cofactors.append(G)

    return torch.stack(cofactors)


# Clow-based algorithm. Cofactor matrix (entrywise derivative of the determinant).
# G[i, j] = d det / d A[i, j] is the transpose of the classical adjugate:
# adj(A) = G.T = det(A) * inv(A).
def MPcofactor(A):
    n = A.shape[0]
    cofactors = MPcofactors(A)
    cofactor = cofactors[-1] * (-1)**n
    return cofactor


# Clow sequences with the prefix property: Getting to Samuelson's method
def CVcoefs(A):
    n = A.shape[0]
    sign = -1 if n % 2 else 1

    P = torch.stack([A.new_tensor(-1), A[n - 1, n - 1]])
    for i in range(n - 2, -1, -1):
        r = A[i, i + 1:]
        s = A[i + 1:, i]
        M = A[i + 1:, i + 1:]
        D = [A.new_zeros(())] * (n - i - 1) + [A.new_tensor(-1), A[i, i], r @ s]
        rM = r
        for _ in range(n - i - 2):
            rM = rM @ M
            D.append(rM @ s)
        P = _correlate_valid(torch.stack(D), torch.flip(P, [0]))

    return sign * P


# Clow sequences with the prefix property
def CVdet(A):
    n = A.shape[0]
    coefs = CVcoefs(A)
    det = coefs[-1] * (-1)**n
    return det


# Chistov's Algorithm
def CHcoefs(A):
    n = A.shape[0]

    B = A.new_ones((n, n + 1))
    C = [_evector(n - i, A) for i in range(n)]
    for i in range(n):
        C = [c @ A[k:, k:] for k, c in enumerate(C)]
        B[:, i + 1] = torch.stack([c[0] for c in C])

    d = B[0]
    for i in range(1, n):
        d = _convolve(d, B[i])[:n + 1]

    e = _evector(n + 1, A)
    for i in range(1, n + 1):
        e[i] = -(d[1:i + 1] * torch.flip(e[:i], [0])).sum()

    return e


# Clow sequences with the prefix property
def CHdet(A):
    n = A.shape[0]
    coefs = CHcoefs(A)
    det = coefs[-1] * (-1)**n
    return det


# Bivariate Cayley-Hamilton recursion (Ikenmeyer 2025)
#   chi_{n,d} = chi_{n-1,d} + sum_{i=1}^{d} (-1)^{i+1} chi_{n,d-i} * pow_{n,i}
# where chi_{n,d} is the sum of d-by-d principal minors of the leading n-by-n
# submatrix and pow_{n,i} = [X_n^i]_{n,n}.
def BCHcoefs(A):
    N = A.shape[0]

    chi = [A.new_ones(())]
    for n in range(1, N + 1):
        Xn = A[:n, :n]
        # pows[i] = [X_n^{i+1}]_{n,n} via the row-vector recurrence r_k = e_n^T X_n^k.
        pows = A.new_zeros(n)
        r = A.new_zeros(n)
        r[n - 1] = 1
        for i in range(n):
            r = r @ Xn
            pows[i] = r[n - 1]

        new_chi = [A.new_ones(())]
        for d in range(1, n + 1):
            val = chi[d] if d < n else A.new_zeros(())
            for i in range(1, d + 1):
                term = new_chi[d - i] * pows[i - 1]
                val = val + term if i % 2 == 1 else val - term
            new_chi.append(val)
        chi = new_chi

    return torch.stack([(-1)**d * chi[d] for d in range(N + 1)])


# Bivariate Cayley-Hamilton recursion
def BCHdet(A):
    n = A.shape[0]
    coefs = BCHcoefs(A)
    det = coefs[-1] * (-1)**n
    return det


# Bivariate Cayley-Hamilton recursion. Gradients of every coefficient.
#
# Ikenmeyer (2025), Theorem 5.1: (grad chi_{n,d+1})^T = sum_i (-1)^i chi_{n,d-i} X_n^i.
# In the coefs convention this is a Horner / Faddeev-LeVerrier recursion on the
# adjugate-expansion matrices S_e = sum_{i=0}^{e-1} coefs[e-1-i] A^i:
#   S_e = coefs[e-1] * I + A @ S_{e-1},   cofactors[e] = -S_e.T   (S_0 = 0).
def BCHcofactors(A):
    N = A.shape[0]
    coefs = BCHcoefs(A)

    Id = torch.eye(N, dtype=A.dtype, device=A.device)
    cofactors = [A.new_zeros((N, N))]   # d coefs[0]/dA = d(1)/dA = 0
    S = A.new_zeros((N, N))             # S_0 = 0
    for e in range(1, N + 1):
        S = coefs[e - 1] * Id + A @ S   # S_e = coefs[e-1] I + A S_{e-1}
        cofactors.append(-S.T)

    return torch.stack(cofactors)


# Bivariate Cayley-Hamilton recursion. Cofactor matrix d det / dA.
def BCHcofactor(A):
    n = A.shape[0]
    cofactors = BCHcofactors(A)
    cofactor = cofactors[-1] * (-1)**n
    return cofactor


if __name__ == "__main__":
    torch.manual_seed(0)
    n = 6

    # Float: every method should match torch.linalg.det.
    A = torch.randn(n, n, dtype=torch.float64)
    ref = torch.linalg.det(A)
    print(f"reference torch.linalg.det = {float(ref):+.6f}")
    for f in (FLdet, DPdet, MPdet, CVdet, CHdet, BCHdet):
        print(f"  {f.__name__:7s} {float(f(A)):+.6f}")

    # Exact integer: all methods should agree (small n to avoid int64 overflow).
    Ai = torch.randint(-5, 6, (n, n))
    dets = {f.__name__: int(f(Ai)) for f in (BRdet, FLdet, DPdet, MPdet, CVdet, CHdet, BCHdet)}
    print("integer determinants:", dets)

    # Cofactor identity: A @ cofactor(A).T == det(A) * I.
    G = MPcofactor(A)
    print("max |A @ G.T - det*I| =", float((A @ G.T - ref * torch.eye(n, dtype=A.dtype)).abs().max()))
