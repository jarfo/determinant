"""Batched torch port of determinant.py.

Every function takes a batch of square matrices ``A`` of shape ``(B, n, n)`` and
vectorizes over the leading batch dimension ``B`` with batched tensor ops; the
inner ``O(n)`` algorithm loops are unchanged. Return shapes gain a leading ``B``:

- ``*coefs``     -> ``(B, n+1)``      characteristic-polynomial coefficients
- ``*det``       -> ``(B,)``          determinants
- ``*cofactors`` -> ``(B, n+1, n, n)``  coefficient gradients (resolvent expansion)
- ``*cofactor``  -> ``(B, n, n)``     cofactor matrix ``d det / dA``
- ``DPmatrix``   -> ``M (B, m, m), r (B, 1, m), s (B, m, 1)``

All algorithms except Bareiss are straight-line in ``n``, so they batch by
broadcasting / batched matmul. ``BRdet`` (Bareiss) has data-dependent pivoting,
handled with masked per-element row swaps; it stays exact-integer only. As in
the scalar port, integer tensors are fixed-width ``int64`` (no arbitrary
precision), so large integer matrices can overflow.
"""

import math

import torch
import torch.nn.functional as F


def _evector(Bsz, length, ref, i=0):
    e = ref.new_zeros((Bsz, length))
    e[:, i] = 1
    return e


def _convolve(a, v):
    # Batched full 1-D convolution: a (B, La), v (B, Lv) -> (B, La+Lv-1).
    Bsz, La = a.shape
    Lv = v.shape[-1]
    if a.is_floating_point() or a.is_complex():
        # conv1d cross-correlates per group; flip the kernel and pad with Lv-1
        # zeros each side so the 'valid' output becomes a full convolution.
        w = v.flip(-1).unsqueeze(1)  # (B, 1, Lv)
        out = F.conv1d(a.unsqueeze(0), w, padding=Lv - 1, groups=Bsz)
        return out.squeeze(0)
    # conv1d is float-only; accumulate explicitly for exact-integer inputs.
    out = a.new_zeros((Bsz, La + Lv - 1))
    for j in range(Lv):
        out[:, j:j + La] += v[:, j:j + 1] * a
    return out


def _correlate_valid(a, v):
    # Batched numpy.correlate(., ., 'valid'): out[b,k] = sum_j a[b,k+j] v[b,j].
    Lv = v.shape[-1]
    if a.is_floating_point() or a.is_complex():
        # Per-group cross-correlation is exactly the 'valid' correlation.
        Bsz = a.shape[0]
        out = F.conv1d(a.unsqueeze(0), v.unsqueeze(1), groups=Bsz)
        return out.squeeze(0)
    # conv1d is float-only; accumulate explicitly for exact-integer inputs.
    Lout = a.shape[-1] - Lv + 1
    return torch.stack([(a[:, k:k + Lv] * v).sum(1) for k in range(Lout)], dim=1)


def _eye(Bsz, n, ref):
    return torch.eye(n, dtype=ref.dtype, device=ref.device).expand(Bsz, n, n)


# Bareiss algorithm (exact-integer only)
def BRdet(A):
    A = A.clone()  # modified in place
    Bsz, N, _ = A.shape
    sign = A.new_ones(Bsz)
    prev = A.new_ones(Bsz)
    zero_det = torch.zeros(Bsz, dtype=torch.bool, device=A.device)
    for i in range(N - 1):
        pivot = A[:, i, i]
        below = A[:, i + 1:, i]
        nz = below != 0
        has_below = nz.any(dim=1)
        first = torch.argmax(nz.to(torch.int64), dim=1)  # first nonzero row below (0 if none)

        # Zero pivot with no nonzero below => that batch element is singular.
        zero_det = zero_det | ((pivot == 0) & ~has_below)
        # Zero pivot with a nonzero below => swap rows i and i+1+first.
        need_swap = (pivot == 0) & has_below & ~zero_det
        if need_swap.any():
            bidx = need_swap.nonzero(as_tuple=True)[0]
            ridx = i + 1 + first[bidx]
            tmp = A[bidx, i, :].clone()
            A[bidx, i, :] = A[bidx, ridx, :]
            A[bidx, ridx, :] = tmp
            sign[bidx] = -sign[bidx]

        piv = A[:, i, i].view(Bsz, 1, 1)
        num = A[:, i + 1:, i + 1:] * piv - A[:, i + 1:, i:i + 1] * A[:, i:i + 1, i + 1:]
        A[:, i + 1:, i + 1:] = torch.div(num, prev.view(Bsz, 1, 1), rounding_mode='floor')
        prev = A[:, i, i].clone()
        prev = torch.where(prev == 0, torch.ones_like(prev), prev)  # guard discarded (singular) elems

    det = sign * A[:, -1, -1]
    return torch.where(zero_det, torch.zeros_like(det), det)


# Faddeev–LeVerrier algorithm
def FLcoefs(A):
    Bsz, n, _ = A.shape
    exact = not (A.is_floating_point() or A.is_complex())
    coefs = A.new_zeros((Bsz, n + 1))
    traces = A.new_zeros((Bsz, n))

    coefs[:, n] = 1
    Apow = _eye(Bsz, n, A)
    for m in range(1, n + 1):
        Apow = Apow @ A
        traces[:, m - 1] = torch.diagonal(Apow, dim1=-2, dim2=-1).sum(-1)
        coef = -(coefs[:, n - m + 1:] * traces[:, :m]).sum(dim=1)
        coefs[:, n - m] = torch.div(coef, m, rounding_mode='floor') if exact else coef / m

    return torch.flip(coefs, [1])


# Faddeev–LeVerrier algorithm
def FLdet(A):
    n = A.shape[-1]
    coefs = FLcoefs(A)
    det = coefs[:, -1] * (-1)**n
    return det


# Clow-based algorithm. Dynamic Programming According to Length
def DPcoefs(A):
    Bsz, n, _ = A.shape
    D = _eye(Bsz, n, A)
    diag = A.new_zeros((Bsz, n))
    coefs = [A.new_ones(Bsz)]
    for i in range(n):
        D = torch.triu(D @ A)
        cumsum = torch.cumsum(torch.diagonal(D, dim1=-2, dim2=-1), dim=1)
        coefs.append(-cumsum[:, -1])
        diag[:, 1:] = -cumsum[:, :-1]
        D.diagonal(dim1=-2, dim2=-1).copy_(diag)

    return torch.stack(coefs, dim=1)


# Clow-based algorithm. Dynamic Programming According to Length
def DPdet(A):
    n = A.shape[-1]
    coefs = DPcoefs(A)
    det = coefs[:, -1] * (-1)**n
    return det


# Clow-based algorithm. Explicit Matrix power method for the Dynamic Programming According to Length
def DPmatrix(A):
    Bsz, n, _ = A.shape
    m = ((n + 1) * n) // 2 + 1

    r = A.new_zeros((Bsz, 1, m))
    r[:, 0, m - 1] = 1

    s = A.new_zeros((Bsz, m, 1))
    icol = 0
    for i in range(n + 1):
        s[:, icol, 0] = 1
        icol += n - i

    M = A.new_zeros((Bsz, m, m))
    d = 0
    for i in range(n + 1):
        p = 0
        for j in range(i):
            M[:, d, p:p + n - j] = -A[:, j, j:]
            p += n - j
        if i < n:
            M[:, d + 1: d + n - i, d: d + n - i] = A[:, i + 1:, i:]
        d += n - i

    return M, r, s


# Clow-based algorithm. Explicit Matrix power method for the Dynamic Programming According to Length
def MPcoefs(A):
    Bsz, n, _ = A.shape
    M, r, s = DPmatrix(A)

    Ms = s
    coefs = [A.new_ones(Bsz)]
    for _ in range(n):
        Ms = M @ Ms
        coefs.append((r @ Ms)[:, 0, 0])

    return torch.stack(coefs, dim=1)


# Clow-based algorithm. Explicit Matrix power method for the Dynamic Programming According to Length
def MPdet(A):
    n = A.shape[-1]
    coefs = MPcoefs(A)
    det = coefs[:, -1] * (-1)**n
    return det


# Clow-based algorithm. Gradients of every characteristic-polynomial coefficient.
# cofactors[:, i] = d coefs[:, i] / dA; the determinant's cofactor matrix is the
# last one (see MPcofactor). See determinant.MPcofactors for the derivation.
def MPcofactors(A):
    Bsz, n, _ = A.shape
    M, r, s = DPmatrix(A)
    m = M.shape[-1]

    # Forward sweep L[:, k] = r @ M^k and backward sweep R[:, k] = M^k @ s, k = 0..n-1.
    L = A.new_zeros((Bsz, n, m))
    R = A.new_zeros((Bsz, n, m))
    Mk_r, Mk_s = r, s
    for k in range(n):
        L[:, k, :] = Mk_r[:, 0, :]
        R[:, k, :] = Mk_s[:, :, 0]
        Mk_r = Mk_r @ M
        Mk_s = M @ Mk_s

    cofactors = [A.new_zeros((Bsz, n, n))]
    for i in range(1, n + 1):
        # W[:, u, v] = sum_{k=0}^{i-1} L[:, k, u] * R[:, i-1-k, v]
        W = A.new_zeros((Bsz, m, m))
        for k in range(i):
            W += L[:, k, :].unsqueeze(2) * R[:, i - 1 - k, :].unsqueeze(1)

        # Mirror DPmatrix's construction, routing W back to the gradient with sign.
        G = A.new_zeros((Bsz, n, n))
        d = 0
        for b in range(n + 1):
            p = 0
            for j in range(b):
                G[:, j, j:] -= W[:, d, p:p + n - j]
                p += n - j
            if b < n:
                G[:, b + 1:, b:] += W[:, d + 1:d + n - b, d:d + n - b]
            d += n - b
        cofactors.append(G)

    return torch.stack(cofactors, dim=1)


# Clow-based algorithm. Cofactor matrix (entrywise derivative of the determinant).
def MPcofactor(A):
    n = A.shape[-1]
    cofactors = MPcofactors(A)
    cofactor = cofactors[:, -1] * (-1)**n
    return cofactor


# Clow sequences with the prefix property: Getting to Samuelson's method
def CVcoefs(A):
    Bsz, n, _ = A.shape
    sign = -1 if n % 2 else 1

    P = torch.stack([-A.new_ones(Bsz), A[:, n - 1, n - 1]], dim=1)  # (B, 2)
    for i in range(n - 2, -1, -1):
        r = A[:, i, i + 1:]
        s = A[:, i + 1:, i]
        M = A[:, i + 1:, i + 1:]
        D = [A.new_zeros(Bsz)] * (n - i - 1) + [-A.new_ones(Bsz), A[:, i, i], (r * s).sum(1)]
        rM = r
        for _ in range(n - i - 2):
            rM = (rM.unsqueeze(1) @ M).squeeze(1)
            D.append((rM * s).sum(1))
        P = _correlate_valid(torch.stack(D, dim=1), torch.flip(P, [1]))

    return sign * P


# Clow sequences with the prefix property
def CVdet(A):
    n = A.shape[-1]
    coefs = CVcoefs(A)
    det = coefs[:, -1] * (-1)**n
    return det


# Chistov's Algorithm
def CHcoefs(A):
    Bsz, n, _ = A.shape

    Bmat = A.new_ones((Bsz, n, n + 1))
    C = [_evector(Bsz, n - i, A) for i in range(n)]
    for i in range(n):
        C = [(c.unsqueeze(1) @ A[:, k:, k:]).squeeze(1) for k, c in enumerate(C)]
        Bmat[:, :, i + 1] = torch.stack([c[:, 0] for c in C], dim=1)

    d = Bmat[:, 0, :]
    for i in range(1, n):
        d = _convolve(d, Bmat[:, i, :])[:, :n + 1]

    e = _evector(Bsz, n + 1, A)
    for i in range(1, n + 1):
        e[:, i] = -(d[:, 1:i + 1] * torch.flip(e[:, :i], [1])).sum(1)

    return e


# Clow sequences with the prefix property
def CHdet(A):
    n = A.shape[-1]
    coefs = CHcoefs(A)
    det = coefs[:, -1] * (-1)**n
    return det


# Bivariate Cayley-Hamilton recursion (Ikenmeyer 2025)
def BCHcoefs(A):
    Bsz, N, _ = A.shape

    chi = [A.new_ones(Bsz)]
    for n in range(1, N + 1):
        Xn = A[:, :n, :n]
        # pows[:, i] = [X_n^{i+1}]_{n,n} via the row-vector recurrence r_k = e_n^T X_n^k.
        pows = A.new_zeros((Bsz, n))
        r = A.new_zeros((Bsz, n))
        r[:, n - 1] = 1
        for i in range(n):
            r = (r.unsqueeze(1) @ Xn).squeeze(1)
            pows[:, i] = r[:, n - 1]

        new_chi = [A.new_ones(Bsz)]
        for d in range(1, n + 1):
            val = chi[d] if d < n else A.new_zeros(Bsz)
            for i in range(1, d + 1):
                term = new_chi[d - i] * pows[:, i - 1]
                val = val + term if i % 2 == 1 else val - term
            new_chi.append(val)
        chi = new_chi

    return torch.stack([(-1)**d * chi[d] for d in range(N + 1)], dim=1)


# Bivariate Cayley-Hamilton recursion
def BCHdet(A):
    n = A.shape[-1]
    coefs = BCHcoefs(A)
    det = coefs[:, -1] * (-1)**n
    return det


# Bivariate Cayley-Hamilton recursion. Gradients of every coefficient.
# Ikenmeyer (2025), Theorem 5.1, as a Horner / Faddeev-LeVerrier recursion on the
# adjugate-expansion matrices S_e = sum_{i=0}^{e-1} coefs[e-1-i] A^i:
#   S_e = coefs[e-1] * I + A @ S_{e-1},   cofactors[e] = -S_e^T   (S_0 = 0).
def BCHcofactors(A):
    Bsz, N, _ = A.shape
    coefs = BCHcoefs(A)

    Id = _eye(Bsz, N, A)
    cofactors = [A.new_zeros((Bsz, N, N))]
    S = A.new_zeros((Bsz, N, N))
    for e in range(1, N + 1):
        S = coefs[:, e - 1].view(Bsz, 1, 1) * Id + A @ S
        cofactors.append(-S.mT)

    return torch.stack(cofactors, dim=1)


# Bivariate Cayley-Hamilton recursion. Cofactor matrix d det / dA.
def BCHcofactor(A):
    n = A.shape[-1]
    cofactors = BCHcofactors(A)
    cofactor = cofactors[:, -1] * (-1)**n
    return cofactor


# Bird's algorithm (Bird 2011). Direct division-free determinant: iterate
# F <- mu(F) @ A from F = A, where mu(X) is upper triangular with the strict
# upper part copied and mu(X)[i, i] = -sum_{k>i} X[k, k]; then
# F_n[0, 0] = (-1)^{n-1} det(A). See determinant.BIdet.
def _mu(X):
    M = torch.triu(X).clone()                       # diagonal + strict upper
    d = torch.diagonal(X, dim1=-2, dim2=-1)         # (B, n)
    rev = torch.flip(torch.cumsum(torch.flip(d, [-1]), -1), [-1])  # rev[i]=sum_{k>=i}
    trail = torch.zeros_like(d)
    trail[:, :-1] = rev[:, 1:]                       # trail[i] = sum_{k>=i+1} d[k]
    M.diagonal(dim1=-2, dim2=-1).copy_(-trail)
    return M


# Bird's algorithm
def BIdet(A):
    n = A.shape[-1]
    F = A
    for _ in range(n - 1):
        F = _mu(F) @ A
    return (-1)**(n - 1) * F[:, 0, 0]


# Bird's algorithm, characteristic-polynomial coefficients. Run the iteration on
# the pencil lambda*I - A, carrying each entry as an ascending-power coefficient
# vector (last axis). mu(F) @ (lambda*I - A) = lambda*mu(F) - mu(F)@A collapses
# to a degree shift plus a matmul over A. See determinant.BIcoefs.
def _mu_poly(F):
    Bsz, n, _, D = F.shape
    M = torch.zeros_like(F)
    iu = torch.triu_indices(n, n, 1, device=F.device)
    M[:, iu[0], iu[1], :] = F[:, iu[0], iu[1], :]    # strict upper part copied
    diag = torch.diagonal(F, dim1=1, dim2=2).permute(0, 2, 1)  # (B, n, D)
    rev = torch.flip(torch.cumsum(torch.flip(diag, [1]), 1), [1])  # rev[i]=sum_{k>=i}
    trail = torch.zeros_like(diag)
    trail[:, :-1, :] = rev[:, 1:, :]                 # trail[i] = sum_{k>i} F[k,k,:]
    idx = torch.arange(n, device=F.device)
    M[:, idx, idx, :] = -trail
    return M


# Bird's algorithm
def BIcoefs(A):
    Bsz, n, _ = A.shape
    F = A.new_zeros((Bsz, n, n, 2))
    F[..., 0] = -A
    idx = torch.arange(n, device=A.device)
    F[:, idx, idx, 1] = 1
    for _ in range(n - 1):
        muF = _mu_poly(F)
        D = muF.shape[-1]
        newF = A.new_zeros((Bsz, n, n, D + 1))
        newF[..., 1:] = muF                          # lambda * mu(F)
        newF[..., :D] -= torch.einsum('bikd,bkj->bijd', muF, A)  # - mu(F) @ A
        F = newF

    poly = (-1)**(n - 1) * F[:, 0, 0, :]             # (B, n+1) ascending
    return torch.flip(poly, [-1])                    # descending (leading first)


# Strassen's avoidance of divisions (power-series elimination); Kaltofen 1992
# adds a baby-step/giant-step Krylov speedup, not reproduced here. Eliminate
# I + zA over R[z]/(z^{n+1}); pivots are units (constant term 1) since
# I + zA == I (mod z), so pivot inversion is a division-free power-series
# inverse. det(I + zA) = sum_k E_k z^k gives the determinant and full charpoly.
# See determinant.STcoefs.
def _ps_mul(a, b, T):
    # Truncated power-series product: a (B, Ta), b (B, Tb) -> (B, T).
    Ta = a.shape[-1]
    Tb = b.shape[-1]
    out = a.new_zeros((a.shape[0], T))
    for j in range(Tb):
        L = min(Ta, T - j)
        if L > 0:
            out[:, j:j + L] += b[:, j:j + 1] * a[:, :L]
    return out


def _ps_inv(p, T):
    # Power-series inverse with unit constant term p[:, 0] == 1.
    b = p.new_zeros((p.shape[0], T))
    b[:, 0] = 1
    for k in range(1, T):
        acc = p.new_zeros(p.shape[0])
        for j in range(1, k + 1):
            acc = acc + p[:, j] * b[:, k - j]
        b[:, k] = -acc
    return b


# Strassen's avoidance of divisions
def STcoefs(A):
    Bsz, n, _ = A.shape
    T = n + 1

    M = A.new_zeros((Bsz, n, n, T))
    M[:, :, :, 1] = A
    idx = torch.arange(n, device=A.device)
    M[:, idx, idx, 0] = 1

    det = A.new_zeros((Bsz, T))
    det[:, 0] = 1
    for i in range(n):
        piv = M[:, i, i, :]
        det = _ps_mul(det, piv, T)
        pinv = _ps_inv(piv, T)
        for j in range(i + 1, n):
            factor = _ps_mul(M[:, j, i, :], pinv, T)
            for k in range(i, n):
                M[:, j, k, :] = M[:, j, k, :] - _ps_mul(factor, M[:, i, k, :], T)

    # det == [E_0, ..., E_n]; charpoly coefs (leading first) are (-1)^k E_k.
    signs = torch.tensor([(-1)**k for k in range(T)], dtype=A.dtype, device=A.device)
    return det * signs


# Strassen's avoidance of divisions
def STdet(A):
    n = A.shape[-1]
    coefs = STcoefs(A)
    det = coefs[:, -1] * (-1)**n
    return det


# Kaltofen's algorithm (baby-step/giant-step Krylov, division-free). Wiedemann
# characteristic-polynomial computation over the pencil B(z) = S + z(A - S) (S
# the nilpotent shift), made division-free by Strassen's power-series technique:
# the Hankel matrix of a_k(z) = e_1^T B(z)^k e_n is the exchange matrix at z=0,
# so (after row reversal) the solve has unit pivots. See determinant.KAcoefs.
def _ps_matmul(M1, M2, T):
    # (B,A,K,T) . (B,K,C,T) -> (B,A,C,T): z^d coeff is sum_{d1+d2=d} of the
    # batched matrix product M1[..., d1] @ M2[..., d2].
    Bsz, Arows = M1.shape[0], M1.shape[1]
    Ccols = M2.shape[2]
    res = M1.new_zeros((Bsz, Arows, Ccols, T))
    for d1 in range(T):
        for d2 in range(T - d1):
            res[:, :, :, d1 + d2] = res[:, :, :, d1 + d2] + M1[:, :, :, d1] @ M2[:, :, :, d2]
    return res


def _hankel_solve(a, n, T):
    # Solve H c = b over R[z]/(z^T): H[i,j]=a[i+j], b[i]=-a[n+i]. Row-reverse so
    # the z=0 system is the identity, then Gauss-Jordan with unit pivots.
    H = a.new_zeros((a.shape[0], n, n, T))
    b = a.new_zeros((a.shape[0], n, T))
    for i in range(n):
        for j in range(n):
            H[:, i, j, :] = a[:, i + j, :]
        b[:, i, :] = -a[:, n + i, :]
    H = H.flip(1).clone()
    b = b.flip(1).clone()
    for k in range(n):
        inv_piv = _ps_inv(H[:, k, k, :], T)
        for j in range(n):
            H[:, k, j, :] = _ps_mul(H[:, k, j, :], inv_piv, T)
        b[:, k, :] = _ps_mul(b[:, k, :], inv_piv, T)
        for i in range(n):
            if i != k:
                factor = H[:, i, k, :].clone()       # clone: the j-loop overwrites H[:,i,k]
                for j in range(n):
                    H[:, i, j, :] = H[:, i, j, :] - _ps_mul(factor, H[:, k, j, :], T)
                b[:, i, :] = b[:, i, :] - _ps_mul(factor, b[:, k, :], T)
    return b                                          # b[:, i] = c_i(z)


# Kaltofen's algorithm
def KAcoefs(A):
    Bsz, n, _ = A.shape
    T = n + 1

    B = A.new_zeros((Bsz, n, n, T))
    B[:, :, :, 1] = A
    for i in range(n - 1):
        B[:, i, i + 1, 0] = 1
        B[:, i, i + 1, 1] = A[:, i, i + 1] - 1

    u = A.new_zeros((Bsz, 1, n, T))
    v = A.new_zeros((Bsz, n, 1, T))
    u[:, 0, 0, 0] = 1
    v[:, n - 1, 0, 0] = 1

    limit = 2 * n
    r = math.ceil(math.sqrt(limit))
    s = math.ceil(limit / r)

    v_steps = [v]
    for _ in range(1, r):
        v_steps.append(_ps_matmul(B, v_steps[-1], T))
    Z = B
    for _ in range(r - 1):
        Z = _ps_matmul(Z, B, T)
    u_steps = [u]
    for _ in range(1, s):
        u_steps.append(_ps_matmul(u_steps[-1], Z, T))

    a = A.new_zeros((Bsz, limit, T))
    for k in range(s):
        for j in range(r):
            idx = k * r + j
            if idx < limit:
                a[:, idx, :] = _ps_matmul(u_steps[k], v_steps[j], T)[:, 0, 0, :]

    c = _hankel_solve(a, n, T)
    coefs = [A.new_ones(Bsz)]
    for i in range(n - 1, -1, -1):
        coefs.append(c[:, i, :].sum(dim=1))
    return torch.stack(coefs, dim=1)


# Kaltofen's algorithm
def KAdet(A):
    n = A.shape[-1]
    coefs = KAcoefs(A)
    det = coefs[:, -1] * (-1)**n
    return det


if __name__ == "__main__":
    torch.manual_seed(0)
    Bsz, n = 4, 6

    # Float: every method should match torch.linalg.det (which is itself batched).
    A = torch.randn(Bsz, n, n, dtype=torch.float64)
    ref = torch.linalg.det(A)
    print(f"reference torch.linalg.det = {ref.tolist()}")
    for f in (FLdet, DPdet, MPdet, CVdet, CHdet, BCHdet, BIdet, STdet, KAdet):
        err = float((f(A) - ref).abs().max())
        print(f"  {f.__name__:7s} max|err| = {err:.2e}")

    # Exact integer: all methods should agree (small n to avoid int64 overflow).
    Ai = torch.randint(-5, 6, (Bsz, n, n))
    dets = torch.stack([f(Ai) for f in (BRdet, FLdet, DPdet, MPdet, CVdet, CHdet, BCHdet, BIdet, STdet, KAdet)])
    print("integer determinants agree across methods:", bool((dets == dets[0]).all()))

    # Cofactor identity: A @ cofactor(A).mT == det(A) * I, per batch element.
    G = MPcofactor(A)
    I = _eye(Bsz, n, A)
    err = float((A @ G.mT - ref.view(Bsz, 1, 1) * I).abs().max())
    print("cofactor identity max|err| =", err)
