"""Unit tests for every public function in ``determinant.determinant``.

These extend the ad-hoc ``assert`` checks that used to live in the module's
``__main__`` block. Ground truth is SymPy for symbolic and exact-integer inputs
and NumPy for floating-point inputs.
"""

import numpy as np
import pytest
import sympy as sp

from determinant.determinant import (
    BCHcharpoly,
    BCHcoefs,
    BCHcofactor,
    BCHcofactors,
    BCHdet,
    BRdet,
    CHcharpoly,
    CHcoefs,
    CHdet,
    CVcharpoly,
    CVcoefs,
    CVdet,
    DPcharpoly,
    DPcoefs,
    DPdet,
    DPmatrix,
    FLcharpoly,
    FLcoefs,
    FLdet,
    MPcharpoly,
    MPcoefs,
    MPcofactor,
    MPcofactors,
    MPdet,
)

# Bareiss (BRdet) uses integer division, so it only handles exact-integer
# input. Every other method also supports symbolic and floating-point input.
DET_FUNCS = [BRdet, FLdet, DPdet, MPdet, CVdet, CHdet, BCHdet]
GENERAL_DET_FUNCS = [FLdet, DPdet, MPdet, CVdet, CHdet, BCHdet]
COEFS_FUNCS = [FLcoefs, DPcoefs, MPcoefs, CVcoefs, CHcoefs, BCHcoefs]
CHARPOLY_FUNCS = [FLcharpoly, DPcharpoly, MPcharpoly, CVcharpoly, CHcharpoly, BCHcharpoly]

# Cofactor matrix (d det / dA) and the full list of coefficient gradients,
# computed both via the matrix-power (MP) and Cayley-Hamilton (BCH) approaches.
COFACTOR_FUNCS = [MPcofactor, BCHcofactor]
COFACTORS_FUNCS = [MPcofactors, BCHcofactors]

SYMBOLIC_SIZES = [1, 2, 3, 4]
INTEGER_SIZES = [1, 2, 3, 5, 7, 10]
FLOAT_SIZES = [2, 3, 4, 6]

_by_name = {"ids": lambda f: f.__name__}


def symbolic_matrix(n):
    return sp.symarray("a", (n, n))


def integer_matrix(n, seed=0):
    rng = np.random.RandomState(seed + n)
    return rng.randint(-9, 10, size=(n, n)).astype(object)


def float_matrix(n, seed=0):
    rng = np.random.RandomState(100 + seed + n)
    return rng.uniform(-3.0, 3.0, size=(n, n))


# --------------------------------------------------------------------------- #
# Characteristic polynomial                                                   #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", SYMBOLIC_SIZES)
@pytest.mark.parametrize("charpoly", CHARPOLY_FUNCS, **_by_name)
def test_charpoly_matches_sympy(charpoly, n):
    M = symbolic_matrix(n)
    assert charpoly(M) == sp.Matrix(M).charpoly()


# --------------------------------------------------------------------------- #
# Characteristic-polynomial coefficients                                      #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", SYMBOLIC_SIZES)
@pytest.mark.parametrize("coefs", COEFS_FUNCS, **_by_name)
def test_coefs_symbolic_matches_sympy(coefs, n):
    M = symbolic_matrix(n)
    expected = sp.Matrix(M).charpoly().all_coeffs()
    got = list(coefs(M))
    assert len(got) == n + 1
    assert all(sp.expand(g - e) == 0 for g, e in zip(got, expected))


@pytest.mark.parametrize("n", INTEGER_SIZES)
@pytest.mark.parametrize("coefs", COEFS_FUNCS, **_by_name)
def test_coefs_integer_matches_sympy(coefs, n):
    A = integer_matrix(n)
    expected = [int(c) for c in sp.Matrix(A).charpoly().all_coeffs()]
    assert [int(c) for c in coefs(A)] == expected


# --------------------------------------------------------------------------- #
# Determinant                                                                 #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", SYMBOLIC_SIZES)
@pytest.mark.parametrize("det", GENERAL_DET_FUNCS, **_by_name)
def test_det_symbolic_matches_sympy(det, n):
    M = symbolic_matrix(n)
    assert sp.expand(det(M) - sp.Matrix(M).det()) == 0


@pytest.mark.parametrize("n", INTEGER_SIZES)
@pytest.mark.parametrize("det", DET_FUNCS, **_by_name)
def test_det_integer_matches_sympy(det, n):
    A = integer_matrix(n)
    assert int(det(A)) == int(sp.Matrix(A).det())


@pytest.mark.parametrize("n", FLOAT_SIZES)
@pytest.mark.parametrize("det", GENERAL_DET_FUNCS, **_by_name)
def test_det_float_matches_numpy(det, n):
    A = float_matrix(n)
    assert float(det(A)) == pytest.approx(np.linalg.det(A), rel=1e-6)


# --------------------------------------------------------------------------- #
# DPmatrix (the explicit matrix-power construction behind MPdet/MPcofactor)   #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", INTEGER_SIZES)
def test_dpmatrix_shapes(n):
    A = integer_matrix(n)
    M, r, s = DPmatrix(A)
    m = (n + 1) * n // 2 + 1
    assert M.shape == (m, m)
    assert r.shape == (1, m)
    assert s.shape == (m, 1)


@pytest.mark.parametrize("n", INTEGER_SIZES)
def test_dpmatrix_reproduces_determinant(n):
    # det(A) = (-1)^n * r @ M^n @ s
    A = integer_matrix(n)
    M, r, s = DPmatrix(A)
    v = s
    for _ in range(n):
        v = M @ v
    det = (-1) ** n * (r @ v)[0, 0]
    assert int(det) == int(sp.Matrix(A).det())


# --------------------------------------------------------------------------- #
# Cofactor matrix (entrywise derivative of the determinant): MP and BCH        #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", SYMBOLIC_SIZES)
@pytest.mark.parametrize("cofactor", COFACTOR_FUNCS, **_by_name)
def test_cofactor_is_derivative_of_det(cofactor, n):
    M = symbolic_matrix(n)
    det = sp.Matrix(M).det()
    G = cofactor(M)
    assert all(
        sp.expand(G[i, j] - sp.diff(det, M[i, j])) == 0
        for i in range(n)
        for j in range(n)
    )


@pytest.mark.parametrize("n", SYMBOLIC_SIZES)
@pytest.mark.parametrize("cofactor", COFACTOR_FUNCS, **_by_name)
def test_cofactor_equals_cofactor_matrix(cofactor, n):
    # G[i, j] is the cofactor C_ij, i.e. the transpose of the classical adjugate.
    M = symbolic_matrix(n)
    adj = sp.Matrix(M).adjugate()
    G = cofactor(M)
    assert all(
        sp.expand(G[i, j] - adj[j, i]) == 0
        for i in range(n)
        for j in range(n)
    )


@pytest.mark.parametrize("n", INTEGER_SIZES)
@pytest.mark.parametrize("cofactor", COFACTOR_FUNCS, **_by_name)
def test_cofactor_integer_identity(cofactor, n):
    # A @ adj(A)^T == A @ G.T == det(A) * I, exactly.
    A = integer_matrix(n)
    G = cofactor(A)
    expected = np.zeros((n, n), dtype=object)
    np.fill_diagonal(expected, int(MPdet(A)))
    assert np.array_equal(A @ G.T, expected)


@pytest.mark.parametrize("n", FLOAT_SIZES)
@pytest.mark.parametrize("cofactor", COFACTOR_FUNCS, **_by_name)
def test_cofactor_float_matches_inverse(cofactor, n):
    A = float_matrix(n)
    G = np.array(cofactor(A), dtype=float)
    ref = np.linalg.det(A) * np.linalg.inv(A).T
    assert np.allclose(G, ref, rtol=1e-6, atol=1e-9)


# --------------------------------------------------------------------------- #
# Coefficient gradients (resolvent/adjugate expansion): MP and BCH            #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", SYMBOLIC_SIZES)
@pytest.mark.parametrize("cofactors", COFACTORS_FUNCS, **_by_name)
def test_cofactors_are_coef_gradients(cofactors, n):
    # cofactors(A)[i] == d MPcoefs(A)[i] / dA, entrywise.
    M = symbolic_matrix(n)
    coefs = MPcoefs(M)
    G = cofactors(M)
    assert len(G) == n + 1
    assert all(
        sp.expand(G[i][a, b] - sp.diff(coefs[i], M[a, b])) == 0
        for i in range(n + 1)
        for a in range(n)
        for b in range(n)
    )


@pytest.mark.parametrize("n", SYMBOLIC_SIZES)
@pytest.mark.parametrize("cofactors", COFACTORS_FUNCS, **_by_name)
def test_cofactors_resolvent_expansion(cofactors, n):
    # adj(lambda*I - A) == sum_i (-cofactors(A)[i].T) * lambda^(n-i).
    M = symbolic_matrix(n)
    lam = sp.Symbol("lam")
    expected = (lam * sp.eye(n) - sp.Matrix(M)).adjugate()
    G = cofactors(M)
    got = sp.zeros(n, n)
    for i in range(n + 1):
        got += -sp.Matrix(G[i].tolist()).T * lam ** (n - i)
    assert all(sp.expand(expected[a, b] - got[a, b]) == 0 for a in range(n) for b in range(n))


@pytest.mark.parametrize("n", INTEGER_SIZES)
@pytest.mark.parametrize("cofactor, cofactors", list(zip(COFACTOR_FUNCS, COFACTORS_FUNCS)),
                         ids=lambda f: f.__name__)
def test_cofactor_reads_last_cofactors(cofactor, cofactors, n):
    # cofactor(A) == cofactors(A)[-1] * (-1)^n, mirroring MPdet/MPcoefs.
    A = integer_matrix(n)
    assert np.array_equal(cofactor(A), cofactors(A)[-1] * (-1) ** n)


@pytest.mark.parametrize("n", INTEGER_SIZES)
def test_bch_matches_mp_cofactors(n):
    # The two algorithms compute identical coefficient gradients (exactly).
    A = integer_matrix(n)
    assert all(np.array_equal(b, m) for b, m in zip(BCHcofactors(A), MPcofactors(A)))


@pytest.mark.parametrize("n", SYMBOLIC_SIZES)
@pytest.mark.parametrize("cofactors", COFACTORS_FUNCS, **_by_name)
def test_cofactors_satisfy_bivariate_cayley_hamilton(cofactors, n):
    # Ikenmeyer (2025) Theorem 5.1: (grad chi_{n,d+1})^T == sum_i (-1)^i chi_{n,d-i} A^i.
    # In the coefs convention coefs[d] = (-1)^d chi_{N,d}, so grad chi_{N,d} = (-1)^d G[d].
    M = symbolic_matrix(n)
    A = sp.Matrix(M)
    coefs = MPcoefs(M)
    chi = [(-1) ** d * coefs[d] for d in range(n + 1)]
    G = cofactors(M)
    grad_chi = [(-1) ** d * sp.Matrix(G[d].tolist()) for d in range(n + 1)]
    powers = [A**i for i in range(n)]
    for d in range(n):  # Theorem 5.1 for d = 0..n-1
        lhs = grad_chi[d + 1].T
        rhs = sp.zeros(n, n)
        for i in range(d + 1):
            rhs += (-1) ** i * chi[d - i] * powers[i]
        assert all(sp.expand(lhs[a, b] - rhs[a, b]) == 0 for a in range(n) for b in range(n))


# --------------------------------------------------------------------------- #
# Edge cases                                                                  #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("det", DET_FUNCS, **_by_name)
def test_scalar_1x1(det):
    assert int(det(np.array([[7]], dtype=object))) == 7


def test_mpcofactor_1x1_is_one():
    # det([[a]]) = a, so d/da = 1.
    assert int(MPcofactor(np.array([[7]], dtype=object))[0, 0]) == 1


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("det", DET_FUNCS, **_by_name)
def test_identity_determinant(det, n):
    assert int(det(np.eye(n, dtype=object))) == 1


@pytest.mark.parametrize("det", DET_FUNCS, **_by_name)
def test_singular_determinant(det):
    # Row 2 is twice row 1, so the matrix is singular.
    A = np.array([[1, 2, 3], [2, 4, 6], [7, 8, 9]], dtype=object)
    assert int(det(A)) == 0
