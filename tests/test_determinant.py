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
# MPcofactor (cofactor matrix = entrywise derivative of the determinant)      #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", SYMBOLIC_SIZES)
def test_mpcofactor_is_derivative_of_det(n):
    M = symbolic_matrix(n)
    det = sp.Matrix(M).det()
    G = MPcofactor(M)
    assert all(
        sp.expand(G[i, j] - sp.diff(det, M[i, j])) == 0
        for i in range(n)
        for j in range(n)
    )


@pytest.mark.parametrize("n", SYMBOLIC_SIZES)
def test_mpcofactor_equals_cofactor(n):
    # G[i, j] is the cofactor C_ij, i.e. the transpose of the classical adjugate.
    M = symbolic_matrix(n)
    adj = sp.Matrix(M).adjugate()
    G = MPcofactor(M)
    assert all(
        sp.expand(G[i, j] - adj[j, i]) == 0
        for i in range(n)
        for j in range(n)
    )


@pytest.mark.parametrize("n", INTEGER_SIZES)
def test_mpcofactor_integer_identity(n):
    # A @ adj(A)^T == A @ G.T == det(A) * I, exactly.
    A = integer_matrix(n)
    G = MPcofactor(A)
    expected = np.zeros((n, n), dtype=object)
    np.fill_diagonal(expected, int(MPdet(A)))
    assert np.array_equal(A @ G.T, expected)


@pytest.mark.parametrize("n", FLOAT_SIZES)
def test_mpcofactor_float_matches_inverse(n):
    A = float_matrix(n)
    G = np.array(MPcofactor(A), dtype=float)
    ref = np.linalg.det(A) * np.linalg.inv(A).T
    assert np.allclose(G, ref, rtol=1e-6, atol=1e-9)


# --------------------------------------------------------------------------- #
# MPcofactors (gradients of every coefficient = resolvent/adjugate expansion) #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", SYMBOLIC_SIZES)
def test_mpcofactors_are_coef_gradients(n):
    # MPcofactors(A)[i] == d MPcoefs(A)[i] / dA, entrywise.
    M = symbolic_matrix(n)
    coefs = MPcoefs(M)
    G = MPcofactors(M)
    assert len(G) == n + 1
    assert all(
        sp.expand(G[i][a, b] - sp.diff(coefs[i], M[a, b])) == 0
        for i in range(n + 1)
        for a in range(n)
        for b in range(n)
    )


@pytest.mark.parametrize("n", SYMBOLIC_SIZES)
def test_mpcofactors_resolvent_expansion(n):
    # adj(lambda*I - A) == sum_i (-MPcofactors(A)[i].T) * lambda^(n-i).
    M = symbolic_matrix(n)
    lam = sp.Symbol("lam")
    expected = (lam * sp.eye(n) - sp.Matrix(M)).adjugate()
    G = MPcofactors(M)
    got = sp.zeros(n, n)
    for i in range(n + 1):
        got += -sp.Matrix(G[i].tolist()).T * lam ** (n - i)
    assert all(sp.expand(expected[a, b] - got[a, b]) == 0 for a in range(n) for b in range(n))


@pytest.mark.parametrize("n", INTEGER_SIZES)
def test_mpcofactor_reads_last_cofactors(n):
    # MPcofactor(A) == MPcofactors(A)[-1] * (-1)^n, mirroring MPdet/MPcoefs.
    A = integer_matrix(n)
    assert np.array_equal(MPcofactor(A), MPcofactors(A)[-1] * (-1) ** n)


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
