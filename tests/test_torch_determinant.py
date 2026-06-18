"""Unit tests for the torch-only port in determinant/torch_determinant.py.

Ground truth: torch.linalg.det / inv for floats, and the numpy/sympy reference
implementation (determinant.determinant) for exact integers. The whole module
skips if torch is not installed.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from determinant import determinant as ref          # noqa: E402  numpy/sympy reference
from determinant import torch_determinant as td      # noqa: E402  torch port under test

# Bareiss (BRdet) uses integer division, so it is exact-integer only.
DET_FUNCS = [td.BRdet, td.FLdet, td.DPdet, td.MPdet, td.CVdet, td.CHdet, td.BCHdet]
GENERAL_DET_FUNCS = [td.FLdet, td.DPdet, td.MPdet, td.CVdet, td.CHdet, td.BCHdet]
COEFS_FUNCS = [td.FLcoefs, td.DPcoefs, td.MPcoefs, td.CVcoefs, td.CHcoefs, td.BCHcoefs]
COFACTOR_FUNCS = [td.MPcofactor, td.BCHcofactor]
COFACTORS_FUNCS = [td.MPcofactors, td.BCHcofactors]

# Sizes kept modest so exact integer arithmetic stays within int64.
INT_SIZES = [1, 2, 3, 5, 7]
FLOAT_SIZES = [2, 3, 4, 6]

_by_name = {"ids": lambda f: f.__name__}


def int_pair(n, seed=0):
    rng = np.random.RandomState(seed + n)
    a = rng.randint(-9, 10, size=(n, n))
    return torch.tensor(a, dtype=torch.int64), a.astype(object)


def float_pair(n, seed=0):
    rng = np.random.RandomState(100 + seed + n)
    a = rng.uniform(-3.0, 3.0, size=(n, n))
    return torch.tensor(a, dtype=torch.float64), a


# --------------------------------------------------------------------------- #
# Determinant                                                                 #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", FLOAT_SIZES)
@pytest.mark.parametrize("det", GENERAL_DET_FUNCS, **_by_name)
def test_det_float_matches_linalg(det, n):
    A, _ = float_pair(n)
    assert float(det(A)) == pytest.approx(float(torch.linalg.det(A)), rel=1e-6)


@pytest.mark.parametrize("n", INT_SIZES)
@pytest.mark.parametrize("det", DET_FUNCS, **_by_name)
def test_det_integer_matches_reference(det, n):
    A, A_np = int_pair(n)
    assert int(det(A)) == int(getattr(ref, det.__name__)(A_np))


# --------------------------------------------------------------------------- #
# Coefficients                                                                #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", INT_SIZES)
@pytest.mark.parametrize("coefs", COEFS_FUNCS, **_by_name)
def test_coefs_integer_match_reference(coefs, n):
    A, A_np = int_pair(n)
    got = [int(c) for c in coefs(A)]
    expected = [int(c) for c in getattr(ref, coefs.__name__)(A_np)]
    assert got == expected


@pytest.mark.parametrize("n", FLOAT_SIZES)
def test_coefs_agree_with_each_other(n):
    A, _ = float_pair(n)
    ref_coefs = td.MPcoefs(A)
    for coefs in COEFS_FUNCS:
        assert torch.allclose(coefs(A), ref_coefs, rtol=1e-6, atol=1e-9)


# --------------------------------------------------------------------------- #
# Cofactor matrix and coefficient gradients                                   #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", INT_SIZES)
@pytest.mark.parametrize("cofactor", COFACTOR_FUNCS, **_by_name)
def test_cofactor_integer_identity(cofactor, n):
    # A @ G.T == det(A) * I, exactly.
    A, _ = int_pair(n)
    G = cofactor(A)
    expected = td.MPdet(A) * torch.eye(n, dtype=torch.int64)
    assert torch.equal(A @ G.T, expected)


@pytest.mark.parametrize("n", FLOAT_SIZES)
@pytest.mark.parametrize("cofactor", COFACTOR_FUNCS, **_by_name)
def test_cofactor_float_matches_inverse(cofactor, n):
    A, _ = float_pair(n)
    G = cofactor(A)
    ref_G = torch.linalg.det(A) * torch.linalg.inv(A).T
    assert torch.allclose(G, ref_G, rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize("n", INT_SIZES)
@pytest.mark.parametrize("cofactor", COFACTOR_FUNCS, **_by_name)
def test_cofactor_matches_reference(cofactor, n):
    A, A_np = int_pair(n)
    got = cofactor(A).tolist()
    expected = np.array(getattr(ref, cofactor.__name__)(A_np), dtype=object).tolist()
    assert got == expected


@pytest.mark.parametrize("n", INT_SIZES)
@pytest.mark.parametrize("cofactors", COFACTORS_FUNCS, **_by_name)
def test_cofactors_match_reference(cofactors, n):
    A, A_np = int_pair(n)
    got = cofactors(A)
    expected = getattr(ref, cofactors.__name__)(A_np)
    assert got.shape[0] == n + 1
    assert all(got[d].tolist() == np.array(expected[d], dtype=object).tolist()
               for d in range(n + 1))


@pytest.mark.parametrize("n", INT_SIZES)
def test_bch_matches_mp_cofactors(n):
    A, _ = int_pair(n)
    assert torch.equal(td.BCHcofactors(A), td.MPcofactors(A))


# --------------------------------------------------------------------------- #
# dtype / device propagation                                                  #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("det", GENERAL_DET_FUNCS, **_by_name)
def test_dtype_preserved(det):
    for dtype in (torch.float32, torch.float64):
        A = torch.randn(4, 4, dtype=dtype)
        assert det(A).dtype == dtype


def test_returns_are_tensors_with_expected_shapes():
    A, _ = float_pair(5)
    assert td.MPcoefs(A).shape == (6,)            # 1-D, length n+1
    assert td.MPdet(A).shape == ()                # scalar
    assert td.MPcofactors(A).shape == (6, 5, 5)   # (n+1, n, n)
    assert td.MPcofactor(A).shape == (5, 5)       # (n, n)


# --------------------------------------------------------------------------- #
# Edge cases                                                                  #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("det", DET_FUNCS, **_by_name)
def test_scalar_1x1(det):
    assert int(det(torch.tensor([[7]], dtype=torch.int64))) == 7


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("det", DET_FUNCS, **_by_name)
def test_identity_determinant(det, n):
    assert int(det(torch.eye(n, dtype=torch.int64))) == 1


@pytest.mark.parametrize("det", DET_FUNCS, **_by_name)
def test_singular_determinant(det):
    A = torch.tensor([[1, 2, 3], [2, 4, 6], [7, 8, 9]], dtype=torch.int64)  # row2 = 2*row1
    assert int(det(A)) == 0
