"""Unit tests for the batched torch port in determinant/torch_determinant.py.

Every function takes a batch of matrices of shape (B, n, n). Ground truth is
torch.linalg.det / inv (natively batched) for floats and the numpy/sympy
reference (determinant.determinant), applied per batch element, for exact
integers. The whole module skips if torch is not installed.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from determinant import determinant as ref          # noqa: E402  numpy/sympy reference
from determinant import torch_determinant as td      # noqa: E402  batched torch port under test

# Bareiss (BRdet) uses integer division, so it is exact-integer only.
DET_FUNCS = [td.BRdet, td.FLdet, td.DPdet, td.MPdet, td.CVdet, td.CHdet, td.BCHdet, td.BIdet, td.STdet, td.KAdet]
GENERAL_DET_FUNCS = [td.FLdet, td.DPdet, td.MPdet, td.CVdet, td.CHdet, td.BCHdet, td.BIdet, td.STdet, td.KAdet]
COEFS_FUNCS = [td.FLcoefs, td.DPcoefs, td.MPcoefs, td.CVcoefs, td.CHcoefs, td.BCHcoefs, td.BIcoefs, td.STcoefs, td.KAcoefs]
COFACTOR_FUNCS = [td.MPcofactor, td.BCHcofactor]
COFACTORS_FUNCS = [td.MPcofactors, td.BCHcofactors]

BATCH = 4
INT_SIZES = [1, 2, 3, 5, 7]
FLOAT_SIZES = [2, 3, 4, 6]

_by_name = {"ids": lambda f: f.__name__}


def int_batch(n, batch=BATCH, seed=0):
    rng = np.random.RandomState(seed + n)
    a = rng.randint(-9, 10, size=(batch, n, n))
    return torch.tensor(a, dtype=torch.int64), a


def float_batch(n, batch=BATCH, seed=0):
    rng = np.random.RandomState(100 + seed + n)
    a = rng.uniform(-3.0, 3.0, size=(batch, n, n))
    return torch.tensor(a, dtype=torch.float64), a


def ref_apply(name, a):
    # Apply a reference (numpy/object) function to every matrix in the batch.
    return [getattr(ref, name)(a[k].astype(object)) for k in range(a.shape[0])]


# --------------------------------------------------------------------------- #
# Determinant                                                                 #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", FLOAT_SIZES)
@pytest.mark.parametrize("det", GENERAL_DET_FUNCS, **_by_name)
def test_det_float_matches_linalg(det, n):
    A, _ = float_batch(n)
    assert torch.allclose(det(A), torch.linalg.det(A), rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize("n", INT_SIZES)
@pytest.mark.parametrize("det", DET_FUNCS, **_by_name)
def test_det_integer_matches_reference(det, n):
    A, a = int_batch(n)
    got = det(A).tolist()
    expected = [int(x) for x in ref_apply(det.__name__, a)]
    assert got == expected


# --------------------------------------------------------------------------- #
# Coefficients                                                                #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", INT_SIZES)
@pytest.mark.parametrize("coefs", COEFS_FUNCS, **_by_name)
def test_coefs_integer_match_reference(coefs, n):
    A, a = int_batch(n)
    got = coefs(A).tolist()
    expected = [[int(c) for c in row] for row in ref_apply(coefs.__name__, a)]
    assert got == expected


@pytest.mark.parametrize("n", FLOAT_SIZES)
def test_coefs_agree_with_each_other(n):
    A, _ = float_batch(n)
    ref_coefs = td.MPcoefs(A)
    for coefs in COEFS_FUNCS:
        assert torch.allclose(coefs(A), ref_coefs, rtol=1e-6, atol=1e-9)


# --------------------------------------------------------------------------- #
# Cofactor matrix and coefficient gradients                                   #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", INT_SIZES)
@pytest.mark.parametrize("cofactor", COFACTOR_FUNCS, **_by_name)
def test_cofactor_integer_identity(cofactor, n):
    # A @ G.mT == det(A) * I, per batch element, exactly.
    A, _ = int_batch(n)
    G = cofactor(A)
    expected = td.MPdet(A).view(-1, 1, 1) * torch.eye(n, dtype=torch.int64)
    assert torch.equal(A @ G.mT, expected)


@pytest.mark.parametrize("n", FLOAT_SIZES)
@pytest.mark.parametrize("cofactor", COFACTOR_FUNCS, **_by_name)
def test_cofactor_float_matches_inverse(cofactor, n):
    A, _ = float_batch(n)
    G = cofactor(A)
    ref_G = torch.linalg.det(A).view(-1, 1, 1) * torch.linalg.inv(A).mT
    assert torch.allclose(G, ref_G, rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize("n", INT_SIZES)
@pytest.mark.parametrize("cofactor", COFACTOR_FUNCS, **_by_name)
def test_cofactor_matches_reference(cofactor, n):
    A, a = int_batch(n)
    got = cofactor(A).tolist()
    expected = [np.array(g, dtype=object).tolist() for g in ref_apply(cofactor.__name__, a)]
    assert got == expected


@pytest.mark.parametrize("n", INT_SIZES)
@pytest.mark.parametrize("cofactors", COFACTORS_FUNCS, **_by_name)
def test_cofactors_match_reference(cofactors, n):
    A, a = int_batch(n)
    got = cofactors(A)
    expected = ref_apply(cofactors.__name__, a)  # list (len B) of lists (len n+1) of matrices
    assert got.shape == (BATCH, n + 1, n, n)
    assert all(
        got[k, d].tolist() == np.array(expected[k][d], dtype=object).tolist()
        for k in range(BATCH)
        for d in range(n + 1)
    )


@pytest.mark.parametrize("n", INT_SIZES)
def test_bch_matches_mp_cofactors(n):
    A, _ = int_batch(n)
    assert torch.equal(td.BCHcofactors(A), td.MPcofactors(A))


# --------------------------------------------------------------------------- #
# dtype / shape                                                               #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("det", GENERAL_DET_FUNCS, **_by_name)
def test_dtype_preserved(det):
    for dtype in (torch.float32, torch.float64):
        A = torch.randn(BATCH, 4, 4, dtype=dtype)
        assert det(A).dtype == dtype


def test_returns_are_tensors_with_expected_shapes():
    A, _ = float_batch(5)
    assert td.MPcoefs(A).shape == (BATCH, 6)              # (B, n+1)
    assert td.MPdet(A).shape == (BATCH,)                  # (B,)
    assert td.MPcofactors(A).shape == (BATCH, 6, 5, 5)    # (B, n+1, n, n)
    assert td.MPcofactor(A).shape == (BATCH, 5, 5)        # (B, n, n)


# --------------------------------------------------------------------------- #
# Edge cases                                                                  #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("det", DET_FUNCS, **_by_name)
def test_scalar_1x1(det):
    A = torch.tensor([7, -3, 0, 5], dtype=torch.int64).view(4, 1, 1)
    assert det(A).tolist() == [7, -3, 0, 5]


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("det", DET_FUNCS, **_by_name)
def test_identity_determinant(det, n):
    A = torch.eye(n, dtype=torch.int64).expand(BATCH, n, n)
    assert det(A).tolist() == [1] * BATCH


@pytest.mark.parametrize("det", DET_FUNCS, **_by_name)
def test_singular_in_batch(det):
    # A batch mixing a singular matrix (row2 = 2*row1) with a regular one.
    singular = [[1, 2, 3], [2, 4, 6], [7, 8, 9]]
    regular = [[2, 0, 1], [3, 1, 0], [5, 1, 1]]
    A = torch.tensor([singular, regular], dtype=torch.int64)
    expected = [0, int(ref.BRdet(np.array(regular, dtype=object)))]
    assert det(A).tolist() == expected
