"""Benchmark the batched torch determinant methods in determinant/torch_determinant.py.

Each method processes a whole batch of shape (B, n, n) per call. Times every
method against torch.linalg.det (which is itself batched), checks correctness
first, and prints min wall-clock per call (i.e. per batch). By default it
benchmarks float64 on CPU; use --int for exact-integer methods (including
Bareiss) and --device / --dtype to target other backends.

Examples
--------
    python torch_benchmark.py
    python torch_benchmark.py --batch 256 --n 16 --repeat 7
    python torch_benchmark.py --int --n 10
    python torch_benchmark.py --device mps --dtype float32
"""

import argparse
import timeit

import numpy as np
import torch
from flint import fmpz_mat

from determinant import torch_determinant as td

FLOAT_METHODS = [td.FLdet, td.DPdet, td.MPdet, td.CVdet, td.CHdet, td.BCHdet]
INT_METHODS = [td.BRdet, td.FLdet, td.DPdet, td.MPdet, td.CVdet, td.CHdet, td.BCHdet]
COFACTOR_METHODS = [td.MPcofactor, td.BCHcofactor]


def min_ms(fn, A, number, repeat):
    times = timeit.repeat(lambda: fn(A), number=number, repeat=repeat)
    return min(times) / number * 1e3


def err_stats(out, ref):
    """Relative-error statistics of `out` against the exact reference `ref`."""
    diff = np.abs(out.astype(np.float64) - ref.astype(np.float64))
    denom = np.abs(ref.astype(np.float64))
    rel = diff / np.where(denom > 0, denom, 1.0)
    return f"max {rel.max():.2e}  mean {rel.mean():.2e}"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch", type=int, default=6400, help="batch size B (default 64)")
    parser.add_argument("--n", type=int, default=30, help="matrix size (default 12)")
    parser.add_argument("--number", type=int, default=10, help="calls per timing run")
    parser.add_argument("--repeat", type=int, default=5, help="timing runs (min is reported)")
    parser.add_argument("--device", default="cpu", help="torch device (cpu, mps, cuda)")
    parser.add_argument("--dtype", default="float64", choices=["float32", "float64"])
    parser.add_argument("--int", action="store_true", dest="integer",
                        help="benchmark exact-integer methods (int64) instead of float")
    parser.add_argument("--cofactors", action="store_true",
                        help="also benchmark the cofactor-matrix methods")
    args = parser.parse_args()

    torch.manual_seed(0)
    device = torch.device(args.device)
    A = torch.randint(-5, 6, (args.batch, args.n, args.n), dtype=torch.int64, device=device)
    Ad = A.float() if args.device == "mps" else A.double()
    # Exact integer determinant (B,), per batch element, via python-flint.
    ref = np.array([int(fmpz_mat(A[k].cpu().tolist()).det()) for k in range(args.batch)])

    if args.integer:
        methods = INT_METHODS
    else:
        A = A.to(dtype=getattr(torch, args.dtype))
        methods = FLOAT_METHODS

    print(f"batch={args.batch}  n={args.n}  dtype={A.dtype}  device={A.device}  number={args.number}  repeat={args.repeat}")
    err_col = "correct" if args.integer else "rel error (max/mean)"
    print(f"{'method':16s} {'min ms/call':>12s} {'us/matrix':>11s} {'vs linalg':>10s}   {err_col}")
    print("-" * 66)

    base_in = Ad if args.integer else A
    base = min_ms(torch.linalg.det, base_in, args.number, args.repeat)
    base_out = torch.linalg.det(base_in).cpu().numpy()
    print(f"{'torch.linalg.det':16s} {base:12.4f} {base / args.batch * 1e3:11.4f} {1.0:9.2f}x {err_stats(base_out, ref)}")

    for fn in methods:
        t = min_ms(fn, A, args.number, args.repeat)
        out = fn(A)
        if isinstance(out, torch.Tensor):
            out = out.cpu().numpy()
        if args.integer:
            status = "OK" if np.array_equal(out, ref) else "MISMATCH"
        else:
            status = err_stats(out, ref)
        print(f"{fn.__name__:16s} {t:12.4f} {t / args.batch * 1e3:11.4f} {t / base:9.2f}x {status}")

    if args.cofactors:
        print("-" * 66)
        eye = np.eye(args.n, dtype=object)
        for fn in COFACTOR_METHODS:
            t = min_ms(fn, A, args.number, args.repeat)
            prod = A @ fn(A).mT
            if isinstance(prod, torch.Tensor):
                prod = prod.cpu().numpy()
            target = ref.reshape(-1, 1, 1) * eye
            if args.integer:
                status = "OK" if np.array_equal(prod, target) else "MISMATCH"
            else:
                status = err_stats(prod, target)
            print(f"{fn.__name__:16s} {t:12.4f} {t / args.batch * 1e3:11.4f} {t / base:9.2f}x   "
                  f"{status}")


if __name__ == "__main__":
    main()
