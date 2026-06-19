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

import torch

from determinant import determinant as det  # numpy/sympy reference (exact integers)
from determinant import torch_determinant as td

FLOAT_METHODS = [td.FLdet, td.DPdet, td.MPdet, td.CVdet, td.CHdet, td.BCHdet]
INT_METHODS = [td.BRdet, td.FLdet, td.DPdet, td.MPdet, td.CVdet, td.CHdet, td.BCHdet]
COFACTOR_METHODS = [td.MPcofactor, td.BCHcofactor]


def min_ms(fn, A, number, repeat):
    times = timeit.repeat(lambda: fn(A), number=number, repeat=repeat)
    return min(times) / number * 1e3


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch", type=int, default=64, help="batch size B (default 64)")
    parser.add_argument("--n", type=int, default=12, help="matrix size (default 12)")
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
    if args.integer:
        methods = INT_METHODS
        # Exact integer reference (B,), per batch element, via the numpy implementation.
        ref = torch.tensor(
            [int(det.DPdet(A[k].cpu().numpy().astype(object))) for k in range(args.batch)],
            dtype=torch.int64, device=device)
    else:
        A = A.to(dtype=getattr(torch, args.dtype))
        methods = FLOAT_METHODS
        ref = torch.linalg.det(Ad)  # (B,) batched float reference

    print(f"batch={args.batch}  n={args.n}  dtype={A.dtype}  device={A.device}  "
          f"number={args.number}  repeat={args.repeat}")
    print(f"{'method':16s} {'min ms/call':>12s} {'us/matrix':>11s} {'vs linalg':>10s}   correct")
    print("-" * 66)

    base = min_ms(torch.linalg.det, Ad if args.integer else A, args.number, args.repeat)
    print(f"{'torch.linalg.det':16s} {base:12.4f} {base / args.batch * 1e3:11.4f} {1.0:9.2f}x   --")

    for fn in methods:
        t = min_ms(fn, A, args.number, args.repeat)
        out = fn(A).cpu()
        if args.integer:
            ok = torch.equal(out, ref.cpu())
        else:
            ok = torch.allclose(out, ref.to(out.dtype).cpu(), rtol=1e-4, atol=1e-6)
        print(f"{fn.__name__:16s} {t:12.4f} {t / args.batch * 1e3:11.4f} {t / base:9.2f}x   "
              f"{'OK' if ok else 'MISMATCH'}")

    if args.cofactors:
        print("-" * 66)
        eye = torch.eye(args.n, dtype=A.dtype, device=device).expand(args.batch, args.n, args.n)
        for fn in COFACTOR_METHODS:
            t = min_ms(fn, A, args.number, args.repeat)
            prod = A @ fn(A).mT
            if args.integer:
                ok = torch.equal(prod, ref.view(-1, 1, 1) * eye)
            else:
                ok = torch.allclose(prod, ref.to(A.dtype).view(-1, 1, 1) * eye, rtol=1e-5, atol=1e-6)
            print(f"{fn.__name__:16s} {t:12.4f} {t / args.batch * 1e3:11.4f} {t / base:9.2f}x   "
                  f"{'OK' if ok else 'MISMATCH'}")


if __name__ == "__main__":
    main()
