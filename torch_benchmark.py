"""Benchmark the torch determinant methods in determinant/torch_determinant.py.

Times every determinant method against torch.linalg.det on a random matrix,
checks correctness first, and prints min wall-clock per call. By default it
benchmarks float64 on CPU; use --int to benchmark exact-integer methods
(including Bareiss) and --device / --dtype to target other backends.

Examples
--------
    python torch_benchmark.py
    python torch_benchmark.py --n 16 --repeat 7 --number 20
    python torch_benchmark.py --int --n 10
    python torch_benchmark.py --device mps --dtype float32
"""

import argparse
import timeit

import torch

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

    if args.integer:
        A = torch.randint(-5, 6, (args.n, args.n), dtype=torch.int64, device=device)
        methods = INT_METHODS
        # Reference: exact determinant via float64, then round to int.
        ref_val = int(round(float(torch.linalg.det(A.double()))))
        ref_label = "round(linalg.det)"
    else:
        dtype = getattr(torch, args.dtype)
        A = torch.randn(args.n, args.n, dtype=dtype, device=device)
        methods = FLOAT_METHODS
        ref_val = torch.linalg.det(A)
        ref_label = "torch.linalg.det"

    print(f"n={args.n}  dtype={A.dtype}  device={A.device}  "
          f"number={args.number}  repeat={args.repeat}")
    print(f"{'method':16s} {'min ms/call':>12s} {'vs linalg':>10s}   correct")
    print("-" * 54)

    base = min_ms(torch.linalg.det, A.double() if args.integer else A, args.number, args.repeat)
    print(f"{ref_label:16s} {base:12.4f} {1.0:9.2f}x   --")

    for fn in methods:
        t = min_ms(fn, A, args.number, args.repeat)
        if args.integer:
            ok = int(fn(A)) == ref_val
        else:
            ok = torch.allclose(fn(A), ref_val, rtol=1e-5, atol=1e-8)
        print(f"{fn.__name__:16s} {t:12.4f} {t / base:9.2f}x   {'OK' if ok else 'MISMATCH'}")

    if args.cofactors:
        print("-" * 54)
        for fn in COFACTOR_METHODS:
            t = min_ms(fn, A, args.number, args.repeat)
            G = fn(A)
            if args.integer:
                ok = torch.equal(A @ G.T, ref_val * torch.eye(args.n, dtype=A.dtype, device=device))
            else:
                ok = torch.allclose(A @ G.T, ref_val * torch.eye(args.n, dtype=A.dtype, device=device),
                                    rtol=1e-5, atol=1e-7)
            print(f"{fn.__name__:16s} {t:12.4f} {t / base:9.2f}x   {'OK' if ok else 'MISMATCH'}")


if __name__ == "__main__":
    main()
