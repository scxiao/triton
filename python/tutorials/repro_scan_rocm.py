#!/usr/bin/env python3
"""
Reproducer for Triton cross-wavefront scan regression on AMD MI300X (gfx942).

Bug: tl.associative_scan over axis=1 with the 2D broadcast_to + carry-loop
pattern returns wrong results on ~3.2 % of elements.

  Bad:  triton 3.8.0 @ 81a46fa (main, 2026-06-30)
  Good: triton @ ba5c151 (nightly reference)

Affected ops: torch.cumsum, torch.logcumsumexp (any cross-warp AddPartialReduce
path in lib/Conversion/TritonGPUToLLVM/ScanOpToLLVM.cpp).

Config that reproduces:
  dtype=float32, XBLOCK=1, R0_BLOCK=2048, num_warps=8, grid=(100,),
  r0_numel=4000 (> one wavefront of 2048).

  Reproduction also requires tt.divisibility=16 specialization on all pointer
  and integer arguments — standard for PyTorch-allocated CUDA/HIP tensors.

Usage:
  python repro_scan_rocm.py            # cumsum (add) variant
  python repro_scan_rocm.py --logadd   # logcumsumexp (logaddexp) variant
  python repro_scan_rocm.py --both     # run both and report
"""
import argparse
import math
import sys

import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# triton_helpers.select_one  — pick the one element per row where mask is True
# (identical semantic to torch._inductor.triton_helpers.select_one)
# ---------------------------------------------------------------------------
@triton.jit
def _select_one(x, mask, dim: tl.constexpr, keep_dims: tl.constexpr):
    """Return the single element of x[..., dim] where mask is True."""
    # tl.zeros_like doesn't exist; use x * 0 to get a zero tensor of the same
    # shape and dtype without introducing a new constexpr dependency.
    return tl.sum(tl.where(mask, x, x * 0), dim, keep_dims=keep_dims)


# ---------------------------------------------------------------------------
# Combine functions
# ---------------------------------------------------------------------------
@triton.jit
def _add(a, b):
    return a + b


@triton.jit
def _logaddexp(a, b):
    max_ab = tl.maximum(a, b)
    # log(exp(a) + exp(b))  — numerically stable
    return max_ab + tl.log(tl.exp(a - max_ab) + tl.exp(b - max_ab))


# ---------------------------------------------------------------------------
# Kernels
# The two kernels below are the exact @triton.jit code that Inductor generates
# for torch.cumsum / torch.logcumsumexp on a 1-D reduction of length 4000.
#
# Key structural requirements for reproduction:
#   1. 2-D tile [XBLOCK, R0_BLOCK].
#   2. tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK]) before the scan.
#   3. tl.associative_scan(..., axis=1, combine_fn=...) — inner (row) scan.
#   4. Cross-loop carry: tmp3 accumulates the last element of each chunk.
#   5. Launch with XBLOCK=1, R0_BLOCK=2048, num_warps=8, r0_numel>R0_BLOCK.
#   6. All pointers/integers divisible by 16 (automatic for torch allocations).
# ---------------------------------------------------------------------------
@triton.jit
def cumsum_kernel(
    in_ptr0, out_ptr0,
    xnumel, r0_numel,
    XBLOCK: tl.constexpr,
    R0_BLOCK: tl.constexpr,
):
    """Inductor-generated cumulative-sum (add) kernel — exact form."""
    xnumel = 100        # noqa: F841 — Inductor emits these as specializations
    r0_numel = 4000     # noqa: F841
    RBLOCK: tl.constexpr = R0_BLOCK

    xoffset = tl.program_id(0) * XBLOCK
    xindex  = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask   = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    x0      = xindex

    # carry accumulator: last scan value from the previous chunk
    tmp3 = tl.full([XBLOCK, 1], float('nan'), tl.float32)

    for r0_offset in tl.range(0, r0_numel, R0_BLOCK, num_stages=2):
        r0_index = r0_offset + r0_base
        r0_mask  = r0_index < r0_numel
        roffset  = r0_offset

        tmp0 = tl.load(
            in_ptr0 + (r0_index + 4000 * x0),
            r0_mask & xmask,
            eviction_policy='evict_first',
            other=0.0,
        )
        # ── The broadcast_to + associative_scan is the failing path ──
        tmp2   = tl.broadcast_to(tmp0.to(tl.float32), [XBLOCK, R0_BLOCK])
        tmp4,  = tl.associative_scan((tmp2,), 1, _add)

        # last element of this chunk (the local carry-out)
        tmp5   = _select_one(tmp4, r0_base == (RBLOCK - 1), dim=-1, keep_dims=True)

        # apply inter-chunk carry
        tmp8   = tl.where(roffset > 0, tmp3 + tmp4, tmp4)
        tmp3   = tl.where(roffset > 0, tmp3 + tmp5, tmp5)

        tl.store(out_ptr0 + (r0_index + 4000 * x0), tmp8, r0_mask & xmask)


@triton.jit
def logcumsumexp_kernel(
    in_ptr0, out_ptr0,
    xnumel, r0_numel,
    XBLOCK: tl.constexpr,
    R0_BLOCK: tl.constexpr,
):
    """Inductor-generated log-cumsum-exp (logaddexp) kernel — exact form."""
    xnumel = 100        # noqa: F841
    r0_numel = 4000     # noqa: F841
    RBLOCK: tl.constexpr = R0_BLOCK

    xoffset = tl.program_id(0) * XBLOCK
    xindex  = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask   = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    x0      = xindex

    tmp3 = tl.full([XBLOCK, 1], float('-inf'), tl.float32)

    for r0_offset in tl.range(0, r0_numel, R0_BLOCK, num_stages=2):
        r0_index = r0_offset + r0_base
        r0_mask  = r0_index < r0_numel
        roffset  = r0_offset

        tmp0 = tl.load(
            in_ptr0 + (r0_index + 4000 * x0),
            r0_mask & xmask,
            eviction_policy='evict_first',
            other=float('-inf'),
        )
        tmp2  = tl.broadcast_to(tmp0.to(tl.float32), [XBLOCK, R0_BLOCK])
        tmp4, = tl.associative_scan((tmp2,), 1, _logaddexp)
        tmp5  = _select_one(tmp4, r0_base == (RBLOCK - 1), dim=-1, keep_dims=True)
        tmp8  = tl.where(roffset > 0, _logaddexp(tmp3, tmp4), tmp4)
        tmp3  = tl.where(roffset > 0, _logaddexp(tmp3, tmp5), tmp5)

        tl.store(out_ptr0 + (r0_index + 4000 * x0), tmp8, r0_mask & xmask)


# ---------------------------------------------------------------------------
# Launch helpers
# ---------------------------------------------------------------------------
_XBLOCK   = 1
_R0_BLOCK = 2048
_NUM_WARPS = 8
_XNUMEL   = 100
_R0_NUMEL = 4000


def _check_alignment(t: torch.Tensor, name: str) -> None:
    """Assert the 16-byte alignment that Inductor/PyTorch guarantees."""
    assert t.data_ptr() % 16 == 0, (
        f"{name} must be 16-byte aligned (got data_ptr()={t.data_ptr():#x}); "
        "allocate with torch.empty / torch.randn on a CUDA/HIP device."
    )


def run_cumsum(inp: torch.Tensor) -> torch.Tensor:
    """Launch the add-scan (cumsum) kernel with Inductor's exact launch config."""
    out = torch.empty_like(inp)
    _check_alignment(inp, "inp")
    _check_alignment(out, "out")
    # grid = (xnumel,) — one program per row, matching Inductor's reduction grid
    cumsum_kernel[(_XNUMEL,)](
        inp, out, _XNUMEL, _R0_NUMEL,
        XBLOCK=_XBLOCK, R0_BLOCK=_R0_BLOCK,
        num_warps=_NUM_WARPS,
    )
    return out


def run_logcumsumexp(inp: torch.Tensor) -> torch.Tensor:
    """Launch the logaddexp-scan kernel with Inductor's exact launch config."""
    out = torch.empty_like(inp)
    _check_alignment(inp, "inp")
    _check_alignment(out, "out")
    logcumsumexp_kernel[(_XNUMEL,)](
        inp, out, _XNUMEL, _R0_NUMEL,
        XBLOCK=_XBLOCK, R0_BLOCK=_R0_BLOCK,
        num_warps=_NUM_WARPS,
    )
    return out


# ---------------------------------------------------------------------------
# Reference implementations
# ---------------------------------------------------------------------------
def ref_cumsum(inp: torch.Tensor) -> torch.Tensor:
    return torch.cumsum(inp, dim=1)


def ref_logcumsumexp(inp: torch.Tensor) -> torch.Tensor:
    return torch.logcumsumexp(inp, dim=1)


# ---------------------------------------------------------------------------
# Comparison utility
# ---------------------------------------------------------------------------
def compare(name: str, out: torch.Tensor, ref: torch.Tensor,
            rtol: float = 1e-4, atol: float = 1e-4) -> int:
    """Print a pass/fail report.  Returns the number of wrong elements."""
    mismatch = ~torch.isclose(out, ref, rtol=rtol, atol=atol)
    n_wrong  = int(mismatch.sum())
    n_total  = mismatch.numel()
    pct      = 100.0 * n_wrong / n_total

    if n_wrong:
        idx    = mismatch.nonzero(as_tuple=False)[0].tolist()
        r, c   = idx
        print(
            f"[FAIL] {name}: {n_wrong}/{n_total} wrong elements ({pct:.2f}%)\n"
            f"       first mismatch at [{r}, {c}]:\n"
            f"         kernel  = {out[r, c].item():.8f}\n"
            f"         ref     = {ref[r, c].item():.8f}\n"
            f"         abs_err = {abs(out[r, c].item() - ref[r, c].item()):.3e}"
        )
    else:
        print(f"[PASS] {name}: all {n_total} elements match")

    return n_wrong


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reproducer for Triton scan regression on AMD MI300X (gfx942)."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--logadd", action="store_true",
                       help="Run only the logcumsumexp (logaddexp) variant.")
    group.add_argument("--both", action="store_true",
                       help="Run both the cumsum and logcumsumexp variants.")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: no CUDA/HIP device found — this reproducer requires a GPU.")
        return 2

    device   = "cuda"
    seed     = 42
    torch.manual_seed(seed)

    print(f"Triton version : {triton.__version__}")
    print(f"Device         : {torch.cuda.get_device_name(0)}")
    print(f"Shape          : [{_XNUMEL}, {_R0_NUMEL}]  "
          f"(XBLOCK={_XBLOCK}, R0_BLOCK={_R0_BLOCK}, num_warps={_NUM_WARPS})\n")

    total_wrong = 0

    run_cumsum_variant = not args.logadd
    run_logadd_variant = args.logadd or args.both

    # ── cumsum variant ──────────────────────────────────────────────────────
    if run_cumsum_variant:
        inp = torch.randn(_XNUMEL, _R0_NUMEL, dtype=torch.float32, device=device)
        ref = ref_cumsum(inp)
        out = run_cumsum(inp)
        total_wrong += compare("cumsum (add combine)", out, ref)

    # ── logcumsumexp variant ────────────────────────────────────────────────
    if run_logadd_variant:
        # Small values keep exp() in range; mirror Inductor's typical input range
        inp = torch.randn(_XNUMEL, _R0_NUMEL, dtype=torch.float32, device=device)
        ref = ref_logcumsumexp(inp)
        out = run_logcumsumexp(inp)
        total_wrong += compare("logcumsumexp (logaddexp combine)", out, ref)

    return 0 if total_wrong == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
