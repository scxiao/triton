#!/usr/bin/env python3
"""
Auto-generated benchmark from AOTI wrapper.cpp

  AMD: triton_per_fused_add_clamp_mean_mul_pow_rsqrt_silu_38
    type: persistent_reduction, size_hints: {'x': 16, 'r0_': 256}
    xnumel=None (16), literals=[]
    AUTOTUNE: XBLOCK=16, num_warps=2, time_us=11.440000000000001

  NV: triton_per_fused_add_clamp_mean_mul_pow_rsqrt_silu_44
    type: persistent_reduction, size_hints: {'x': 16, 'r0_': 256}
    xnumel=None (16), literals=[]
    AUTOTUNE: XBLOCK=1, num_warps=2, time_us=7.776

Usage:
    python3 repro_triton_per_fused_silu_44.py
    python3 repro_triton_per_fused_silu_44.py --platform amd
    python3 repro_triton_per_fused_silu_44.py --platform nv
"""

import argparse
import torch
import triton
import triton.language as tl


try:
    from torch._inductor.runtime import triton_helpers, triton_heuristics
    from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
except ImportError:
    triton_helpers = None
    libdevice = None
    tl_math = None

# ============================================================
# AMD: triton_per_fused_add_clamp_mean_mul_pow_rsqrt_silu_38
# xnumel=16, size_hints={'x': 16, 'r0_': 256}
# AUTOTUNE: XBLOCK=16, num_warps=2, time_us=11.440000000000001
# ============================================================
@triton.jit
def triton_per_fused_add_clamp_mean_mul_pow_rsqrt_silu_38(in_out_ptr0, in_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):

        r0_numel = 256
        R0_BLOCK: tl.constexpr = 256
        rnumel = r0_numel
        RBLOCK: tl.constexpr = R0_BLOCK
        xoffset = tl.program_id(0) * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
        xmask = xindex < xnumel
        r0_index = tl.arange(0, R0_BLOCK)[None, :]
        r0_offset = 0
        r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        x0 = xindex
        tmp0 = tl.load(in_out_ptr0 + (r0_1 + 256*x0), xmask, other=0.0).to(tl.float32)
        tmp17 = tl.load(in_ptr0 + (r0_1), None, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tl.full([1, 1], -65504.0, tl.float32)
        tmp3 = tl.maximum(tmp1, tmp2, tl.PropagateNan.ALL)
        tmp4 = tl.full([1, 1], 65504.0, tl.float32)
        tmp5 = tl.minimum(tmp3, tmp4, tl.PropagateNan.ALL)
        tmp6 = tmp5 * tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
        tmp9 = tl.where(xmask, tmp7, 0)
        tmp10 = tl.sum(tmp9, 1)[:, None].to(tl.float32)
        tmp11 = tl.full([1, 1], 256.0, tl.float32)
        tmp12 = (tmp10 / tmp11)
        tmp13 = tl.full([1, 1], 9.999999747378752e-06, tl.float32)
        tmp14 = tmp12 + tmp13
        tmp15 = tl.rsqrt(tmp14)
        tmp16 = tmp5 * tmp15
        tmp18 = tmp17.to(tl.float32)
        tmp19 = tmp16 * tmp18
        tmp20 = -tmp19
        tmp21 = libdevice.exp(tmp20)
        tmp22 = tl.full([1, 1], 1.0, tl.float32)
        tmp23 = tmp21 + tmp22
        tmp24 = (tmp19 / tmp23)
        tmp25 = tmp24.to(tl.float32)
        tl.store(in_out_ptr0 + (r0_1 + 256*x0), tmp25, xmask)
    

# ============================================================
# NV: triton_per_fused_add_clamp_mean_mul_pow_rsqrt_silu_44
# xnumel=16, size_hints={'x': 16, 'r0_': 256}
# AUTOTUNE: XBLOCK=1, num_warps=2, time_us=7.776
# ============================================================
@triton.jit
def triton_per_fused_add_clamp_mean_mul_pow_rsqrt_silu_44(in_out_ptr0, in_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):

        r0_numel = 256
        R0_BLOCK: tl.constexpr = 256
        rnumel = r0_numel
        RBLOCK: tl.constexpr = R0_BLOCK
        xoffset = tl.program_id(0).to(tl.int64) * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:, None].to(tl.int64)
        xmask = xindex < xnumel
        r0_index = tl.arange(0, R0_BLOCK)[None, :].to(tl.int64)
        r0_offset = 0
        r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        x0 = xindex
        tmp0 = tl.load(in_out_ptr0 + (r0_1 + 256*x0), xmask, other=0.0).to(tl.float32)
        tmp17 = tl.load(in_ptr0 + (r0_1), None, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tl.full([1, 1], -65504.0, tl.float32)
        tmp3 = triton_helpers.maximum(tmp1, tmp2)
        tmp4 = tl.full([1, 1], 65504.0, tl.float32)
        tmp5 = triton_helpers.minimum(tmp3, tmp4)
        tmp6 = tmp5 * tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
        tmp9 = tl.where(xmask, tmp7, 0)
        tmp10 = tl.sum(tmp9, 1)[:, None].to(tl.float32)
        tmp11 = tl.full([1, 1], 256.0, tl.float32)
        tmp12 = (tmp10 / tmp11)
        tmp13 = tl.full([1, 1], 9.999999747378752e-06, tl.float32)
        tmp14 = tmp12 + tmp13
        tmp15 = libdevice.rsqrt(tmp14)
        tmp16 = tmp5 * tmp15
        tmp18 = tmp17.to(tl.float32)
        tmp19 = tmp16 * tmp18
        tmp20 = -tmp19
        tmp21 = libdevice.exp(tmp20)
        tmp22 = tl.full([1, 1], 1.0, tl.float32)
        tmp23 = tmp21 + tmp22
        tmp24 = (tmp19 / tmp23)
        tmp25 = tmp24.to(tl.float32)
        tl.store(in_out_ptr0 + (r0_1 + 256*x0), tmp25, xmask)
    

def benchmark_kernel(kernel_fn, grid_size, kernel_args, constexpr_kwargs, label, iters=100):
    """Benchmark matching Inductor's InductorBenchmarker methodology."""
    # Warmup
    kernel_fn[grid_size](*kernel_args, **constexpr_kwargs)
    torch.cuda.synchronize()

    # L2 flush buffer
    l2_size = torch.cuda.get_device_properties(0).L2_cache_size
    flush_buf = torch.empty(l2_size // 4, dtype=torch.int, device="cuda")

    # Estimation (5 iters)
    est_events = []
    for _ in range(5):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        flush_buf.zero_()
        s.record()
        kernel_fn[grid_size](*kernel_args, **constexpr_kwargs)
        e.record()
        est_events.append((s, e))
    torch.cuda.synchronize()
    est_times = [s.elapsed_time(e) for s, e in est_events]
    est_min = min(est_times)

    # Adjust iters
    if est_min > 0:
        iters = max(min(iters, int(25 // est_min)), 1)

    # Memory warmup
    for _ in range(100):
        flush_buf.zero_()

    # Benchmark
    bench_events = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        flush_buf.zero_()
        s.record()
        kernel_fn[grid_size](*kernel_args, **constexpr_kwargs)
        e.record()
        bench_events.append((s, e))
    torch.cuda.synchronize()
    bench_times = [s.elapsed_time(e) for s, e in bench_events]

    all_us = sorted([(t * 1000) for t in est_times + bench_times])
    bench_us = sorted([t * 1000 for t in bench_times])
    del flush_buf

    return {
        "min": all_us[0],
        "median": bench_us[len(bench_us) // 2],
        "p10": bench_us[len(bench_us) // 10],
        "p90": bench_us[len(bench_us) * 9 // 10],
        "iters": iters,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--platform", choices=['both', 'amd', 'nv'], default="both")
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()

    device_name = torch.cuda.get_device_name(0)
    print(f"Device: {device_name}")
    print(f"PyTorch: {torch.__version__}")
    print()

    if args.platform in ("amd", "both"):
        print("--- AMD (triton_per_fused_add_clamp_mean_mul_pow_rsqrt_silu_38) ---")
        xnumel = 16
        r0_numel = 256
        alloc_size = 65536
        in0 = torch.randn(alloc_size, device="cuda", dtype=torch.float16)
        in1 = torch.randn(alloc_size, device="cuda", dtype=torch.float16)
        constexpr_kwargs = {"XBLOCK": 16, "num_warps": 2}
        print(f"  xnumel={xnumel}, r0_numel={r0_numel}, {constexpr_kwargs}")
        grid = (xnumel,)
        r = benchmark_kernel(triton_per_fused_add_clamp_mean_mul_pow_rsqrt_silu_38, grid, [in0, in1, xnumel, r0_numel], constexpr_kwargs, "AMD", iters=args.iters)
        print(f"  Min: {r['min']:.1f}us  Median: {r['median']:.1f}us  P10: {r['p10']:.1f}us  P90: {r['p90']:.1f}us  Iters: {r['iters']}")
        print()

    if args.platform in ("nv", "both"):
        print("--- NV (triton_per_fused_add_clamp_mean_mul_pow_rsqrt_silu_44) ---")
        xnumel = 16
        r0_numel = 256
        alloc_size = 65536
        in0 = torch.randn(alloc_size, device="cuda", dtype=torch.float16)
        in1 = torch.randn(alloc_size, device="cuda", dtype=torch.float16)
        constexpr_kwargs = {"XBLOCK": 1, "num_warps": 2}
        print(f"  xnumel={xnumel}, r0_numel={r0_numel}, {constexpr_kwargs}")
        grid = (xnumel,)
        r = benchmark_kernel(triton_per_fused_add_clamp_mean_mul_pow_rsqrt_silu_44, grid, [in0, in1, xnumel, r0_numel], constexpr_kwargs, "NV", iters=args.iters)
        print(f"  Min: {r['min']:.1f}us  Median: {r['median']:.1f}us  P10: {r['p10']:.1f}us  P90: {r['p90']:.1f}us  Iters: {r['iters']}")
        print()


if __name__ == "__main__":
    main()
