#!/usr/bin/env python3
"""
Auto-generated benchmark from AOTI wrapper.cpp

  AMD: triton_poi_fused_add_arange_index_put_new_zeros_171
    type: pointwise, size_hints: {'x': 262144}
    xnumel=32L*s13 (352), literals=[]
    AUTOTUNE: XBLOCK=2048, num_warps=8, time_us=12.48

  NV: triton_poi_fused_add_arange_index_put_new_zeros_182
    type: pointwise, size_hints: {'x': 262144}
    xnumel=32L*s13 (352), literals=[]
    AUTOTUNE: XBLOCK=1024, num_warps=4, time_us=8.256

Usage:
    python3 repro_triton_poi_fused_add_arange_index_put.py
    python3 repro_triton_poi_fused_add_arange_index_put.py --platform amd
    python3 repro_triton_poi_fused_add_arange_index_put.py --platform nv
"""

import argparse
import torch
import triton
import triton.language as tl
from torch.profiler import profile, ProfilerActivity
import os
import contextlib


try:
    from torch._inductor.runtime import triton_helpers, triton_heuristics
    from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
except ImportError:
    triton_helpers = None
    libdevice = None
    tl_math = None


def profiler_or_nullcontext(kernel_name: str, enabled: bool, with_stack: bool):
    def _kineto_trace_handler(p: torch.profiler.profile) -> None:
        trace_url = "/tmp/libkineto_activities_{}_{}.json".format(
            os.getpid(),
            kernel_name,
        )

        print(
             p.key_averages(group_by_input_shape=True).table(
                  sort_by='self_cuda_time_total'
             )
        )
        print(f"trace url: {trace_url}")
        p.export_chrome_trace(trace_url)


    return (
         profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
         on_trace_ready=_kineto_trace_handler,
         with_stack=with_stack,
         record_shapes=True,
         )
         if enabled
         else contextlib.nullcontext()
    )


# ============================================================
# AMD: triton_poi_fused_add_arange_index_put_new_zeros_171
# xnumel=352, size_hints={'x': 262144}
# AUTOTUNE: XBLOCK=2048, num_warps=8, time_us=12.48
# ============================================================
@triton.jit
def triton_poi_fused_add_arange_index_put_new_zeros_171(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):

        xoffset = tl.program_id(0) * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x2 = xindex
        tmp0 = tl.load(in_ptr0 + (x2), xmask, eviction_policy='evict_last').to(tl.float32)
        tl.store(out_ptr0 + (x2), tmp0, xmask)
    

# ============================================================
# NV: triton_poi_fused_add_arange_index_put_new_zeros_182
# xnumel=352, size_hints={'x': 262144}
# AUTOTUNE: XBLOCK=1024, num_warps=4, time_us=8.256
# ============================================================
@triton.jit
def triton_poi_fused_add_arange_index_put_new_zeros_182(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):

        xoffset = tl.program_id(0) * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x2 = xindex
        tmp0 = tl.load(in_ptr0 + (x2), xmask, eviction_policy='evict_last').to(tl.float32)
        tl.store(out_ptr0 + (x2), tmp0, xmask)
    

def benchmark_kernel(kernel_fn, grid_size, kernel_args, constexpr_kwargs, label, iters=100, profiling_mode=False, profiling_with_stack=False):
    """Benchmark matching Inductor's InductorBenchmarker methodology."""
    # Warmup
    kernel_fn[grid_size](*kernel_args, **constexpr_kwargs)
    torch.cuda.synchronize()

    # L2 flush buffer
    l2_size = torch.cuda.get_device_properties(0).L2_cache_size
    flush_buf = torch.empty(l2_size // 4, dtype=torch.int, device="cuda")

    with profiler_or_nullcontext(kernel_fn.__name__, profiling_mode, profiling_with_stack):
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
    parser.add_argument("--profiling_mode", '--profiling_mode', action='store_true', default=False)
    parser.add_argument("--profiling_with_stack", '--profiling_with_stack', action='store_true', default=False)
    args = parser.parse_args()

    device_name = torch.cuda.get_device_name(0)
    print(f"Device: {device_name}")
    print(f"PyTorch: {torch.__version__}")
    print()

    enabled = args.profiling_mode
    with_stack = args.profiling_with_stack

    if args.platform in ("amd", "both"):
        print("--- AMD (triton_poi_fused_add_arange_index_put_new_zeros_171) ---")
        xnumel = 352
        alloc_size = 65536
        in0 = torch.randn(alloc_size, device="cuda", dtype=torch.float16)
        out0 = torch.empty(xnumel, device="cuda", dtype=torch.float32)
        constexpr_kwargs = {"XBLOCK": 2048, "num_warps": 8}
        print(f"  xnumel={xnumel}, {constexpr_kwargs}")
        grid = (xnumel,)
        r = benchmark_kernel(triton_poi_fused_add_arange_index_put_new_zeros_171, grid, [in0, out0, xnumel], constexpr_kwargs, "AMD", iters=args.iters, profiling_mode=enabled, profiling_with_stack=with_stack)
        print(f"  Min: {r['min']:.1f}us  Median: {r['median']:.1f}us  P10: {r['p10']:.1f}us  P90: {r['p90']:.1f}us  Iters: {r['iters']}")
        print()

    if args.platform in ("nv", "both"):
        print("--- NV (triton_poi_fused_add_arange_index_put_new_zeros_182) ---")
        xnumel = 352
        alloc_size = 65536
        in0 = torch.randn(alloc_size, device="cuda", dtype=torch.float16)
        out0 = torch.empty(xnumel, device="cuda", dtype=torch.float32)
        constexpr_kwargs = {"XBLOCK": 1024, "num_warps": 4}
        print(f"  xnumel={xnumel}, {constexpr_kwargs}")
        grid = (xnumel,)
        r = benchmark_kernel(triton_poi_fused_add_arange_index_put_new_zeros_182, grid, [in0, out0, xnumel], constexpr_kwargs, "NV", iters=args.iters, profiling_mode=enabled, profiling_with_stack=with_stack)
        print(f"  Min: {r['min']:.1f}us  Median: {r['median']:.1f}us  P10: {r['p10']:.1f}us  P90: {r['p90']:.1f}us  Iters: {r['iters']}")
        print()


if __name__ == "__main__":
    main()
