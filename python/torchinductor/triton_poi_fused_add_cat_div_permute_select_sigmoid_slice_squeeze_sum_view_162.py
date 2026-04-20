#!/usr/bin/env python3
"""
Auto-generated benchmark from AOTI wrapper.cpp

  AMD: triton_poi_fused_add_cat_div_permute_select_sigmoid_slice_squeeze_sum_view_162
    type: pointwise, size_hints: {'x': 65536}
    xnumel=5L*s13 (55), literals=[]
    AUTOTUNE: XBLOCK=256, num_warps=4, time_us=6.6

  NV: triton_poi_fused_add_cat_div_permute_select_sigmoid_slice_squeeze_sum_view_173
    type: pointwise, size_hints: {'x': 65536}
    xnumel=5L*s13 (55), literals=[]
    AUTOTUNE: XBLOCK=512, num_warps=4, time_us=5.12

Usage:
    python3 repro_row2_poi_copy.py
    python3 repro_row2_poi_copy.py --platform amd
    python3 repro_row2_poi_copy.py --platform nv
"""

import argparse
import torch
import triton
import triton.language as tl
from torch.profiler import profile, ProfilerActivity
import os
import contextlib


try:
    from torch._inductor.runtime import triton_helpers
except ImportError:
    triton_helpers = None


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
# AMD: triton_poi_fused_add_cat_div_permute_select_sigmoid_slice_squeeze_sum_view_162
# xnumel=55, size_hints={'x': 65536}
# AUTOTUNE: XBLOCK=256, num_warps=4, time_us=6.6
# ============================================================
@triton.jit
def triton_poi_fused_add_cat_div_permute_select_sigmoid_slice_squeeze_sum_view_162(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):

        xoffset = tl.program_id(0) * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x0 = xindex
        tmp0 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
        tl.store(out_ptr0 + (x0), tmp0, xmask)
    

# ============================================================
# NV: triton_poi_fused_add_cat_div_permute_select_sigmoid_slice_squeeze_sum_view_173
# xnumel=55, size_hints={'x': 65536}
# AUTOTUNE: XBLOCK=512, num_warps=4, time_us=5.12
# ============================================================
@triton.jit
def triton_poi_fused_add_cat_div_permute_select_sigmoid_slice_squeeze_sum_view_173(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):

        xoffset = tl.program_id(0) * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x0 = xindex
        tmp0 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
        tl.store(out_ptr0 + (x0), tmp0, xmask)
    

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
        print("--- AMD (triton_poi_fused_add_cat_div_permute_select_sigmoid_slice_squeeze_sum_view_162) ---")
        xnumel = 55
        alloc_size = 65536
        in0 = torch.randn(alloc_size, device="cuda", dtype=torch.float16)
        out0 = torch.empty(xnumel, device="cuda", dtype=torch.float32)
        constexpr_kwargs = {"XBLOCK": 1024, "num_warps": 4}
        print(f"  xnumel={xnumel}, {constexpr_kwargs}")
        grid = (xnumel,)
        r = benchmark_kernel(triton_poi_fused_add_cat_div_permute_select_sigmoid_slice_squeeze_sum_view_162, grid, [in0, out0, xnumel], constexpr_kwargs, "AMD", iters=args.iters)
        print(f"  Min: {r['min']:.1f}us  Median: {r['median']:.1f}us  P10: {r['p10']:.1f}us  P90: {r['p90']:.1f}us  Iters: {r['iters']}")
        print()

    if args.platform in ("nv", "both"):
        print("--- NV (triton_poi_fused_add_cat_div_permute_select_sigmoid_slice_squeeze_sum_view_173) ---")
        xnumel = 55
        alloc_size = 65536
        in0 = torch.randn(alloc_size, device="cuda", dtype=torch.float16)
        out0 = torch.empty(xnumel, device="cuda", dtype=torch.float32)
        constexpr_kwargs = {"XBLOCK": 512, "num_warps": 4}
        print(f"  xnumel={xnumel}, {constexpr_kwargs}")
        grid = (xnumel,)
        r = benchmark_kernel(triton_poi_fused_add_cat_div_permute_select_sigmoid_slice_squeeze_sum_view_173, grid, [in0, out0, xnumel], constexpr_kwargs, "NV", iters=args.iters)
        print(f"  Min: {r['min']:.1f}us  Median: {r['median']:.1f}us  P10: {r['p10']:.1f}us  P90: {r['p90']:.1f}us  Iters: {r['iters']}")
        print()


if __name__ == "__main__":
    main()
