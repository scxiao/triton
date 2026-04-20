#!/usr/bin/env python3
"""
Auto-generated benchmark from AOTI wrapper.cpp

  AMD: triton_red_fused_add_clamp_mean_mul_pow_rsqrt_silu_45
    type: reduction, size_hints: {'x': 16, 'r0_': 2048}
    xnumel=None (16), literals=[]
    AUTOTUNE: XBLOCK=2, R0_BLOCK=2048, num_warps=8, time_us=12.761

  NV: triton_red_fused_add_clamp_mean_mul_pow_rsqrt_silu_51
    type: reduction, size_hints: {'x': 16, 'r0_': 2048}
    xnumel=None (16), literals=[]
    AUTOTUNE: XBLOCK=1, R0_BLOCK=2048, num_warps=16, time_us=8.448

Usage:
    python3 repro_triton_red_fused_silu_51.py
    python3 repro_triton_red_fused_silu_51.py --platform amd
    python3 repro_triton_red_fused_silu_51.py --platform nv
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
# AMD: triton_red_fused_add_clamp_mean_mul_pow_rsqrt_silu_45
# xnumel=16, size_hints={'x': 16, 'r0_': 2048}
# AUTOTUNE: XBLOCK=2, R0_BLOCK=2048, num_warps=8, time_us=12.761
# ============================================================
@triton.jit
def triton_red_fused_add_clamp_mean_mul_pow_rsqrt_silu_45(in_out_ptr0, in_ptr0, in_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):

        r0_numel = 2048
        rnumel = r0_numel
        RBLOCK: tl.constexpr = R0_BLOCK
        xoffset = tl.program_id(0) * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
        xmask = xindex < xnumel
        r0_base = tl.arange(0, R0_BLOCK)[None, :]
        rbase = r0_base
        x0 = xindex
        _tmp10 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK, num_stages = 2):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_1 = r0_index
            tmp0 = tl.load(in_out_ptr0 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
            tmp1 = tl.load(in_ptr0 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
            tmp2 = tmp0 + tmp1
            tmp3 = tmp2.to(tl.float32)
            tmp4 = tl.full([1, 1], -65504.0, tl.float32)
            tmp5 = tl.maximum(tmp3, tmp4, tl.PropagateNan.ALL)
            tmp6 = tl.full([1, 1], 65504.0, tl.float32)
            tmp7 = tl.minimum(tmp5, tmp6, tl.PropagateNan.ALL)
            tmp8 = tmp7 * tmp7
            tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
            tmp11 = _tmp10 + tmp9
            _tmp10 = tl.where(r0_mask & xmask, tmp11, _tmp10)
        tmp10 = tl.sum(_tmp10, 1)[:, None]
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK, num_stages = 2):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_1 = r0_index
            tmp12 = tl.load(in_out_ptr0 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
            tmp13 = tl.load(in_ptr0 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
            tmp26 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
            tmp14 = tmp12 + tmp13
            tmp15 = tmp14.to(tl.float32)
            tmp16 = tl.full([1, 1], -65504.0, tl.float32)
            tmp17 = tl.maximum(tmp15, tmp16, tl.PropagateNan.ALL)
            tmp18 = tl.full([1, 1], 65504.0, tl.float32)
            tmp19 = tl.minimum(tmp17, tmp18, tl.PropagateNan.ALL)
            tmp20 = tl.full([1, 1], 2048.0, tl.float32)
            tmp21 = (tmp10 / tmp20)
            tmp22 = tl.full([1, 1], 9.999999747378752e-06, tl.float32)
            tmp23 = tmp21 + tmp22
            tmp24 = tl.rsqrt(tmp23)
            tmp25 = tmp19 * tmp24
            tmp27 = tmp26.to(tl.float32)
            tmp28 = tmp25 * tmp27
            tmp29 = -tmp28
            tmp30 = libdevice.exp(tmp29)
            tmp31 = tl.full([1, 1], 1.0, tl.float32)
            tmp32 = tmp30 + tmp31
            tmp33 = (tmp28 / tmp32)
            tmp34 = tmp33.to(tl.float32)
            tl.store(in_out_ptr0 + (r0_1 + 2048*x0), tmp34, r0_mask & xmask)
    

# ============================================================
# NV: triton_red_fused_add_clamp_mean_mul_pow_rsqrt_silu_51
# xnumel=16, size_hints={'x': 16, 'r0_': 2048}
# AUTOTUNE: XBLOCK=1, R0_BLOCK=2048, num_warps=16, time_us=8.448
# ============================================================
@triton.jit
def triton_red_fused_add_clamp_mean_mul_pow_rsqrt_silu_51(in_out_ptr0, in_ptr0, in_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):

        r0_numel = 2048
        rnumel = r0_numel
        RBLOCK: tl.constexpr = R0_BLOCK
        xoffset = tl.program_id(0).to(tl.int64) * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:, None].to(tl.int64)
        xmask = xindex < xnumel
        r0_base = tl.arange(0, R0_BLOCK)[None, :].to(tl.int64)
        rbase = r0_base
        x0 = xindex
        _tmp10 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_1 = r0_index
            tmp0 = tl.load(in_out_ptr0 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
            tmp1 = tl.load(in_ptr0 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
            tmp2 = tmp0 + tmp1
            tmp3 = tmp2.to(tl.float32)
            tmp4 = tl.full([1, 1], -65504.0, tl.float32)
            tmp5 = triton_helpers.maximum(tmp3, tmp4)
            tmp6 = tl.full([1, 1], 65504.0, tl.float32)
            tmp7 = triton_helpers.minimum(tmp5, tmp6)
            tmp8 = tmp7 * tmp7
            tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
            tmp11 = _tmp10 + tmp9
            _tmp10 = tl.where(r0_mask & xmask, tmp11, _tmp10)
        tmp10 = tl.sum(_tmp10, 1)[:, None]
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_1 = r0_index
            tmp12 = tl.load(in_out_ptr0 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
            tmp13 = tl.load(in_ptr0 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
            tmp26 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
            tmp14 = tmp12 + tmp13
            tmp15 = tmp14.to(tl.float32)
            tmp16 = tl.full([1, 1], -65504.0, tl.float32)
            tmp17 = triton_helpers.maximum(tmp15, tmp16)
            tmp18 = tl.full([1, 1], 65504.0, tl.float32)
            tmp19 = triton_helpers.minimum(tmp17, tmp18)
            tmp20 = tl.full([1, 1], 2048.0, tl.float32)
            tmp21 = (tmp10 / tmp20)
            tmp22 = tl.full([1, 1], 9.999999747378752e-06, tl.float32)
            tmp23 = tmp21 + tmp22
            tmp24 = libdevice.rsqrt(tmp23)
            tmp25 = tmp19 * tmp24
            tmp27 = tmp26.to(tl.float32)
            tmp28 = tmp25 * tmp27
            tmp29 = -tmp28
            tmp30 = libdevice.exp(tmp29)
            tmp31 = tl.full([1, 1], 1.0, tl.float32)
            tmp32 = tmp30 + tmp31
            tmp33 = (tmp28 / tmp32)
            tmp34 = tmp33.to(tl.float32)
            tl.store(in_out_ptr0 + (r0_1 + 2048*x0), tmp34, r0_mask & xmask)
    

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
        print("--- AMD (triton_red_fused_add_clamp_mean_mul_pow_rsqrt_silu_45) ---")
        xnumel = 16
        r0_numel = 2048
        alloc_size = 65536
        in0 = torch.randn(alloc_size, device="cuda", dtype=torch.float16)
        in1 = torch.randn(alloc_size, device="cuda", dtype=torch.float16)
        in2 = torch.randn(alloc_size, device="cuda", dtype=torch.float16)
        constexpr_kwargs = {"XBLOCK": 2, "R0_BLOCK": 2048, "num_warps": 8}
        print(f"  xnumel={xnumel}, r0_numel={r0_numel}, {constexpr_kwargs}")
        grid = (xnumel,)
        r = benchmark_kernel(triton_red_fused_add_clamp_mean_mul_pow_rsqrt_silu_45, grid, [in0, in1, in2, xnumel, r0_numel], constexpr_kwargs, "AMD", iters=args.iters, profiling_mode=enabled, profiling_with_stack=with_stack)
        print(f"  Min: {r['min']:.1f}us  Median: {r['median']:.1f}us  P10: {r['p10']:.1f}us  P90: {r['p90']:.1f}us  Iters: {r['iters']}")
        print()

    if args.platform in ("nv", "both"):
        print("--- NV (triton_red_fused_add_clamp_mean_mul_pow_rsqrt_silu_51) ---")
        xnumel = 16
        r0_numel = 2048
        alloc_size = 65536
        in0 = torch.randn(alloc_size, device="cuda", dtype=torch.float16)
        in1 = torch.randn(alloc_size, device="cuda", dtype=torch.float16)
        in2 = torch.randn(alloc_size, device="cuda", dtype=torch.float16)
        constexpr_kwargs = {"XBLOCK": 1, "R0_BLOCK": 2048, "num_warps": 16}
        print(f"  xnumel={xnumel}, r0_numel={r0_numel}, {constexpr_kwargs}")
        grid = (xnumel,)
        r = benchmark_kernel(triton_red_fused_add_clamp_mean_mul_pow_rsqrt_silu_51, grid, [in0, in1, in2, xnumel, r0_numel], constexpr_kwargs, "NV", iters=args.iters, profiling_mode=enabled, profiling_with_stack=with_stack)
        print(f"  Min: {r['min']:.1f}us  Median: {r['median']:.1f}us  P10: {r['p10']:.1f}us  P90: {r['p90']:.1f}us  Iters: {r['iters']}")
        print()


if __name__ == "__main__":
    main()
