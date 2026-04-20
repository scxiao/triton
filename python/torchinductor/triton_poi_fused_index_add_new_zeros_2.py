# KERNEL CALLS: 1

import triton
import triton.language as tl

# import triton.language.extra.tlx as tlx  # noqa: F401

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

from torch._dynamo.testing import rand_strided
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch

@triton_heuristics.pointwise(
    size_hints={'x': 268435456}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*i64', 'in_ptr3': '*i64', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='hip', index=0, multi_processor_count=256, cc='gfx950', major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=64), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_add_new_zeros_2', 'mutated_arg_names': ['out_ptr0', 'out_ptr1', 'out_ptr2'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': True, 'num_load': 4, 'num_store': 3, 'num_reduction': 0, 'backend_hash': '0213FDD8D981B15D560CE2B172B91C13750BF18688DF3ADD5CDE2386AAEBCFD2', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': True, 'min_split_scan_rblock': 256, 'spill_threshold': 32, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'is_hip': True, 'is_fbcode': True, 'kernel_num_gb': 0.917976472, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_index_add_new_zeros_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 64
    x2 = xindex
    x0 = (xindex % 64)
    tmp0 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr1 + (x2), xmask)
    tmp7 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 2501, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert(((0 <= tmp4) & (tmp4 < 2501)) | ~(xmask), "index out of bounds: 0 <= tmp4 < 2501")
    tmp8 = tmp7 + tmp1
    tmp9 = tmp7 < 0
    tmp10 = tl.where(tmp9, tmp8, tmp7)
    tl.device_assert(((0 <= tmp10) & (tmp10 < 2501)) | ~(xmask), "index out of bounds: 0 <= tmp10 < 2501")
    tmp13 = tl.full([XBLOCK], 9600, tl.int32)
    tmp14 = tmp12 + tmp13
    tmp15 = tmp12 < 0
    tmp16 = tl.where(tmp15, tmp14, tmp12)
    tl.device_assert(((0 <= tmp16) & (tmp16 < 9600)) | ~(xmask), "index out of bounds: 0 <= tmp16 < 9600")
    tl.atomic_add(out_ptr0 + (x0 + 64*tmp4), tmp6, xmask, sem='relaxed')
    tl.atomic_add(out_ptr1 + (x0 + 64*tmp10), tmp6, xmask, sem='relaxed')
    tl.atomic_add(out_ptr2 + (x0 + 64*tmp16), tmp6, xmask, sem='relaxed')


def get_args():
    arg_0 = rand_strided((3265137,), (1,), device='cuda:0', dtype=torch.int64)
    arg_1 = rand_strided((3265137, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((3265137,), (1,), device='cuda:0', dtype=torch.int64)
    arg_3 = rand_strided((3265137,), (1,), device='cuda:0', dtype=torch.int64)
    arg_4 = rand_strided((2501, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((2501, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg_6 = rand_strided((9600, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, 208968768,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream7 = get_raw_stream(0)
        triton_poi_fused_index_add_new_zeros_2.run(*args, stream=stream7)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused_index_add_new_zeros_2.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=1000, warmup=100, return_mode="min")
    num_gb = 0.917976472
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
