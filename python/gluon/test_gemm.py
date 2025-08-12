import torch

import triton
import triton.language as tl

from triton.experimental import gluon
from triton.experimental.gluon import language as ttgl
from triton.experimental.gluon.language.nvidia import blackwell
from triton.experimental.gluon.language.nvidia import hopper
from triton.experimental.gluon.language.nvidia.blackwell import mbarrier, tma, TensorMemoryLayout, async_copy
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton._filecheck import filecheck_test, run_parser
import triton.language as tl
from triton._internal_testing import is_cuda, is_ampere_or_newer, is_blackwell, is_hopper, is_hopper_or_newer
from triton.compiler.errors import CompilationError, CompileTimeAssertionFailure


@gluon.jit
def dot_kernel_v1(a_ptr, b_ptr, c_ptr,
                M, N, K,
                stride_am, stride_ak,  #
                stride_bk, stride_bn,  #
                stride_cm, stride_cn,
                BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
                GROUP_SIZE_M: tl.constexpr
               ):
    blocked: ttgl.constexpr = ttgl.BlockedLayout(size_per_thread=[1, 8], threads_per_warp=[8, 8], warps_per_cta=[4, 1],
                                                order=[1, 0])
    shared: ttgl.constexpr =  ttgl.SwizzledSharedLayout(vec=8, per_phase=2, max_phase=4, order=[1, 0])
    #type of c, ret_ty is set https://github.com/zwu-2025/triton/blob/main/python/triton/language/semantic.py#L1543 with blocked_type
    #so we use blocked_type in here to pass the type validation in the frontend.
    mfma_layout: ttgl.constexpr = ttgl.amd.AMDMFMALayout(version=4, instr_shape=[32, 32], transposed=True, warps_per_cta=[2, 2],
       tiles_per_warp=[2, 2], #not used
       ctas_per_cga=[1, 1], cta_split_num=[1, 1], cta_order=[1, 0]
                                                         )
    dot_a_layout: ttgl.constexpr = ttgl.DotOperandLayout(operand_index=0, parent=mfma_layout, k_width=8)
    dot_b_layout: ttgl.constexpr = ttgl.DotOperandLayout(operand_index=1, parent=mfma_layout, k_width=8)

    #shared1 = #ttg.swizzled_shared<{vec = 8, perPhase = 4, maxPhase = 4, order = [1, 0]}>

    pid = ttgl.program_id(axis=0)
    num_pid_m = ttgl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = ttgl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # offs_am = (pid_m * BLOCK_SIZE_M + ttgl.arange(0, BLOCK_SIZE_M, layout=ttgl.SliceLayout(1, blocked))) % M
    # offs_bn = (pid_n * BLOCK_SIZE_N + ttgl.arange(0, BLOCK_SIZE_N, layout=ttgl.SliceLayout(0, blocked))) % N

    offs_am = (pid_m * BLOCK_SIZE_M + ttgl.arange(0, BLOCK_SIZE_M, layout=ttgl.SliceLayout(1, blocked)))
    offs_bn = (pid_n * BLOCK_SIZE_N + ttgl.arange(0, BLOCK_SIZE_N, layout=ttgl.SliceLayout(0, blocked)))

    offs_ak = ttgl.arange(0, BLOCK_SIZE_K, layout=ttgl.SliceLayout(0, blocked))
    offs_bk = ttgl.arange(0, BLOCK_SIZE_K, layout=ttgl.SliceLayout(1, blocked))
    offs_a = offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak
    offs_b = offs_bk[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    ttgl.static_assert(offs_a.type.layout == blocked)

    smem_a = ttgl.allocate_shared_memory(a_ptr.dtype.element_ty, [BLOCK_SIZE_M, BLOCK_SIZE_K], shared)
    a = ttgl.amd.cdna3.buffer_load(ptr=a_ptr, offsets=offs_a, mask=(offs_ak[None, :] < K) & (offs_am[:, None] < M))
    smem_a.store(a)

    smem_b = ttgl.allocate_shared_memory(b_ptr.dtype.element_ty, [BLOCK_SIZE_K, BLOCK_SIZE_N], shared)
    b = ttgl.amd.cdna3.buffer_load(ptr=b_ptr, offsets=offs_b, mask=(offs_bk[:, None] < K) & (offs_bn[None, :] < N))
    smem_b.store(b)

    acc = ttgl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), ttgl.float32, mfma_layout)
    for k in range(0, ttgl.cdiv(K, BLOCK_SIZE_K)):
        a_mask = offs_ak[None, :] < K - k * BLOCK_SIZE_K
        b_mask = offs_bk[:, None] < K - k * BLOCK_SIZE_K

        blk_a = smem_a.load(layout=dot_a_layout) #ttgl.amd.cdna3.buffer_load(ptr=a_ptr, offsets=offs_a, mask=a_mask)
        blk_b = smem_b.load(layout=dot_b_layout) #ttgl.amd.cdna3.buffer_load(ptr=b_ptr, offsets=offs_b, mask=b_mask)

        #a2 = ttgl.convert_layout(a, layout=dot_a_layout)
        #b2 = ttgl.convert_layout(b, layout=dot_b_layout)
        # acc = ttgl.amd.cdna3.dot(blk_a, blk_b, acc)
        # acc = tl.dot(blk_a, blk_b, acc)
        acc = ttgl.amd.cdna3.mfma(blk_a, blk_b, acc)
        a_ptr += BLOCK_SIZE_K * stride_ak
        b_ptr += BLOCK_SIZE_K * stride_bk

        next_a = ttgl.amd.cdna3.buffer_load(ptr=a_ptr, offsets=offs_a, mask=(offs_ak[None, :] < K) & (offs_am[:, None] < M))
        smem_a.store(next_a)

        next_b = ttgl.amd.cdna3.buffer_load(ptr=b_ptr, offsets=offs_b, mask=(offs_bk[:, None] < K) & (offs_bn[None, :] < N))
        smem_b.store(next_b)

    c_layout: ttgl.constexpr = blocked # ll
    c = acc
    c = ttgl.convert_layout(c, layout=c_layout)
    c = c.to(a_ptr.dtype.element_ty)

    offs_cm = pid_m * BLOCK_SIZE_M + ttgl.arange(0, BLOCK_SIZE_M, layout=ttgl.SliceLayout(1, c_layout))
    offs_cn = pid_n * BLOCK_SIZE_N + ttgl.arange(0, BLOCK_SIZE_N, layout=ttgl.SliceLayout(0, c_layout))
    offs_c = offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    ttgl.amd.cdna3.buffer_store(stored_value=c, ptr=c_ptr, offsets=offs_c, mask=c_mask)


def dot(a, b):
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)

    # dot_kernel[1, ](a, b, c,
    dot_kernel_v1[1, ](a, b, c,
        M, N, K,  #
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        c.stride(0), c.stride(1),  #
        BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=64,
        GROUP_SIZE_M=4,
    )

    return c


def dot_torch(a, b):
    return torch.matmul(a, b)

# M = 64#128
# N = 64#128
# K = 64

M = 64
N = 64
K = 128

a = torch.randn((M, K), device='cuda', dtype=torch.float32)
b = torch.randn((K, N), device='cuda', dtype=torch.float32)

tri = dot(a, b)
ref = dot_torch(a, b)
torch.testing.assert_close(tri.to(torch.float32), ref.to(torch.float32))
print(f'✅Pass')

