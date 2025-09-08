import torch

import triton
import triton.language as tl

from triton.experimental import gluon
from triton.experimental.gluon import language as ttgl
from triton._filecheck import filecheck_test, run_parser
import triton.language as tl


# @gluon.jit
# def dot_kernel_v1(a_ptr, b_ptr, c_ptr,
#                 M, N, K,
#                 stride_am, stride_ak,  #
#                 stride_bn, stride_bk,  #
#                 stride_cm, stride_cn,
#                 BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
#                 GROUP_SIZE_M: tl.constexpr
#                ):
#     blocked: ttgl.constexpr = ttgl.BlockedLayout(size_per_thread=[1, 8], threads_per_warp=[8, 8], warps_per_cta=[4, 1],
#                                                 order=[1, 0])
#     shared: ttgl.constexpr =  ttgl.SwizzledSharedLayout(vec=8, per_phase=2, max_phase=4, order=[1, 0])
#     #type of c, ret_ty is set https://github.com/zwu-2025/triton/blob/main/python/triton/language/semantic.py#L1543 with blocked_type
#     #so we use blocked_type in here to pass the type validation in the frontend.
#     mfma_layout: ttgl.constexpr = ttgl.amd.AMDMFMALayout(version=4, instr_shape=[32, 32], transposed=True, warps_per_cta=[2, 2],
#        tiles_per_warp=[2, 2], #not used
#        ctas_per_cga=[1, 1], cta_split_num=[1, 1], cta_order=[1, 0]
#                                                          )
#     dot_a_layout: ttgl.constexpr = ttgl.DotOperandLayout(operand_index=0, parent=mfma_layout, k_width=8)
#     dot_b_layout: ttgl.constexpr = ttgl.DotOperandLayout(operand_index=1, parent=mfma_layout, k_width=8)

#     #shared1 = #ttg.swizzled_shared<{vec = 8, perPhase = 4, maxPhase = 4, order = [1, 0]}>

#     pid = ttgl.program_id(axis=0)
#     num_pid_m = ttgl.cdiv(M, BLOCK_SIZE_M)
#     num_pid_n = ttgl.cdiv(N, BLOCK_SIZE_N)
#     num_pid_in_group = GROUP_SIZE_M * num_pid_n
#     group_id = pid // num_pid_in_group
#     first_pid_m = group_id * GROUP_SIZE_M
#     group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
#     pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
#     pid_n = (pid % num_pid_in_group) // group_size_m

#     # offs_am = (pid_m * BLOCK_SIZE_M + ttgl.arange(0, BLOCK_SIZE_M, layout=ttgl.SliceLayout(1, blocked))) % M
#     # offs_bn = (pid_n * BLOCK_SIZE_N + ttgl.arange(0, BLOCK_SIZE_N, layout=ttgl.SliceLayout(0, blocked))) % N

#     offs_am = (pid_m * BLOCK_SIZE_M + ttgl.arange(0, BLOCK_SIZE_M, layout=ttgl.SliceLayout(1, blocked)))
#     offs_bn = (pid_n * BLOCK_SIZE_N + ttgl.arange(0, BLOCK_SIZE_N, layout=ttgl.SliceLayout(1, blocked)))

#     offs_ak = ttgl.arange(0, BLOCK_SIZE_K, layout=ttgl.SliceLayout(0, blocked))
#     offs_bk = ttgl.arange(0, BLOCK_SIZE_K, layout=ttgl.SliceLayout(0, blocked))
#     offs_a = offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak
#     offs_b = offs_bn[:, None] * stride_bn + offs_bk[None, :] * stride_bk

#     ttgl.static_assert(offs_a.type.layout == blocked)

#     smem_a = ttgl.allocate_shared_memory(a_ptr.dtype.element_ty, [BLOCK_SIZE_M, BLOCK_SIZE_K], shared)
#     a = ttgl.amd.cdna3.buffer_load(ptr=a_ptr, offsets=offs_a, mask=(offs_ak[None, :] < K) & (offs_am[:, None] < M))
#     smem_a.store(a)

#     smem_b = ttgl.allocate_shared_memory(b_ptr.dtype.element_ty, [BLOCK_SIZE_N, BLOCK_SIZE_K], shared)
#     b = ttgl.amd.cdna3.buffer_load(ptr=b_ptr, offsets=offs_b, mask=(offs_bk[None, :] < K) & (offs_bn[:, None] < N))
#     smem_b.store(b)

#     acc = ttgl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), ttgl.float32, mfma_layout)
#     for k in range(0, ttgl.cdiv(K, BLOCK_SIZE_K)):
#         # a_mask = offs_ak[None, :] < K - k * BLOCK_SIZE_K
#         # b_mask = offs_bk[:, None] < K - k * BLOCK_SIZE_K

#         blk_a = smem_a.load(layout=dot_a_layout) #ttgl.amd.cdna3.buffer_load(ptr=a_ptr, offsets=offs_a, mask=a_mask)
#         blk_b = smem_b.load(layout=blocked) #ttgl.amd.cdna3.buffer_load(ptr=b_ptr, offsets=offs_b, mask=b_mask)
#         blk_b = ttgl.convert_layout(blk_b.trans(), dot_b_layout)
#         #a2 = ttgl.convert_layout(a, layout=dot_a_layout)
#         #b2 = ttgl.convert_layout(b, layout=dot_b_layout)
#         # acc = ttgl.amd.cdna3.dot(blk_a, blk_b, acc)
#         # acc = tl.dot(blk_a, blk_b, acc)
#         acc = ttgl.amd.cdna3.mfma(blk_a, blk_b, acc)
#         a_ptr += BLOCK_SIZE_K * stride_ak
#         b_ptr += BLOCK_SIZE_K * stride_bk

#         next_a = ttgl.amd.cdna3.buffer_load(ptr=a_ptr, offsets=offs_a, mask=(offs_ak[None, :] < K) & (offs_am[:, None] < M))
#         smem_a.store(next_a)

#         next_b = ttgl.amd.cdna3.buffer_load(ptr=b_ptr, offsets=offs_b, mask=(offs_bk[None, :] < K) & (offs_bn[:, None] < N))
#         smem_b.store(next_b)

#     c_layout: ttgl.constexpr = blocked # ll
#     c = acc
#     c = ttgl.convert_layout(c, layout=c_layout)
#     c = c.to(a_ptr.dtype.element_ty)

#     offs_cm = pid_m * BLOCK_SIZE_M + ttgl.arange(0, BLOCK_SIZE_M, layout=ttgl.SliceLayout(1, c_layout))
#     offs_cn = pid_n * BLOCK_SIZE_N + ttgl.arange(0, BLOCK_SIZE_N, layout=ttgl.SliceLayout(0, c_layout))
#     offs_c = offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
#     c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
#     ttgl.amd.cdna3.buffer_store(stored_value=c, ptr=c_ptr, offsets=offs_c, mask=c_mask)


@gluon.jit
def dot_kernel_v2(a_ptr, b_ptr, c_ptr,
                M, N, K,
                stride_am, stride_ak,  #
                stride_bn, stride_bk,  #
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
       ctas_per_cga=[1, 1], cta_split_num=[1, 1], cta_order=[1, 0])

    # linear layout    
    linear: ttgl.constexpr = ttgl.DistributedLinearLayout(reg_bases=[[0, 1], [0, 2], [0, 4]], 
                                                              lane_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [0, 16], [0, 8]],
                                                              warp_bases=[[0, 0], [0, 0]],
                                                              block_bases=[], shape=[16, 32])

    # linear layout    
    linear1: ttgl.constexpr = ttgl.DistributedLinearLayout(reg_bases=[[0, 1], [0, 2], [0, 4], [0, 32], [16, 0]], 
                                                              lane_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [0, 8], [0, 16]],
                                                              warp_bases=[[0, 0], [0, 0]],
                                                              block_bases=[], shape=[32, 64])

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
    offs_bn = (pid_n * BLOCK_SIZE_N + ttgl.arange(0, BLOCK_SIZE_N, layout=ttgl.SliceLayout(1, blocked)))

    offs_ak = ttgl.arange(0, BLOCK_SIZE_K, layout=ttgl.SliceLayout(0, blocked))
    offs_bk = ttgl.arange(0, BLOCK_SIZE_K, layout=ttgl.SliceLayout(0, blocked))
    offs_a = offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak
    offs_b = offs_bn[:, None] * stride_bn + offs_bk[None, :] * stride_bk

    ttgl.static_assert(offs_a.type.layout == blocked)

    smem_a = ttgl.allocate_shared_memory(a_ptr.dtype.element_ty, [BLOCK_SIZE_M, BLOCK_SIZE_K], shared)

    smem_b = ttgl.allocate_shared_memory(b_ptr.dtype.element_ty, [BLOCK_SIZE_N, BLOCK_SIZE_K], shared)

    acc = ttgl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), ttgl.float32, mfma_layout)
    for k in range(0, ttgl.cdiv(K, BLOCK_SIZE_K)):
        # a_mask = offs_ak[None, :] < K - k * BLOCK_SIZE_K
        # b_mask = offs_bk[:, None] < K - k * BLOCK_SIZE_K
        a = ttgl.amd.cdna3.buffer_load(ptr=a_ptr, offsets=offs_a, mask=(offs_ak[None, :] < K) & (offs_am[:, None] < M))
        smem_a.store(a)
        b = ttgl.amd.cdna3.buffer_load(ptr=b_ptr, offsets=offs_b, mask=(offs_bk[None, :] < K) & (offs_bn[:, None] < N))
        smem_b.store(b)

        blk_a = smem_a.load(layout=dot_a_layout) #ttgl.amd.cdna3.buffer_load(ptr=a_ptr, offsets=offs_a, mask=a_mask)
        blk_b0 = smem_b.load(layout=linear1) #ttgl.amd.cdna3.buffer_load(ptr=b_ptr, offsets=offs_b, mask=b_mask)
        # blk_b = ttgl.convert_layout(blk_b.trans(), dot_b_layout)
        blk_b = ttgl.convert_layout(blk_b0.trans(), dot_b_layout)
        #a2 = ttgl.convert_layout(a, layout=dot_a_layout)
        #b2 = ttgl.convert_layout(b, layout=dot_b_layout)
        # acc = ttgl.amd.cdna3.dot(blk_a, blk_b, acc)
        # acc = tl.dot(blk_a, blk_b, acc)
        acc = ttgl.amd.cdna3.mfma(blk_a, blk_b, acc)
        a_ptr += BLOCK_SIZE_K * stride_ak
        b_ptr += BLOCK_SIZE_K * stride_bk

        # next_a = ttgl.amd.cdna3.buffer_load(ptr=a_ptr, offsets=offs_a, mask=(offs_ak[None, :] < K) & (offs_am[:, None] < M))
        # smem_a.store(next_a)

        # next_b = ttgl.amd.cdna3.buffer_load(ptr=b_ptr, offsets=offs_b, mask=(offs_bk[None, :] < K) & (offs_bn[:, None] < N))
        # smem_b.store(next_b)

    c_layout: ttgl.constexpr = blocked # ll
    c = acc
    c = ttgl.convert_layout(c, layout=c_layout)
    c = c.to(a_ptr.dtype.element_ty)

    offs_cm = pid_m * BLOCK_SIZE_M + ttgl.arange(0, BLOCK_SIZE_M, layout=ttgl.SliceLayout(1, c_layout))
    offs_cn = pid_n * BLOCK_SIZE_N + ttgl.arange(0, BLOCK_SIZE_N, layout=ttgl.SliceLayout(0, c_layout))
    offs_c = offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    ttgl.amd.cdna3.buffer_store(stored_value=c, ptr=c_ptr, offsets=offs_c, mask=c_mask)


@gluon.jit
def dot_kernel_v3(a_ptr, b_ptr, c_ptr,
                M, N, K,
                stride_am, stride_ak,  #
                stride_bn, stride_bk,  #
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
    offs_bn = (pid_n * BLOCK_SIZE_N + ttgl.arange(0, BLOCK_SIZE_N, layout=ttgl.SliceLayout(1, blocked)))

    offs_ak = ttgl.arange(0, BLOCK_SIZE_K, layout=ttgl.SliceLayout(0, blocked))
    offs_bk = ttgl.arange(0, BLOCK_SIZE_K, layout=ttgl.SliceLayout(0, blocked))
    offs_a = offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak
    offs_b = offs_bn[:, None] * stride_bn + offs_bk[None, :] * stride_bk

    ttgl.static_assert(offs_a.type.layout == blocked)

    smem_a = ttgl.allocate_shared_memory(a_ptr.dtype.element_ty, [BLOCK_SIZE_M, BLOCK_SIZE_K], shared)

    smem_b = ttgl.allocate_shared_memory(b_ptr.dtype.element_ty, [BLOCK_SIZE_N, BLOCK_SIZE_K], shared)

    acc = ttgl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), ttgl.float32, mfma_layout)
    for k in range(0, ttgl.cdiv(K, BLOCK_SIZE_K)):
        # a_mask = offs_ak[None, :] < K - k * BLOCK_SIZE_K
        # b_mask = offs_bk[:, None] < K - k * BLOCK_SIZE_K
        a = ttgl.amd.cdna3.buffer_load(ptr=a_ptr, offsets=offs_a, mask=(offs_ak[None, :] < K) & (offs_am[:, None] < M))
        smem_a.store(a)
        b = ttgl.amd.cdna3.buffer_load(ptr=b_ptr, offsets=offs_b, mask=(offs_bk[None, :] < K) & (offs_bn[:, None] < N))
        # smem_b.store(b)

        blk_a = smem_a.load(layout=dot_a_layout) #ttgl.amd.cdna3.buffer_load(ptr=a_ptr, offsets=offs_a, mask=a_mask)
        # blk_b = smem_b.load(layout=blocked) #ttgl.amd.cdna3.buffer_load(ptr=b_ptr, offsets=offs_b, mask=b_mask)
        # blk_b = ttgl.convert_layout(blk_b.trans(), dot_b_layout)
        blk_b = ttgl.convert_layout(b.trans(), dot_b_layout)
        #a2 = ttgl.convert_layout(a, layout=dot_a_layout)
        #b2 = ttgl.convert_layout(b, layout=dot_b_layout)
        # acc = ttgl.amd.cdna3.dot(blk_a, blk_b, acc)
        # acc = tl.dot(blk_a, blk_b, acc)
        acc = ttgl.amd.cdna3.mfma(blk_a, blk_b, acc)
        a_ptr += BLOCK_SIZE_K * stride_ak
        b_ptr += BLOCK_SIZE_K * stride_bk

        # next_a = ttgl.amd.cdna3.buffer_load(ptr=a_ptr, offsets=offs_a, mask=(offs_ak[None, :] < K) & (offs_am[:, None] < M))
        # smem_a.store(next_a)

        # next_b = ttgl.amd.cdna3.buffer_load(ptr=b_ptr, offsets=offs_b, mask=(offs_bk[None, :] < K) & (offs_bn[:, None] < N))
        # smem_b.store(next_b)

    c_layout: ttgl.constexpr = blocked # ll
    c = acc
    c = ttgl.convert_layout(c, layout=c_layout)
    c = c.to(a_ptr.dtype.element_ty)

    offs_cm = pid_m * BLOCK_SIZE_M + ttgl.arange(0, BLOCK_SIZE_M, layout=ttgl.SliceLayout(1, c_layout))
    offs_cn = pid_n * BLOCK_SIZE_N + ttgl.arange(0, BLOCK_SIZE_N, layout=ttgl.SliceLayout(0, c_layout))
    offs_c = offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    ttgl.amd.cdna3.buffer_store(stored_value=c, ptr=c_ptr, offsets=offs_c, mask=c_mask)


@triton.jit
def dot_kernel_triton(a_ptr, b_ptr, c_ptr,
                M, N, K,
                stride_am, stride_ak,  #
                stride_bn, stride_bk,  #
                stride_cm, stride_cn,
                BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
                GROUP_SIZE_M: tl.constexpr
               ):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # offs_am = (pid_m * BLOCK_SIZE_M + ttgl.arange(0, BLOCK_SIZE_M, layout=ttgl.SliceLayout(1, blocked))) % M
    # offs_bn = (pid_n * BLOCK_SIZE_N + ttgl.arange(0, BLOCK_SIZE_N, layout=ttgl.SliceLayout(0, blocked))) % N

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))

    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_a = offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    offs_b = offs_bn[:, None] * stride_bn + offs_k[None, :] * stride_bk
    a_ptrs = a_ptr + offs_a
    b_ptrs = b_ptr + offs_b

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # a_mask = offs_ak[None, :] < K - k * BLOCK_SIZE_K
        # b_mask = offs_bk[:, None] < K - k * BLOCK_SIZE_K
        blk_a = tl.load(a_ptrs, mask=(offs_k[None, :] < K - k * BLOCK_SIZE_K) & (offs_am[:, None] < M), other=0.0)
        blk_b = tl.load(b_ptrs, mask=(offs_k[None, :] < K - k * BLOCK_SIZE_K) & (offs_bn[:, None] < N), other=0.0)

        acc = tl.dot(blk_a, tl.trans(blk_b), acc)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    acc = acc.to(a_ptr.dtype.element_ty)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_c = offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_ptrs = c_ptr + offs_c
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def dot_triton(a, b):
    M, K = a.shape
    N, _ = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    block_m = 16
    block_n = 32
    grid = (triton.cdiv(M, block_m) * triton.cdiv(N, block_n), )

    dot_kernel_triton[grid](a, b, c,
        M, N, K,  #
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        c.stride(0), c.stride(1),  #
        BLOCK_SIZE_M=block_m, BLOCK_SIZE_N=block_n, BLOCK_SIZE_K=64,
        GROUP_SIZE_M=4,
    )

    return c


def dot_gluon(a, b):
    M, K = a.shape
    N, _ = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    block_m = 16
    block_n = 32
    grid = (triton.cdiv(M, block_m) * triton.cdiv(N, block_n), )

    dot_kernel_v2[grid](a, b, c,
    # dot_kernel_triton[grid](a, b, c,
        M, N, K,  #
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        c.stride(0), c.stride(1),  #
        BLOCK_SIZE_M=block_m, BLOCK_SIZE_N=block_n, BLOCK_SIZE_K=64,
        GROUP_SIZE_M=4,
    )

    return c


def dot_torch(a, b):
    return torch.matmul(a, torch.transpose(b, 0, 1))

M = 160
N = 130
K = 128

data_type = torch.float16

a = torch.randn((M, K), device='cuda', dtype=data_type)
b = torch.randn((N, K), device='cuda', dtype=data_type)

tri = dot_gluon(a, b)
ref = dot_torch(a, b)
# torch.testing.assert_close(tri.to(torch.float32), ref.to(torch.float32))
torch.testing.assert_close(tri, ref)
print(f'✅Pass')

