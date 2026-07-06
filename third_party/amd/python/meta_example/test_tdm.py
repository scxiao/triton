import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd.gfx1250 import async_copy as cp
from triton.experimental.gluon.language.amd.gfx1250 import get_wmma_scale_layout, PartitionedSharedLayout, _valid_dtype_combinations

# -------------- 1. kernel calling async_copy --------------------
@gluon.jit
def kernel_async_copy_local_prefetch(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,    
):
    """
    GEMM kernel using async_copy for gfx1250.

    Computes C = A @ B where:
    - A is (M, K) row-major (K-contiguous)
    - B is (K, N) column-major (K-contiguous), created via torch.randn(N, K).T
    - C is (M, N) in float32
    """    

    pid = gl.program_id(axis=0)
    num_pid_n = gl.cdiv(N, BLOCK_N)

    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    
    gLoadLayoutA : gl.constexpr = gl.BlockedLayout(
        [1, 8],
        [2, 16],
        [4, 1],
        [1, 0],
    )
    
    gLoadLayoutB : gl.constexpr = gl.BlockedLayout(
        [8, 1],
        [16, 2],
        [1, 4],
        [0, 1],
    )
    
    sharedLayoutA: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[256, 16]],
        [BLOCK_M, BLOCK_K],
        [1, 0],
    )
    
    sharedLayoutB: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[256, 16]],
        [BLOCK_K, BLOCK_N],
        [0, 1],
    )

    smemA = gl.allocate_shared_memory(a_ptr.dtype.element_ty, [2, BLOCK_M, BLOCK_K], layout=sharedLayoutA)
    smemB = gl.allocate_shared_memory(b_ptr.dtype.element_ty, [2, BLOCK_K, BLOCK_N], layout=sharedLayoutB)

    offs_am = gl.arange(0, BLOCK_M, gl.SliceLayout(1, gLoadLayoutA))
    offs_ak = gl.arange(0, BLOCK_K, gl.SliceLayout(0, gLoadLayoutA))

    offs_bn = gl.arange(0, BLOCK_N, gl.SliceLayout(0, gLoadLayoutB))
    offs_bk = gl.arange(0, BLOCK_K, gl.SliceLayout(1, gLoadLayoutB))

    a_base = a_ptr + pid_m * BLOCK_M * stride_am
    b_base = b_ptr + pid_n * BLOCK_N * stride_bn
    # async_copy.global_to_shared requires full pointer tensors
    a_ptrs = a_base + offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak
    b_ptrs = b_base + offs_bk[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    
    wmmaLayout: gl.constexpr = gl.amd.AMDWMMALayout(
        version=3, transposed=True, warp_bases=[[0, 1], [1, 0]], instr_shape=[16, 16, 32]
    )
    
    dotOpLayoutA: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=wmmaLayout, k_width=8)
    dotOpLayoutB: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=wmmaLayout, k_width=8)

    acc = gl.zeros((BLOCK_M, BLOCK_N), gl.float32, wmmaLayout)

    iterMax = gl.cdiv(K, BLOCK_K)

    # load A, B to lds buffer 0
    g_idx = 0
    cp.global_to_shared(smemA.index(g_idx), a_ptrs)
    cp.global_to_shared(smemB.index(g_idx), b_ptrs)
    cp.commit_group()
    a_ptrs += BLOCK_K * stride_ak
    b_ptrs += BLOCK_K * stride_bk
    
    # load A, B to lds buffer 0
    g_idx = 1    
    cp.global_to_shared(smemA.index(g_idx), a_ptrs)
    cp.global_to_shared(smemB.index(g_idx), b_ptrs)
    cp.commit_group()
    a_ptrs += BLOCK_K * stride_ak
    b_ptrs += BLOCK_K * stride_bk

    # wait buffer 0 to complete    
    cp.wait_group(1)

    # local_load a0, b0 -> buffer 0
    l_idx = 0
    a = smemA.index(l_idx).load(layout=dotOpLayoutA)
    b = smemB.index(l_idx).load(layout=dotOpLayoutB)

    for k in range(0, iterMax - 1):
        l_idx = (k + 1) % 2
        g_idx = k % 2
        
        acc = gl.amd.gfx1250.wmma(a, b, acc)
        
        # wait for all outstanding TDM loads to complete
        cp.wait_group(0)

        # cp.global_to_shared(smemA.index(g_idx), a_ptrs, mask=(k < iterMax - 2))
        # cp.global_to_shared(smemB.index(g_idx), b_ptrs, mask=(k < iterMax - 2))
        # cp.global_to_shared(smemA.index(g_idx), a_ptrs, mask=offs_ak[None,:] < K - (k + 2) * BLOCK_K)
        # cp.global_to_shared(smemB.index(g_idx), b_ptrs, mask=offs_bk[:,None] < K - (k + 2) * BLOCK_K)
        cp.global_to_shared(smemA.index(g_idx), a_ptrs)
        cp.global_to_shared(smemB.index(g_idx), b_ptrs)
        cp.commit_group()
         
        a_next = smemA.index(l_idx).load(layout=dotOpLayoutA)
        b_next = smemB.index(l_idx).load(layout=dotOpLayoutB)

        a = a_next
        b = b_next
        
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # epilogue
    acc = gl.amd.gfx1250.wmma(a, b, acc)
    
    # store results back
    gStoreLayoutC: gl.constexpr = wmmaLayout
    c = gl.convert_layout(acc, layout=gStoreLayoutC)
    offs_cm = pid_m * BLOCK_M + gl.arange(0, BLOCK_M, gl.SliceLayout(1, gStoreLayoutC))
    offs_cn = pid_n * BLOCK_N + gl.arange(0, BLOCK_N, gl.SliceLayout(0, gStoreLayoutC))
    offs_c = stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    gl.amd.gfx1250.buffer_store(c, c_ptr, offs_c, mask=c_mask)


def matmul_async_copy_local_prefetch(a, b):
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    
    M, K = a.shape
    _, N = b.shape
    
    BLOCK_M, BLOCK_N, BLOCK_K = 256, 256, 128
    num_warps = 4
    c = torch.zeros((M, N), dtype=torch.float32, device='cuda')
    stride_am, stride_ak = a.stride(0), a.stride(1)
    stride_bk, stride_bn = b.stride(0), b.stride(1)
    stride_cm, stride_cn = c.stride(0), c.stride(1)

    GRID_MN = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    grid = (GRID_MN,)
    kernel_async_copy_local_prefetch[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    return c


# 2. -------------------------- kernel tdm local_prefetch ---------------------
@gluon.jit
def kernel_tdm_local_prefetch(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,    
):
    """
    GEMM kernel with local prefetch pipeline using TDM for gfx1250.

    Computes C = A @ B where:
    - A is (M, K) row-major (K-contiguous)
    - B is (K, N) row-major after b.contiguous()
    - C is (M, N) in float32
    """

    pid = gl.program_id(axis=0)
    num_pid_n = gl.cdiv(N, BLOCK_N)

    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    sharedLayoutA: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[128, 8]],
        [BLOCK_M, BLOCK_K],
        [1, 0],
    )
    
    sharedLayoutB: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[256, 16]],
        [BLOCK_K, BLOCK_N],
        [1, 0],
    )

    a_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base = a_ptr + pid_m * BLOCK_M * stride_am,
        shape = (M, K),
        strides = (stride_am, stride_ak),
        block_shape = (BLOCK_M, BLOCK_K),
        layout = sharedLayoutA
    )
    
    b_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base = b_ptr + pid_n * BLOCK_N * stride_bn,
        shape = (K, N),
        strides = (stride_bk, stride_bn),
        block_shape = (BLOCK_K, BLOCK_N),
        layout = sharedLayoutB
    )
    
    smemA = gl.allocate_shared_memory(a_ptr.dtype.element_ty, [2, BLOCK_M, BLOCK_K], layout=sharedLayoutA)
    smemB = gl.allocate_shared_memory(b_ptr.dtype.element_ty, [2, BLOCK_K, BLOCK_N], layout=sharedLayoutB)
    
    wmmaLayout: gl.constexpr = gl.amd.AMDWMMALayout(
        version=3, transposed=True, warp_bases=[[0, 1], [1, 0]], instr_shape=[16, 16, 32]
    )
    
    dotOpLayoutA: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=wmmaLayout, k_width=8)
    dotOpLayoutB: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=wmmaLayout, k_width=8)

    acc = gl.zeros((BLOCK_M, BLOCK_N), gl.float32, wmmaLayout)
    iterMax = gl.cdiv(K, BLOCK_K)

    # load A, B to lds buffer 0
    gl.amd.gfx1250.tdm.async_load(a_desc, [0, 0], smemA.index(0))
    gl.amd.gfx1250.tdm.async_load(b_desc, [0, 0], smemB.index(0))

    # load A, B to lds buffer 0
    gl.amd.gfx1250.tdm.async_load(a_desc, [0, BLOCK_K], smemA.index(1))
    gl.amd.gfx1250.tdm.async_load(b_desc, [BLOCK_K, 0], smemB.index(1))

    # wait async copy to lds buffer 0 to finish
    gl.amd.gfx1250.tdm.async_wait(2)

    # local_load a0, b0 -> buffer 0
    a = smemA.index(0).load(layout=dotOpLayoutA)
    b = smemB.index(0).load(layout=dotOpLayoutB)

    for k in range(0, iterMax - 1):
        l_idx = (k + 1) % 2
        g_idx = k % 2
        
        acc = gl.amd.gfx1250.wmma(a, b, acc)
        
        # wait for all outstanding TDM loads to complete
        gl.amd.gfx1250.tdm.async_wait(0)
        
        pred = k - iterMax + 2
        pred = (pred >> 31) & 1
        gl.amd.gfx1250.tdm.async_load(a_desc, [0, (k + 2) * BLOCK_K], smemA.index(g_idx), pred=pred)
        gl.amd.gfx1250.tdm.async_load(b_desc, [(k + 2) * BLOCK_K, 0], smemB.index(g_idx), pred=pred)

        # local load next tile
        a_next = smemA.index(l_idx).load(layout=dotOpLayoutA)
        b_next = smemB.index(l_idx).load(layout=dotOpLayoutB)
        
        # move to next tile
        a = a_next
        b = b_next
        
    # epilogue
    acc = gl.amd.gfx1250.wmma(a, b, acc)
    
    # store results back
    gStoreLayoutC: gl.constexpr = wmmaLayout
    c = gl.convert_layout(acc, layout=gStoreLayoutC)
    offs_cm = pid_m * BLOCK_M + gl.arange(0, BLOCK_M, gl.SliceLayout(1, gStoreLayoutC))
    offs_cn = pid_n * BLOCK_N + gl.arange(0, BLOCK_N, gl.SliceLayout(0, gStoreLayoutC))
    offs_c = stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    gl.amd.gfx1250.buffer_store(c, c_ptr, offs_c, mask=c_mask)


def matmul_tdm_local_prefetch(a, b):
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    
    M, K = a.shape
    _, N = b.shape
    
    b = b.contiguous()
    BLOCK_M, BLOCK_N, BLOCK_K = 256, 256, 128
    num_warps = 4
    c = torch.zeros((M, N), dtype=torch.float32, device='cuda')
    stride_am, stride_ak = a.stride(0), a.stride(1)
    stride_bk, stride_bn = b.stride(0), b.stride(1)
    stride_cm, stride_cn = c.stride(0), c.stride(1)

    GRID_MN = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    grid = (GRID_MN, 1)
    kernel_tdm_local_prefetch[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    return c

# 3. ------------------------ TDM with partition shared layout local prefetch ---------------------------
@gluon.jit
def kernel_tdm_local_prefetch_partition_layout(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,    
):
    """
    GEMM kernel with local prefetch pipeline using TDM for gfx1250.

    Computes C = A @ B where:
    - A is (M, K) row-major (K-contiguous)
    - B is (K, N) row-major after b.contiguous()
    - C is (M, N) in float32
    """

    pid = gl.program_id(axis=0)
    num_pid_n = gl.cdiv(N, BLOCK_N)

    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    
    # shared layout for A input
    NUM_PARTITIONS: gl.constexpr = 2
    NUM_GROUPS: gl.constexpr = 2
    PARTITION_DIM_A: gl.constexpr = 0
    inner_shape_m: gl.constexpr = BLOCK_M // (NUM_PARTITIONS * NUM_GROUPS)
    inner_shape_k: gl.constexpr = BLOCK_K
    inner_layoutA: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[32, 8]], [inner_shape_m, inner_shape_k],
                                                                             [1, 0])
    sharedLayoutA: gl.constexpr = PartitionedSharedLayout(NUM_PARTITIONS, NUM_GROUPS, PARTITION_DIM_A, inner_layoutA)

    # shared layout for B input
    # NUM_PARTITIONS: gl.constexpr = 2
    # NUM_GROUPS: gl.constexpr = 2
    PARTITION_DIM_B: gl.constexpr = 1
    inner_shape_n: gl.constexpr = BLOCK_N // (NUM_PARTITIONS * NUM_GROUPS)
    # inner_shape_k: gl.constexpr = BLOCK_K
    inner_layoutB: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[64, 16]], [inner_shape_k, inner_shape_n],
                                                                             [1, 0])
    sharedLayoutB: gl.constexpr = PartitionedSharedLayout(NUM_PARTITIONS, NUM_GROUPS, PARTITION_DIM_B, inner_layoutB)

    a_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base = a_ptr + pid_m * BLOCK_M * stride_am,
        shape = (M, K),
        strides = (stride_am, stride_ak),
        block_shape = (BLOCK_M, BLOCK_K),
        layout = sharedLayoutA
    )
    
    b_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base = b_ptr + pid_n * BLOCK_N * stride_bn,
        shape = (K, N),
        strides = (stride_bk, stride_bn),
        block_shape = (BLOCK_K, BLOCK_N),
        layout = sharedLayoutB
    )
    
    smemA = gl.allocate_shared_memory(a_ptr.dtype.element_ty, [2, BLOCK_M, BLOCK_K], layout=sharedLayoutA)
    smemB = gl.allocate_shared_memory(b_ptr.dtype.element_ty, [2, BLOCK_K, BLOCK_N], layout=sharedLayoutB)
    
    wmmaLayout: gl.constexpr = gl.amd.AMDWMMALayout(
        version=3, transposed=True, warp_bases=[[0, 1], [1, 0]], instr_shape=[16, 16, 32]
    )
    
    dotOpLayoutA: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=wmmaLayout, k_width=8)
    dotOpLayoutB: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=wmmaLayout, k_width=8)

    acc = gl.zeros((BLOCK_M, BLOCK_N), gl.float32, wmmaLayout)
    iterMax = gl.cdiv(K, BLOCK_K)

    # load A, B to lds buffer 0
    gl.amd.gfx1250.tdm.async_load(a_desc, [0, 0], smemA.index(0))
    gl.amd.gfx1250.tdm.async_load(b_desc, [0, 0], smemB.index(0))

    # load A, B to lds buffer 0
    gl.amd.gfx1250.tdm.async_load(a_desc, [0, BLOCK_K], smemA.index(1))
    gl.amd.gfx1250.tdm.async_load(b_desc, [BLOCK_K, 0], smemB.index(1))

    # wait async copy to lds buffer 0 to finish
    gl.amd.gfx1250.tdm.async_wait(2)

    # local_load a0, b0 -> buffer 0
    a = smemA.index(0).load(layout=dotOpLayoutA)
    b = smemB.index(0).load(layout=dotOpLayoutB)

    for k in range(0, iterMax - 1):
        l_idx = (k + 1) % 2
        g_idx = k % 2
        
        acc = gl.amd.gfx1250.wmma(a, b, acc)
        
        # wait for all outstanding TDM loads to complete
        gl.amd.gfx1250.tdm.async_wait(0)
        
        pred = k - iterMax + 2
        pred = (pred >> 31) & 1
        gl.amd.gfx1250.tdm.async_load(a_desc, [0, (k + 2) * BLOCK_K], smemA.index(g_idx), pred=pred)
        gl.amd.gfx1250.tdm.async_load(b_desc, [(k + 2) * BLOCK_K, 0], smemB.index(g_idx), pred=pred)

        # local load next tile
        a_next = smemA.index(l_idx).load(layout=dotOpLayoutA)
        b_next = smemB.index(l_idx).load(layout=dotOpLayoutB)
        
        # move to next tile
        a = a_next
        b = b_next
        
    # epilogue
    acc = gl.amd.gfx1250.wmma(a, b, acc)
    
    # store results back
    gStoreLayoutC: gl.constexpr = wmmaLayout
    c = gl.convert_layout(acc, layout=gStoreLayoutC)
    offs_cm = pid_m * BLOCK_M + gl.arange(0, BLOCK_M, gl.SliceLayout(1, gStoreLayoutC))
    offs_cn = pid_n * BLOCK_N + gl.arange(0, BLOCK_N, gl.SliceLayout(0, gStoreLayoutC))
    offs_c = stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    gl.amd.gfx1250.buffer_store(c, c_ptr, offs_c, mask=c_mask)


def matmul_tdm_local_prefetch_partition_layout(a, b):
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    
    M, K = a.shape
    _, N = b.shape
    
    b = b.contiguous()
    BLOCK_M, BLOCK_N, BLOCK_K = 256, 256, 128
    num_warps = 4
    c = torch.zeros((M, N), dtype=torch.float32, device='cuda')
    stride_am, stride_ak = a.stride(0), a.stride(1)
    stride_bk, stride_bn = b.stride(0), b.stride(1)
    stride_cm, stride_cn = c.stride(0), c.stride(1)

    GRID_MN = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    grid = (GRID_MN, 1)
    kernel_tdm_local_prefetch_partition_layout[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    return c


def test_correctness(matmul, M, N, K, dtype, trans_b=False):
    a = torch.randn((M, K), dtype=dtype, device='cuda')
    if trans_b:
        b = torch.randn((K, N), dtype=dtype, device='cuda')
    else:
        b = torch.randn((N, K), dtype=dtype, device='cuda').T

    c_torch = a.to(torch.float32) @ b.to(torch.float32)
    c_triton = matmul(a, b)
    
    if torch.allclose(c_triton.float(), c_torch, atol=1e-1, rtol=0):
        print(f"{M=} {N=} {K=} {dtype=}: ✅ Triton and Torch match")
    else:
        print(f"{M=} {N=} {K=} {dtype=}: ❌ Triton and Torch differ")
        _d = (c_triton.float() - c_torch).abs()
        _nan_mask = torch.isnan(c_triton)
        _nan_rows = _nan_mask.any(dim=1).sum().item()
        print(f"  max_abs_diff={_d.max().item():.4g} "
                f"nan_frac={_nan_mask.float().mean().item():.4g} "
                f"rows_with_nan={_nan_rows}/{c_triton.shape[0]}")
        print("  triton[0,:8] =", c_triton[0, :8].tolist())
        print("  torch [0,:8] =", c_torch[0, :8].tolist())


test_correctness(matmul_async_copy_local_prefetch, 4096, 4096, 512, torch.bfloat16)
test_correctness(matmul_tdm_local_prefetch, 4096, 4096, 4096, torch.bfloat16)
test_correctness(matmul_tdm_local_prefetch_partition_layout, 4096, 4096, 4096, torch.bfloat16)


def main():
    configs = [
        triton.testing.Benchmark(
            x_names=["M", "N", "K"],
            x_vals=[512 * i for i in range(4, 8)],
            line_arg="kernels",
            line_vals=["async_copy", "tdm", "tdm_partition"],
            line_names=["async_copy", "tdm", "tdm_partition"],
            styles=[("green", "-"), ("yellow", "--"), ("red", "--")],
            ylabel="TFLOPS",
            plot_name=f"matmul-performance",
            args={},
        )
    ]

    @triton.testing.perf_report(configs)
    def benchmark(M, N, K, kernels):
        torch_dtype = torch.float16
        a = torch.randn((M, K), dtype=torch_dtype, device='cuda')
        b = torch.randn((N, K), dtype=torch_dtype, device='cuda').T
        quantiles = [0.5, 0.2, 0.8]

        ms, min_ms, max_max = -1, -1, -1
        if kernels == "async_copy":
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul_async_copy_local_prefetch(a, b), quantiles=quantiles)
        if kernels == "tdm":
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul_tdm_local_prefetch(a, b), quantiles=quantiles)
        if kernels == "tdm_partition":
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul_tdm_local_prefetch_partition_layout(a, b), quantiles=quantiles)

        def perf(ms):
            return 2 * M * N * K * 1e-12 / (ms * 1e-3)

        return perf(ms), perf(max_ms), perf(min_ms)

    benchmark.run(show_plots=False, print_data=True)


if __name__ == "__main__":
    main()
