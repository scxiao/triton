// RUN: triton-opt %s -split-input-file -tritonamdgpu-reorder-instructions | FileCheck %s

// Check that we order load, local_alloc and local_load one after another. This is useful
// for making sure that Q tensor in FA is hoisted out of the main loop and kept in registers
// throughout the computation.
// CHECK-LABEL: order_load_alloc_local_load
//       CHECK:   %[[LOAD:.+]] = tt.load
//       CHECK-NEXT:   %[[ALLOC:.+]] = triton_gpu.local_alloc %[[LOAD]]
//       CHECK-NEXT:   triton_gpu.local_load %[[ALLOC]]
#blocked = #triton_gpu.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 8], order = [0, 1]}>
#mma = #triton_gpu.amd_mfma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [32, 32], isTransposed = true}>
#shared = #triton_gpu.shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [0, 1], hasLeadingOffset = false}>
module attributes {"triton_gpu.num-warps" = 8 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {
  tt.func public @order_load_alloc_local_load(%arg0: tensor<32x32x!tt.ptr<f32>, #blocked>) attributes {noinline = false} {
    %9 = tt.load %arg0 : tensor<32x32x!tt.ptr<f32>, #blocked>
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %10 = triton_gpu.local_alloc %9 : (tensor<32x32xf32, #blocked>) -> !tt.memdesc<32x32xf32, #shared>
    %cst_0 = arith.constant dense<1.230000e+02> : tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %11 = triton_gpu.local_load %10 : !tt.memdesc<32x32xf32, #shared> -> tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
    %12 = tt.dot %11, %cst_0, %cst : tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<32x32xf32, #mma>
    %13 = triton_gpu.convert_layout %12 : tensor<32x32xf32, #mma> -> tensor<32x32xf32, #blocked>
    tt.store %arg0, %13 : tensor<32x32x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #triton_gpu.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = []}>
#shared = #triton_gpu.shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [1, 0], hasLeadingOffset = false}>
#shared1 = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0], hasLeadingOffset = false}>
#shared2 = #triton_gpu.shared<{vec = 8, perPhase = 4, maxPhase = 2, order = [1, 0], hasLeadingOffset = false}>
#shared3 = #triton_gpu.shared<{vec = 4, perPhase = 1, maxPhase = 8, order = [1, 0], hasLeadingOffset = false}>
#shared4 = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 4, order = [1, 0], hasLeadingOffset = false}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, triton_gpu.target = "cuda:80"} {

// CHECK-LABEL:  tt.func @matmul_loop
// CHECK:  %{{.*}}:7 = scf.for %[[ARG5:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ARG6:.*]] = %{{.*}}, %[[ARG7:.*]] = %{{.*}}, %[[ARG8:.*]] = %{{.*}}, %[[ARG9:.*]] = %{{.*}}-1_i32, %[[ARG10:.*]] = %{{.*}}, %[[ARG11:.*]] = %{{.*}}, %[[ARG12:.*]] = %{{.*}})
// CHECK:  %[[SUBI_20:.*]] = arith.subi %{{.*}}, %{{.*}}
// CHECK:  %[[CMPI_21:.*]] = arith.cmpi slt, %[[ARG5]], %[[SUBI_20]]
// CHECK:  %[[SPLAT_22:.*]] = tt.splat %[[CMPI_21]]
// CHECK:  %[[ADDPTR_23:.*]] = tt.addptr %[[ARG6]], %{{.*}}
// CHECK:  %[[LOAD_24:.*]] = tt.load %[[ADDPTR_23]], %[[SPLAT_22]]
// CHECK:  %[[SPLAT_25:.*]] = tt.splat %[[CMPI_21]]
// CHECK:  %[[ADDPTR_26:.*]] = tt.addptr %[[ARG7]], %{{.*}}
// CHECK:  %[[LOAD_27:.*]] = tt.load %[[ADDPTR_26]], %[[SPLAT_25]], %{{.*}}
// CHECK:  %[[ADDI_28:.*]] = arith.addi %[[ARG9]], %{{.*}}
// CHECK:  %[[CMPI_29:.*]] = arith.cmpi slt, %[[ADDI_28]], %{{.*}}
// CHECK:  %[[SELECT_30:.*]] = arith.select %[[CMPI_29]], %[[ADDI_28]], %{{.*}}
// CHECK:  %[[LOCAL_LOAD_31:.*]] = triton_gpu.local_load %[[ARG11]]
// CHECK:  %[[LOCAL_LOAD_32:.*]] = triton_gpu.local_load %[[ARG12]]
// CHECK:  %[[MULF_33:.*]] = arith.mulf %[[LOCAL_LOAD_32]], %{{.*}}
// CHECK:  %[[DOT_34:.*]] = tt.dot %[[LOCAL_LOAD_31]], %[[MULF_33]], %[[ARG8]]
// CHECK:  %[[ADDI_35:.*]] = arith.addi %[[ARG10]], %{{.*}}
// CHECK:  %[[CMPI_36:.*]] = arith.cmpi slt, %[[ADDI_35]], %{{.*}}
// CHECK:  %[[SELECT_37:.*]] = arith.select %[[CMPI_36]], %[[ADDI_35]], %{{.*}}
// CHECK:  %[[MEMDESC_SUBVIEW_38:.*]] = triton_gpu.memdesc_subview %{{.*}}[%[[SELECT_37]], %{{.*}}, %{{.*}}]
// CHECK:  triton_gpu.local_store %[[LOAD_24]], %[[MEMDESC_SUBVIEW_38]]
// CHECK:  %[[MEMDESC_SUBVIEW_39:.*]] = triton_gpu.memdesc_subview %{{.*}}[%[[SELECT_37]], %{{.*}}, %{{.*}}]
// CHECK:  triton_gpu.local_store %[[LOAD_27]], %[[MEMDESC_SUBVIEW_39]]
// CHECK:  scf.yield %[[ADDPTR_23]], %[[ADDPTR_26]], %[[DOT_34]], %[[SELECT_30]], %[[SELECT_37]], %[[MEMDESC_SUBVIEW_38]], %[[MEMDESC_SUBVIEW_39]]
// CHECK:  }

  tt.func @matmul_loop(%arg0: index, %arg1: index, %arg2: index, %arg3: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f16> {tt.divisibility = 16 : i32}) -> tensor<128x128xf32, #mma> {
    %c1_i32 = arith.constant 1 : i32
    %0 = arith.cmpi slt, %arg0, %arg1 : index
    %1 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %2 = tt.expand_dims %1 {axis = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
    %3 = tt.broadcast %2 : tensor<1x128xi32, #blocked> -> tensor<32x128xi32, #blocked>
    %4 = tt.splat %arg4 : !tt.ptr<f16> -> tensor<32x128x!tt.ptr<f16>, #blocked>
    %cst = arith.constant dense<0.000000e+00> : tensor<32x128xf16, #blocked>
    %5 = tt.splat %0 : i1 -> tensor<32x128xi1, #blocked>
    %6 = tt.addptr %4, %3 : tensor<32x128x!tt.ptr<f16>, #blocked>, tensor<32x128xi32, #blocked>
    %7 = tt.load %6, %5, %cst : tensor<32x128x!tt.ptr<f16>, #blocked>
    %8 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %9 = tt.expand_dims %8 {axis = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x32xi32, #blocked1>
    %10 = tt.broadcast %9 : tensor<1x32xi32, #blocked1> -> tensor<128x32xi32, #blocked1>
    %11 = tt.splat %arg3 : !tt.ptr<f16> -> tensor<128x32x!tt.ptr<f16>, #blocked1>
    %12 = tt.splat %0 : i1 -> tensor<128x32xi1, #blocked1>
    %13 = tt.addptr %11, %10 : tensor<128x32x!tt.ptr<f16>, #blocked1>, tensor<128x32xi32, #blocked1>
    %14 = tt.load %13, %12 : tensor<128x32x!tt.ptr<f16>, #blocked1>
    %c0_i32 = arith.constant 0 : i32
    %c-1_i32 = arith.constant -1 : i32
    %cst_0 = arith.constant dense<4.000000e+00> : tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %cst_1 = arith.constant dense<4> : tensor<32x128xi32, #blocked>
    %cst_2 = arith.constant dense<4> : tensor<128x32xi32, #blocked1>
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %15 = triton_gpu.local_alloc  : () -> !tt.memdesc<1x128x32xf16, #shared, #triton_gpu.shared_memory, mutable>
    %16 = triton_gpu.local_alloc  : () -> !tt.memdesc<1x32x128xf16, #shared1, #triton_gpu.shared_memory, mutable>
    %17 = triton_gpu.memdesc_subview %15[%c0_i32, %c0_i32, %c0_i32] : !tt.memdesc<1x128x32xf16, #shared, #triton_gpu.shared_memory, mutable> -> !tt.memdesc<128x32xf16, #shared, #triton_gpu.shared_memory, mutable>
    triton_gpu.local_store %14, %17 : tensor<128x32xf16, #blocked1> -> !tt.memdesc<128x32xf16, #shared, #triton_gpu.shared_memory, mutable>
    %18 = triton_gpu.memdesc_subview %16[%c0_i32, %c0_i32, %c0_i32] : !tt.memdesc<1x32x128xf16, #shared1, #triton_gpu.shared_memory, mutable> -> !tt.memdesc<32x128xf16, #shared1, #triton_gpu.shared_memory, mutable>
    triton_gpu.local_store %7, %18 : tensor<32x128xf16, #blocked> -> !tt.memdesc<32x128xf16, #shared1, #triton_gpu.shared_memory, mutable>
    %19:7 = scf.for %arg5 = %arg0 to %arg1 step %arg2 iter_args(%arg6 = %13, %arg7 = %6, %arg8 = %cst_3, %arg9 = %c-1_i32, %arg10 = %c0_i32, %arg11 = %17, %arg12 = %18) -> (tensor<128x32x!tt.ptr<f16>, #blocked1>, tensor<32x128x!tt.ptr<f16>, #blocked>, tensor<128x128xf32, #mma>, i32, i32, !tt.memdesc<128x32xf16, #shared, #triton_gpu.shared_memory, mutable>, !tt.memdesc<32x128xf16, #shared1, #triton_gpu.shared_memory, mutable>) {
      %20 = arith.subi %arg1, %arg2 : index
      %21 = arith.cmpi slt, %arg5, %20 : index
      %22 = tt.splat %21 : i1 -> tensor<32x128xi1, #blocked>
      %23 = tt.addptr %arg7, %cst_1 : tensor<32x128x!tt.ptr<f16>, #blocked>, tensor<32x128xi32, #blocked>
      %24 = tt.load %23, %22, %cst : tensor<32x128x!tt.ptr<f16>, #blocked>
      %25 = tt.splat %21 : i1 -> tensor<128x32xi1, #blocked1>
      %26 = tt.addptr %arg6, %cst_2 : tensor<128x32x!tt.ptr<f16>, #blocked1>, tensor<128x32xi32, #blocked1>
      %27 = tt.load %26, %25 : tensor<128x32x!tt.ptr<f16>, #blocked1>
      %28 = arith.addi %arg9, %c1_i32 : i32
      %29 = arith.cmpi slt, %28, %c1_i32 : i32
      %30 = arith.select %29, %28, %c0_i32 : i32
      %31 = triton_gpu.local_load %arg11 : !tt.memdesc<128x32xf16, #shared, #triton_gpu.shared_memory, mutable> -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %32 = triton_gpu.local_load %arg12 : !tt.memdesc<32x128xf16, #shared1, #triton_gpu.shared_memory, mutable> -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %33 = arith.mulf %32, %cst_0 : tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %34 = tt.dot %31, %33, %arg8 : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x128xf32, #mma>
      %35 = arith.addi %arg10, %c1_i32 : i32
      %36 = arith.cmpi slt, %35, %c1_i32 : i32
      %37 = arith.select %36, %35, %c0_i32 : i32
      %38 = triton_gpu.memdesc_subview %15[%37, %c0_i32, %c0_i32] : !tt.memdesc<1x128x32xf16, #shared, #triton_gpu.shared_memory, mutable> -> !tt.memdesc<128x32xf16, #shared, #triton_gpu.shared_memory, mutable>
      triton_gpu.local_store %27, %38 : tensor<128x32xf16, #blocked1> -> !tt.memdesc<128x32xf16, #shared, #triton_gpu.shared_memory, mutable>
      %39 = triton_gpu.memdesc_subview %16[%37, %c0_i32, %c0_i32] : !tt.memdesc<1x32x128xf16, #shared1, #triton_gpu.shared_memory, mutable> -> !tt.memdesc<32x128xf16, #shared1, #triton_gpu.shared_memory, mutable>
      triton_gpu.local_store %24, %39 : tensor<32x128xf16, #blocked> -> !tt.memdesc<32x128xf16, #shared1, #triton_gpu.shared_memory, mutable>
      scf.yield %26, %23, %34, %30, %37, %38, %39 : tensor<128x32x!tt.ptr<f16>, #blocked1>, tensor<32x128x!tt.ptr<f16>, #blocked>, tensor<128x128xf32, #mma>, i32, i32, !tt.memdesc<128x32xf16, #shared, #triton_gpu.shared_memory, mutable>, !tt.memdesc<32x128xf16, #shared1, #triton_gpu.shared_memory, mutable>
    }
    triton_gpu.local_dealloc %15 : !tt.memdesc<1x128x32xf16, #shared, #triton_gpu.shared_memory, mutable>
    triton_gpu.local_dealloc %16 : !tt.memdesc<1x32x128xf16, #shared1, #triton_gpu.shared_memory, mutable>
    tt.return %19#2 : tensor<128x128xf32, #mma>
  }

// CHECK-LABEL:  tt.func @matmul_loop_nested
// CHECK:  %[[FOR_0:.*]] = scf.for %[[ARG5:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ARG6:.*]] = %{{.*}})

// CHECK:  %[[SPLAT_1:.*]] = tt.splat %{{.*}}
// CHECK:  %[[CMPI_2:.*]] = arith.cmpi slt, %{{.*}}, %{{.*}}
// CHECK:  %[[MAKE_RANGE_3:.*]] = tt.make_range {end = 32 : i32, start = 0 : i32}
// CHECK:  %[[EXPAND_DIMS_4:.*]] = tt.expand_dims %[[MAKE_RANGE_3]] {axis = 0 : i32}
// CHECK:  %[[BROADCAST_5:.*]] = tt.broadcast %[[EXPAND_DIMS_4]]
// CHECK:  %[[SPLAT_6:.*]] = tt.splat %[[CMPI_2]]
// CHECK:  %[[ADDPTR_7:.*]] = tt.addptr %[[SPLAT_1]], %[[BROADCAST_5]]
// CHECK:  %[[LOAD_8:.*]] = tt.load %[[ADDPTR_7]], %[[SPLAT_6]], %{{.*}}
// CHECK:  %[[MAKE_RANGE_9:.*]] = tt.make_range {end = 128 : i32, start = 0 : i32}
// CHECK:  %[[EXPAND_DIMS_10:.*]] = tt.expand_dims %[[MAKE_RANGE_9]] {axis = 0 : i32}
// CHECK:  %[[BROADCAST_11:.*]] = tt.broadcast %[[EXPAND_DIMS_10]]
// CHECK:  %[[SPLAT_12:.*]] = tt.splat %{{.*}}
// CHECK:  %[[SPLAT_13:.*]] = tt.splat %[[CMPI_2]]
// CHECK:  %[[ADDPTR_14:.*]] = tt.addptr %[[SPLAT_12]], %[[BROADCAST_11]]
// CHECK:  %[[LOAD_15:.*]] = tt.load %[[ADDPTR_14]], %[[SPLAT_13]], %{{.*}}
// CHECK:  %[[LOCAL_ALLOC_16:.*]] = triton_gpu.local_alloc
// CHECK:  %[[LOCAL_ALLOC_17:.*]] = triton_gpu.local_alloc
// CHECK:  %[[MEMDESC_SUBVIEW_18:.*]] = triton_gpu.memdesc_subview %[[LOCAL_ALLOC_16]][%{{.*}}, %{{.*}}, %{{.*}}]
// CHECK:  triton_gpu.local_store %[[LOAD_8]], %[[MEMDESC_SUBVIEW_18]]
// CHECK:  %[[MEMDESC_SUBVIEW_19:.*]] = triton_gpu.memdesc_subview %[[LOCAL_ALLOC_17]][%{{.*}}, %{{.*}}, %{{.*}}]
// CHECK:  triton_gpu.local_store %[[LOAD_15]], %[[MEMDESC_SUBVIEW_19]]
// CHECK:  %{{.*}}:7 = scf.for %[[ARG7:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ARG8:.*]] = %[[ADDPTR_7]], %[[ARG9:.*]] = %[[ADDPTR_14]], %[[ARG10:.*]] = %[[ARG6]], %[[ARG11:.*]] = %{{.*}}-1_i32, %[[ARG12:.*]] = %{{.*}}, %[[ARG13:.*]] = %[[MEMDESC_SUBVIEW_18]], %[[ARG14:.*]] = %[[MEMDESC_SUBVIEW_19]])

// CHECK:  %[[SUBI_21:.*]] = arith.subi %{{.*}}, %{{.*}}
// CHECK:  %[[CMPI_22:.*]] = arith.cmpi slt, %[[ARG7]], %[[SUBI_21]]
// CHECK:  %[[SPLAT_23:.*]] = tt.splat %[[CMPI_22]]
// CHECK:  %[[ADDPTR_24:.*]] = tt.addptr %[[ARG8]], %{{.*}}
// CHECK:  %[[LOAD_25:.*]] = tt.load %[[ADDPTR_24]], %[[SPLAT_23]], %{{.*}}
// CHECK:  %[[SPLAT_26:.*]] = tt.splat %[[CMPI_22]]
// CHECK:  %[[ADDPTR_27:.*]] = tt.addptr %[[ARG9]], %{{.*}}
// CHECK:  %[[LOAD_28:.*]] = tt.load %[[ADDPTR_27]], %[[SPLAT_26]], %{{.*}}
// CHECK:  %[[ADDI_29:.*]] = arith.addi %[[ARG11]], %{{.*}}
// CHECK:  %[[CMPI_30:.*]] = arith.cmpi slt, %[[ADDI_29]], %{{.*}}
// CHECK:  %[[SELECT_31:.*]] = arith.select %[[CMPI_30]], %[[ADDI_29]], %{{.*}}
// CHECK:  %[[LOCAL_LOAD_32:.*]] = triton_gpu.local_load %[[ARG13]]
// CHECK:  %[[LOCAL_LOAD_33:.*]] = triton_gpu.local_load %[[ARG14]]
// CHECK:  %[[DOT_34:.*]] = tt.dot %[[LOCAL_LOAD_32]], %[[LOCAL_LOAD_33]], %[[ARG10]]
// CHECK:  %[[ADDI_35:.*]] = arith.addi %[[ARG12]], %{{.*}}
// CHECK:  %[[CMPI_36:.*]] = arith.cmpi slt, %[[ADDI_35]], %{{.*}}
// CHECK:  %[[SELECT_37:.*]] = arith.select %[[CMPI_36]], %[[ADDI_35]], %{{.*}}
// CHECK:  %[[MEMDESC_SUBVIEW_38:.*]] = triton_gpu.memdesc_subview %[[LOCAL_ALLOC_16]][%[[SELECT_37]], %{{.*}}, %{{.*}}]
// CHECK:  triton_gpu.local_store %[[LOAD_25]], %[[MEMDESC_SUBVIEW_38]]
// CHECK:  %[[MEMDESC_SUBVIEW_39:.*]] = triton_gpu.memdesc_subview %[[LOCAL_ALLOC_17]][%[[SELECT_37]], %{{.*}}, %{{.*}}]
// CHECK:  triton_gpu.local_store %[[LOAD_28]], %[[MEMDESC_SUBVIEW_39]]
// CHECK:  scf.yield %[[ADDPTR_24]], %[[ADDPTR_27]], %[[DOT_34]], %[[SELECT_31]], %[[SELECT_37]], %[[MEMDESC_SUBVIEW_38]], %[[MEMDESC_SUBVIEW_39]]
// CHECK:  }
// CHECK:  triton_gpu.local_dealloc %[[LOCAL_ALLOC_16]]
// CHECK:  triton_gpu.local_dealloc %[[LOCAL_ALLOC_17]]
// CHECK:  scf.yield %{{.*}}#2
// CHECK:  }

  tt.func @matmul_loop_nested(%arg0: index, %arg1: index, %arg2: index, %arg3: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f16> {tt.divisibility = 16 : i32}) -> tensor<128x128xf32, #mma> {
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c-1_i32 = arith.constant -1 : i32
    %cst = arith.constant dense<4> : tensor<32x128xi32, #blocked>
    %cst_0 = arith.constant dense<4> : tensor<128x32xi32, #blocked1>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<32x128xf16, #blocked>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<128x32xf16, #blocked1>
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %0 = scf.for %arg5 = %arg0 to %arg1 step %arg2 iter_args(%arg6 = %cst_3) -> (tensor<128x128xf32, #mma>) {
      %1 = tt.splat %arg3 : !tt.ptr<f16> -> tensor<128x32x!tt.ptr<f16>, #blocked1>
      %2 = arith.cmpi slt, %arg0, %arg1 : index
      %3 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
      %4 = tt.expand_dims %3 {axis = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
      %5 = tt.broadcast %4 : tensor<1x128xi32, #blocked> -> tensor<32x128xi32, #blocked>
      %6 = tt.splat %arg4 : !tt.ptr<f16> -> tensor<32x128x!tt.ptr<f16>, #blocked>
      %7 = tt.splat %2 : i1 -> tensor<32x128xi1, #blocked>
      %8 = tt.addptr %6, %5 : tensor<32x128x!tt.ptr<f16>, #blocked>, tensor<32x128xi32, #blocked>
      %9 = tt.load %8, %7, %cst_1 : tensor<32x128x!tt.ptr<f16>, #blocked>
      %10 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
      %11 = tt.expand_dims %10 {axis = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x32xi32, #blocked1>
      %12 = tt.broadcast %11 : tensor<1x32xi32, #blocked1> -> tensor<128x32xi32, #blocked1>
      %13 = tt.splat %2 : i1 -> tensor<128x32xi1, #blocked1>
      %14 = tt.addptr %1, %12 : tensor<128x32x!tt.ptr<f16>, #blocked1>, tensor<128x32xi32, #blocked1>
      %15 = tt.load %14, %13, %cst_2 : tensor<128x32x!tt.ptr<f16>, #blocked1>
      %16 = triton_gpu.local_alloc  : () -> !tt.memdesc<1x128x32xf16, #shared, #triton_gpu.shared_memory, mutable>
      %17 = triton_gpu.local_alloc  : () -> !tt.memdesc<1x32x128xf16, #shared1, #triton_gpu.shared_memory, mutable>
      %18 = triton_gpu.memdesc_subview %16[%c0_i32, %c0_i32, %c0_i32] : !tt.memdesc<1x128x32xf16, #shared, #triton_gpu.shared_memory, mutable> -> !tt.memdesc<128x32xf16, #shared, #triton_gpu.shared_memory, mutable>
      triton_gpu.local_store %15, %18 : tensor<128x32xf16, #blocked1> -> !tt.memdesc<128x32xf16, #shared, #triton_gpu.shared_memory, mutable>
      %19 = triton_gpu.memdesc_subview %17[%c0_i32, %c0_i32, %c0_i32] : !tt.memdesc<1x32x128xf16, #shared1, #triton_gpu.shared_memory, mutable> -> !tt.memdesc<32x128xf16, #shared1, #triton_gpu.shared_memory, mutable>
      triton_gpu.local_store %9, %19 : tensor<32x128xf16, #blocked> -> !tt.memdesc<32x128xf16, #shared1, #triton_gpu.shared_memory, mutable>
      %20:7 = scf.for %arg7 = %arg0 to %arg1 step %arg2 iter_args(%arg8 = %14, %arg9 = %8, %arg10 = %arg6, %arg11 = %c-1_i32, %arg12 = %c0_i32, %arg13 = %18, %arg14 = %19) -> (tensor<128x32x!tt.ptr<f16>, #blocked1>, tensor<32x128x!tt.ptr<f16>, #blocked>, tensor<128x128xf32, #mma>, i32, i32, !tt.memdesc<128x32xf16, #shared, #triton_gpu.shared_memory, mutable>, !tt.memdesc<32x128xf16, #shared1, #triton_gpu.shared_memory, mutable>) {
        %21 = arith.subi %arg1, %arg2 : index
        %22 = arith.cmpi slt, %arg7, %21 : index
        %23 = tt.splat %22 : i1 -> tensor<32x128xi1, #blocked>
        %24 = tt.addptr %arg9, %cst : tensor<32x128x!tt.ptr<f16>, #blocked>, tensor<32x128xi32, #blocked>
        %25 = tt.load %24, %23, %cst_1 : tensor<32x128x!tt.ptr<f16>, #blocked>
        %26 = tt.splat %22 : i1 -> tensor<128x32xi1, #blocked1>
        %27 = tt.addptr %arg8, %cst_0 : tensor<128x32x!tt.ptr<f16>, #blocked1>, tensor<128x32xi32, #blocked1>
        %28 = tt.load %27, %26, %cst_2 : tensor<128x32x!tt.ptr<f16>, #blocked1>
        %29 = arith.addi %arg11, %c1_i32 : i32
        %30 = arith.cmpi slt, %29, %c1_i32 : i32
        %31 = arith.select %30, %29, %c0_i32 : i32
        %32 = triton_gpu.local_load %arg13 : !tt.memdesc<128x32xf16, #shared, #triton_gpu.shared_memory, mutable> -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
        %33 = triton_gpu.local_load %arg14 : !tt.memdesc<32x128xf16, #shared1, #triton_gpu.shared_memory, mutable> -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
        %34 = tt.dot %32, %33, %arg10 : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x128xf32, #mma>
        %35 = arith.addi %arg12, %c1_i32 : i32
        %36 = arith.cmpi slt, %35, %c1_i32 : i32
        %37 = arith.select %36, %35, %c0_i32 : i32
        %38 = triton_gpu.memdesc_subview %16[%37, %c0_i32, %c0_i32] : !tt.memdesc<1x128x32xf16, #shared, #triton_gpu.shared_memory, mutable> -> !tt.memdesc<128x32xf16, #shared, #triton_gpu.shared_memory, mutable>
        triton_gpu.local_store %28, %38 : tensor<128x32xf16, #blocked1> -> !tt.memdesc<128x32xf16, #shared, #triton_gpu.shared_memory, mutable>
        %39 = triton_gpu.memdesc_subview %17[%37, %c0_i32, %c0_i32] : !tt.memdesc<1x32x128xf16, #shared1, #triton_gpu.shared_memory, mutable> -> !tt.memdesc<32x128xf16, #shared1, #triton_gpu.shared_memory, mutable>
        triton_gpu.local_store %25, %39 : tensor<32x128xf16, #blocked> -> !tt.memdesc<32x128xf16, #shared1, #triton_gpu.shared_memory, mutable>
        scf.yield %27, %24, %34, %31, %37, %38, %39 : tensor<128x32x!tt.ptr<f16>, #blocked1>, tensor<32x128x!tt.ptr<f16>, #blocked>, tensor<128x128xf32, #mma>, i32, i32, !tt.memdesc<128x32xf16, #shared, #triton_gpu.shared_memory, mutable>, !tt.memdesc<32x128xf16, #shared1, #triton_gpu.shared_memory, mutable>
      }
      triton_gpu.local_dealloc %16 : !tt.memdesc<1x128x32xf16, #shared, #triton_gpu.shared_memory, mutable>
      triton_gpu.local_dealloc %17 : !tt.memdesc<1x32x128xf16, #shared1, #triton_gpu.shared_memory, mutable>
      scf.yield %20#2 : tensor<128x128xf32, #mma>
    }
    tt.return %0 : tensor<128x128xf32, #mma>
  }

// CHECK-LABEL:  tt.func @matmul_loop_single_pipeline
// CHECK:  %{{.*}}:5 = scf.for %[[ARG5:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ARG6:.*]] = %{{.*}}, %[[ARG7:.*]] = %{{.*}}, %[[ARG8:.*]] = %{{.*}}-1_i32, %[[ARG9:.*]] = %{{.*}}, %[[ARG10:.*]] = %{{.*}})

// CHECK:  %[[SUBI_17:.*]] = arith.subi %{{.*}}, %{{.*}}
// CHECK:  %[[CMPI_18:.*]] = arith.cmpi slt, %[[ARG5]], %[[SUBI_17]]
// CHECK:  %[[SPLAT_19:.*]] = tt.splat %[[CMPI_18]]
// CHECK:  %[[ADDPTR_20:.*]] = tt.addptr %[[ARG6]], %{{.*}}
// CHECK:  %[[LOAD_21:.*]] = tt.load %[[ADDPTR_20]], %[[SPLAT_19]], %{{.*}}
// CHECK:  %[[ADDI_22:.*]] = arith.addi %[[ARG8]], %{{.*}}
// CHECK:  %[[CMPI_23:.*]] = arith.cmpi slt, %[[ADDI_22]], %{{.*}}
// CHECK:  %[[SELECT_24:.*]] = arith.select %[[CMPI_23]], %[[ADDI_22]], %{{.*}}
// CHECK:  %[[LOCAL_LOAD_25:.*]] = triton_gpu.local_load %[[ARG10]]
// CHECK:  %[[CONVERT_LAYOUT_26:.*]] = triton_gpu.convert_layout %{{.*}}
// CHECK:  %[[DOT_27:.*]] = tt.dot %[[CONVERT_LAYOUT_26]], %[[LOCAL_LOAD_25]], %[[ARG7]]
// CHECK:  %[[ADDI_28:.*]] = arith.addi %[[ARG9]], %{{.*}}
// CHECK:  %[[CMPI_29:.*]] = arith.cmpi slt, %[[ADDI_28]], %{{.*}}
// CHECK:  %[[SELECT_30:.*]] = arith.select %[[CMPI_29]], %[[ADDI_28]], %{{.*}}
// CHECK:  %[[MEMDESC_SUBVIEW_31:.*]] = triton_gpu.memdesc_subview %{{.*}}[%[[SELECT_30]], %{{.*}}, %{{.*}}]
// CHECK:  triton_gpu.local_store %[[LOAD_21]], %[[MEMDESC_SUBVIEW_31]]
// CHECK:  scf.yield %[[ADDPTR_20]], %[[DOT_27]], %[[SELECT_24]], %[[SELECT_30]], %[[MEMDESC_SUBVIEW_31]]
// CHECK:  }

  tt.func @matmul_loop_single_pipeline(%arg0: index, %arg1: index, %arg2: index, %arg3: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f16> {tt.divisibility = 16 : i32}) -> tensor<128x128xf32, #mma> {
    %c1_i32 = arith.constant 1 : i32
    %0 = arith.cmpi slt, %arg0, %arg1 : index
    %1 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %2 = tt.expand_dims %1 {axis = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
    %3 = tt.broadcast %2 : tensor<1x128xi32, #blocked> -> tensor<32x128xi32, #blocked>
    %4 = tt.splat %arg4 : !tt.ptr<f16> -> tensor<32x128x!tt.ptr<f16>, #blocked>
    %cst = arith.constant dense<0.000000e+00> : tensor<32x128xf16, #blocked>
    %5 = tt.splat %0 : i1 -> tensor<32x128xi1, #blocked>
    %6 = tt.addptr %4, %3 : tensor<32x128x!tt.ptr<f16>, #blocked>, tensor<32x128xi32, #blocked>
    %7 = tt.load %6, %5, %cst : tensor<32x128x!tt.ptr<f16>, #blocked>
    %8 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %9 = tt.expand_dims %8 {axis = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x32xi32, #blocked1>
    %10 = tt.broadcast %9 : tensor<1x32xi32, #blocked1> -> tensor<128x32xi32, #blocked1>
    %11 = tt.splat %arg3 : !tt.ptr<f16> -> tensor<128x32x!tt.ptr<f16>, #blocked1>
    %12 = tt.addptr %11, %10 : tensor<128x32x!tt.ptr<f16>, #blocked1>, tensor<128x32xi32, #blocked1>
    %13 = tt.load %12 : tensor<128x32x!tt.ptr<f16>, #blocked1>
    %c0_i32 = arith.constant 0 : i32
    %c-1_i32 = arith.constant -1 : i32
    %cst_0 = arith.constant dense<4> : tensor<32x128xi32, #blocked>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %14 = triton_gpu.local_alloc  : () -> !tt.memdesc<1x32x128xf16, #shared1, #triton_gpu.shared_memory, mutable>
    %15 = triton_gpu.memdesc_subview %14[%c0_i32, %c0_i32, %c0_i32] : !tt.memdesc<1x32x128xf16, #shared1, #triton_gpu.shared_memory, mutable> -> !tt.memdesc<32x128xf16, #shared1, #triton_gpu.shared_memory, mutable>
    triton_gpu.local_store %7, %15 : tensor<32x128xf16, #blocked> -> !tt.memdesc<32x128xf16, #shared1, #triton_gpu.shared_memory, mutable>
    %16:5 = scf.for %arg5 = %arg0 to %arg1 step %arg2 iter_args(%arg6 = %6, %arg7 = %cst_1, %arg8 = %c-1_i32, %arg9 = %c0_i32, %arg10 = %15) -> (tensor<32x128x!tt.ptr<f16>, #blocked>, tensor<128x128xf32, #mma>, i32, i32, !tt.memdesc<32x128xf16, #shared1, #triton_gpu.shared_memory, mutable>) {
      %17 = arith.subi %arg1, %arg2 : index
      %18 = arith.cmpi slt, %arg5, %17 : index
      %19 = tt.splat %18 : i1 -> tensor<32x128xi1, #blocked>
      %20 = tt.addptr %arg6, %cst_0 : tensor<32x128x!tt.ptr<f16>, #blocked>, tensor<32x128xi32, #blocked>
      %21 = tt.load %20, %19, %cst : tensor<32x128x!tt.ptr<f16>, #blocked>
      %22 = arith.addi %arg8, %c1_i32 : i32
      %23 = arith.cmpi slt, %22, %c1_i32 : i32
      %24 = arith.select %23, %22, %c0_i32 : i32
      %25 = triton_gpu.local_load %arg10 : !tt.memdesc<32x128xf16, #shared1, #triton_gpu.shared_memory, mutable> -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %26 = triton_gpu.convert_layout %13 : tensor<128x32xf16, #blocked1> -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %27 = tt.dot %26, %25, %arg7 : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x128xf32, #mma>
      %28 = arith.addi %arg9, %c1_i32 : i32
      %29 = arith.cmpi slt, %28, %c1_i32 : i32
      %30 = arith.select %29, %28, %c0_i32 : i32
      %31 = triton_gpu.memdesc_subview %14[%30, %c0_i32, %c0_i32] : !tt.memdesc<1x32x128xf16, #shared1, #triton_gpu.shared_memory, mutable> -> !tt.memdesc<32x128xf16, #shared1, #triton_gpu.shared_memory, mutable>
      triton_gpu.local_store %21, %31 : tensor<32x128xf16, #blocked> -> !tt.memdesc<32x128xf16, #shared1, #triton_gpu.shared_memory, mutable>
      scf.yield %20, %27, %24, %30, %31 : tensor<32x128x!tt.ptr<f16>, #blocked>, tensor<128x128xf32, #mma>, i32, i32, !tt.memdesc<32x128xf16, #shared1, #triton_gpu.shared_memory, mutable>
    }
    triton_gpu.local_dealloc %14 : !tt.memdesc<1x32x128xf16, #shared1, #triton_gpu.shared_memory, mutable>
    tt.return %16#1 : tensor<128x128xf32, #mma>
  }

// CHECK-LABEL:  tt.func @indirect_bmm_scalar
// CHECK:  %{{.*}}:9 = scf.for %[[ARG6:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ARG7:.*]] = %{{.*}}, %[[ARG8:.*]] = %{{.*}}, %[[ARG9:.*]] = %{{.*}}, %[[ARG10:.*]] = %{{.*}}-1_i32, %[[ARG11:.*]] = %{{.*}}, %[[ARG12:.*]] = %{{.*}}, %[[ARG13:.*]] = %{{.*}}, %[[ARG14:.*]] = %{{.*}}, %[[ARG15:.*]] = %{{.*}})

// CHECK:  %[[SUBI_25:.*]] = arith.subi %{{.*}}, %{{.*}}
// CHECK:  %[[CMPI_26:.*]] = arith.cmpi slt, %[[ARG6]], %[[SUBI_25]]
// CHECK:  %[[SPLAT_27:.*]] = tt.splat %[[CMPI_26]]
// CHECK:  %[[ADDPTR_28:.*]] = tt.addptr %[[ARG8]], %{{.*}}
// CHECK:  %[[LOAD_29:.*]] = tt.load %[[ADDPTR_28]], %[[SPLAT_27]]
// CHECK:  %[[ADDPTR_30:.*]] = tt.addptr %[[ARG9]], %{{.*}}
// CHECK:  %[[LOAD_31:.*]] = tt.load %[[ADDPTR_30]], %[[CMPI_26]]
// CHECK:  %[[MULI_32:.*]] = arith.muli %{{.*}}, %[[LOAD_31]]
// CHECK:  %[[SPLAT_33:.*]] = tt.splat %[[MULI_32]]
// CHECK:  %[[SPLAT_34:.*]] = tt.splat %[[CMPI_26]]
// CHECK:  %[[ADDPTR_35:.*]] = tt.addptr %{{.*}}, %[[SPLAT_33]]
// CHECK:  %[[LOAD_36:.*]] = tt.load %[[ADDPTR_35]], %[[SPLAT_34]]
// CHECK:  %[[ADDI_37:.*]] = arith.addi %[[ARG11]], %{{.*}}
// CHECK:  %[[CMPI_38:.*]] = arith.cmpi slt, %[[ADDI_37]], %{{.*}}
// CHECK:  %[[SELECT_39:.*]] = arith.select %[[CMPI_38]], %[[ADDI_37]], %{{.*}}
// CHECK:  %[[MEMDESC_SUBVIEW_40:.*]] = triton_gpu.memdesc_subview %{{.*}}[%[[SELECT_39]], %{{.*}}, %{{.*}}]
// CHECK:  triton_gpu.local_store %[[ARG14]], %[[MEMDESC_SUBVIEW_40]]
// CHECK:  %[[MEMDESC_SUBVIEW_41:.*]] = triton_gpu.memdesc_subview %{{.*}}[%[[SELECT_39]], %{{.*}}, %{{.*}}]
// CHECK:  triton_gpu.local_store %[[ARG15]], %[[MEMDESC_SUBVIEW_41]]
// CHECK:  %[[ADDI_42:.*]] = arith.addi %[[ARG10]], %{{.*}}
// CHECK:  %[[CMPI_43:.*]] = arith.cmpi slt, %[[ADDI_42]], %{{.*}}
// CHECK:  %[[SELECT_44:.*]] = arith.select %[[CMPI_43]], %[[ADDI_42]], %{{.*}}
// CHECK:  %[[LOCAL_LOAD_45:.*]] = triton_gpu.local_load %[[ARG12]]
// CHECK:  %[[LOCAL_LOAD_46:.*]] = triton_gpu.local_load %[[ARG13]]
// CHECK:  %[[DOT_47:.*]] = tt.dot %[[LOCAL_LOAD_45]], %[[LOCAL_LOAD_46]], %[[ARG7]]
// CHECK:  scf.yield %[[DOT_47]], %[[ADDPTR_28]], %[[ADDPTR_30]], %[[SELECT_44]], %[[SELECT_39]], %[[MEMDESC_SUBVIEW_40]], %[[MEMDESC_SUBVIEW_41]], %[[LOAD_29]], %[[LOAD_36]]
// CHECK:  }

  tt.func @indirect_bmm_scalar(%arg0: i64 {tt.divisibility = 16 : i32}, %arg1: index, %arg2: tensor<16x16x!tt.ptr<f16>, #blocked1> {tt.contiguity = 2 : i32, tt.divisibility = 16 : i32}, %arg3: !tt.ptr<i64>, %arg4: tensor<16x16xi32, #blocked1> {tt.constancy = 16 : i32, tt.divisibility = 16 : i32}, %arg5: tensor<16x16x!tt.ptr<f16>, #blocked> {tt.contiguity = 16 : i32, tt.divisibility = 16 : i32}) -> tensor<16x16xf32, #mma> {
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %0 = arith.cmpi sgt, %arg1, %c1 : index
    %c1_i32 = arith.constant 1 : i32
    %1 = tt.addptr %arg3, %c1_i32 : !tt.ptr<i64>, i32
    %2 = tt.load %1, %0 : !tt.ptr<i64>
    %3 = arith.muli %arg0, %2 : i64
    %4 = tt.splat %3 : i64 -> tensor<16x16xi64, #blocked>
    %5 = tt.splat %0 : i1 -> tensor<16x16xi1, #blocked>
    %6 = tt.addptr %arg5, %4 : tensor<16x16x!tt.ptr<f16>, #blocked>, tensor<16x16xi64, #blocked>
    %7 = tt.load %6, %5 : tensor<16x16x!tt.ptr<f16>, #blocked>
    %8 = tt.splat %0 : i1 -> tensor<16x16xi1, #blocked1>
    %9 = tt.addptr %arg2, %arg4 : tensor<16x16x!tt.ptr<f16>, #blocked1>, tensor<16x16xi32, #blocked1>
    %10 = tt.load %9, %8 : tensor<16x16x!tt.ptr<f16>, #blocked1>
    %c0 = arith.constant 0 : index
    %11 = arith.cmpi sgt, %arg1, %c0 : index
    %12 = tt.load %arg3, %11 : !tt.ptr<i64>
    %13 = arith.muli %arg0, %12 : i64
    %14 = tt.splat %13 : i64 -> tensor<16x16xi64, #blocked>
    %15 = tt.splat %11 : i1 -> tensor<16x16xi1, #blocked>
    %16 = tt.addptr %arg5, %14 : tensor<16x16x!tt.ptr<f16>, #blocked>, tensor<16x16xi64, #blocked>
    %17 = tt.load %16, %15 : tensor<16x16x!tt.ptr<f16>, #blocked>
    %18 = tt.splat %11 : i1 -> tensor<16x16xi1, #blocked1>
    %19 = tt.load %arg2, %18 : tensor<16x16x!tt.ptr<f16>, #blocked1>
    %c2_i32 = arith.constant 2 : i32
    %c0_i32 = arith.constant 0 : i32
    %c-1_i32 = arith.constant -1 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #mma>
    %20 = triton_gpu.local_alloc  : () -> !tt.memdesc<2x16x16xf16, #shared2, #triton_gpu.shared_memory, mutable>
    %21 = triton_gpu.local_alloc  : () -> !tt.memdesc<2x16x16xf16, #shared2, #triton_gpu.shared_memory, mutable>
    %22 = triton_gpu.memdesc_subview %20[%c0_i32, %c0_i32, %c0_i32] : !tt.memdesc<2x16x16xf16, #shared2, #triton_gpu.shared_memory, mutable> -> !tt.memdesc<16x16xf16, #shared2, #triton_gpu.shared_memory, mutable>
    triton_gpu.local_store %19, %22 : tensor<16x16xf16, #blocked1> -> !tt.memdesc<16x16xf16, #shared2, #triton_gpu.shared_memory, mutable>
    %23 = triton_gpu.memdesc_subview %21[%c0_i32, %c0_i32, %c0_i32] : !tt.memdesc<2x16x16xf16, #shared2, #triton_gpu.shared_memory, mutable> -> !tt.memdesc<16x16xf16, #shared2, #triton_gpu.shared_memory, mutable>
    triton_gpu.local_store %17, %23 : tensor<16x16xf16, #blocked> -> !tt.memdesc<16x16xf16, #shared2, #triton_gpu.shared_memory, mutable>
    %24:9 = scf.for %arg6 = %c0 to %arg1 step %c1 iter_args(%arg7 = %cst, %arg8 = %9, %arg9 = %1, %arg10 = %c-1_i32, %arg11 = %c0_i32, %arg12 = %22, %arg13 = %23, %arg14 = %10, %arg15 = %7) -> (tensor<16x16xf32, #mma>, tensor<16x16x!tt.ptr<f16>, #blocked1>, !tt.ptr<i64>, i32, i32, !tt.memdesc<16x16xf16, #shared2, #triton_gpu.shared_memory, mutable>, !tt.memdesc<16x16xf16, #shared2, #triton_gpu.shared_memory, mutable>, tensor<16x16xf16, #blocked1>, tensor<16x16xf16, #blocked>) {
      %25 = arith.subi %arg1, %c2 : index
      %26 = arith.cmpi slt, %arg6, %25 : index
      %27 = tt.addptr %arg9, %c1_i32 : !tt.ptr<i64>, i32
      %28 = tt.load %27, %26 : !tt.ptr<i64>
      %29 = arith.muli %arg0, %28 : i64
      %30 = tt.splat %29 : i64 -> tensor<16x16xi64, #blocked>
      %31 = tt.splat %26 : i1 -> tensor<16x16xi1, #blocked>
      %32 = tt.addptr %arg5, %30 : tensor<16x16x!tt.ptr<f16>, #blocked>, tensor<16x16xi64, #blocked>
      %33 = tt.load %32, %31 : tensor<16x16x!tt.ptr<f16>, #blocked>
      %34 = tt.splat %26 : i1 -> tensor<16x16xi1, #blocked1>
      %35 = tt.addptr %arg8, %arg4 : tensor<16x16x!tt.ptr<f16>, #blocked1>, tensor<16x16xi32, #blocked1>
      %36 = tt.load %35, %34 : tensor<16x16x!tt.ptr<f16>, #blocked1>
      %37 = arith.addi %arg11, %c1_i32 : i32
      %38 = arith.cmpi slt, %37, %c2_i32 : i32
      %39 = arith.select %38, %37, %c0_i32 : i32
      %40 = triton_gpu.memdesc_subview %21[%39, %c0_i32, %c0_i32] : !tt.memdesc<2x16x16xf16, #shared2, #triton_gpu.shared_memory, mutable> -> !tt.memdesc<16x16xf16, #shared2, #triton_gpu.shared_memory, mutable>
      triton_gpu.local_store %arg15, %40 : tensor<16x16xf16, #blocked> -> !tt.memdesc<16x16xf16, #shared2, #triton_gpu.shared_memory, mutable>
      %41 = triton_gpu.memdesc_subview %20[%39, %c0_i32, %c0_i32] : !tt.memdesc<2x16x16xf16, #shared2, #triton_gpu.shared_memory, mutable> -> !tt.memdesc<16x16xf16, #shared2, #triton_gpu.shared_memory, mutable>
      triton_gpu.local_store %arg14, %41 : tensor<16x16xf16, #blocked1> -> !tt.memdesc<16x16xf16, #shared2, #triton_gpu.shared_memory, mutable>
      %42 = arith.addi %arg10, %c1_i32 : i32
      %43 = arith.cmpi slt, %42, %c2_i32 : i32
      %44 = arith.select %43, %42, %c0_i32 : i32
      %45 = triton_gpu.local_load %arg12 : !tt.memdesc<16x16xf16, #shared2, #triton_gpu.shared_memory, mutable> -> tensor<16x16xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %46 = triton_gpu.local_load %arg13 : !tt.memdesc<16x16xf16, #shared2, #triton_gpu.shared_memory, mutable> -> tensor<16x16xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %47 = tt.dot %45, %46, %arg7 : tensor<16x16xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<16x16xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<16x16xf32, #mma>
      scf.yield %47, %35, %27, %44, %39, %41, %40, %36, %33 : tensor<16x16xf32, #mma>, tensor<16x16x!tt.ptr<f16>, #blocked1>, !tt.ptr<i64>, i32, i32, !tt.memdesc<16x16xf16, #shared2, #triton_gpu.shared_memory, mutable>, !tt.memdesc<16x16xf16, #shared2, #triton_gpu.shared_memory, mutable>, tensor<16x16xf16, #blocked1>, tensor<16x16xf16, #blocked>
    }
    triton_gpu.local_dealloc %20 : !tt.memdesc<2x16x16xf16, #shared2, #triton_gpu.shared_memory, mutable>
    triton_gpu.local_dealloc %21 : !tt.memdesc<2x16x16xf16, #shared2, #triton_gpu.shared_memory, mutable>
    tt.return %24#0 : tensor<16x16xf32, #mma>
  }

// CHECK-LABEL:  tt.func @indirect_bmm_scalar_dist_one
// CHECK:  %{{.*}}:8 = scf.for %[[ARG6:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ARG7:.*]] = %{{.*}}, %[[ARG8:.*]] = %{{.*}}, %[[ARG9:.*]] = %{{.*}}, %[[ARG10:.*]] = %{{.*}}, %[[ARG11:.*]] = %{{.*}}-1_i32, %[[ARG12:.*]] = %{{.*}}, %[[ARG13:.*]] = %{{.*}}, %[[ARG14:.*]] = %{{.*}})

// CHECK:  %[[SUBI_17:.*]] = arith.subi %{{.*}}, %{{.*}}
// CHECK:  %[[CMPI_18:.*]] = arith.cmpi slt, %[[ARG6]], %[[SUBI_17]]
// CHECK:  %[[SPLAT_19:.*]] = tt.splat %[[CMPI_18]]
// CHECK:  %[[ADDPTR_20:.*]] = tt.addptr %[[ARG8]], %{{.*}}
// CHECK:  %[[LOAD_21:.*]] = tt.load %[[ADDPTR_20]], %[[SPLAT_19]]
// CHECK:  %[[LOAD_22:.*]] = tt.load %[[ARG9]], %[[CMPI_18]]
// CHECK:  %[[MULI_23:.*]] = arith.muli %{{.*}}, %[[ARG10]]
// CHECK:  %[[SPLAT_24:.*]] = tt.splat %[[MULI_23]]
// CHECK:  %[[SPLAT_25:.*]] = tt.splat %[[CMPI_18]]
// CHECK:  %[[ADDPTR_26:.*]] = tt.addptr %{{.*}}, %[[SPLAT_24]]
// CHECK:  %[[LOAD_27:.*]] = tt.load %[[ADDPTR_26]], %[[SPLAT_25]]
// CHECK:  %[[ADDI_28:.*]] = arith.addi %[[ARG11]], %{{.*}}
// CHECK:  %[[CMPI_29:.*]] = arith.cmpi slt, %[[ADDI_28]], %{{.*}}
// CHECK:  %[[SELECT_30:.*]] = arith.select %[[CMPI_29]], %[[ADDI_28]], %{{.*}}
// CHECK:  %[[LOCAL_LOAD_31:.*]] = triton_gpu.local_load %[[ARG13]]
// CHECK:  %[[LOCAL_LOAD_32:.*]] = triton_gpu.local_load %[[ARG14]]
// CHECK:  %[[DOT_33:.*]] = tt.dot %[[LOCAL_LOAD_31]], %[[LOCAL_LOAD_32]], %[[ARG7]]
// CHECK:  %[[ADDPTR_34:.*]] = tt.addptr %[[ARG9]], %{{.*}}
// CHECK:  %[[ADDI_35:.*]] = arith.addi %[[ARG12]], %{{.*}}
// CHECK:  %[[CMPI_36:.*]] = arith.cmpi slt, %[[ADDI_35]], %{{.*}}
// CHECK:  %[[SELECT_37:.*]] = arith.select %[[CMPI_36]], %[[ADDI_35]], %{{.*}}
// CHECK:  %[[MEMDESC_SUBVIEW_38:.*]] = triton_gpu.memdesc_subview %{{.*}}[%[[SELECT_37]], %{{.*}}, %{{.*}}]
// CHECK:  triton_gpu.local_store %[[LOAD_21]], %[[MEMDESC_SUBVIEW_38]]
// CHECK:  %[[MEMDESC_SUBVIEW_39:.*]] = triton_gpu.memdesc_subview %{{.*}}[%[[SELECT_37]], %{{.*}}, %{{.*}}]
// CHECK:  triton_gpu.local_store %[[LOAD_27]], %[[MEMDESC_SUBVIEW_39]]
// CHECK:  scf.yield %[[DOT_33]], %[[ADDPTR_20]], %[[ADDPTR_34]], %[[LOAD_22]], %[[SELECT_30]], %[[SELECT_37]], %[[MEMDESC_SUBVIEW_38]], %[[MEMDESC_SUBVIEW_39]]
// CHECK:  }

  tt.func @indirect_bmm_scalar_dist_one(%arg0: i64 {tt.divisibility = 16 : i32}, %arg1: index, %arg2: tensor<16x16x!tt.ptr<f16>, #blocked1> {tt.contiguity = 2 : i32, tt.divisibility = 16 : i32}, %arg3: !tt.ptr<i64>, %arg4: tensor<16x16xi32, #blocked1> {tt.constancy = 16 : i32, tt.divisibility = 16 : i32}, %arg5: tensor<16x16x!tt.ptr<f16>, #blocked> {tt.contiguity = 16 : i32, tt.divisibility = 16 : i32}) -> tensor<16x16xf32, #mma> {
    %c0_i32 = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    %0 = arith.cmpi sgt, %arg1, %c0 : index
    %1 = tt.load %arg3 : !tt.ptr<i64>
    %2 = arith.muli %arg0, %1 : i64
    %3 = tt.splat %2 : i64 -> tensor<16x16xi64, #blocked>
    %4 = tt.splat %0 : i1 -> tensor<16x16xi1, #blocked>
    %5 = tt.addptr %arg5, %3 : tensor<16x16x!tt.ptr<f16>, #blocked>, tensor<16x16xi64, #blocked>
    %6 = tt.load %5, %4 : tensor<16x16x!tt.ptr<f16>, #blocked>
    %c1_i32 = arith.constant 1 : i32
    %7 = tt.addptr %arg3, %c1_i32 : !tt.ptr<i64>, i32
    %8 = tt.load %7, %0 : !tt.ptr<i64>
    %9 = tt.splat %0 : i1 -> tensor<16x16xi1, #blocked1>
    %10 = tt.load %arg2, %9 : tensor<16x16x!tt.ptr<f16>, #blocked1>
    %c-1_i32 = arith.constant -1 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #mma>
    %c1 = arith.constant 1 : index
    %11 = triton_gpu.local_alloc  : () -> !tt.memdesc<1x16x16xf16, #shared2, #triton_gpu.shared_memory, mutable>
    %12 = triton_gpu.local_alloc  : () -> !tt.memdesc<1x16x16xf16, #shared2, #triton_gpu.shared_memory, mutable>
    %13 = tt.addptr %7, %c1_i32 : !tt.ptr<i64>, i32
    %14 = triton_gpu.memdesc_subview %11[%c0_i32, %c0_i32, %c0_i32] : !tt.memdesc<1x16x16xf16, #shared2, #triton_gpu.shared_memory, mutable> -> !tt.memdesc<16x16xf16, #shared2, #triton_gpu.shared_memory, mutable>
    triton_gpu.local_store %10, %14 : tensor<16x16xf16, #blocked1> -> !tt.memdesc<16x16xf16, #shared2, #triton_gpu.shared_memory, mutable>
    %15 = triton_gpu.memdesc_subview %12[%c0_i32, %c0_i32, %c0_i32] : !tt.memdesc<1x16x16xf16, #shared2, #triton_gpu.shared_memory, mutable> -> !tt.memdesc<16x16xf16, #shared2, #triton_gpu.shared_memory, mutable>
    triton_gpu.local_store %6, %15 : tensor<16x16xf16, #blocked> -> !tt.memdesc<16x16xf16, #shared2, #triton_gpu.shared_memory, mutable>
    %16:8 = scf.for %arg6 = %c0 to %arg1 step %c1 iter_args(%arg7 = %cst, %arg8 = %arg2, %arg9 = %13, %arg10 = %8, %arg11 = %c-1_i32, %arg12 = %c0_i32, %arg13 = %14, %arg14 = %15) -> (tensor<16x16xf32, #mma>, tensor<16x16x!tt.ptr<f16>, #blocked1>, !tt.ptr<i64>, i64, i32, i32, !tt.memdesc<16x16xf16, #shared2, #triton_gpu.shared_memory, mutable>, !tt.memdesc<16x16xf16, #shared2, #triton_gpu.shared_memory, mutable>) {
      %17 = arith.subi %arg1, %c1 : index
      %18 = arith.cmpi slt, %arg6, %17 : index
      %19 = arith.muli %arg0, %arg10 : i64
      %20 = tt.splat %19 : i64 -> tensor<16x16xi64, #blocked>
      %21 = tt.splat %18 : i1 -> tensor<16x16xi1, #blocked>
      %22 = tt.addptr %arg5, %20 : tensor<16x16x!tt.ptr<f16>, #blocked>, tensor<16x16xi64, #blocked>
      %23 = tt.load %22, %21 : tensor<16x16x!tt.ptr<f16>, #blocked>
      %24 = tt.load %arg9, %18 : !tt.ptr<i64>
      %25 = tt.splat %18 : i1 -> tensor<16x16xi1, #blocked1>
      %26 = tt.addptr %arg8, %arg4 : tensor<16x16x!tt.ptr<f16>, #blocked1>, tensor<16x16xi32, #blocked1>
      %27 = tt.load %26, %25 : tensor<16x16x!tt.ptr<f16>, #blocked1>
      %28 = arith.addi %arg11, %c1_i32 : i32
      %29 = arith.cmpi slt, %28, %c1_i32 : i32
      %30 = arith.select %29, %28, %c0_i32 : i32
      %31 = triton_gpu.local_load %arg13 : !tt.memdesc<16x16xf16, #shared2, #triton_gpu.shared_memory, mutable> -> tensor<16x16xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %32 = triton_gpu.local_load %arg14 : !tt.memdesc<16x16xf16, #shared2, #triton_gpu.shared_memory, mutable> -> tensor<16x16xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %33 = tt.dot %31, %32, %arg7 : tensor<16x16xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<16x16xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<16x16xf32, #mma>
      %34 = tt.addptr %arg9, %c1_i32 : !tt.ptr<i64>, i32
      %35 = arith.addi %arg12, %c1_i32 : i32
      %36 = arith.cmpi slt, %35, %c1_i32 : i32
      %37 = arith.select %36, %35, %c0_i32 : i32
      %38 = triton_gpu.memdesc_subview %11[%37, %c0_i32, %c0_i32] : !tt.memdesc<1x16x16xf16, #shared2, #triton_gpu.shared_memory, mutable> -> !tt.memdesc<16x16xf16, #shared2, #triton_gpu.shared_memory, mutable>
      triton_gpu.local_store %27, %38 : tensor<16x16xf16, #blocked1> -> !tt.memdesc<16x16xf16, #shared2, #triton_gpu.shared_memory, mutable>
      %39 = triton_gpu.memdesc_subview %12[%37, %c0_i32, %c0_i32] : !tt.memdesc<1x16x16xf16, #shared2, #triton_gpu.shared_memory, mutable> -> !tt.memdesc<16x16xf16, #shared2, #triton_gpu.shared_memory, mutable>
      triton_gpu.local_store %23, %39 : tensor<16x16xf16, #blocked> -> !tt.memdesc<16x16xf16, #shared2, #triton_gpu.shared_memory, mutable>
      scf.yield %33, %26, %34, %24, %30, %37, %38, %39 : tensor<16x16xf32, #mma>, tensor<16x16x!tt.ptr<f16>, #blocked1>, !tt.ptr<i64>, i64, i32, i32, !tt.memdesc<16x16xf16, #shared2, #triton_gpu.shared_memory, mutable>, !tt.memdesc<16x16xf16, #shared2, #triton_gpu.shared_memory, mutable>
    }
    triton_gpu.local_dealloc %11 : !tt.memdesc<1x16x16xf16, #shared2, #triton_gpu.shared_memory, mutable>
    triton_gpu.local_dealloc %12 : !tt.memdesc<1x16x16xf16, #shared2, #triton_gpu.shared_memory, mutable>
    tt.return %16#0 : tensor<16x16xf32, #mma>
  }

// CHECK-LABEL:  tt.func @indirect_bmm_vector
// CHECK:  %{{.*}}:8 = scf.for %[[ARG6:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ARG7:.*]] = %{{.*}}, %[[ARG8:.*]] = %{{.*}}, %[[ARG9:.*]] = %{{.*}}, %[[ARG10:.*]] = %{{.*}}-1_i32, %[[ARG11:.*]] = %{{.*}}, %[[ARG12:.*]] = %{{.*}}, %[[ARG13:.*]] = %{{.*}}, %[[ARG14:.*]] = %{{.*}})

// CHECK:  %[[SUBI_20:.*]] = arith.subi %{{.*}}, %{{.*}}
// CHECK:  %[[SUBI_21:.*]] = arith.subi %{{.*}}, %{{.*}}
// CHECK:  %[[CMPI_22:.*]] = arith.cmpi slt, %[[ARG6]], %[[SUBI_21]]
// CHECK:  %[[SPLAT_23:.*]] = tt.splat %[[CMPI_22]]
// CHECK:  %[[ADDPTR_24:.*]] = tt.addptr %[[ARG8]], %{{.*}}
// CHECK:  %[[LOAD_25:.*]] = tt.load %[[ADDPTR_24]], %[[SPLAT_23]]
// CHECK:  %[[EXPAND_DIMS_26:.*]] = tt.expand_dims %[[ARG14]] {axis = 1 : i32}
// CHECK:  %[[BROADCAST_27:.*]] = tt.broadcast %[[EXPAND_DIMS_26]]
// CHECK:  %[[MULI_28:.*]] = arith.muli %{{.*}}, %[[BROADCAST_27]]
// CHECK:  %[[SPLAT_29:.*]] = tt.splat %[[CMPI_22]]
// CHECK:  %[[ADDPTR_30:.*]] = tt.addptr %{{.*}}, %[[MULI_28]]
// CHECK:  %[[LOAD_31:.*]] = tt.load %[[ADDPTR_30]], %[[SPLAT_29]]
// CHECK:  %[[CMPI_32:.*]] = arith.cmpi slt, %[[ARG6]], %[[SUBI_20]]
// CHECK:  %[[SPLAT_33:.*]] = tt.splat %[[CMPI_32]]
// CHECK:  %[[ADDPTR_34:.*]] = tt.addptr %[[ARG9]], %{{.*}}
// CHECK:  %[[LOAD_35:.*]] = tt.load %[[ADDPTR_34]], %[[SPLAT_33]]
// CHECK:  %[[ADDI_36:.*]] = arith.addi %[[ARG10]], %{{.*}}
// CHECK:  %[[CMPI_37:.*]] = arith.cmpi slt, %[[ADDI_36]], %{{.*}}
// CHECK:  %[[SELECT_38:.*]] = arith.select %[[CMPI_37]], %[[ADDI_36]], %{{.*}}
// CHECK:  %[[LOCAL_LOAD_39:.*]] = triton_gpu.local_load %[[ARG12]]
// CHECK:  %[[LOCAL_LOAD_40:.*]] = triton_gpu.local_load %[[ARG13]]
// CHECK:  %[[DOT_41:.*]] = tt.dot %[[LOCAL_LOAD_39]], %[[LOCAL_LOAD_40]], %[[ARG7]]
// CHECK:  %[[ADDI_42:.*]] = arith.addi %[[ARG11]], %{{.*}}
// CHECK:  %[[CMPI_43:.*]] = arith.cmpi slt, %[[ADDI_42]], %{{.*}}
// CHECK:  %[[SELECT_44:.*]] = arith.select %[[CMPI_43]], %[[ADDI_42]], %{{.*}}
// CHECK:  %[[MEMDESC_SUBVIEW_45:.*]] = triton_gpu.memdesc_subview %{{.*}}[%[[SELECT_44]], %{{.*}}, %{{.*}}]
// CHECK:  triton_gpu.local_store %[[LOAD_25]], %[[MEMDESC_SUBVIEW_45]]
// CHECK:  %[[MEMDESC_SUBVIEW_46:.*]] = triton_gpu.memdesc_subview %{{.*}}[%[[SELECT_44]], %{{.*}}, %{{.*}}]
// CHECK:  triton_gpu.local_store %[[LOAD_31]], %[[MEMDESC_SUBVIEW_46]]
// CHECK:  scf.yield %[[DOT_41]], %[[ADDPTR_24]], %[[ADDPTR_34]], %[[SELECT_38]], %[[SELECT_44]], %[[MEMDESC_SUBVIEW_45]], %[[MEMDESC_SUBVIEW_46]], %[[LOAD_35]]
// CHECK:  }

  tt.func @indirect_bmm_vector(%arg0: tensor<16x16xi64, #blocked> {tt.constancy = 16 : i32, tt.divisibility = 16 : i32}, %arg1: index, %arg2: tensor<16x16x!tt.ptr<f16>, #blocked1> {tt.contiguity = 2 : i32, tt.divisibility = 16 : i32}, %arg3: tensor<16x!tt.ptr<i64>, #triton_gpu.slice<{dim = 1, parent = #blocked}>>, %arg4: tensor<16x16xi32, #blocked1> {tt.constancy = 16 : i32, tt.divisibility = 16 : i32}, %arg5: tensor<16x16x!tt.ptr<f16>, #blocked> {tt.contiguity = 16 : i32, tt.divisibility = 16 : i32}) -> tensor<16x16xf32, #mma> {
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %0 = arith.cmpi sgt, %arg1, %c1 : index
    %cst = arith.constant dense<1> : tensor<16xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %1 = tt.splat %0 : i1 -> tensor<16xi1, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %2 = tt.addptr %arg3, %cst : tensor<16x!tt.ptr<i64>, #triton_gpu.slice<{dim = 1, parent = #blocked}>>, tensor<16xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %3 = tt.load %2, %1 : tensor<16x!tt.ptr<i64>, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %c0 = arith.constant 0 : index
    %4 = arith.cmpi sgt, %arg1, %c0 : index
    %5 = tt.splat %4 : i1 -> tensor<16xi1, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %6 = tt.load %arg3, %5 : tensor<16x!tt.ptr<i64>, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %7 = tt.expand_dims %6 {axis = 1 : i32} : tensor<16xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<16x1xi64, #blocked>
    %8 = tt.broadcast %7 : tensor<16x1xi64, #blocked> -> tensor<16x16xi64, #blocked>
    %9 = arith.muli %arg0, %8 : tensor<16x16xi64, #blocked>
    %10 = tt.splat %4 : i1 -> tensor<16x16xi1, #blocked>
    %11 = tt.addptr %arg5, %9 : tensor<16x16x!tt.ptr<f16>, #blocked>, tensor<16x16xi64, #blocked>
    %12 = tt.load %11, %10 : tensor<16x16x!tt.ptr<f16>, #blocked>
    %13 = tt.splat %4 : i1 -> tensor<16x16xi1, #blocked1>
    %14 = tt.load %arg2, %13 : tensor<16x16x!tt.ptr<f16>, #blocked1>
    %c0_i32 = arith.constant 0 : i32
    %c-1_i32 = arith.constant -1 : i32
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #mma>
    %c1_i32 = arith.constant 1 : i32
    %15 = triton_gpu.local_alloc  : () -> !tt.memdesc<1x16x16xf16, #shared2, #triton_gpu.shared_memory, mutable>
    %16 = triton_gpu.local_alloc  : () -> !tt.memdesc<1x16x16xf16, #shared2, #triton_gpu.shared_memory, mutable>
    %17 = triton_gpu.memdesc_subview %15[%c0_i32, %c0_i32, %c0_i32] : !tt.memdesc<1x16x16xf16, #shared2, #triton_gpu.shared_memory, mutable> -> !tt.memdesc<16x16xf16, #shared2, #triton_gpu.shared_memory, mutable>
    triton_gpu.local_store %14, %17 : tensor<16x16xf16, #blocked1> -> !tt.memdesc<16x16xf16, #shared2, #triton_gpu.shared_memory, mutable>
    %18 = triton_gpu.memdesc_subview %16[%c0_i32, %c0_i32, %c0_i32] : !tt.memdesc<1x16x16xf16, #shared2, #triton_gpu.shared_memory, mutable> -> !tt.memdesc<16x16xf16, #shared2, #triton_gpu.shared_memory, mutable>
    triton_gpu.local_store %12, %18 : tensor<16x16xf16, #blocked> -> !tt.memdesc<16x16xf16, #shared2, #triton_gpu.shared_memory, mutable>
    %19:8 = scf.for %arg6 = %c0 to %arg1 step %c1 iter_args(%arg7 = %cst_0, %arg8 = %arg2, %arg9 = %2, %arg10 = %c-1_i32, %arg11 = %c0_i32, %arg12 = %17, %arg13 = %18, %arg14 = %3) -> (tensor<16x16xf32, #mma>, tensor<16x16x!tt.ptr<f16>, #blocked1>, tensor<16x!tt.ptr<i64>, #triton_gpu.slice<{dim = 1, parent = #blocked}>>, i32, i32, !tt.memdesc<16x16xf16, #shared2, #triton_gpu.shared_memory, mutable>, !tt.memdesc<16x16xf16, #shared2, #triton_gpu.shared_memory, mutable>, tensor<16xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>>) {
      %20 = arith.subi %arg1, %c2 : index
      %21 = arith.cmpi slt, %arg6, %20 : index
      %22 = tt.splat %21 : i1 -> tensor<16xi1, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
      %23 = tt.addptr %arg9, %cst : tensor<16x!tt.ptr<i64>, #triton_gpu.slice<{dim = 1, parent = #blocked}>>, tensor<16xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
      %24 = tt.load %23, %22 : tensor<16x!tt.ptr<i64>, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
      %25 = arith.subi %arg1, %c1 : index
      %26 = arith.cmpi slt, %arg6, %25 : index
      %27 = tt.expand_dims %arg14 {axis = 1 : i32} : tensor<16xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<16x1xi64, #blocked>
      %28 = tt.broadcast %27 : tensor<16x1xi64, #blocked> -> tensor<16x16xi64, #blocked>
      %29 = arith.muli %arg0, %28 : tensor<16x16xi64, #blocked>
      %30 = tt.splat %26 : i1 -> tensor<16x16xi1, #blocked>
      %31 = tt.addptr %arg5, %29 : tensor<16x16x!tt.ptr<f16>, #blocked>, tensor<16x16xi64, #blocked>
      %32 = tt.load %31, %30 : tensor<16x16x!tt.ptr<f16>, #blocked>
      %33 = tt.splat %26 : i1 -> tensor<16x16xi1, #blocked1>
      %34 = tt.addptr %arg8, %arg4 : tensor<16x16x!tt.ptr<f16>, #blocked1>, tensor<16x16xi32, #blocked1>
      %35 = tt.load %34, %33 : tensor<16x16x!tt.ptr<f16>, #blocked1>
      %36 = arith.addi %arg10, %c1_i32 : i32
      %37 = arith.cmpi slt, %36, %c1_i32 : i32
      %38 = arith.select %37, %36, %c0_i32 : i32
      %39 = triton_gpu.local_load %arg12 : !tt.memdesc<16x16xf16, #shared2, #triton_gpu.shared_memory, mutable> -> tensor<16x16xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %40 = triton_gpu.local_load %arg13 : !tt.memdesc<16x16xf16, #shared2, #triton_gpu.shared_memory, mutable> -> tensor<16x16xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %41 = tt.dot %39, %40, %arg7 : tensor<16x16xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<16x16xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<16x16xf32, #mma>
      %42 = arith.addi %arg11, %c1_i32 : i32
      %43 = arith.cmpi slt, %42, %c1_i32 : i32
      %44 = arith.select %43, %42, %c0_i32 : i32
      %45 = triton_gpu.memdesc_subview %15[%44, %c0_i32, %c0_i32] : !tt.memdesc<1x16x16xf16, #shared2, #triton_gpu.shared_memory, mutable> -> !tt.memdesc<16x16xf16, #shared2, #triton_gpu.shared_memory, mutable>
      triton_gpu.local_store %35, %45 : tensor<16x16xf16, #blocked1> -> !tt.memdesc<16x16xf16, #shared2, #triton_gpu.shared_memory, mutable>
      %46 = triton_gpu.memdesc_subview %16[%44, %c0_i32, %c0_i32] : !tt.memdesc<1x16x16xf16, #shared2, #triton_gpu.shared_memory, mutable> -> !tt.memdesc<16x16xf16, #shared2, #triton_gpu.shared_memory, mutable>
      triton_gpu.local_store %32, %46 : tensor<16x16xf16, #blocked> -> !tt.memdesc<16x16xf16, #shared2, #triton_gpu.shared_memory, mutable>
      scf.yield %41, %34, %23, %38, %44, %45, %46, %24 : tensor<16x16xf32, #mma>, tensor<16x16x!tt.ptr<f16>, #blocked1>, tensor<16x!tt.ptr<i64>, #triton_gpu.slice<{dim = 1, parent = #blocked}>>, i32, i32, !tt.memdesc<16x16xf16, #shared2, #triton_gpu.shared_memory, mutable>, !tt.memdesc<16x16xf16, #shared2, #triton_gpu.shared_memory, mutable>, tensor<16xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    }
    triton_gpu.local_dealloc %15 : !tt.memdesc<1x16x16xf16, #shared2, #triton_gpu.shared_memory, mutable>
    triton_gpu.local_dealloc %16 : !tt.memdesc<1x16x16xf16, #shared2, #triton_gpu.shared_memory, mutable>
    tt.return %19#0 : tensor<16x16xf32, #mma>
  }

// CHECK-LABEL:  tt.func @post_load_inv
// CHECK:  %{{.*}}:5 = scf.for %[[ARG9:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ARG10:.*]] = %{{.*}}, %[[ARG11:.*]] = %{{.*}}-1_i32, %[[ARG12:.*]] = %{{.*}}, %[[ARG13:.*]] = %{{.*}}, %[[ARG14:.*]] = %{{.*}})

// CHECK:  %[[CMPI_19:.*]] = arith.cmpi slt, %[[ARG9]], %{{.*}}
// CHECK:  %[[ADDI_20:.*]] = arith.addi %[[ARG9]], %{{.*}}
// CHECK:  %[[INDEX_CAST_21:.*]] = arith.index_cast %[[ADDI_20]]
// CHECK:  %[[MULI_22:.*]] = arith.muli %[[INDEX_CAST_21]], %{{.*}}
// CHECK:  %[[SUBI_23:.*]] = arith.subi %{{.*}}, %[[MULI_22]]
// CHECK:  %[[SPLAT_24:.*]] = tt.splat %[[SUBI_23]]
// CHECK:  %[[CMPI_25:.*]] = arith.cmpi slt, %{{.*}}, %[[SPLAT_24]]
// CHECK:  %[[BROADCAST_26:.*]] = tt.broadcast %[[CMPI_25]]
// CHECK:  %[[SPLAT_27:.*]] = tt.splat %[[CMPI_19]]
// CHECK:  %[[INDEX_CAST_28:.*]] = arith.index_cast %[[ARG9]]
// CHECK:  %[[ADDI_29:.*]] = arith.addi %[[INDEX_CAST_28]], %{{.*}}
// CHECK:  %[[MULI_30:.*]] = arith.muli %[[ADDI_29]], %{{.*}}
// CHECK:  %[[SPLAT_31:.*]] = tt.splat %[[MULI_30]]
// CHECK:  %[[ANDI_32:.*]] = arith.andi %[[SPLAT_27]], %[[BROADCAST_26]]
// CHECK:  %[[ADDPTR_33:.*]] = tt.addptr %{{.*}}, %[[SPLAT_31]]
// CHECK:  %[[LOAD_34:.*]] = tt.load %[[ADDPTR_33]], %[[ANDI_32]], %{{.*}}
// CHECK:  %[[SPLAT_35:.*]] = tt.splat %[[SUBI_23]]
// CHECK:  %[[CMPI_36:.*]] = arith.cmpi slt, %{{.*}}, %[[SPLAT_35]]
// CHECK:  %[[BROADCAST_37:.*]] = tt.broadcast %[[CMPI_36]]
// CHECK:  %[[SPLAT_38:.*]] = tt.splat %[[CMPI_19]]
// CHECK:  %[[MULI_39:.*]] = arith.muli %[[MULI_30]], %{{.*}}
// CHECK:  %[[SPLAT_40:.*]] = tt.splat %[[MULI_39]]
// CHECK:  %[[ANDI_41:.*]] = arith.andi %[[SPLAT_38]], %[[BROADCAST_37]]
// CHECK:  %[[ADDPTR_42:.*]] = tt.addptr %{{.*}}, %[[SPLAT_40]]
// CHECK:  %[[LOAD_43:.*]] = tt.load %[[ADDPTR_42]], %[[ANDI_41]], %{{.*}}
// CHECK:  %[[ADDI_44:.*]] = arith.addi %[[ARG11]], %{{.*}}
// CHECK:  %[[CMPI_45:.*]] = arith.cmpi slt, %[[ADDI_44]], %{{.*}}
// CHECK:  %[[SELECT_46:.*]] = arith.select %[[CMPI_45]], %[[ADDI_44]], %{{.*}}
// CHECK:  %[[LOCAL_LOAD_47:.*]] = triton_gpu.local_load %[[ARG13]]
// CHECK:  %[[LOCAL_LOAD_48:.*]] = triton_gpu.local_load %[[ARG14]]
// CHECK:  %[[DOT_49:.*]] = tt.dot %[[LOCAL_LOAD_47]], %[[LOCAL_LOAD_48]], %[[ARG10]]
// CHECK:  %[[ADDI_50:.*]] = arith.addi %[[ARG12]], %{{.*}}
// CHECK:  %[[CMPI_51:.*]] = arith.cmpi slt, %[[ADDI_50]], %{{.*}}
// CHECK:  %[[SELECT_52:.*]] = arith.select %[[CMPI_51]], %[[ADDI_50]], %{{.*}}
// CHECK:  %[[MEMDESC_SUBVIEW_53:.*]] = triton_gpu.memdesc_subview %{{.*}}[%[[SELECT_52]], %{{.*}}, %{{.*}}]
// CHECK:  triton_gpu.local_store %[[LOAD_34]], %[[MEMDESC_SUBVIEW_53]]
// CHECK:  %[[MEMDESC_SUBVIEW_54:.*]] = triton_gpu.memdesc_subview %{{.*}}[%[[SELECT_52]], %{{.*}}, %{{.*}}]
// CHECK:  triton_gpu.local_store %[[LOAD_43]], %[[MEMDESC_SUBVIEW_54]]
// CHECK:  scf.yield %[[DOT_49]], %[[SELECT_46]], %[[SELECT_52]], %[[MEMDESC_SUBVIEW_53]], %[[MEMDESC_SUBVIEW_54]]
// CHECK:  }

  tt.func @post_load_inv(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}) -> tensor<32x32xf32, #mma> {
    %c899 = arith.constant 899 : index
    %0 = tt.splat %arg5 : i32 -> tensor<32x1xi32, #blocked1>
    %1 = tt.splat %arg4 : i32 -> tensor<32x1xi32, #blocked1>
    %2 = arith.cmpi slt, %1, %0 : tensor<32x1xi32, #blocked1>
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #blocked1>
    %3 = tt.broadcast %2 : tensor<32x1xi1, #blocked1> -> tensor<32x32xi1, #blocked1>
    %4 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #blocked1>
    %5 = tt.load %4, %3, %cst : tensor<32x32x!tt.ptr<f32>, #blocked1>
    %6 = tt.splat %arg5 : i32 -> tensor<1x32xi32, #blocked1>
    %7 = tt.splat %arg3 : i32 -> tensor<1x32xi32, #blocked1>
    %8 = arith.cmpi slt, %7, %6 : tensor<1x32xi32, #blocked1>
    %9 = tt.broadcast %8 : tensor<1x32xi1, #blocked1> -> tensor<32x32xi1, #blocked1>
    %10 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #blocked1>
    %11 = tt.load %10, %9, %cst : tensor<32x32x!tt.ptr<f32>, #blocked1>
    %c0_i32 = arith.constant 0 : i32
    %c-1_i32 = arith.constant -1 : i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1_i32 = arith.constant 1 : i32
    %c32_i32 = arith.constant 32 : i32
    %c900 = arith.constant 900 : index
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %12 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #blocked1>
    %13 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #blocked1>
    %14 = triton_gpu.local_alloc  : () -> !tt.memdesc<1x32x32xf32, #shared3, #triton_gpu.shared_memory, mutable>
    %15 = triton_gpu.local_alloc  : () -> !tt.memdesc<1x32x32xf32, #shared4, #triton_gpu.shared_memory, mutable>
    %16 = triton_gpu.memdesc_subview %14[%c0_i32, %c0_i32, %c0_i32] : !tt.memdesc<1x32x32xf32, #shared3, #triton_gpu.shared_memory, mutable> -> !tt.memdesc<32x32xf32, #shared3, #triton_gpu.shared_memory, mutable>
    triton_gpu.local_store %11, %16 : tensor<32x32xf32, #blocked1> -> !tt.memdesc<32x32xf32, #shared3, #triton_gpu.shared_memory, mutable>
    %17 = triton_gpu.memdesc_subview %15[%c0_i32, %c0_i32, %c0_i32] : !tt.memdesc<1x32x32xf32, #shared4, #triton_gpu.shared_memory, mutable> -> !tt.memdesc<32x32xf32, #shared4, #triton_gpu.shared_memory, mutable>
    triton_gpu.local_store %5, %17 : tensor<32x32xf32, #blocked1> -> !tt.memdesc<32x32xf32, #shared4, #triton_gpu.shared_memory, mutable>
    %18:5 = scf.for %arg9 = %c0 to %c900 step %c1 iter_args(%arg10 = %cst_0, %arg11 = %c-1_i32, %arg12 = %c0_i32, %arg13 = %16, %arg14 = %17) -> (tensor<32x32xf32, #mma>, i32, i32, !tt.memdesc<32x32xf32, #shared3, #triton_gpu.shared_memory, mutable>, !tt.memdesc<32x32xf32, #shared4, #triton_gpu.shared_memory, mutable>) {
      %19 = arith.cmpi slt, %arg9, %c899 : index
      %20 = arith.addi %arg9, %c1 : index
      %21 = arith.index_cast %20 : index to i32
      %22 = arith.muli %21, %c32_i32 : i32
      %23 = arith.subi %arg5, %22 : i32
      %24 = tt.splat %23 : i32 -> tensor<32x1xi32, #blocked1>
      %25 = arith.cmpi slt, %1, %24 : tensor<32x1xi32, #blocked1>
      %26 = tt.broadcast %25 : tensor<32x1xi1, #blocked1> -> tensor<32x32xi1, #blocked1>
      %27 = tt.splat %19 : i1 -> tensor<32x32xi1, #blocked1>
      %28 = arith.index_cast %arg9 : index to i32
      %29 = arith.addi %28, %c1_i32 : i32
      %30 = arith.muli %29, %c32_i32 : i32
      %31 = arith.muli %30, %arg7 : i32
      %32 = tt.splat %31 : i32 -> tensor<32x32xi32, #blocked1>
      %33 = arith.andi %27, %26 : tensor<32x32xi1, #blocked1>
      %34 = tt.addptr %13, %32 : tensor<32x32x!tt.ptr<f32>, #blocked1>, tensor<32x32xi32, #blocked1>
      %35 = tt.load %34, %33, %cst : tensor<32x32x!tt.ptr<f32>, #blocked1>
      %36 = tt.splat %23 : i32 -> tensor<1x32xi32, #blocked1>
      %37 = arith.cmpi slt, %7, %36 : tensor<1x32xi32, #blocked1>
      %38 = tt.broadcast %37 : tensor<1x32xi1, #blocked1> -> tensor<32x32xi1, #blocked1>
      %39 = tt.splat %19 : i1 -> tensor<32x32xi1, #blocked1>
      %40 = tt.splat %30 : i32 -> tensor<32x32xi32, #blocked1>
      %41 = arith.andi %39, %38 : tensor<32x32xi1, #blocked1>
      %42 = tt.addptr %12, %40 : tensor<32x32x!tt.ptr<f32>, #blocked1>, tensor<32x32xi32, #blocked1>
      %43 = tt.load %42, %41, %cst : tensor<32x32x!tt.ptr<f32>, #blocked1>
      %44 = arith.addi %arg11, %c1_i32 : i32
      %45 = arith.cmpi slt, %44, %c1_i32 : i32
      %46 = arith.select %45, %44, %c0_i32 : i32
      %47 = triton_gpu.local_load %arg13 : !tt.memdesc<32x32xf32, #shared3, #triton_gpu.shared_memory, mutable> -> tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
      %48 = triton_gpu.local_load %arg14 : !tt.memdesc<32x32xf32, #shared4, #triton_gpu.shared_memory, mutable> -> tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>
      %49 = tt.dot %47, %48, %arg10 : tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>> -> tensor<32x32xf32, #mma>
      %50 = arith.addi %arg12, %c1_i32 : i32
      %51 = arith.cmpi slt, %50, %c1_i32 : i32
      %52 = arith.select %51, %50, %c0_i32 : i32
      %53 = triton_gpu.memdesc_subview %14[%52, %c0_i32, %c0_i32] : !tt.memdesc<1x32x32xf32, #shared3, #triton_gpu.shared_memory, mutable> -> !tt.memdesc<32x32xf32, #shared3, #triton_gpu.shared_memory, mutable>
      triton_gpu.local_store %43, %53 : tensor<32x32xf32, #blocked1> -> !tt.memdesc<32x32xf32, #shared3, #triton_gpu.shared_memory, mutable>
      %54 = triton_gpu.memdesc_subview %15[%52, %c0_i32, %c0_i32] : !tt.memdesc<1x32x32xf32, #shared4, #triton_gpu.shared_memory, mutable> -> !tt.memdesc<32x32xf32, #shared4, #triton_gpu.shared_memory, mutable>
      triton_gpu.local_store %35, %54 : tensor<32x32xf32, #blocked1> -> !tt.memdesc<32x32xf32, #shared4, #triton_gpu.shared_memory, mutable>
      scf.yield %49, %46, %52, %53, %54 : tensor<32x32xf32, #mma>, i32, i32, !tt.memdesc<32x32xf32, #shared3, #triton_gpu.shared_memory, mutable>, !tt.memdesc<32x32xf32, #shared4, #triton_gpu.shared_memory, mutable>
    }
    triton_gpu.local_dealloc %14 : !tt.memdesc<1x32x32xf32, #shared3, #triton_gpu.shared_memory, mutable>
    triton_gpu.local_dealloc %15 : !tt.memdesc<1x32x32xf32, #shared4, #triton_gpu.shared_memory, mutable>
    tt.return %18#0 : tensor<32x32xf32, #mma>
  }

// CHECK-LABEL:  tt.func @cross_iter_dep
// CHECK:  %{{.*}}:5 = scf.for %[[ARG9:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ARG10:.*]] = %{{.*}}, %[[ARG11:.*]] = %{{.*}}, %[[ARG12:.*]] = %{{.*}}, %[[ARG13:.*]] = %{{.*}}, %[[ARG14:.*]] = %{{.*}})

// CHECK:  %[[INDEX_CAST_9:.*]] = arith.index_cast %[[ARG9]]
// CHECK:  %[[MULI_10:.*]] = arith.muli %[[INDEX_CAST_9]], %{{.*}}
// CHECK:  %[[SUBI_11:.*]] = arith.subi %{{.*}}, %[[MULI_10]]
// CHECK:  %[[SPLAT_12:.*]] = tt.splat %[[SUBI_11]]
// CHECK:  %[[CMPI_13:.*]] = arith.cmpi slt, %{{.*}}, %[[SPLAT_12]]
// CHECK:  %[[BROADCAST_14:.*]] = tt.broadcast %[[CMPI_13]]
// CHECK:  %[[LOAD_15:.*]] = tt.load %[[ARG11]], %[[BROADCAST_14]], %{{.*}}
// CHECK:  %[[SPLAT_16:.*]] = tt.splat %[[SUBI_11]]
// CHECK:  %[[CMPI_17:.*]] = arith.cmpi slt, %{{.*}}, %[[SPLAT_16]]
// CHECK:  %[[BROADCAST_18:.*]] = tt.broadcast %[[CMPI_17]]
// CHECK:  %[[LOAD_19:.*]] = tt.load %[[ARG12]], %[[BROADCAST_18]], %{{.*}}
// CHECK:  %[[CONVERT_LAYOUT_20:.*]] = triton_gpu.convert_layout %[[LOAD_15]]
// CHECK:  %[[CONVERT_LAYOUT_21:.*]] = triton_gpu.convert_layout %[[LOAD_19]]
// CHECK:  %[[DOT_22:.*]] = tt.dot %[[CONVERT_LAYOUT_20]], %[[CONVERT_LAYOUT_21]], %[[ARG10]]
// CHECK:  %[[INDEX_CAST_23:.*]] = arith.index_cast %[[ARG9]]
// CHECK:  %[[ADDI_24:.*]] = arith.addi %[[INDEX_CAST_23]], %{{.*}}
// CHECK:  %[[MULI_25:.*]] = arith.muli %[[ADDI_24]], %{{.*}}
// CHECK:  %[[SPLAT_26:.*]] = tt.splat %[[MULI_25]]
// CHECK:  %[[ADDPTR_27:.*]] = tt.addptr %{{.*}}, %[[SPLAT_26]]
// CHECK:  %[[MULI_28:.*]] = arith.muli %[[MULI_25]], %{{.*}}
// CHECK:  %[[SPLAT_29:.*]] = tt.splat %[[MULI_28]]
// CHECK:  %[[ADDPTR_30:.*]] = tt.addptr %{{.*}}, %[[SPLAT_29]]
// CHECK:  scf.yield %[[DOT_22]], %[[ARG13]], %[[ARG14]], %[[ADDPTR_27]], %[[ADDPTR_30]]
// CHECK:  }

  tt.func @cross_iter_dep(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}) -> tensor<32x32xf32, #mma> {
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %c1 = arith.constant 1 : index
    %c2_i32 = arith.constant 2 : i32
    %c32_i32 = arith.constant 32 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #blocked1>
    %0 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #blocked1>
    %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #blocked1>
    %2 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #blocked1>
    %3 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #blocked1>
    %4 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #blocked1>
    %5 = tt.splat %arg3 : i32 -> tensor<1x32xi32, #blocked1>
    %6 = tt.splat %arg4 : i32 -> tensor<32x1xi32, #blocked1>
    %7 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #blocked1>
    %8:5 = scf.for %arg9 = %c0 to %c32 step %c1 iter_args(%arg10 = %cst, %arg11 = %0, %arg12 = %1, %arg13 = %3, %arg14 = %4) -> (tensor<32x32xf32, #mma>, tensor<32x32x!tt.ptr<f32>, #blocked1>, tensor<32x32x!tt.ptr<f32>, #blocked1>, tensor<32x32x!tt.ptr<f32>, #blocked1>, tensor<32x32x!tt.ptr<f32>, #blocked1>) {
      %9 = arith.index_cast %arg9 : index to i32
      %10 = arith.muli %9, %c32_i32 : i32
      %11 = arith.subi %arg5, %10 : i32
      %12 = tt.splat %11 : i32 -> tensor<32x1xi32, #blocked1>
      %13 = arith.cmpi slt, %6, %12 : tensor<32x1xi32, #blocked1>
      %14 = tt.broadcast %13 : tensor<32x1xi1, #blocked1> -> tensor<32x32xi1, #blocked1>
      %15 = tt.load %arg12, %14, %cst_0 : tensor<32x32x!tt.ptr<f32>, #blocked1>
      %16 = tt.splat %11 : i32 -> tensor<1x32xi32, #blocked1>
      %17 = arith.cmpi slt, %5, %16 : tensor<1x32xi32, #blocked1>
      %18 = tt.broadcast %17 : tensor<1x32xi1, #blocked1> -> tensor<32x32xi1, #blocked1>
      %19 = tt.load %arg11, %18, %cst_0 : tensor<32x32x!tt.ptr<f32>, #blocked1>
      %20 = triton_gpu.convert_layout %19 : tensor<32x32xf32, #blocked1> -> tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
      %21 = triton_gpu.convert_layout %15 : tensor<32x32xf32, #blocked1> -> tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>
      %22 = tt.dot %20, %21, %arg10 : tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>> -> tensor<32x32xf32, #mma>
      %23 = arith.index_cast %arg9 : index to i32
      %24 = arith.addi %23, %c2_i32 : i32
      %25 = arith.muli %24, %c32_i32 : i32
      %26 = tt.splat %25 : i32 -> tensor<32x32xi32, #blocked1>
      %27 = tt.addptr %7, %26 : tensor<32x32x!tt.ptr<f32>, #blocked1>, tensor<32x32xi32, #blocked1>
      %28 = arith.muli %25, %arg7 : i32
      %29 = tt.splat %28 : i32 -> tensor<32x32xi32, #blocked1>
      %30 = tt.addptr %2, %29 : tensor<32x32x!tt.ptr<f32>, #blocked1>, tensor<32x32xi32, #blocked1>
      scf.yield %22, %arg13, %arg14, %27, %30 : tensor<32x32xf32, #mma>, tensor<32x32x!tt.ptr<f32>, #blocked1>, tensor<32x32x!tt.ptr<f32>, #blocked1>, tensor<32x32x!tt.ptr<f32>, #blocked1>, tensor<32x32x!tt.ptr<f32>, #blocked1>
    }
    tt.return %8#0 : tensor<32x32xf32, #mma>
  }

// CHECK-LABEL:  tt.func @dep_arg_two_uses
// CHECK:  %{{.*}}:5 = scf.for %[[ARG3:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ARG4:.*]] = %{{.*}}, %[[ARG5:.*]] = %{{.*}}, %[[ARG6:.*]] = %{{.*}}, %[[ARG7:.*]] = %{{.*}}, %[[ARG8:.*]] = %{{.*}})

// CHECK:  %[[SUBI_8:.*]] = arith.subi %{{.*}}, %[[ARG3]]
// CHECK:  %[[INDEX_CAST_9:.*]] = arith.index_cast %[[SUBI_8]]
// CHECK:  %[[SPLAT_10:.*]] = tt.splat %[[INDEX_CAST_9]]
// CHECK:  %[[CMPI_11:.*]] = arith.cmpi slt, %{{.*}}, %[[SPLAT_10]]
// CHECK:  %[[EXPAND_DIMS_12:.*]] = tt.expand_dims %[[CMPI_11]] {axis = 0 : i32}
// CHECK:  %[[EXPAND_DIMS_13:.*]] = tt.expand_dims %[[ARG5]] {axis = 0 : i32}
// CHECK:  %[[EXTSI_14:.*]] = arith.extsi %[[EXPAND_DIMS_13]]
// CHECK:  %[[MULI_15:.*]] = arith.muli %[[EXTSI_14]], %{{.*}}
// CHECK:  %[[BROADCAST_16:.*]] = tt.broadcast %[[MULI_15]]
// CHECK:  %[[BROADCAST_17:.*]] = tt.broadcast %[[EXPAND_DIMS_12]]
// CHECK:  %[[ADDPTR_18:.*]] = tt.addptr %[[ARG4]], %[[BROADCAST_16]]
// CHECK:  %[[LOAD_19:.*]] = tt.load %[[ADDPTR_18]], %[[BROADCAST_17]]
// CHECK:  %[[SPLAT_20:.*]] = tt.splat %[[ARG6]]
// CHECK:  %[[ADDPTR_21:.*]] = tt.addptr %[[SPLAT_20]], %{{.*}}
// CHECK:  %[[LOAD_22:.*]] = tt.load %[[ADDPTR_21]]
// CHECK:  %[[SPLAT_23:.*]] = tt.splat %[[INDEX_CAST_9]]
// CHECK:  %[[CMPI_24:.*]] = arith.cmpi slt, %{{.*}}, %[[SPLAT_23]]
// CHECK:  %[[EXPAND_DIMS_25:.*]] = tt.expand_dims %[[CMPI_24]] {axis = 1 : i32}
// CHECK:  %[[BROADCAST_26:.*]] = tt.broadcast %[[EXPAND_DIMS_25]]
// CHECK:  %[[LOAD_27:.*]] = tt.load %[[ARG8]], %[[BROADCAST_26]], %{{.*}}
// CHECK:  %[[EXPAND_DIMS_28:.*]] = tt.expand_dims %[[ARG5]] {axis = 0 : i32}
// CHECK:  %[[EXTSI_29:.*]] = arith.extsi %[[EXPAND_DIMS_28]]
// CHECK:  %[[MULI_30:.*]] = arith.muli %[[EXTSI_29]], %{{.*}}
// CHECK:  %[[BROADCAST_31:.*]] = tt.broadcast %[[MULI_30]]
// CHECK:  %[[ADDPTR_32:.*]] = tt.addptr %[[ARG4]], %[[BROADCAST_31]]
// CHECK:  %[[ADDPTR_33:.*]] = tt.addptr %[[ARG6]], %{{.*}}
// CHECK:  %[[CONVERT_LAYOUT_34:.*]] = triton_gpu.convert_layout %[[LOAD_19]]
// CHECK:  %[[CONVERT_LAYOUT_35:.*]] = triton_gpu.convert_layout %[[LOAD_27]]
// CHECK:  %[[DOT_36:.*]] = tt.dot %[[CONVERT_LAYOUT_34]], %[[CONVERT_LAYOUT_35]], %[[ARG7]]
// CHECK:  %[[ADDPTR_37:.*]] = tt.addptr %[[ARG8]], %{{.*}}
// CHECK:  scf.yield %[[ADDPTR_32]], %[[LOAD_22]], %[[ADDPTR_33]], %[[DOT_36]], %[[ADDPTR_37]]
// CHECK:  }

  tt.func @dep_arg_two_uses(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}) -> tensor<128x128xf32, #mma> {
    %cst = arith.constant dense<64> : tensor<32x128xi64, #blocked>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<32x128xf16, #blocked>
    %c32_i32 = arith.constant 32 : i32
    %cst_1 = arith.constant dense<64> : tensor<1x32xi64, #blocked1>
    %c0 = arith.constant 0 : index
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %c32 = arith.constant 32 : index
    %c100 = arith.constant 100 : index
    %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %1 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %2 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %3 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %4 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<128x32x!tt.ptr<f16>, #blocked1>
    %5 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<32x128x!tt.ptr<f16>, #blocked>
    %6 = tt.addptr %arg1, %c32_i32 : !tt.ptr<i32>, i32
    %7:5 = scf.for %arg3 = %c0 to %c100 step %c32 iter_args(%arg4 = %4, %arg5 = %3, %arg6 = %6, %arg7 = %cst_2, %arg8 = %5) -> (tensor<128x32x!tt.ptr<f16>, #blocked1>, tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>, !tt.ptr<i32>, tensor<128x128xf32, #mma>, tensor<32x128x!tt.ptr<f16>, #blocked>) {
      %8 = arith.subi %c100, %arg3 : index
      %9 = arith.index_cast %8 : index to i32
      %10 = tt.splat %9 : i32 -> tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
      %11 = arith.cmpi slt, %2, %10 : tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
      %12 = tt.expand_dims %11 {axis = 1 : i32} : tensor<32xi1, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<32x1xi1, #blocked>
      %13 = tt.broadcast %12 : tensor<32x1xi1, #blocked> -> tensor<32x128xi1, #blocked>
      %14 = tt.load %arg8, %13, %cst_0 : tensor<32x128x!tt.ptr<f16>, #blocked>
      %15 = tt.splat %arg6 : !tt.ptr<i32> -> tensor<32x!tt.ptr<i32>, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
      %16 = tt.addptr %15, %0 : tensor<32x!tt.ptr<i32>, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>, tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
      %17 = tt.load %16 : tensor<32x!tt.ptr<i32>, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
      %18 = tt.splat %9 : i32 -> tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
      %19 = arith.cmpi slt, %1, %18 : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
      %20 = tt.expand_dims %19 {axis = 0 : i32} : tensor<32xi1, #triton_gpu.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x32xi1, #blocked1>
      %21 = tt.expand_dims %arg5 {axis = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x32xi32, #blocked1>
      %22 = arith.extsi %21 : tensor<1x32xi32, #blocked1> to tensor<1x32xi64, #blocked1>
      %23 = arith.muli %22, %cst_1 : tensor<1x32xi64, #blocked1>
      %24 = tt.broadcast %23 : tensor<1x32xi64, #blocked1> -> tensor<128x32xi64, #blocked1>
      %25 = tt.broadcast %20 : tensor<1x32xi1, #blocked1> -> tensor<128x32xi1, #blocked1>
      %26 = tt.addptr %arg4, %24 : tensor<128x32x!tt.ptr<f16>, #blocked1>, tensor<128x32xi64, #blocked1>
      %27 = tt.load %26, %25 : tensor<128x32x!tt.ptr<f16>, #blocked1>
      %28 = tt.expand_dims %arg5 {axis = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x32xi32, #blocked1>
      %29 = arith.extsi %28 : tensor<1x32xi32, #blocked1> to tensor<1x32xi64, #blocked1>
      %30 = arith.muli %29, %cst_1 : tensor<1x32xi64, #blocked1>
      %31 = tt.broadcast %30 : tensor<1x32xi64, #blocked1> -> tensor<128x32xi64, #blocked1>
      %32 = tt.addptr %arg4, %31 : tensor<128x32x!tt.ptr<f16>, #blocked1>, tensor<128x32xi64, #blocked1>
      %33 = tt.addptr %arg6, %c32_i32 : !tt.ptr<i32>, i32
      %34 = triton_gpu.convert_layout %27 : tensor<128x32xf16, #blocked1> -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %35 = triton_gpu.convert_layout %14 : tensor<32x128xf16, #blocked> -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %36 = tt.dot %34, %35, %arg7 : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x128xf32, #mma>
      %37 = tt.addptr %arg8, %cst : tensor<32x128x!tt.ptr<f16>, #blocked>, tensor<32x128xi64, #blocked>
      scf.yield %32, %17, %33, %36, %37 : tensor<128x32x!tt.ptr<f16>, #blocked1>, tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>, !tt.ptr<i32>, tensor<128x128xf32, #mma>, tensor<32x128x!tt.ptr<f16>, #blocked>
    }
    tt.return %7#3 : tensor<128x128xf32, #mma>
  }
}

// -----
#blocked = #triton_gpu.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #triton_gpu.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>
#shared = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1], hasLeadingOffset = false}>
#shared1 = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0], hasLeadingOffset = false}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, triton_gpu.target = "cuda:80", "triton_gpu.threads-per-warp" = 32 : i32} {

// CHECK-LABEL:  tt.func @load_two_users
// CHECK:  %{{.*}}:5 = scf.for %[[ARG2:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ARG3:.*]] = %{{.*}}, %[[ARG4:.*]] = %{{.*}}, %[[ARG5:.*]] = %{{.*}}-1_i32, %[[ARG6:.*]] = %{{.*}}, %[[ARG7:.*]] = %{{.*}})

// CHECK:  %[[CMPI_21:.*]] = arith.cmpi slt, %[[ARG2]], %{{.*}}
// CHECK:  %[[SPLAT_22:.*]] = tt.splat %[[CMPI_21]]
// CHECK:  %[[LOAD_23:.*]] = tt.load %{{.*}}, %[[SPLAT_22]]
// CHECK:  %[[ADDI_24:.*]] = arith.addi %[[ARG5]], %{{.*}}
// CHECK:  %[[CMPI_25:.*]] = arith.cmpi slt, %[[ADDI_24]], %{{.*}}
// CHECK:  %[[SELECT_26:.*]] = arith.select %[[CMPI_25]], %[[ADDI_24]], %{{.*}}
// CHECK:  %[[CONVERT_LAYOUT_27:.*]] = triton_gpu.convert_layout %{{.*}}
// CHECK:  %[[LOCAL_LOAD_28:.*]] = triton_gpu.local_load %[[ARG7]]
// CHECK:  %[[DOT_29:.*]] = tt.dot %[[CONVERT_LAYOUT_27]], %[[LOCAL_LOAD_28]], %{{.*}}
// CHECK:  %[[TRUNCF_30:.*]] = arith.truncf %[[DOT_29]]
// CHECK:  %[[CONVERT_LAYOUT_31:.*]] = triton_gpu.convert_layout %[[TRUNCF_30]]
// CHECK:  %[[TRANS_32:.*]] = tt.trans %[[ARG7]] {order = array<i32: 1, 0>}
// CHECK:  %[[LOCAL_LOAD_33:.*]] = triton_gpu.local_load %[[TRANS_32]]
// CHECK:  %[[DOT_34:.*]] = tt.dot %[[CONVERT_LAYOUT_31]], %[[LOCAL_LOAD_33]], %[[ARG4]]
// CHECK:  %[[ADDI_35:.*]] = arith.addi %[[ARG6]], %{{.*}}
// CHECK:  %[[CMPI_36:.*]] = arith.cmpi slt, %[[ADDI_35]], %{{.*}}
// CHECK:  %[[SELECT_37:.*]] = arith.select %[[CMPI_36]], %[[ADDI_35]], %{{.*}}
// CHECK:  %[[MEMDESC_SUBVIEW_38:.*]] = triton_gpu.memdesc_subview %{{.*}}[%[[SELECT_37]], %{{.*}}, %{{.*}}]
// CHECK:  triton_gpu.local_store %[[LOAD_23]], %[[MEMDESC_SUBVIEW_38]]
// CHECK:  scf.yield %[[DOT_29]], %[[DOT_34]], %[[SELECT_26]], %[[SELECT_37]], %[[MEMDESC_SUBVIEW_38]]
// CHECK:  }

  tt.func @load_two_users(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}) -> (tensor<128x16xf32, #mma>, tensor<128x64xf32, #mma>) {
    %c7_i32 = arith.constant 7 : i32
    %0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %c0_i64 = arith.constant 0 : i64
    %2 = tt.addptr %arg0, %c0_i64 : !tt.ptr<f16>, i64
    %cst = arith.constant dense<0> : tensor<1x16xi32, #blocked>
    %3 = tt.splat %2 : !tt.ptr<f16> -> tensor<1x16x!tt.ptr<f16>, #blocked>
    %4 = tt.addptr %3, %cst : tensor<1x16x!tt.ptr<f16>, #blocked>, tensor<1x16xi32, #blocked>
    %5 = tt.broadcast %1 : tensor<64x1xi32, #blocked> -> tensor<64x16xi32, #blocked>
    %6 = tt.broadcast %4 : tensor<1x16x!tt.ptr<f16>, #blocked> -> tensor<64x16x!tt.ptr<f16>, #blocked>
    %7 = tt.addptr %6, %5 : tensor<64x16x!tt.ptr<f16>, #blocked>, tensor<64x16xi32, #blocked>
    %8 = tt.load %7 : tensor<64x16x!tt.ptr<f16>, #blocked>
    %9 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %10 = tt.expand_dims %9 {axis = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
    %11 = tt.addptr %arg1, %c0_i64 : !tt.ptr<f16>, i64
    %cst_0 = arith.constant dense<0> : tensor<128x1xi32, #blocked1>
    %12 = tt.splat %11 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>, #blocked1>
    %13 = tt.addptr %12, %cst_0 : tensor<128x1x!tt.ptr<f16>, #blocked1>, tensor<128x1xi32, #blocked1>
    %14 = tt.broadcast %10 : tensor<1x64xi32, #blocked1> -> tensor<128x64xi32, #blocked1>
    %15 = tt.broadcast %13 : tensor<128x1x!tt.ptr<f16>, #blocked1> -> tensor<128x64x!tt.ptr<f16>, #blocked1>
    %16 = tt.addptr %15, %14 : tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<128x64xi32, #blocked1>
    %17 = tt.load %16 : tensor<128x64x!tt.ptr<f16>, #blocked1>
    %c-1_i32 = arith.constant -1 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<128x16xf32, #mma>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #mma>
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32
    %18 = triton_gpu.local_alloc  : () -> !tt.memdesc<1x64x16xf16, #shared, #triton_gpu.shared_memory, mutable>
    %19 = triton_gpu.memdesc_subview %18[%c0_i32, %c0_i32, %c0_i32] : !tt.memdesc<1x64x16xf16, #shared, #triton_gpu.shared_memory, mutable> -> !tt.memdesc<64x16xf16, #shared, #triton_gpu.shared_memory, mutable>
    triton_gpu.local_store %8, %19 : tensor<64x16xf16, #blocked> -> !tt.memdesc<64x16xf16, #shared, #triton_gpu.shared_memory, mutable>
    %20:5 = scf.for %arg2 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg3 = %cst_1, %arg4 = %cst_2, %arg5 = %c-1_i32, %arg6 = %c0_i32, %arg7 = %19) -> (tensor<128x16xf32, #mma>, tensor<128x64xf32, #mma>, i32, i32, !tt.memdesc<64x16xf16, #shared, #triton_gpu.shared_memory, mutable>)  : i32 {
      %21 = arith.cmpi slt, %arg2, %c7_i32 : i32
      %22 = tt.splat %21 : i1 -> tensor<64x16xi1, #blocked>
      %23 = tt.load %7, %22 : tensor<64x16x!tt.ptr<f16>, #blocked>
      %24 = arith.addi %arg5, %c1_i32 : i32
      %25 = arith.cmpi slt, %24, %c1_i32 : i32
      %26 = arith.select %25, %24, %c0_i32 : i32
      %27 = triton_gpu.convert_layout %17 : tensor<128x64xf16, #blocked1> -> tensor<128x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %28 = triton_gpu.local_load %arg7 : !tt.memdesc<64x16xf16, #shared, #triton_gpu.shared_memory, mutable> -> tensor<64x16xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %29 = tt.dot %27, %28, %cst_1 : tensor<128x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<64x16xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x16xf32, #mma>
      %30 = arith.truncf %29 : tensor<128x16xf32, #mma> to tensor<128x16xf16, #mma>
      %31 = triton_gpu.convert_layout %30 : tensor<128x16xf16, #mma> -> tensor<128x16xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %32 = tt.trans %arg7 {order = array<i32: 1, 0>} : !tt.memdesc<64x16xf16, #shared, #triton_gpu.shared_memory, mutable> -> !tt.memdesc<16x64xf16, #shared1, #triton_gpu.shared_memory>
      %33 = triton_gpu.local_load %32 : !tt.memdesc<16x64xf16, #shared1, #triton_gpu.shared_memory> -> tensor<16x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %34 = tt.dot %31, %33, %arg4 : tensor<128x16xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<16x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x64xf32, #mma>
      %35 = arith.addi %arg6, %c1_i32 : i32
      %36 = arith.cmpi slt, %35, %c1_i32 : i32
      %37 = arith.select %36, %35, %c0_i32 : i32
      %38 = triton_gpu.memdesc_subview %18[%37, %c0_i32, %c0_i32] : !tt.memdesc<1x64x16xf16, #shared, #triton_gpu.shared_memory, mutable> -> !tt.memdesc<64x16xf16, #shared, #triton_gpu.shared_memory, mutable>
      triton_gpu.local_store %23, %38 : tensor<64x16xf16, #blocked> -> !tt.memdesc<64x16xf16, #shared, #triton_gpu.shared_memory, mutable>
      scf.yield %29, %34, %26, %37, %38 : tensor<128x16xf32, #mma>, tensor<128x64xf32, #mma>, i32, i32, !tt.memdesc<64x16xf16, #shared, #triton_gpu.shared_memory, mutable>
    }
    triton_gpu.local_dealloc %18 : !tt.memdesc<1x64x16xf16, #shared, #triton_gpu.shared_memory, mutable>
    tt.return %20#0, %20#1 : tensor<128x16xf32, #mma>, tensor<128x64xf32, #mma>
  }
}

// -----
#blocked = #triton_gpu.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #triton_gpu.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>
#shared = #triton_gpu.shared<{vec = 4, perPhase = 1, maxPhase = 2, order = [0, 1], hasLeadingOffset = false}>
#shared1 = #triton_gpu.shared<{vec = 4, perPhase = 1, maxPhase = 2, order = [1, 0], hasLeadingOffset = false}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, triton_gpu.target = "cuda:80", "triton_gpu.threads-per-warp" = 32 : i32} {

// CHECK-LABEL:  tt.func @load_two_users_incompatible_layouts
// CHECK:  %{{.*}}:5 = scf.for %[[ARG2:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ARG3:.*]] = %{{.*}}, %[[ARG4:.*]] = %{{.*}}, %[[ARG5:.*]] = %{{.*}}-1_i32, %[[ARG6:.*]] = %{{.*}}-1_i32, %[[ARG7:.*]] = %{{.*}})

// CHECK:  %[[CMPI_19:.*]] = arith.cmpi slt, %[[ARG2]], %{{.*}}
// CHECK:  %[[SPLAT_20:.*]] = tt.splat %[[CMPI_19]]
// CHECK:  %[[LOAD_21:.*]] = tt.load %{{.*}}, %[[SPLAT_20]]
// CHECK:  %[[ADDI_22:.*]] = arith.addi %[[ARG5]], %{{.*}}
// CHECK:  %[[CMPI_23:.*]] = arith.cmpi slt, %[[ADDI_22]], %{{.*}}
// CHECK:  %[[SELECT_24:.*]] = arith.select %[[CMPI_23]], %[[ADDI_22]], %{{.*}}
// CHECK:  %[[ADDI_25:.*]] = arith.addi %[[ARG6]], %{{.*}}
// CHECK:  %[[CMPI_26:.*]] = arith.cmpi slt, %[[ADDI_25]], %{{.*}}
// CHECK:  %[[SELECT_27:.*]] = arith.select %[[CMPI_26]], %[[ADDI_25]], %{{.*}}
// CHECK:  %[[CONVERT_LAYOUT_28:.*]] = triton_gpu.convert_layout %{{.*}}
// CHECK:  %[[CONVERT_LAYOUT_29:.*]] = triton_gpu.convert_layout %[[ARG7]]
// CHECK:  %[[DOT_30:.*]] = tt.dot %[[CONVERT_LAYOUT_28]], %[[CONVERT_LAYOUT_29]], %{{.*}}
// CHECK:  %[[TRUNCF_31:.*]] = arith.truncf %[[DOT_30]]
// CHECK:  %[[CONVERT_LAYOUT_32:.*]] = triton_gpu.convert_layout %[[TRUNCF_31]]
// CHECK:  %[[LOCAL_ALLOC_33:.*]] = triton_gpu.local_alloc %[[ARG7]]
// CHECK:  %[[TRANS_34:.*]] = tt.trans %[[LOCAL_ALLOC_33]] {order = array<i32: 1, 0>}
// CHECK:  %[[LOCAL_LOAD_35:.*]] = triton_gpu.local_load %[[TRANS_34]]
// CHECK:  %[[DOT_36:.*]] = tt.dot %[[CONVERT_LAYOUT_32]], %[[LOCAL_LOAD_35]], %[[ARG4]]
// CHECK:  scf.yield %[[DOT_30]], %[[DOT_36]], %[[SELECT_24]], %[[SELECT_27]], %[[LOAD_21]]
// CHECK:  }

  tt.func @load_two_users_incompatible_layouts(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}) -> (tensor<128x16xf32, #mma>, tensor<128x64xf32, #mma>) {
    %c7_i32 = arith.constant 7 : i32
    %0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %c0_i64 = arith.constant 0 : i64
    %2 = tt.addptr %arg0, %c0_i64 : !tt.ptr<f16>, i64
    %cst = arith.constant dense<0> : tensor<1x16xi32, #blocked>
    %3 = tt.splat %2 : !tt.ptr<f16> -> tensor<1x16x!tt.ptr<f16>, #blocked>
    %4 = tt.addptr %3, %cst : tensor<1x16x!tt.ptr<f16>, #blocked>, tensor<1x16xi32, #blocked>
    %5 = tt.broadcast %1 : tensor<64x1xi32, #blocked> -> tensor<64x16xi32, #blocked>
    %6 = tt.broadcast %4 : tensor<1x16x!tt.ptr<f16>, #blocked> -> tensor<64x16x!tt.ptr<f16>, #blocked>
    %7 = tt.addptr %6, %5 : tensor<64x16x!tt.ptr<f16>, #blocked>, tensor<64x16xi32, #blocked>
    %8 = tt.load %7 : tensor<64x16x!tt.ptr<f16>, #blocked>
    %9 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %10 = tt.expand_dims %9 {axis = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
    %11 = tt.addptr %arg1, %c0_i64 : !tt.ptr<f16>, i64
    %cst_0 = arith.constant dense<0> : tensor<128x1xi32, #blocked1>
    %12 = tt.splat %11 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>, #blocked1>
    %13 = tt.addptr %12, %cst_0 : tensor<128x1x!tt.ptr<f16>, #blocked1>, tensor<128x1xi32, #blocked1>
    %14 = tt.broadcast %10 : tensor<1x64xi32, #blocked1> -> tensor<128x64xi32, #blocked1>
    %15 = tt.broadcast %13 : tensor<128x1x!tt.ptr<f16>, #blocked1> -> tensor<128x64x!tt.ptr<f16>, #blocked1>
    %16 = tt.addptr %15, %14 : tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<128x64xi32, #blocked1>
    %17 = tt.load %16 : tensor<128x64x!tt.ptr<f16>, #blocked1>
    %c-1_i32 = arith.constant -1 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<128x16xf32, #mma>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #mma>
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32
    %18:5 = scf.for %arg2 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg3 = %cst_1, %arg4 = %cst_2, %arg5 = %c-1_i32, %arg6 = %c-1_i32, %arg7 = %8) -> (tensor<128x16xf32, #mma>, tensor<128x64xf32, #mma>, i32, i32, tensor<64x16xf16, #blocked>)  : i32 {
      %19 = arith.cmpi slt, %arg2, %c7_i32 : i32
      %20 = tt.splat %19 : i1 -> tensor<64x16xi1, #blocked>
      %21 = tt.load %7, %20 : tensor<64x16x!tt.ptr<f16>, #blocked>
      %22 = arith.addi %arg5, %c1_i32 : i32
      %23 = arith.cmpi slt, %22, %c1_i32 : i32
      %24 = arith.select %23, %22, %c0_i32 : i32
      %25 = arith.addi %arg6, %c1_i32 : i32
      %26 = arith.cmpi slt, %25, %c1_i32 : i32
      %27 = arith.select %26, %25, %c0_i32 : i32
      %28 = triton_gpu.convert_layout %17 : tensor<128x64xf16, #blocked1> -> tensor<128x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %29 = triton_gpu.convert_layout %arg7 : tensor<64x16xf16, #blocked> -> tensor<64x16xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %30 = tt.dot %28, %29, %cst_1 : tensor<128x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<64x16xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x16xf32, #mma>
      %31 = arith.truncf %30 : tensor<128x16xf32, #mma> to tensor<128x16xf16, #mma>
      %32 = triton_gpu.convert_layout %31 : tensor<128x16xf16, #mma> -> tensor<128x16xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %33 = triton_gpu.local_alloc %arg7 : (tensor<64x16xf16, #blocked>) -> !tt.memdesc<64x16xf16, #shared, #triton_gpu.shared_memory>
      %34 = tt.trans %33 {order = array<i32: 1, 0>} : !tt.memdesc<64x16xf16, #shared, #triton_gpu.shared_memory> -> !tt.memdesc<16x64xf16, #shared1, #triton_gpu.shared_memory>
      %35 = triton_gpu.local_load %34 : !tt.memdesc<16x64xf16, #shared1, #triton_gpu.shared_memory> -> tensor<16x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %36 = tt.dot %32, %35, %arg4 : tensor<128x16xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<16x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x64xf32, #mma>
      scf.yield %30, %36, %24, %27, %21 : tensor<128x16xf32, #mma>, tensor<128x64xf32, #mma>, i32, i32, tensor<64x16xf16, #blocked>
    }
    tt.return %18#0, %18#1 : tensor<128x16xf32, #mma>, tensor<128x64xf32, #mma>
  }
}

// -----
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #triton_gpu.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 8]}>
#shared = #triton_gpu.shared<{vec = 4, perPhase = 1, maxPhase = 8, order = [1, 0], hasLeadingOffset = false}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, triton_gpu.target = "cuda:80", "triton_gpu.threads-per-warp" = 32 : i32} {

// CHECK-LABEL:  tt.func public @nested_loops
// CHECK:  scf.for %[[ARG4:.*]] = %{{.*}} to %{{.*}} step %{{.*}}  : i32 {

// CHECK:  %[[MULI_9:.*]] = arith.muli %[[ARG4]], %{{.*}}
// CHECK:  %[[SPLAT_10:.*]] = tt.splat %[[MULI_9]]
// CHECK:  %[[ADDI_11:.*]] = arith.addi %[[SPLAT_10]], %{{.*}}
// CHECK:  %[[EXPAND_DIMS_12:.*]] = tt.expand_dims %[[ADDI_11]] {axis = 0 : i32}
// CHECK:  %[[BROADCAST_13:.*]] = tt.broadcast %[[EXPAND_DIMS_12]]
// CHECK:  %[[ADDPTR_14:.*]] = tt.addptr %{{.*}}, %[[BROADCAST_13]]
// CHECK:  %[[LOAD_15:.*]] = tt.load %[[ADDPTR_14]]
// CHECK:  %[[EXPAND_DIMS_16:.*]] = tt.expand_dims %{{.*}} {axis = 0 : i32}
// CHECK:  %[[SPLAT_17:.*]] = tt.splat %[[MULI_9]]
// CHECK:  %[[ADDI_18:.*]] = arith.addi %[[SPLAT_17]], %{{.*}}
// CHECK:  %[[EXPAND_DIMS_19:.*]] = tt.expand_dims %[[ADDI_18]] {axis = 1 : i32}
// CHECK:  %[[MULI_20:.*]] = arith.muli %[[EXPAND_DIMS_19]], %{{.*}}
// CHECK:  %[[ADDPTR_21:.*]] = tt.addptr %{{.*}}, %[[MULI_20]]
// CHECK:  %[[BROADCAST_22:.*]] = tt.broadcast %[[EXPAND_DIMS_16]]
// CHECK:  %[[BROADCAST_23:.*]] = tt.broadcast %[[ADDPTR_21]]
// CHECK:  %[[ADDPTR_24:.*]] = tt.addptr %[[BROADCAST_23]], %[[BROADCAST_22]]
// CHECK:  %[[LOAD_25:.*]] = tt.load %[[ADDPTR_24]]
// CHECK:  %[[ADDPTR_26:.*]] = tt.addptr %{{.*}}, %[[MULI_20]]
// CHECK:  %[[BROADCAST_27:.*]] = tt.broadcast %[[ADDPTR_26]]
// CHECK:  %[[LOCAL_ALLOC_28:.*]] = triton_gpu.local_alloc
// CHECK:  %[[MEMDESC_SUBVIEW_29:.*]] = triton_gpu.memdesc_subview %[[LOCAL_ALLOC_28]][%{{.*}}, %{{.*}}, %{{.*}}]
// CHECK:  triton_gpu.local_store %[[LOAD_25]], %[[MEMDESC_SUBVIEW_29]]
// CHECK:  %{{.*}}:4 = scf.for %[[ARG5:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ARG6:.*]] = %{{.*}}-1_i32, %[[ARG7:.*]] = %{{.*}}, %[[ARG8:.*]] = %[[MEMDESC_SUBVIEW_29]], %[[ARG9:.*]] = %[[BROADCAST_22]])

// CHECK:  %[[CMPI_31:.*]] = arith.cmpi slt, %[[ARG5]], %{{.*}}
// CHECK:  %[[ADDI_32:.*]] = arith.addi %[[ARG5]], %{{.*}}
// CHECK:  %[[MULI_33:.*]] = arith.muli %[[ADDI_32]], %{{.*}}
// CHECK:  %[[SPLAT_34:.*]] = tt.splat %[[MULI_33]]
// CHECK:  %[[ADDI_35:.*]] = arith.addi %[[SPLAT_34]], %{{.*}}
// CHECK:  %[[EXPAND_DIMS_36:.*]] = tt.expand_dims %[[ADDI_35]] {axis = 0 : i32}
// CHECK:  %[[BROADCAST_37:.*]] = tt.broadcast %[[EXPAND_DIMS_36]]
// CHECK:  %[[SPLAT_38:.*]] = tt.splat %[[CMPI_31]]
// CHECK:  %[[ADDPTR_39:.*]] = tt.addptr %[[BROADCAST_23]], %[[BROADCAST_37]]
// CHECK:  %[[LOAD_40:.*]] = tt.load %[[ADDPTR_39]], %[[SPLAT_38]]
// CHECK:  %[[ADDI_41:.*]] = arith.addi %[[ARG6]], %{{.*}}
// CHECK:  %[[CMPI_42:.*]] = arith.cmpi slt, %[[ADDI_41]], %{{.*}}
// CHECK:  %[[SELECT_43:.*]] = arith.select %[[CMPI_42]], %[[ADDI_41]], %{{.*}}
// CHECK:  %[[LOCAL_LOAD_44:.*]] = triton_gpu.local_load %[[ARG8]]
// CHECK:  %[[CONVERT_LAYOUT_45:.*]] = triton_gpu.convert_layout %[[LOAD_15]]
// CHECK:  %[[DOT_46:.*]] = tt.dot %[[LOCAL_LOAD_44]], %[[CONVERT_LAYOUT_45]], %{{.*}}
// CHECK:  %[[ADDPTR_47:.*]] = tt.addptr %[[BROADCAST_27]], %[[ARG9]]
// CHECK:  %[[CONVERT_LAYOUT_48:.*]] = triton_gpu.convert_layout %[[DOT_46]]
// CHECK:  tt.store %[[ADDPTR_47]], %[[CONVERT_LAYOUT_48]]
// CHECK:  %[[ADDI_49:.*]] = arith.addi %[[ARG7]], %{{.*}}
// CHECK:  %[[CMPI_50:.*]] = arith.cmpi slt, %[[ADDI_49]], %{{.*}}
// CHECK:  %[[SELECT_51:.*]] = arith.select %[[CMPI_50]], %[[ADDI_49]], %{{.*}}
// CHECK:  %[[MEMDESC_SUBVIEW_52:.*]] = triton_gpu.memdesc_subview %[[LOCAL_ALLOC_28]][%[[SELECT_51]], %{{.*}}, %{{.*}}]
// CHECK:  triton_gpu.local_store %[[LOAD_40]], %[[MEMDESC_SUBVIEW_52]]
// CHECK:  scf.yield %[[SELECT_43]], %[[SELECT_51]], %[[MEMDESC_SUBVIEW_52]], %[[BROADCAST_37]]
// CHECK:  }

  tt.func public @nested_loops(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c9_i32 = arith.constant 9 : i32
    %c-1_i32 = arith.constant -1 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %cst_0 = arith.constant dense<320> : tensor<32x1xi32, #blocked>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c32_i32 = arith.constant 32 : i32
    %c10_i32 = arith.constant 10 : i32
    %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %1 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %2 = tt.expand_dims %1 {axis = 1 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<32x1xi32, #blocked>
    %3 = arith.muli %2, %cst_0 : tensor<32x1xi32, #blocked>
    %4 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<32x1x!tt.ptr<f32>, #blocked>
    %5 = tt.addptr %4, %3 : tensor<32x1x!tt.ptr<f32>, #blocked>, tensor<32x1xi32, #blocked>
    %6 = tt.broadcast %5 : tensor<32x1x!tt.ptr<f32>, #blocked> -> tensor<32x32x!tt.ptr<f32>, #blocked>
    %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x1x!tt.ptr<f32>, #blocked>
    %8 = tt.splat %arg3 : !tt.ptr<f32> -> tensor<32x1x!tt.ptr<f32>, #blocked>
    scf.for %arg4 = %c0_i32 to %c10_i32 step %c1_i32  : i32 {
      %9 = arith.muli %arg4, %c32_i32 : i32
      %10 = tt.expand_dims %0 {axis = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi32, #blocked>
      %11 = tt.splat %9 : i32 -> tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
      %12 = arith.addi %11, %1 : tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
      %13 = tt.expand_dims %12 {axis = 1 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<32x1xi32, #blocked>
      %14 = arith.muli %13, %cst_0 : tensor<32x1xi32, #blocked>
      %15 = tt.addptr %7, %14 : tensor<32x1x!tt.ptr<f32>, #blocked>, tensor<32x1xi32, #blocked>
      %16 = tt.broadcast %10 : tensor<1x32xi32, #blocked> -> tensor<32x32xi32, #blocked>
      %17 = tt.broadcast %15 : tensor<32x1x!tt.ptr<f32>, #blocked> -> tensor<32x32x!tt.ptr<f32>, #blocked>
      %18 = tt.addptr %17, %16 : tensor<32x32x!tt.ptr<f32>, #blocked>, tensor<32x32xi32, #blocked>
      %19 = tt.load %18 : tensor<32x32x!tt.ptr<f32>, #blocked>
      %20 = tt.splat %9 : i32 -> tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
      %21 = arith.addi %20, %0 : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
      %22 = tt.expand_dims %21 {axis = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi32, #blocked>
      %23 = tt.broadcast %22 : tensor<1x32xi32, #blocked> -> tensor<32x32xi32, #blocked>
      %24 = tt.addptr %6, %23 : tensor<32x32x!tt.ptr<f32>, #blocked>, tensor<32x32xi32, #blocked>
      %25 = tt.load %24 : tensor<32x32x!tt.ptr<f32>, #blocked>
      %26 = tt.addptr %8, %14 : tensor<32x1x!tt.ptr<f32>, #blocked>, tensor<32x1xi32, #blocked>
      %27 = tt.broadcast %26 : tensor<32x1x!tt.ptr<f32>, #blocked> -> tensor<32x32x!tt.ptr<f32>, #blocked>
      %28 = triton_gpu.local_alloc  : () -> !tt.memdesc<1x32x32xf32, #shared, #triton_gpu.shared_memory, mutable>
      %29 = triton_gpu.memdesc_subview %28[%c0_i32, %c0_i32, %c0_i32] : !tt.memdesc<1x32x32xf32, #shared, #triton_gpu.shared_memory, mutable> -> !tt.memdesc<32x32xf32, #shared, #triton_gpu.shared_memory, mutable>
      triton_gpu.local_store %19, %29 : tensor<32x32xf32, #blocked> -> !tt.memdesc<32x32xf32, #shared, #triton_gpu.shared_memory, mutable>
      %30:4 = scf.for %arg5 = %c0_i32 to %c10_i32 step %c1_i32 iter_args(%arg6 = %c-1_i32, %arg7 = %c0_i32, %arg8 = %29, %arg9 = %16) -> (i32, i32, !tt.memdesc<32x32xf32, #shared, #triton_gpu.shared_memory, mutable>, tensor<32x32xi32, #blocked>)  : i32 {
        %31 = arith.cmpi slt, %arg5, %c9_i32 : i32
        %32 = arith.addi %arg5, %c1_i32 : i32
        %33 = arith.muli %32, %c32_i32 : i32
        %34 = tt.splat %33 : i32 -> tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
        %35 = arith.addi %34, %0 : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
        %36 = tt.expand_dims %35 {axis = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi32, #blocked>
        %37 = tt.broadcast %36 : tensor<1x32xi32, #blocked> -> tensor<32x32xi32, #blocked>
        %38 = tt.splat %31 : i1 -> tensor<32x32xi1, #blocked>
        %39 = tt.addptr %17, %37 : tensor<32x32x!tt.ptr<f32>, #blocked>, tensor<32x32xi32, #blocked>
        %40 = tt.load %39, %38 : tensor<32x32x!tt.ptr<f32>, #blocked>
        %41 = arith.addi %arg6, %c1_i32 : i32
        %42 = arith.cmpi slt, %41, %c1_i32 : i32
        %43 = arith.select %42, %41, %c0_i32 : i32
        %44 = triton_gpu.local_load %arg8 : !tt.memdesc<32x32xf32, #shared, #triton_gpu.shared_memory, mutable> -> tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
        %45 = triton_gpu.convert_layout %25 : tensor<32x32xf32, #blocked> -> tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>
        %46 = tt.dot %44, %45, %cst : tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>> -> tensor<32x32xf32, #mma>
        %47 = tt.addptr %27, %arg9 : tensor<32x32x!tt.ptr<f32>, #blocked>, tensor<32x32xi32, #blocked>
        %48 = triton_gpu.convert_layout %46 : tensor<32x32xf32, #mma> -> tensor<32x32xf32, #blocked>
        tt.store %47, %48 : tensor<32x32x!tt.ptr<f32>, #blocked>
        %49 = arith.addi %arg7, %c1_i32 : i32
        %50 = arith.cmpi slt, %49, %c1_i32 : i32
        %51 = arith.select %50, %49, %c0_i32 : i32
        %52 = triton_gpu.memdesc_subview %28[%51, %c0_i32, %c0_i32] : !tt.memdesc<1x32x32xf32, #shared, #triton_gpu.shared_memory, mutable> -> !tt.memdesc<32x32xf32, #shared, #triton_gpu.shared_memory, mutable>
        triton_gpu.local_store %40, %52 : tensor<32x32xf32, #blocked> -> !tt.memdesc<32x32xf32, #shared, #triton_gpu.shared_memory, mutable>
        scf.yield %43, %51, %52, %37 : i32, i32, !tt.memdesc<32x32xf32, #shared, #triton_gpu.shared_memory, mutable>, tensor<32x32xi32, #blocked>
      }
      triton_gpu.local_dealloc %28 : !tt.memdesc<1x32x32xf32, #shared, #triton_gpu.shared_memory, mutable>
    }
    tt.return
  }
}

// -----
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [2, 2], order = [0, 1]}>
#mma = #triton_gpu.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>
#shared = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 4, order = [0, 1], hasLeadingOffset = false}>
#shared1 = #triton_gpu.shared<{vec = 4, perPhase = 1, maxPhase = 8, order = [0, 1], hasLeadingOffset = false}>
#shared2 = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 4, order = [1, 0], hasLeadingOffset = false}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, triton_gpu.target = "cuda:80", "triton_gpu.threads-per-warp" = 32 : i32} {

// CHECK-LABEL:  tt.func public @_jagged_hstu_attn_fwd_0d1d2d3d4d5de
// CHECK:  %{{.*}}:5 = scf.for %[[ARG6:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ARG7:.*]] = %{{.*}}, %[[ARG8:.*]] = %{{.*}}-1_i32, %[[ARG9:.*]] = %{{.*}}, %[[ARG10:.*]] = %{{.*}}, %[[ARG11:.*]] = %{{.*}})

// CHECK:  %[[CMPI_76:.*]] = arith.cmpi slt, %[[ARG6]], %{{.*}}
// CHECK:  %[[SPLAT_77:.*]] = tt.splat %[[CMPI_76]]
// CHECK:  %[[LOAD_78:.*]] = tt.load %{{.*}}, %[[SPLAT_77]]
// CHECK:  %[[SPLAT_79:.*]] = tt.splat %[[CMPI_76]]
// CHECK:  %[[LOAD_80:.*]] = tt.load %{{.*}}, %[[SPLAT_79]]
// CHECK:  %[[ADDI_81:.*]] = arith.addi %[[ARG8]], %{{.*}}
// CHECK:  %[[CMPI_82:.*]] = arith.cmpi slt, %[[ADDI_81]], %{{.*}}
// CHECK:  %[[SELECT_83:.*]] = arith.select %[[CMPI_82]], %[[ADDI_81]], %{{.*}}
// CHECK:  %[[CONVERT_LAYOUT_84:.*]] = triton_gpu.convert_layout %{{.*}}
// CHECK:  %[[TRANS_85:.*]] = tt.trans %[[ARG10]] {order = array<i32: 1, 0>}
// CHECK:  %[[LOCAL_LOAD_86:.*]] = triton_gpu.local_load %[[TRANS_85]]
// CHECK:  %[[DOT_87:.*]] = tt.dot %[[CONVERT_LAYOUT_84]], %[[LOCAL_LOAD_86]], %{{.*}}
// CHECK:  %[[CONVERT_LAYOUT_88:.*]] = triton_gpu.convert_layout %[[DOT_87]]
// CHECK:  %[[LOCAL_LOAD_89:.*]] = triton_gpu.local_load %[[ARG11]]
// CHECK:  %[[DOT_90:.*]] = tt.dot %[[CONVERT_LAYOUT_88]], %[[LOCAL_LOAD_89]], %[[ARG7]]
// CHECK:  %[[ADDI_91:.*]] = arith.addi %[[ARG9]], %{{.*}}
// CHECK:  %[[CMPI_92:.*]] = arith.cmpi slt, %[[ADDI_91]], %{{.*}}
// CHECK:  %[[SELECT_93:.*]] = arith.select %[[CMPI_92]], %[[ADDI_91]], %{{.*}}
// CHECK:  %[[MEMDESC_SUBVIEW_94:.*]] = triton_gpu.memdesc_subview %{{.*}}[%[[SELECT_93]], %{{.*}}, %{{.*}}]
// CHECK:  triton_gpu.local_store %[[LOAD_78]], %[[MEMDESC_SUBVIEW_94]]
// CHECK:  %[[MEMDESC_SUBVIEW_95:.*]] = triton_gpu.memdesc_subview %{{.*}}[%[[SELECT_93]], %{{.*}}, %{{.*}}]
// CHECK:  triton_gpu.local_store %[[LOAD_80]], %[[MEMDESC_SUBVIEW_95]]
// CHECK:  scf.yield %[[DOT_90]], %[[SELECT_83]], %[[SELECT_93]], %[[MEMDESC_SUBVIEW_94]], %[[MEMDESC_SUBVIEW_95]]
// CHECK:  }
// CHECK:  triton_gpu.local_dealloc %{{.*}}
// CHECK:  triton_gpu.local_dealloc %{{.*}}
// CHECK:  %[[BROADCAST_70:.*]] = tt.broadcast %{{.*}}
// CHECK:  %[[BROADCAST_71:.*]] = tt.broadcast %{{.*}}
// CHECK:  %[[ADDI_72:.*]] = arith.addi %[[BROADCAST_70]], %[[BROADCAST_71]]
// CHECK:  %[[SPLAT_73:.*]] = tt.splat %{{.*}}
// CHECK:  %[[ADDPTR_74:.*]] = tt.addptr %[[SPLAT_73]], %[[ADDI_72]]
// CHECK:  %[[CONVERT_LAYOUT_75:.*]] = triton_gpu.convert_layout %{{.*}}#0
// CHECK:  tt.store %[[ADDPTR_74]], %[[CONVERT_LAYOUT_75]]

  tt.func public @_jagged_hstu_attn_fwd_0d1d2d3d4d5de(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}) attributes {noinline = false} {
    %c1_i32 = arith.constant 1 : i32
    %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %1 = tt.splat %arg5 : i32 -> tensor<1x32xi32, #blocked>
    %2 = tt.expand_dims %0 {axis = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi32, #blocked>
    %3 = arith.muli %2, %1 : tensor<1x32xi32, #blocked>
    %4 = arith.extsi %3 : tensor<1x32xi32, #blocked> to tensor<1x32xi64, #blocked>
    %5 = tt.get_program_id y : i32
    %6 = arith.muli %5, %arg5 : i32
    %7 = arith.extsi %6 : i32 to i64
    %8 = arith.extsi %arg5 : i32 to i64
    %9 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %10 = tt.expand_dims %9 {axis = 1 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<32x1xi32, #blocked>
    %11 = tt.load %arg3 : !tt.ptr<i64>
    %12 = arith.extsi %10 : tensor<32x1xi32, #blocked> to tensor<32x1xi64, #blocked>
    %13 = tt.splat %11 : i64 -> tensor<32x1xi64, #blocked>
    %14 = tt.splat %8 : i64 -> tensor<32x1xi64, #blocked>
    %15 = arith.addi %13, %12 : tensor<32x1xi64, #blocked>
    %16 = tt.splat %7 : i64 -> tensor<32x1xi64, #blocked>
    %17 = arith.muli %15, %14 : tensor<32x1xi64, #blocked>
    %18 = arith.addi %17, %16 : tensor<32x1xi64, #blocked>
    %19 = tt.broadcast %4 : tensor<1x32xi64, #blocked> -> tensor<32x32xi64, #blocked>
    %20 = tt.broadcast %18 : tensor<32x1xi64, #blocked> -> tensor<32x32xi64, #blocked>
    %21 = arith.addi %20, %19 : tensor<32x32xi64, #blocked>
    %22 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #blocked>
    %23 = tt.addptr %22, %21 : tensor<32x32x!tt.ptr<f32>, #blocked>, tensor<32x32xi64, #blocked>
    %24 = tt.load %23 : tensor<32x32x!tt.ptr<f32>, #blocked>
    %25 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %26 = tt.splat %arg5 : i32 -> tensor<1x64xi32, #blocked>
    %27 = tt.expand_dims %25 {axis = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>
    %28 = arith.muli %27, %26 : tensor<1x64xi32, #blocked>
    %29 = arith.extsi %28 : tensor<1x64xi32, #blocked> to tensor<1x64xi64, #blocked>
    %30 = tt.broadcast %29 : tensor<1x64xi64, #blocked> -> tensor<32x64xi64, #blocked>
    %31 = tt.broadcast %18 : tensor<32x1xi64, #blocked> -> tensor<32x64xi64, #blocked>
    %32 = arith.addi %31, %30 : tensor<32x64xi64, #blocked>
    %33 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<32x64x!tt.ptr<f32>, #blocked>
    %34 = tt.addptr %33, %32 : tensor<32x64x!tt.ptr<f32>, #blocked>, tensor<32x64xi64, #blocked>
    %35 = tt.load %34 : tensor<32x64x!tt.ptr<f32>, #blocked>
    %36 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %37 = tt.splat %arg5 : i32 -> tensor<1x64xi32, #blocked1>
    %38 = tt.expand_dims %36 {axis = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
    %39 = arith.muli %38, %37 : tensor<1x64xi32, #blocked1>
    %40 = arith.extsi %39 : tensor<1x64xi32, #blocked1> to tensor<1x64xi64, #blocked1>
    %c64_i32 = arith.constant 64 : i32
    %41 = tt.get_program_id x : i32
    %42 = arith.muli %41, %c64_i32 : i32
    %43 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %44 = tt.splat %42 : i32 -> tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %45 = arith.addi %44, %43 : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %46 = tt.expand_dims %45 {axis = 1 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>> -> tensor<64x1xi32, #blocked1>
    %47 = arith.extsi %46 : tensor<64x1xi32, #blocked1> to tensor<64x1xi64, #blocked1>
    %48 = tt.splat %11 : i64 -> tensor<64x1xi64, #blocked1>
    %49 = tt.splat %8 : i64 -> tensor<64x1xi64, #blocked1>
    %50 = arith.addi %48, %47 : tensor<64x1xi64, #blocked1>
    %51 = tt.splat %7 : i64 -> tensor<64x1xi64, #blocked1>
    %52 = arith.muli %50, %49 : tensor<64x1xi64, #blocked1>
    %53 = arith.addi %52, %51 : tensor<64x1xi64, #blocked1>
    %54 = tt.broadcast %40 : tensor<1x64xi64, #blocked1> -> tensor<64x64xi64, #blocked1>
    %55 = tt.broadcast %53 : tensor<64x1xi64, #blocked1> -> tensor<64x64xi64, #blocked1>
    %56 = arith.addi %55, %54 : tensor<64x64xi64, #blocked1>
    %57 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x64x!tt.ptr<f32>, #blocked1>
    %58 = tt.addptr %57, %56 : tensor<64x64x!tt.ptr<f32>, #blocked1>, tensor<64x64xi64, #blocked1>
    %59 = tt.load %58 : tensor<64x64x!tt.ptr<f32>, #blocked1>
    %c-1_i32 = arith.constant -1 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<64x32xf32, #mma>
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %60 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %61 = tt.expand_dims %60 {axis = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x32xi32, #blocked1>
    %62 = tt.splat %arg5 : i32 -> tensor<1x32xi32, #blocked1>
    %63 = arith.muli %61, %62 : tensor<1x32xi32, #blocked1>
    %64 = arith.extsi %63 : tensor<1x32xi32, #blocked1> to tensor<1x32xi64, #blocked1>
    %65 = triton_gpu.local_alloc  : () -> !tt.memdesc<1x32x64xf32, #shared, #triton_gpu.shared_memory, mutable>
    %66 = triton_gpu.local_alloc  : () -> !tt.memdesc<1x32x32xf32, #shared1, #triton_gpu.shared_memory, mutable>
    %67 = triton_gpu.memdesc_subview %65[%c0_i32, %c0_i32, %c0_i32] : !tt.memdesc<1x32x64xf32, #shared, #triton_gpu.shared_memory, mutable> -> !tt.memdesc<32x64xf32, #shared, #triton_gpu.shared_memory, mutable>
    triton_gpu.local_store %35, %67 : tensor<32x64xf32, #blocked> -> !tt.memdesc<32x64xf32, #shared, #triton_gpu.shared_memory, mutable>
    %68 = triton_gpu.memdesc_subview %66[%c0_i32, %c0_i32, %c0_i32] : !tt.memdesc<1x32x32xf32, #shared1, #triton_gpu.shared_memory, mutable> -> !tt.memdesc<32x32xf32, #shared1, #triton_gpu.shared_memory, mutable>
    triton_gpu.local_store %24, %68 : tensor<32x32xf32, #blocked> -> !tt.memdesc<32x32xf32, #shared1, #triton_gpu.shared_memory, mutable>
    %69:5 = scf.for %arg6 = %c0_i32 to %c64_i32 step %c32_i32 iter_args(%arg7 = %cst, %arg8 = %c-1_i32, %arg9 = %c0_i32, %arg10 = %67, %arg11 = %68) -> (tensor<64x32xf32, #mma>, i32, i32, !tt.memdesc<32x64xf32, #shared, #triton_gpu.shared_memory, mutable>, !tt.memdesc<32x32xf32, #shared1, #triton_gpu.shared_memory, mutable>)  : i32 {
      %76 = arith.cmpi slt, %arg6, %c32_i32 : i32
      %77 = tt.splat %76 : i1 -> tensor<32x32xi1, #blocked>
      %78 = tt.load %23, %77 : tensor<32x32x!tt.ptr<f32>, #blocked>
      %79 = tt.splat %76 : i1 -> tensor<32x64xi1, #blocked>
      %80 = tt.load %34, %79 : tensor<32x64x!tt.ptr<f32>, #blocked>
      %81 = arith.addi %arg8, %c1_i32 : i32
      %82 = arith.cmpi slt, %81, %c1_i32 : i32
      %83 = arith.select %82, %81, %c0_i32 : i32
      %84 = triton_gpu.convert_layout %59 : tensor<64x64xf32, #blocked1> -> tensor<64x64xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
      %85 = tt.trans %arg10 {order = array<i32: 1, 0>} : !tt.memdesc<32x64xf32, #shared, #triton_gpu.shared_memory, mutable> -> !tt.memdesc<64x32xf32, #shared2, #triton_gpu.shared_memory>
      %86 = triton_gpu.local_load %85 : !tt.memdesc<64x32xf32, #shared2, #triton_gpu.shared_memory> -> tensor<64x32xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>
      %87 = tt.dot %84, %86, %cst : tensor<64x64xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<64x32xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>> -> tensor<64x32xf32, #mma>
      %88 = triton_gpu.convert_layout %87 : tensor<64x32xf32, #mma> -> tensor<64x32xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
      %89 = triton_gpu.local_load %arg11 : !tt.memdesc<32x32xf32, #shared1, #triton_gpu.shared_memory, mutable> -> tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>
      %90 = tt.dot %88, %89, %arg7 : tensor<64x32xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>> -> tensor<64x32xf32, #mma>
      %91 = arith.addi %arg9, %c1_i32 : i32
      %92 = arith.cmpi slt, %91, %c1_i32 : i32
      %93 = arith.select %92, %91, %c0_i32 : i32
      %94 = triton_gpu.memdesc_subview %65[%93, %c0_i32, %c0_i32] : !tt.memdesc<1x32x64xf32, #shared, #triton_gpu.shared_memory, mutable> -> !tt.memdesc<32x64xf32, #shared, #triton_gpu.shared_memory, mutable>
      triton_gpu.local_store %80, %94 : tensor<32x64xf32, #blocked> -> !tt.memdesc<32x64xf32, #shared, #triton_gpu.shared_memory, mutable>
      %95 = triton_gpu.memdesc_subview %66[%93, %c0_i32, %c0_i32] : !tt.memdesc<1x32x32xf32, #shared1, #triton_gpu.shared_memory, mutable> -> !tt.memdesc<32x32xf32, #shared1, #triton_gpu.shared_memory, mutable>
      triton_gpu.local_store %78, %95 : tensor<32x32xf32, #blocked> -> !tt.memdesc<32x32xf32, #shared1, #triton_gpu.shared_memory, mutable>
      scf.yield %90, %83, %93, %94, %95 : tensor<64x32xf32, #mma>, i32, i32, !tt.memdesc<32x64xf32, #shared, #triton_gpu.shared_memory, mutable>, !tt.memdesc<32x32xf32, #shared1, #triton_gpu.shared_memory, mutable>
    }
    triton_gpu.local_dealloc %65 : !tt.memdesc<1x32x64xf32, #shared, #triton_gpu.shared_memory, mutable>
    triton_gpu.local_dealloc %66 : !tt.memdesc<1x32x32xf32, #shared1, #triton_gpu.shared_memory, mutable>
    %70 = tt.broadcast %53 : tensor<64x1xi64, #blocked1> -> tensor<64x32xi64, #blocked1>
    %71 = tt.broadcast %64 : tensor<1x32xi64, #blocked1> -> tensor<64x32xi64, #blocked1>
    %72 = arith.addi %70, %71 : tensor<64x32xi64, #blocked1>
    %73 = tt.splat %arg4 : !tt.ptr<f32> -> tensor<64x32x!tt.ptr<f32>, #blocked1>
    %74 = tt.addptr %73, %72 : tensor<64x32x!tt.ptr<f32>, #blocked1>, tensor<64x32xi64, #blocked1>
    %75 = triton_gpu.convert_layout %69#0 : tensor<64x32xf32, #mma> -> tensor<64x32xf32, #blocked1>
    tt.store %74, %75 : tensor<64x32x!tt.ptr<f32>, #blocked1>
    tt.return
  }
}

// -----
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #triton_gpu.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = []}>
#shared = #triton_gpu.shared<{vec = 8, perPhase = 4, maxPhase = 2, order = [1, 0], hasLeadingOffset = false}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, triton_gpu.target = "cuda:86", "triton_gpu.threads-per-warp" = 32 : i32} {

// CHECK-LABEL:  tt.func @indirect_load_shared_layout
// CHECK:  %{{.*}}:8 = scf.for %[[ARG6:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ARG7:.*]] = %{{.*}}, %[[ARG8:.*]] = %{{.*}}, %[[ARG9:.*]] = %{{.*}}, %[[ARG10:.*]] = %{{.*}}-1_i32, %[[ARG11:.*]] = %{{.*}}, %[[ARG12:.*]] = %{{.*}}, %[[ARG13:.*]] = %{{.*}}, %[[ARG14:.*]] = %{{.*}})

// CHECK:  %[[SUBI_20:.*]] = arith.subi %{{.*}}, %{{.*}}
// CHECK:  %[[SUBI_21:.*]] = arith.subi %{{.*}}, %{{.*}}
// CHECK:  %[[CMPI_22:.*]] = arith.cmpi slt, %[[ARG6]], %[[SUBI_21]]
// CHECK:  %[[SPLAT_23:.*]] = tt.splat %[[CMPI_22]]
// CHECK:  %[[ADDPTR_24:.*]] = tt.addptr %[[ARG8]], %{{.*}}
// CHECK:  %[[LOAD_25:.*]] = tt.load %[[ADDPTR_24]], %[[SPLAT_23]]
// CHECK:  %[[EXPAND_DIMS_26:.*]] = tt.expand_dims %[[ARG14]] {axis = 1 : i32}
// CHECK:  %[[BROADCAST_27:.*]] = tt.broadcast %[[EXPAND_DIMS_26]]
// CHECK:  %[[MULI_28:.*]] = arith.muli %{{.*}}, %[[BROADCAST_27]]
// CHECK:  %[[SPLAT_29:.*]] = tt.splat %[[CMPI_22]]
// CHECK:  %[[ADDPTR_30:.*]] = tt.addptr %{{.*}}, %[[MULI_28]]
// CHECK:  %[[LOAD_31:.*]] = tt.load %[[ADDPTR_30]], %[[SPLAT_29]]
// CHECK:  %[[CMPI_32:.*]] = arith.cmpi slt, %[[ARG6]], %[[SUBI_20]]
// CHECK:  %[[SPLAT_33:.*]] = tt.splat %[[CMPI_32]]
// CHECK:  %[[ADDPTR_34:.*]] = tt.addptr %[[ARG9]], %{{.*}}
// CHECK:  %[[LOAD_35:.*]] = tt.load %[[ADDPTR_34]], %[[SPLAT_33]]
// CHECK:  %[[ADDI_36:.*]] = arith.addi %[[ARG10]], %{{.*}}
// CHECK:  %[[CMPI_37:.*]] = arith.cmpi slt, %[[ADDI_36]], %{{.*}}
// CHECK:  %[[SELECT_38:.*]] = arith.select %[[CMPI_37]], %[[ADDI_36]], %{{.*}}
// CHECK:  %[[LOCAL_LOAD_39:.*]] = triton_gpu.local_load %[[ARG12]]
// CHECK:  %[[LOCAL_LOAD_40:.*]] = triton_gpu.local_load %[[ARG13]]
// CHECK:  %[[DOT_41:.*]] = tt.dot %[[LOCAL_LOAD_39]], %[[LOCAL_LOAD_40]], %[[ARG7]]
// CHECK:  %[[ADDI_42:.*]] = arith.addi %[[ARG11]], %{{.*}}
// CHECK:  %[[CMPI_43:.*]] = arith.cmpi slt, %[[ADDI_42]], %{{.*}}
// CHECK:  %[[SELECT_44:.*]] = arith.select %[[CMPI_43]], %[[ADDI_42]], %{{.*}}
// CHECK:  %[[MEMDESC_SUBVIEW_45:.*]] = triton_gpu.memdesc_subview %{{.*}}[%[[SELECT_44]], %{{.*}}, %{{.*}}]
// CHECK:  triton_gpu.local_store %[[LOAD_25]], %[[MEMDESC_SUBVIEW_45]]
// CHECK:  %[[MEMDESC_SUBVIEW_46:.*]] = triton_gpu.memdesc_subview %{{.*}}[%[[SELECT_44]], %{{.*}}, %{{.*}}]
// CHECK:  triton_gpu.local_store %[[LOAD_31]], %[[MEMDESC_SUBVIEW_46]]
// CHECK:  scf.yield %[[DOT_41]], %[[ADDPTR_24]], %[[ADDPTR_34]], %[[SELECT_38]], %[[SELECT_44]], %[[MEMDESC_SUBVIEW_45]], %[[MEMDESC_SUBVIEW_46]], %[[LOAD_35]]
// CHECK:  }

  tt.func @indirect_load_shared_layout(%arg0: tensor<16x16xi64, #blocked> {tt.constancy = 16 : i32, tt.divisibility = 16 : i32}, %arg1: index, %arg2: tensor<16x16x!tt.ptr<f16>, #blocked1> {tt.contiguity = 2 : i32, tt.divisibility = 16 : i32}, %arg3: tensor<16x!tt.ptr<i64>, #triton_gpu.slice<{dim = 1, parent = #blocked}>>, %arg4: tensor<16x16xi32, #blocked1> {tt.constancy = 16 : i32, tt.divisibility = 16 : i32}, %arg5: tensor<16x16x!tt.ptr<f16>, #blocked> {tt.contiguity = 16 : i32, tt.divisibility = 16 : i32}) -> tensor<16x16xf32, #mma> {
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %0 = arith.cmpi sgt, %arg1, %c1 : index
    %cst = arith.constant dense<1> : tensor<16xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %1 = tt.splat %0 : i1 -> tensor<16xi1, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %2 = tt.addptr %arg3, %cst : tensor<16x!tt.ptr<i64>, #triton_gpu.slice<{dim = 1, parent = #blocked}>>, tensor<16xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %3 = tt.load %2, %1 : tensor<16x!tt.ptr<i64>, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %c0 = arith.constant 0 : index
    %4 = arith.cmpi sgt, %arg1, %c0 : index
    %5 = tt.splat %4 : i1 -> tensor<16xi1, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %6 = tt.load %arg3, %5 : tensor<16x!tt.ptr<i64>, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %7 = tt.expand_dims %6 {axis = 1 : i32} : tensor<16xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<16x1xi64, #blocked>
    %8 = tt.broadcast %7 : tensor<16x1xi64, #blocked> -> tensor<16x16xi64, #blocked>
    %9 = arith.muli %arg0, %8 : tensor<16x16xi64, #blocked>
    %10 = tt.splat %4 : i1 -> tensor<16x16xi1, #blocked>
    %11 = tt.addptr %arg5, %9 : tensor<16x16x!tt.ptr<f16>, #blocked>, tensor<16x16xi64, #blocked>
    %12 = tt.load %11, %10 : tensor<16x16x!tt.ptr<f16>, #blocked>
    %13 = tt.splat %4 : i1 -> tensor<16x16xi1, #blocked1>
    %14 = tt.load %arg2, %13 : tensor<16x16x!tt.ptr<f16>, #blocked1>
    %c0_i32 = arith.constant 0 : i32
    %c-1_i32 = arith.constant -1 : i32
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #mma>
    %c1_i32 = arith.constant 1 : i32
    %15 = triton_gpu.local_alloc  : () -> !tt.memdesc<1x16x16xf16, #shared, #triton_gpu.shared_memory, mutable>
    %16 = triton_gpu.local_alloc  : () -> !tt.memdesc<1x16x16xf16, #shared, #triton_gpu.shared_memory, mutable>
    %17 = triton_gpu.memdesc_subview %15[%c0_i32, %c0_i32, %c0_i32] : !tt.memdesc<1x16x16xf16, #shared, #triton_gpu.shared_memory, mutable> -> !tt.memdesc<16x16xf16, #shared, #triton_gpu.shared_memory, mutable>
    triton_gpu.local_store %14, %17 : tensor<16x16xf16, #blocked1> -> !tt.memdesc<16x16xf16, #shared, #triton_gpu.shared_memory, mutable>
    %18 = triton_gpu.memdesc_subview %16[%c0_i32, %c0_i32, %c0_i32] : !tt.memdesc<1x16x16xf16, #shared, #triton_gpu.shared_memory, mutable> -> !tt.memdesc<16x16xf16, #shared, #triton_gpu.shared_memory, mutable>
    triton_gpu.local_store %12, %18 : tensor<16x16xf16, #blocked> -> !tt.memdesc<16x16xf16, #shared, #triton_gpu.shared_memory, mutable>
    %19:8 = scf.for %arg6 = %c0 to %arg1 step %c1 iter_args(%arg7 = %cst_0, %arg8 = %arg2, %arg9 = %2, %arg10 = %c-1_i32, %arg11 = %c0_i32, %arg12 = %17, %arg13 = %18, %arg14 = %3) -> (tensor<16x16xf32, #mma>, tensor<16x16x!tt.ptr<f16>, #blocked1>, tensor<16x!tt.ptr<i64>, #triton_gpu.slice<{dim = 1, parent = #blocked}>>, i32, i32, !tt.memdesc<16x16xf16, #shared, #triton_gpu.shared_memory, mutable>, !tt.memdesc<16x16xf16, #shared, #triton_gpu.shared_memory, mutable>, tensor<16xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>>) {
      %20 = arith.subi %arg1, %c2 : index
      %21 = arith.cmpi slt, %arg6, %20 : index
      %22 = tt.splat %21 : i1 -> tensor<16xi1, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
      %23 = tt.addptr %arg9, %cst : tensor<16x!tt.ptr<i64>, #triton_gpu.slice<{dim = 1, parent = #blocked}>>, tensor<16xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
      %24 = tt.load %23, %22 : tensor<16x!tt.ptr<i64>, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
      %25 = arith.subi %arg1, %c1 : index
      %26 = arith.cmpi slt, %arg6, %25 : index
      %27 = tt.expand_dims %arg14 {axis = 1 : i32} : tensor<16xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<16x1xi64, #blocked>
      %28 = tt.broadcast %27 : tensor<16x1xi64, #blocked> -> tensor<16x16xi64, #blocked>
      %29 = arith.muli %arg0, %28 : tensor<16x16xi64, #blocked>
      %30 = tt.splat %26 : i1 -> tensor<16x16xi1, #blocked>
      %31 = tt.addptr %arg5, %29 : tensor<16x16x!tt.ptr<f16>, #blocked>, tensor<16x16xi64, #blocked>
      %32 = tt.load %31, %30 : tensor<16x16x!tt.ptr<f16>, #blocked>
      %33 = tt.splat %26 : i1 -> tensor<16x16xi1, #blocked1>
      %34 = tt.addptr %arg8, %arg4 : tensor<16x16x!tt.ptr<f16>, #blocked1>, tensor<16x16xi32, #blocked1>
      %35 = tt.load %34, %33 : tensor<16x16x!tt.ptr<f16>, #blocked1>
      %36 = arith.addi %arg10, %c1_i32 : i32
      %37 = arith.cmpi slt, %36, %c1_i32 : i32
      %38 = arith.select %37, %36, %c0_i32 : i32
      %39 = triton_gpu.local_load %arg12 : !tt.memdesc<16x16xf16, #shared, #triton_gpu.shared_memory, mutable> -> tensor<16x16xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %40 = triton_gpu.local_load %arg13 : !tt.memdesc<16x16xf16, #shared, #triton_gpu.shared_memory, mutable> -> tensor<16x16xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %41 = tt.dot %39, %40, %arg7 : tensor<16x16xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<16x16xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<16x16xf32, #mma>
      %42 = arith.addi %arg11, %c1_i32 : i32
      %43 = arith.cmpi slt, %42, %c1_i32 : i32
      %44 = arith.select %43, %42, %c0_i32 : i32
      %45 = triton_gpu.memdesc_subview %15[%44, %c0_i32, %c0_i32] : !tt.memdesc<1x16x16xf16, #shared, #triton_gpu.shared_memory, mutable> -> !tt.memdesc<16x16xf16, #shared, #triton_gpu.shared_memory, mutable>
      triton_gpu.local_store %35, %45 : tensor<16x16xf16, #blocked1> -> !tt.memdesc<16x16xf16, #shared, #triton_gpu.shared_memory, mutable>
      %46 = triton_gpu.memdesc_subview %16[%44, %c0_i32, %c0_i32] : !tt.memdesc<1x16x16xf16, #shared, #triton_gpu.shared_memory, mutable> -> !tt.memdesc<16x16xf16, #shared, #triton_gpu.shared_memory, mutable>
      triton_gpu.local_store %32, %46 : tensor<16x16xf16, #blocked> -> !tt.memdesc<16x16xf16, #shared, #triton_gpu.shared_memory, mutable>
      scf.yield %41, %34, %23, %38, %44, %45, %46, %24 : tensor<16x16xf32, #mma>, tensor<16x16x!tt.ptr<f16>, #blocked1>, tensor<16x!tt.ptr<i64>, #triton_gpu.slice<{dim = 1, parent = #blocked}>>, i32, i32, !tt.memdesc<16x16xf16, #shared, #triton_gpu.shared_memory, mutable>, !tt.memdesc<16x16xf16, #shared, #triton_gpu.shared_memory, mutable>, tensor<16xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    }
    triton_gpu.local_dealloc %15 : !tt.memdesc<1x16x16xf16, #shared, #triton_gpu.shared_memory, mutable>
    triton_gpu.local_dealloc %16 : !tt.memdesc<1x16x16xf16, #shared, #triton_gpu.shared_memory, mutable>
    tt.return %19#0 : tensor<16x16xf32, #mma>
  }
}

// -----
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #triton_gpu.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 8]}>
#shared = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 4, order = [1, 0], hasLeadingOffset = false}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, triton_gpu.target = "cuda:86", "triton_gpu.threads-per-warp" = 32 : i32} {

// CHECK-LABEL:  tt.func public @kernel_yield_constant
// CHECK:  %{{.*}}:4 = scf.for %[[ARG7:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ARG8:.*]] = %{{.*}}, %[[ARG9:.*]] = %{{.*}}-1_i32, %[[ARG10:.*]] = %{{.*}}, %[[ARG11:.*]] = %{{.*}})

// CHECK:  %[[SUBI_17:.*]] = arith.subi %{{.*}}, %{{.*}}
// CHECK:  %[[ADDI_18:.*]] = arith.addi %[[ARG7]], %{{.*}}
// CHECK:  %[[MULI_19:.*]] = arith.muli %[[ADDI_18]], %{{.*}}
// CHECK:  %[[SUBI_20:.*]] = arith.subi %{{.*}}, %[[MULI_19]]
// CHECK:  %[[SPLAT_21:.*]] = tt.splat %[[SUBI_20]]
// CHECK:  %[[CMPI_22:.*]] = arith.cmpi slt, %{{.*}}, %[[SPLAT_21]]
// CHECK:  %[[CMPI_23:.*]] = arith.cmpi slt, %[[ARG7]], %[[SUBI_17]]
// CHECK:  %[[BROADCAST_24:.*]] = tt.broadcast %[[CMPI_22]]
// CHECK:  %[[SPLAT_25:.*]] = tt.splat %[[CMPI_23]]
// CHECK:  %[[MULI_26:.*]] = arith.muli %[[MULI_19]], %{{.*}}
// CHECK:  %[[SPLAT_27:.*]] = tt.splat %[[MULI_26]]
// CHECK:  %[[ANDI_28:.*]] = arith.andi %[[SPLAT_25]], %[[BROADCAST_24]]
// CHECK:  %[[ADDPTR_29:.*]] = tt.addptr %{{.*}}, %[[SPLAT_27]]
// CHECK:  %[[LOAD_30:.*]] = tt.load %[[ADDPTR_29]], %[[ANDI_28]], %{{.*}}
// CHECK:  %[[ADDI_31:.*]] = arith.addi %[[ARG9]], %{{.*}}
// CHECK:  %[[CMPI_32:.*]] = arith.cmpi slt, %[[ADDI_31]], %{{.*}}
// CHECK:  %[[SELECT_33:.*]] = arith.select %[[CMPI_32]], %[[ADDI_31]], %{{.*}}
// CHECK:  %[[LOCAL_LOAD_34:.*]] = triton_gpu.local_load %[[ARG11]]
// CHECK:  %[[DOT_35:.*]] = tt.dot %{{.*}}, %[[LOCAL_LOAD_34]], %[[ARG8]]
// CHECK:  %[[CONVERT_LAYOUT_36:.*]] = triton_gpu.convert_layout %[[DOT_35]]
// CHECK:  tt.store %{{.*}}, %[[CONVERT_LAYOUT_36]]
// CHECK:  %[[ADDI_37:.*]] = arith.addi %[[ARG10]], %{{.*}}
// CHECK:  %[[CMPI_38:.*]] = arith.cmpi slt, %[[ADDI_37]], %{{.*}}
// CHECK:  %[[SELECT_39:.*]] = arith.select %[[CMPI_38]], %[[ADDI_37]], %{{.*}}
// CHECK:  %[[MEMDESC_SUBVIEW_40:.*]] = triton_gpu.memdesc_subview %{{.*}}[%[[SELECT_39]], %{{.*}}, %{{.*}}]
// CHECK:  triton_gpu.local_store %[[LOAD_30]], %[[MEMDESC_SUBVIEW_40]]
// CHECK:  scf.yield %{{.*}}, %[[SELECT_33]], %[[SELECT_39]], %[[MEMDESC_SUBVIEW_40]]
// CHECK:  }

  tt.func public @kernel_yield_constant(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0> : tensor<32x32xi32, #blocked>
    %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %1 = tt.splat %arg4 : i32 -> tensor<32x1xi32, #blocked>
    %2 = tt.expand_dims %0 {axis = 1 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<32x1xi32, #blocked>
    %3 = arith.cmpi slt, %2, %1 : tensor<32x1xi32, #blocked>
    %c31_i32 = arith.constant 31 : i32
    %c32_i32 = arith.constant 32 : i32
    %4 = arith.addi %arg4, %c31_i32 : i32
    %c0_i32 = arith.constant 0 : i32
    %5 = arith.divsi %4, %c32_i32 : i32
    %6 = arith.cmpi sgt, %5, %c0_i32 : i32
    %7 = tt.broadcast %3 : tensor<32x1xi1, #blocked> -> tensor<32x32xi1, #blocked>
    %8 = tt.splat %6 : i1 -> tensor<32x32xi1, #blocked>
    %9 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #blocked>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #blocked>
    %10 = arith.andi %8, %7 : tensor<32x32xi1, #blocked>
    %11 = tt.addptr %9, %cst : tensor<32x32x!tt.ptr<f32>, #blocked>, tensor<32x32xi32, #blocked>
    %12 = tt.load %11, %10, %cst_0 : tensor<32x32x!tt.ptr<f32>, #blocked>
    %c-1_i32 = arith.constant -1 : i32
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %cst_2 = arith.constant dense<1.000000e+00> : tensor<32x32xf32, #mma>
    %c1_i32 = arith.constant 1 : i32
    %cst_3 = arith.constant dense<2.000000e+00> : tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
    %13 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #blocked>
    %14 = triton_gpu.local_alloc  : () -> !tt.memdesc<1x32x32xf32, #shared, #triton_gpu.shared_memory, mutable>
    %15 = triton_gpu.memdesc_subview %14[%c0_i32, %c0_i32, %c0_i32] : !tt.memdesc<1x32x32xf32, #shared, #triton_gpu.shared_memory, mutable> -> !tt.memdesc<32x32xf32, #shared, #triton_gpu.shared_memory, mutable>
    triton_gpu.local_store %12, %15 : tensor<32x32xf32, #blocked> -> !tt.memdesc<32x32xf32, #shared, #triton_gpu.shared_memory, mutable>
    %16:4 = scf.for %arg7 = %c0_i32 to %5 step %c1_i32 iter_args(%arg8 = %cst_1, %arg9 = %c-1_i32, %arg10 = %c0_i32, %arg11 = %15) -> (tensor<32x32xf32, #mma>, i32, i32, !tt.memdesc<32x32xf32, #shared, #triton_gpu.shared_memory, mutable>)  : i32 {
      %17 = arith.subi %5, %c1_i32 : i32
      %18 = arith.addi %arg7, %c1_i32 : i32
      %19 = arith.muli %18, %c32_i32 : i32
      %20 = arith.subi %arg4, %19 : i32
      %21 = tt.splat %20 : i32 -> tensor<32x1xi32, #blocked>
      %22 = arith.cmpi slt, %2, %21 : tensor<32x1xi32, #blocked>
      %23 = arith.cmpi slt, %arg7, %17 : i32
      %24 = tt.broadcast %22 : tensor<32x1xi1, #blocked> -> tensor<32x32xi1, #blocked>
      %25 = tt.splat %23 : i1 -> tensor<32x32xi1, #blocked>
      %26 = arith.muli %19, %arg5 : i32
      %27 = tt.splat %26 : i32 -> tensor<32x32xi32, #blocked>
      %28 = arith.andi %25, %24 : tensor<32x32xi1, #blocked>
      %29 = tt.addptr %9, %27 : tensor<32x32x!tt.ptr<f32>, #blocked>, tensor<32x32xi32, #blocked>
      %30 = tt.load %29, %28, %cst_0 : tensor<32x32x!tt.ptr<f32>, #blocked>
      %31 = arith.addi %arg9, %c1_i32 : i32
      %32 = arith.cmpi slt, %31, %c1_i32 : i32
      %33 = arith.select %32, %31, %c0_i32 : i32
      %34 = triton_gpu.local_load %arg11 : !tt.memdesc<32x32xf32, #shared, #triton_gpu.shared_memory, mutable> -> tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>
      %35 = tt.dot %cst_3, %34, %arg8 : tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>> -> tensor<32x32xf32, #mma>
      %36 = triton_gpu.convert_layout %35 : tensor<32x32xf32, #mma> -> tensor<32x32xf32, #blocked>
      tt.store %13, %36 : tensor<32x32x!tt.ptr<f32>, #blocked>
      %37 = arith.addi %arg10, %c1_i32 : i32
      %38 = arith.cmpi slt, %37, %c1_i32 : i32
      %39 = arith.select %38, %37, %c0_i32 : i32
      %40 = triton_gpu.memdesc_subview %14[%39, %c0_i32, %c0_i32] : !tt.memdesc<1x32x32xf32, #shared, #triton_gpu.shared_memory, mutable> -> !tt.memdesc<32x32xf32, #shared, #triton_gpu.shared_memory, mutable>
      triton_gpu.local_store %30, %40 : tensor<32x32xf32, #blocked> -> !tt.memdesc<32x32xf32, #shared, #triton_gpu.shared_memory, mutable>
      scf.yield %cst_2, %33, %39, %40 : tensor<32x32xf32, #mma>, i32, i32, !tt.memdesc<32x32xf32, #shared, #triton_gpu.shared_memory, mutable>
    }
    triton_gpu.local_dealloc %14 : !tt.memdesc<1x32x32xf32, #shared, #triton_gpu.shared_memory, mutable>
    tt.return
  }
}

// -----
#blocked = #triton_gpu.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, triton_gpu.target = "cuda:90", "triton_gpu.threads-per-warp" = 32 : i32} {

// CHECK-LABEL:  tt.func public @add_kernel
// CHECK:  %{{.*}}:10 = scf.for %[[ARG4:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ARG5:.*]] = %{{.*}}-1_i32, %[[ARG6:.*]] = %{{.*}}-1_i32, %[[ARG7:.*]] = %{{.*}}, %[[ARG8:.*]] = %{{.*}}, %[[ARG9:.*]] = %{{.*}}, %[[ARG10:.*]] = %{{.*}}, %[[ARG11:.*]] = %{{.*}}, %[[ARG12:.*]] = %{{.*}}, %[[ARG13:.*]] = %{{.*}}, %[[ARG14:.*]] = %{{.*}})

// CHECK:  %[[CMPI_23:.*]] = arith.cmpi slt, %[[ARG4]], %{{.*}}
// CHECK:  %[[ADDI_24:.*]] = arith.addi %[[ARG4]], %{{.*}}
// CHECK:  %[[ADDI_25:.*]] = arith.addi %{{.*}}, %[[ADDI_24]]
// CHECK:  %[[SPLAT_26:.*]] = tt.splat %[[ADDI_25]]
// CHECK:  %[[ADDI_27:.*]] = arith.addi %[[SPLAT_26]], %{{.*}}
// CHECK:  %[[CMPI_28:.*]] = arith.cmpi slt, %[[ADDI_27]], %{{.*}}
// CHECK:  %[[SPLAT_29:.*]] = tt.splat %[[CMPI_23]]
// CHECK:  %[[ANDI_30:.*]] = arith.andi %[[SPLAT_29]], %[[CMPI_28]]
// CHECK:  %[[ADDPTR_31:.*]] = tt.addptr %{{.*}}, %[[ADDI_27]]
// CHECK:  %[[LOAD_32:.*]] = tt.load %[[ADDPTR_31]], %[[ANDI_30]]
// CHECK:  %[[SPLAT_33:.*]] = tt.splat %[[CMPI_23]]
// CHECK:  %[[ANDI_34:.*]] = arith.andi %[[SPLAT_33]], %[[CMPI_28]]
// CHECK:  %[[ADDPTR_35:.*]] = tt.addptr %{{.*}}, %[[ADDI_27]]
// CHECK:  %[[LOAD_36:.*]] = tt.load %[[ADDPTR_35]], %[[ANDI_34]]
// CHECK:  %[[ADDI_37:.*]] = arith.addi %[[ARG5]], %{{.*}}
// CHECK:  %[[CMPI_38:.*]] = arith.cmpi slt, %[[ADDI_37]], %{{.*}}
// CHECK:  %[[SELECT_39:.*]] = arith.select %[[CMPI_38]], %[[ADDI_37]], %{{.*}}
// CHECK:  %[[ADDI_40:.*]] = arith.addi %[[ARG6]], %{{.*}}
// CHECK:  %[[CMPI_41:.*]] = arith.cmpi slt, %[[ADDI_40]], %{{.*}}
// CHECK:  %[[SELECT_42:.*]] = arith.select %[[CMPI_41]], %[[ADDI_40]], %{{.*}}
// CHECK:  %[[ADDF_43:.*]] = arith.addf %[[ARG7]], %[[ARG9]]
// CHECK:  %[[ADDPTR_44:.*]] = tt.addptr %{{.*}}, %[[ARG11]]
// CHECK:  tt.store %[[ADDPTR_44]], %[[ADDF_43]], %[[ARG13]]
// CHECK:  scf.yield %[[SELECT_39]], %[[SELECT_42]], %[[ARG8]], %[[LOAD_32]], %[[ARG10]], %[[LOAD_36]], %[[ARG12]], %[[ADDI_27]], %[[ARG14]], %[[CMPI_28]]
// CHECK:  }

  tt.func public @add_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 16 : i32}) attributes {noinline = false} {
    %c2048_i32 = arith.constant 2048 : i32
    %c1016800_i32 = arith.constant 1016800 : i32
    %0 = tt.get_program_id x : i32
    %c1024_i32 = arith.constant 1024 : i32
    %1 = arith.muli %0, %c1016800_i32 : i32
    %2 = arith.addi %1, %c1024_i32 : i32
    %3 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %4 = tt.splat %2 : i32 -> tensor<1024xi32, #blocked>
    %5 = tt.splat %arg3 : i32 -> tensor<1024xi32, #blocked>
    %6 = arith.addi %4, %3 : tensor<1024xi32, #blocked>
    %7 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %8 = arith.cmpi slt, %6, %5 : tensor<1024xi32, #blocked>
    %9 = tt.addptr %7, %6 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    %10 = tt.load %9, %8 : tensor<1024x!tt.ptr<f32>, #blocked>
    %11 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %12 = tt.addptr %11, %6 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    %13 = tt.load %12, %8 : tensor<1024x!tt.ptr<f32>, #blocked>
    %14 = tt.splat %1 : i32 -> tensor<1024xi32, #blocked>
    %15 = arith.addi %14, %3 : tensor<1024xi32, #blocked>
    %16 = arith.cmpi slt, %15, %5 : tensor<1024xi32, #blocked>
    %17 = tt.addptr %7, %15 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    %18 = tt.load %17, %16 : tensor<1024x!tt.ptr<f32>, #blocked>
    %19 = tt.addptr %11, %15 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    %20 = tt.load %19, %16 : tensor<1024x!tt.ptr<f32>, #blocked>
    %c1014752_i32 = arith.constant 1014752 : i32
    %c2_i32 = arith.constant 2 : i32
    %c1_i32 = arith.constant 1 : i32
    %c-1_i32 = arith.constant -1 : i32
    %c0_i32 = arith.constant 0 : i32
    %21 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %22:10 = scf.for %arg4 = %c0_i32 to %c1016800_i32 step %c1024_i32 iter_args(%arg5 = %c-1_i32, %arg6 = %c-1_i32, %arg7 = %20, %arg8 = %13, %arg9 = %18, %arg10 = %10, %arg11 = %15, %arg12 = %6, %arg13 = %16, %arg14 = %8) -> (i32, i32, tensor<1024xf32, #blocked>, tensor<1024xf32, #blocked>, tensor<1024xf32, #blocked>, tensor<1024xf32, #blocked>, tensor<1024xi32, #blocked>, tensor<1024xi32, #blocked>, tensor<1024xi1, #blocked>, tensor<1024xi1, #blocked>)  : i32 {
      %23 = arith.cmpi slt, %arg4, %c1014752_i32 : i32
      %24 = arith.addi %arg4, %c2048_i32 : i32
      %25 = arith.addi %1, %24 : i32
      %26 = tt.splat %25 : i32 -> tensor<1024xi32, #blocked>
      %27 = arith.addi %26, %3 : tensor<1024xi32, #blocked>
      %28 = arith.cmpi slt, %27, %5 : tensor<1024xi32, #blocked>
      %29 = tt.splat %23 : i1 -> tensor<1024xi1, #blocked>
      %30 = arith.andi %29, %28 : tensor<1024xi1, #blocked>
      %31 = tt.addptr %7, %27 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
      %32 = tt.load %31, %30 : tensor<1024x!tt.ptr<f32>, #blocked>
      %33 = tt.splat %23 : i1 -> tensor<1024xi1, #blocked>
      %34 = arith.andi %33, %28 : tensor<1024xi1, #blocked>
      %35 = tt.addptr %11, %27 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
      %36 = tt.load %35, %34 : tensor<1024x!tt.ptr<f32>, #blocked>
      %37 = arith.addi %arg5, %c1_i32 : i32
      %38 = arith.cmpi slt, %37, %c2_i32 : i32
      %39 = arith.select %38, %37, %c0_i32 : i32
      %40 = arith.addi %arg6, %c1_i32 : i32
      %41 = arith.cmpi slt, %40, %c2_i32 : i32
      %42 = arith.select %41, %40, %c0_i32 : i32
      %43 = arith.addf %arg7, %arg9 : tensor<1024xf32, #blocked>
      %44 = tt.addptr %21, %arg11 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
      tt.store %44, %43, %arg13 : tensor<1024x!tt.ptr<f32>, #blocked>
      scf.yield %39, %42, %arg8, %36, %arg10, %32, %arg12, %27, %arg14, %28 : i32, i32, tensor<1024xf32, #blocked>, tensor<1024xf32, #blocked>, tensor<1024xf32, #blocked>, tensor<1024xf32, #blocked>, tensor<1024xi32, #blocked>, tensor<1024xi32, #blocked>, tensor<1024xi1, #blocked>, tensor<1024xi1, #blocked>
    }
    tt.return
  }
}

// -----
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [2, 1], order = [1, 0]}>
#mma = #triton_gpu.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 2], instrShape = [16, 8]}>
#shared = #triton_gpu.shared<{vec = 4, perPhase = 2, maxPhase = 4, order = [1, 0], hasLeadingOffset = false}>
#shared1 = #triton_gpu.shared<{vec = 4, perPhase = 2, maxPhase = 4, order = [0, 1], hasLeadingOffset = false}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 2 : i32, triton_gpu.target = "cuda:80", "triton_gpu.threads-per-warp" = 32 : i32} {

// CHECK-LABEL:  tt.func public @nested_loops
// CHECK:  scf.for %[[ARG1:.*]] = %{{.*}} to %{{.*}} step %{{.*}}  : i32 {

// CHECK:  %[[LOAD_10:.*]] = tt.load %{{.*}}
// CHECK:  %[[LOAD_11:.*]] = tt.load %{{.*}}
// CHECK:  %[[LOCAL_ALLOC_12:.*]] = triton_gpu.local_alloc %[[LOAD_10]]
// CHECK:  %[[TRANS_13:.*]] = tt.trans %[[LOCAL_ALLOC_12]] {order = array<i32: 1, 0>}
// CHECK:  %[[LOCAL_LOAD_14:.*]] = triton_gpu.local_load %[[TRANS_13]]
// CHECK:  %[[LOCAL_ALLOC_15:.*]] = triton_gpu.local_alloc
// CHECK:  %[[MEMDESC_SUBVIEW_16:.*]] = triton_gpu.memdesc_subview %[[LOCAL_ALLOC_15]][%{{.*}}, %{{.*}}, %{{.*}}]
// CHECK:  triton_gpu.local_store %[[LOAD_11]], %[[MEMDESC_SUBVIEW_16]]
// CHECK:  %{{.*}}:3 = scf.for %[[ARG2:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ARG3:.*]] = %{{.*}}-1_i32, %[[ARG4:.*]] = %{{.*}}, %[[ARG5:.*]] = %[[MEMDESC_SUBVIEW_16]])

// CHECK:  %[[CMPI_18:.*]] = arith.cmpi slt, %[[ARG2]], %{{.*}}
// CHECK:  %[[SPLAT_19:.*]] = tt.splat %[[CMPI_18]]
// CHECK:  %[[LOAD_20:.*]] = tt.load %{{.*}}, %[[SPLAT_19]]
// CHECK:  %[[ADDI_21:.*]] = arith.addi %[[ARG3]], %{{.*}}
// CHECK:  %[[CMPI_22:.*]] = arith.cmpi slt, %[[ADDI_21]], %{{.*}}
// CHECK:  %[[SELECT_23:.*]] = arith.select %[[CMPI_22]], %[[ADDI_21]], %{{.*}}
// CHECK:  %[[LOCAL_LOAD_24:.*]] = triton_gpu.local_load %[[ARG5]]
// CHECK:  %[[DOT_25:.*]] = tt.dot %[[LOCAL_LOAD_24]], %[[LOCAL_LOAD_14]], %{{.*}}
// CHECK:  %[[CONVERT_LAYOUT_26:.*]] = triton_gpu.convert_layout %[[DOT_25]]
// CHECK:  tt.store %{{.*}}, %[[CONVERT_LAYOUT_26]]
// CHECK:  %[[ADDI_27:.*]] = arith.addi %[[ARG4]], %{{.*}}
// CHECK:  %[[CMPI_28:.*]] = arith.cmpi slt, %[[ADDI_27]], %{{.*}}
// CHECK:  %[[SELECT_29:.*]] = arith.select %[[CMPI_28]], %[[ADDI_27]], %{{.*}}
// CHECK:  %[[MEMDESC_SUBVIEW_30:.*]] = triton_gpu.memdesc_subview %[[LOCAL_ALLOC_15]][%[[SELECT_29]], %{{.*}}, %{{.*}}]
// CHECK:  triton_gpu.local_store %[[LOAD_20]], %[[MEMDESC_SUBVIEW_30]]
// CHECK:  scf.yield %[[SELECT_23]], %[[SELECT_29]], %[[MEMDESC_SUBVIEW_30]]
// CHECK:  }

  tt.func public @nested_loops(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c-1_i32 = arith.constant -1 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #mma>
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst_0 = arith.constant dense<16> : tensor<16x1xi32, #blocked>
    %0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<16xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<16x1xi32, #blocked>
    %2 = arith.muli %1, %cst_0 : tensor<16x1xi32, #blocked>
    %3 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<16x1x!tt.ptr<f32>, #blocked>
    %4 = tt.addptr %3, %2 : tensor<16x1x!tt.ptr<f32>, #blocked>, tensor<16x1xi32, #blocked>
    %5 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %6 = tt.expand_dims %5 {axis = 0 : i32} : tensor<16xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> -> tensor<1x16xi32, #blocked>
    %7 = tt.broadcast %4 : tensor<16x1x!tt.ptr<f32>, #blocked> -> tensor<16x16x!tt.ptr<f32>, #blocked>
    %8 = tt.broadcast %6 : tensor<1x16xi32, #blocked> -> tensor<16x16xi32, #blocked>
    %9 = tt.addptr %7, %8 : tensor<16x16x!tt.ptr<f32>, #blocked>, tensor<16x16xi32, #blocked>
    scf.for %arg1 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
      %10 = tt.load %9 : tensor<16x16x!tt.ptr<f32>, #blocked>
      %11 = tt.load %9 : tensor<16x16x!tt.ptr<f32>, #blocked>
      %12 = triton_gpu.local_alloc %10 : (tensor<16x16xf32, #blocked>) -> !tt.memdesc<16x16xf32, #shared, #triton_gpu.shared_memory>
      %13 = tt.trans %12 {order = array<i32: 1, 0>} : !tt.memdesc<16x16xf32, #shared, #triton_gpu.shared_memory> -> !tt.memdesc<16x16xf32, #shared1, #triton_gpu.shared_memory>
      %14 = triton_gpu.local_load %13 : !tt.memdesc<16x16xf32, #shared1, #triton_gpu.shared_memory> -> tensor<16x16xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>
      %15 = triton_gpu.local_alloc  : () -> !tt.memdesc<1x16x16xf32, #shared, #triton_gpu.shared_memory, mutable>
      %16 = triton_gpu.memdesc_subview %15[%c0_i32, %c0_i32, %c0_i32] : !tt.memdesc<1x16x16xf32, #shared, #triton_gpu.shared_memory, mutable> -> !tt.memdesc<16x16xf32, #shared, #triton_gpu.shared_memory, mutable>
      triton_gpu.local_store %11, %16 : tensor<16x16xf32, #blocked> -> !tt.memdesc<16x16xf32, #shared, #triton_gpu.shared_memory, mutable>
      %17:3 = scf.for %arg2 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg3 = %c-1_i32, %arg4 = %c0_i32, %arg5 = %16) -> (i32, i32, !tt.memdesc<16x16xf32, #shared, #triton_gpu.shared_memory, mutable>)  : i32 {
        %18 = arith.cmpi slt, %arg2, %c1_i32 : i32
        %19 = tt.splat %18 : i1 -> tensor<16x16xi1, #blocked>
        %20 = tt.load %9, %19 : tensor<16x16x!tt.ptr<f32>, #blocked>
        %21 = arith.addi %arg3, %c1_i32 : i32
        %22 = arith.cmpi slt, %21, %c1_i32 : i32
        %23 = arith.select %22, %21, %c0_i32 : i32
        %24 = triton_gpu.local_load %arg5 : !tt.memdesc<16x16xf32, #shared, #triton_gpu.shared_memory, mutable> -> tensor<16x16xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
        %25 = tt.dot %24, %14, %cst : tensor<16x16xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<16x16xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>> -> tensor<16x16xf32, #mma>
        %26 = triton_gpu.convert_layout %25 : tensor<16x16xf32, #mma> -> tensor<16x16xf32, #blocked>
        tt.store %9, %26 : tensor<16x16x!tt.ptr<f32>, #blocked>
        %27 = arith.addi %arg4, %c1_i32 : i32
        %28 = arith.cmpi slt, %27, %c1_i32 : i32
        %29 = arith.select %28, %27, %c0_i32 : i32
        %30 = triton_gpu.memdesc_subview %15[%29, %c0_i32, %c0_i32] : !tt.memdesc<1x16x16xf32, #shared, #triton_gpu.shared_memory, mutable> -> !tt.memdesc<16x16xf32, #shared, #triton_gpu.shared_memory, mutable>
        triton_gpu.local_store %20, %30 : tensor<16x16xf32, #blocked> -> !tt.memdesc<16x16xf32, #shared, #triton_gpu.shared_memory, mutable>
        scf.yield %23, %29, %30 : i32, i32, !tt.memdesc<16x16xf32, #shared, #triton_gpu.shared_memory, mutable>
      }
      triton_gpu.local_dealloc %15 : !tt.memdesc<1x16x16xf32, #shared, #triton_gpu.shared_memory, mutable>
    }
    tt.return
  }
}

// -----
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #triton_gpu.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = []}>
#shared = #triton_gpu.shared<{vec = 8, perPhase = 4, maxPhase = 2, order = [1, 0], hasLeadingOffset = false}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, triton_gpu.target = "cuda:80", "triton_gpu.threads-per-warp" = 32 : i32} {

// CHECK-LABEL:  tt.func @load_convert_layout
// CHECK:  %{{.*}}:8 = scf.for %[[ARG6:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ARG7:.*]] = %{{.*}}, %[[ARG8:.*]] = %{{.*}}, %[[ARG9:.*]] = %{{.*}}, %[[ARG10:.*]] = %{{.*}}-1_i32, %[[ARG11:.*]] = %{{.*}}, %[[ARG12:.*]] = %{{.*}}, %[[ARG13:.*]] = %{{.*}}, %[[ARG14:.*]] = %{{.*}})

// CHECK:  %[[SUBI_24:.*]] = arith.subi %{{.*}}, %{{.*}}
// CHECK:  %[[SUBI_25:.*]] = arith.subi %{{.*}}, %{{.*}}
// CHECK:  %[[CMPI_26:.*]] = arith.cmpi slt, %[[ARG6]], %[[SUBI_25]]
// CHECK:  %[[SPLAT_27:.*]] = tt.splat %[[CMPI_26]]
// CHECK:  %[[ADDPTR_28:.*]] = tt.addptr %[[ARG8]], %{{.*}}
// CHECK:  %[[LOAD_29:.*]] = tt.load %[[ADDPTR_28]], %[[SPLAT_27]]
// CHECK:  %[[EXPAND_DIMS_30:.*]] = tt.expand_dims %[[ARG14]] {axis = 1 : i32}
// CHECK:  %[[BROADCAST_31:.*]] = tt.broadcast %[[EXPAND_DIMS_30]]
// CHECK:  %[[MULI_32:.*]] = arith.muli %{{.*}}, %[[BROADCAST_31]]
// CHECK:  %[[SPLAT_33:.*]] = tt.splat %[[CMPI_26]]
// CHECK:  %[[ADDPTR_34:.*]] = tt.addptr %{{.*}}, %[[MULI_32]]
// CHECK:  %[[LOAD_35:.*]] = tt.load %[[ADDPTR_34]], %[[SPLAT_33]]
// CHECK:  %[[CMPI_36:.*]] = arith.cmpi slt, %[[ARG6]], %[[SUBI_24]]
// CHECK:  %[[SPLAT_37:.*]] = tt.splat %[[CMPI_36]]
// CHECK:  %[[ANDI_38:.*]] = arith.andi %[[SPLAT_37]], %{{.*}}
// CHECK:  %[[ADDPTR_39:.*]] = tt.addptr %[[ARG9]], %{{.*}}
// CHECK:  %[[LOAD_40:.*]] = tt.load %[[ADDPTR_39]], %[[ANDI_38]]
// CHECK:  %[[ADDI_41:.*]] = arith.addi %[[ARG10]], %{{.*}}
// CHECK:  %[[CMPI_42:.*]] = arith.cmpi slt, %[[ADDI_41]], %{{.*}}
// CHECK:  %[[SELECT_43:.*]] = arith.select %[[CMPI_42]], %[[ADDI_41]], %{{.*}}
// CHECK:  %[[LOCAL_LOAD_44:.*]] = triton_gpu.local_load %[[ARG12]]
// CHECK:  %[[LOCAL_LOAD_45:.*]] = triton_gpu.local_load %[[ARG13]]
// CHECK:  %[[DOT_46:.*]] = tt.dot %[[LOCAL_LOAD_44]], %[[LOCAL_LOAD_45]], %[[ARG7]]
// CHECK:  %[[ADDI_47:.*]] = arith.addi %[[ARG11]], %{{.*}}
// CHECK:  %[[CMPI_48:.*]] = arith.cmpi slt, %[[ADDI_47]], %{{.*}}
// CHECK:  %[[SELECT_49:.*]] = arith.select %[[CMPI_48]], %[[ADDI_47]], %{{.*}}
// CHECK:  %[[MEMDESC_SUBVIEW_50:.*]] = triton_gpu.memdesc_subview %{{.*}}[%[[SELECT_49]], %{{.*}}, %{{.*}}]
// CHECK:  triton_gpu.local_store %[[LOAD_29]], %[[MEMDESC_SUBVIEW_50]]
// CHECK:  %[[MEMDESC_SUBVIEW_51:.*]] = triton_gpu.memdesc_subview %{{.*}}[%[[SELECT_49]], %{{.*}}, %{{.*}}]
// CHECK:  triton_gpu.local_store %[[LOAD_35]], %[[MEMDESC_SUBVIEW_51]]
// CHECK:  scf.yield %[[DOT_46]], %[[ADDPTR_28]], %[[ADDPTR_39]], %[[SELECT_43]], %[[SELECT_49]], %[[MEMDESC_SUBVIEW_50]], %[[MEMDESC_SUBVIEW_51]], %[[LOAD_40]]
// CHECK:  }

  tt.func @load_convert_layout(%arg0: tensor<16x16xi64, #blocked> {tt.constancy = 16 : i32, tt.divisibility = 16 : i32}, %arg1: index, %arg2: tensor<16x16x!tt.ptr<f16>, #blocked1> {tt.contiguity = 2 : i32, tt.divisibility = 16 : i32}, %arg3: tensor<16x!tt.ptr<i64>, #triton_gpu.slice<{dim = 1, parent = #blocked}>>, %arg4: tensor<16x16xi32, #blocked1> {tt.constancy = 16 : i32, tt.divisibility = 16 : i32}, %arg5: tensor<16x16x!tt.ptr<f16>, #blocked> {tt.contiguity = 16 : i32, tt.divisibility = 16 : i32}) -> tensor<16x16xf32, #mma> {
    %c2 = arith.constant 2 : index
    %cst = arith.constant dense<2> : tensor<16xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %c1 = arith.constant 1 : index
    %1 = arith.cmpi sgt, %arg1, %c1 : index
    %2 = arith.cmpi slt, %0, %cst : tensor<16xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %3 = tt.splat %1 : i1 -> tensor<16xi1, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %cst_0 = arith.constant dense<1> : tensor<16xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %4 = arith.andi %3, %2 : tensor<16xi1, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %5 = tt.addptr %arg3, %cst_0 : tensor<16x!tt.ptr<i64>, #triton_gpu.slice<{dim = 1, parent = #blocked}>>, tensor<16xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %6 = tt.load %5, %4 : tensor<16x!tt.ptr<i64>, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %c0 = arith.constant 0 : index
    %7 = arith.cmpi sgt, %arg1, %c0 : index
    %8 = tt.splat %7 : i1 -> tensor<16xi1, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %9 = arith.andi %8, %2 : tensor<16xi1, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %10 = tt.load %arg3, %9 : tensor<16x!tt.ptr<i64>, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %11 = tt.expand_dims %10 {axis = 1 : i32} : tensor<16xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<16x1xi64, #blocked>
    %12 = tt.broadcast %11 : tensor<16x1xi64, #blocked> -> tensor<16x16xi64, #blocked>
    %13 = arith.muli %arg0, %12 : tensor<16x16xi64, #blocked>
    %14 = tt.splat %7 : i1 -> tensor<16x16xi1, #blocked>
    %15 = tt.addptr %arg5, %13 : tensor<16x16x!tt.ptr<f16>, #blocked>, tensor<16x16xi64, #blocked>
    %16 = tt.load %15, %14 : tensor<16x16x!tt.ptr<f16>, #blocked>
    %17 = tt.splat %7 : i1 -> tensor<16x16xi1, #blocked1>
    %18 = tt.load %arg2, %17 : tensor<16x16x!tt.ptr<f16>, #blocked1>
    %c0_i32 = arith.constant 0 : i32
    %c-1_i32 = arith.constant -1 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #mma>
    %19 = triton_gpu.local_alloc  : () -> !tt.memdesc<1x16x16xf16, #shared, #triton_gpu.shared_memory, mutable>
    %20 = triton_gpu.local_alloc  : () -> !tt.memdesc<1x16x16xf16, #shared, #triton_gpu.shared_memory, mutable>
    %21 = triton_gpu.memdesc_subview %19[%c0_i32, %c0_i32, %c0_i32] : !tt.memdesc<1x16x16xf16, #shared, #triton_gpu.shared_memory, mutable> -> !tt.memdesc<16x16xf16, #shared, #triton_gpu.shared_memory, mutable>
    triton_gpu.local_store %18, %21 : tensor<16x16xf16, #blocked1> -> !tt.memdesc<16x16xf16, #shared, #triton_gpu.shared_memory, mutable>
    %22 = triton_gpu.memdesc_subview %20[%c0_i32, %c0_i32, %c0_i32] : !tt.memdesc<1x16x16xf16, #shared, #triton_gpu.shared_memory, mutable> -> !tt.memdesc<16x16xf16, #shared, #triton_gpu.shared_memory, mutable>
    triton_gpu.local_store %16, %22 : tensor<16x16xf16, #blocked> -> !tt.memdesc<16x16xf16, #shared, #triton_gpu.shared_memory, mutable>
    %23:8 = scf.for %arg6 = %c0 to %arg1 step %c1 iter_args(%arg7 = %cst_1, %arg8 = %arg2, %arg9 = %5, %arg10 = %c-1_i32, %arg11 = %c0_i32, %arg12 = %21, %arg13 = %22, %arg14 = %6) -> (tensor<16x16xf32, #mma>, tensor<16x16x!tt.ptr<f16>, #blocked1>, tensor<16x!tt.ptr<i64>, #triton_gpu.slice<{dim = 1, parent = #blocked}>>, i32, i32, !tt.memdesc<16x16xf16, #shared, #triton_gpu.shared_memory, mutable>, !tt.memdesc<16x16xf16, #shared, #triton_gpu.shared_memory, mutable>, tensor<16xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>>) {
      %24 = arith.subi %arg1, %c2 : index
      %25 = arith.cmpi slt, %arg6, %24 : index
      %26 = tt.splat %25 : i1 -> tensor<16xi1, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
      %27 = arith.andi %26, %2 : tensor<16xi1, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
      %28 = tt.addptr %arg9, %cst_0 : tensor<16x!tt.ptr<i64>, #triton_gpu.slice<{dim = 1, parent = #blocked}>>, tensor<16xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
      %29 = tt.load %28, %27 : tensor<16x!tt.ptr<i64>, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
      %30 = arith.subi %arg1, %c1 : index
      %31 = arith.cmpi slt, %arg6, %30 : index
      %32 = tt.expand_dims %arg14 {axis = 1 : i32} : tensor<16xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<16x1xi64, #blocked>
      %33 = tt.broadcast %32 : tensor<16x1xi64, #blocked> -> tensor<16x16xi64, #blocked>
      %34 = arith.muli %arg0, %33 : tensor<16x16xi64, #blocked>
      %35 = tt.splat %31 : i1 -> tensor<16x16xi1, #blocked>
      %36 = tt.addptr %arg5, %34 : tensor<16x16x!tt.ptr<f16>, #blocked>, tensor<16x16xi64, #blocked>
      %37 = tt.load %36, %35 : tensor<16x16x!tt.ptr<f16>, #blocked>
      %38 = tt.splat %31 : i1 -> tensor<16x16xi1, #blocked1>
      %39 = tt.addptr %arg8, %arg4 : tensor<16x16x!tt.ptr<f16>, #blocked1>, tensor<16x16xi32, #blocked1>
      %40 = tt.load %39, %38 : tensor<16x16x!tt.ptr<f16>, #blocked1>
      %41 = arith.addi %arg10, %c1_i32 : i32
      %42 = arith.cmpi slt, %41, %c1_i32 : i32
      %43 = arith.select %42, %41, %c0_i32 : i32
      %44 = triton_gpu.local_load %arg12 : !tt.memdesc<16x16xf16, #shared, #triton_gpu.shared_memory, mutable> -> tensor<16x16xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %45 = triton_gpu.local_load %arg13 : !tt.memdesc<16x16xf16, #shared, #triton_gpu.shared_memory, mutable> -> tensor<16x16xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %46 = tt.dot %44, %45, %arg7 : tensor<16x16xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<16x16xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<16x16xf32, #mma>
      %47 = arith.addi %arg11, %c1_i32 : i32
      %48 = arith.cmpi slt, %47, %c1_i32 : i32
      %49 = arith.select %48, %47, %c0_i32 : i32
      %50 = triton_gpu.memdesc_subview %19[%49, %c0_i32, %c0_i32] : !tt.memdesc<1x16x16xf16, #shared, #triton_gpu.shared_memory, mutable> -> !tt.memdesc<16x16xf16, #shared, #triton_gpu.shared_memory, mutable>
      triton_gpu.local_store %40, %50 : tensor<16x16xf16, #blocked1> -> !tt.memdesc<16x16xf16, #shared, #triton_gpu.shared_memory, mutable>
      %51 = triton_gpu.memdesc_subview %20[%49, %c0_i32, %c0_i32] : !tt.memdesc<1x16x16xf16, #shared, #triton_gpu.shared_memory, mutable> -> !tt.memdesc<16x16xf16, #shared, #triton_gpu.shared_memory, mutable>
      triton_gpu.local_store %37, %51 : tensor<16x16xf16, #blocked> -> !tt.memdesc<16x16xf16, #shared, #triton_gpu.shared_memory, mutable>
      scf.yield %46, %39, %28, %43, %49, %50, %51, %29 : tensor<16x16xf32, #mma>, tensor<16x16x!tt.ptr<f16>, #blocked1>, tensor<16x!tt.ptr<i64>, #triton_gpu.slice<{dim = 1, parent = #blocked}>>, i32, i32, !tt.memdesc<16x16xf16, #shared, #triton_gpu.shared_memory, mutable>, !tt.memdesc<16x16xf16, #shared, #triton_gpu.shared_memory, mutable>, tensor<16xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    }
    triton_gpu.local_dealloc %19 : !tt.memdesc<1x16x16xf16, #shared, #triton_gpu.shared_memory, mutable>
    triton_gpu.local_dealloc %20 : !tt.memdesc<1x16x16xf16, #shared, #triton_gpu.shared_memory, mutable>
    tt.return %23#0 : tensor<16x16xf32, #mma>
  }
}

// -----
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 2], order = [0, 1]}>
#mma = #triton_gpu.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 1], instrShape = [16, 8]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 2 : i32, triton_gpu.target = "cuda:80", "triton_gpu.threads-per-warp" = 32 : i32} {

// CHECK-LABEL:  tt.func public @matmul_indirect_pipeline
// CHECK:  %{{.*}}:4 = scf.for %[[ARG4:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ARG5:.*]] = %{{.*}}-1_i32, %[[ARG6:.*]] = %{{.*}}-1_i32, %[[ARG7:.*]] = %{{.*}}, %[[ARG8:.*]] = %{{.*}})

// CHECK:  %[[CMPI_20:.*]] = arith.cmpi slt, %[[ARG4]], %{{.*}}
// CHECK:  %[[CMPI_21:.*]] = arith.cmpi slt, %[[ARG4]], %{{.*}}
// CHECK:  %[[SPLAT_22:.*]] = tt.splat %[[CMPI_21]]
// CHECK:  %[[ADDPTR_23:.*]] = tt.addptr %{{.*}}, %[[ARG8]]
// CHECK:  %[[LOAD_24:.*]] = tt.load %[[ADDPTR_23]], %[[SPLAT_22]]
// CHECK:  %[[SPLAT_25:.*]] = tt.splat %[[CMPI_20]]
// CHECK:  %[[LOAD_26:.*]] = tt.load %{{.*}}, %[[SPLAT_25]]
// CHECK:  %[[ADDI_27:.*]] = arith.addi %[[ARG5]], %{{.*}}
// CHECK:  %[[CMPI_28:.*]] = arith.cmpi slt, %[[ADDI_27]], %{{.*}}
// CHECK:  %[[SELECT_29:.*]] = arith.select %[[CMPI_28]], %[[ADDI_27]], %{{.*}}
// CHECK:  %[[ADDI_30:.*]] = arith.addi %[[ARG6]], %{{.*}}
// CHECK:  %[[CMPI_31:.*]] = arith.cmpi slt, %[[ADDI_30]], %{{.*}}
// CHECK:  %[[SELECT_32:.*]] = arith.select %[[CMPI_31]], %[[ADDI_30]], %{{.*}}
// CHECK:  %[[EXPAND_DIMS_33:.*]] = tt.expand_dims %[[ARG7]] {axis = 0 : i32}
// CHECK:  %[[BROADCAST_34:.*]] = tt.broadcast %[[EXPAND_DIMS_33]]
// CHECK:  %[[ADDF_35:.*]] = arith.addf %{{.*}}, %[[BROADCAST_34]]
// CHECK:  %[[CONVERT_LAYOUT_36:.*]] = triton_gpu.convert_layout %{{.*}}
// CHECK:  %[[CONVERT_LAYOUT_37:.*]] = triton_gpu.convert_layout %[[ADDF_35]]
// CHECK:  %[[DOT_38:.*]] = tt.dot %[[CONVERT_LAYOUT_36]], %[[CONVERT_LAYOUT_37]], %{{.*}}
// CHECK:  %[[CONVERT_LAYOUT_39:.*]] = triton_gpu.convert_layout %[[DOT_38]]
// CHECK:  tt.store %{{.*}}, %[[CONVERT_LAYOUT_39]]
// CHECK:  scf.yield %[[SELECT_29]], %[[SELECT_32]], %[[LOAD_24]], %[[LOAD_26]]
// CHECK:  }

  tt.func public @matmul_indirect_pipeline(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c-1_i32 = arith.constant -1 : i32
    %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %1 = tt.splat %arg1 : !tt.ptr<i64> -> tensor<32x!tt.ptr<i64>, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %2 = tt.addptr %1, %0 : tensor<32x!tt.ptr<i64>, #triton_gpu.slice<{dim = 0, parent = #blocked}>>, tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %3 = tt.load %2 : tensor<32x!tt.ptr<i64>, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %4 = tt.load %2 : tensor<32x!tt.ptr<i64>, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %5 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %6 = tt.addptr %5, %4 : tensor<32x!tt.ptr<f32>, #triton_gpu.slice<{dim = 0, parent = #blocked}>>, tensor<32xi64, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %7 = tt.load %6 : tensor<32x!tt.ptr<f32>, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %8 = tt.expand_dims %0 {axis = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi32, #blocked>
    %9 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %10 = tt.expand_dims %9 {axis = 1 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<32x1xi32, #blocked>
    %11 = tt.broadcast %8 : tensor<1x32xi32, #blocked> -> tensor<32x32xi32, #blocked>
    %12 = tt.broadcast %10 : tensor<32x1xi32, #blocked> -> tensor<32x32xi32, #blocked>
    %13 = arith.addi %12, %11 : tensor<32x32xi32, #blocked>
    %14 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #blocked>
    %15 = tt.addptr %14, %13 : tensor<32x32x!tt.ptr<f32>, #blocked>, tensor<32x32xi32, #blocked>
    %16 = tt.load %15 : tensor<32x32x!tt.ptr<f32>, #blocked>
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c0_i32 = arith.constant 0 : i32
    %17 = tt.splat %arg3 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #blocked>
    %18 = tt.addptr %17, %13 : tensor<32x32x!tt.ptr<f32>, #blocked>, tensor<32x32xi32, #blocked>
    %19:4 = scf.for %arg4 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg5 = %c-1_i32, %arg6 = %c-1_i32, %arg7 = %7, %arg8 = %3) -> (i32, i32, tensor<32xf32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>, tensor<32xi64, #triton_gpu.slice<{dim = 0, parent = #blocked}>>)  : i32 {
      %20 = arith.cmpi slt, %arg4, %c0_i32 : i32
      %21 = tt.splat %20 : i1 -> tensor<32xi1, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
      %22 = tt.load %2, %21 : tensor<32x!tt.ptr<i64>, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
      %23 = arith.cmpi slt, %arg4, %c1_i32 : i32
      %24 = tt.splat %23 : i1 -> tensor<32xi1, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
      %25 = tt.addptr %5, %arg8 : tensor<32x!tt.ptr<f32>, #triton_gpu.slice<{dim = 0, parent = #blocked}>>, tensor<32xi64, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
      %26 = tt.load %25, %24 : tensor<32x!tt.ptr<f32>, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
      %27 = arith.addi %arg5, %c1_i32 : i32
      %28 = arith.cmpi slt, %27, %c1_i32 : i32
      %29 = arith.select %28, %27, %c0_i32 : i32
      %30 = arith.addi %arg6, %c1_i32 : i32
      %31 = arith.cmpi slt, %30, %c1_i32 : i32
      %32 = arith.select %31, %30, %c0_i32 : i32
      %33 = tt.expand_dims %arg7 {axis = 0 : i32} : tensor<32xf32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xf32, #blocked>
      %34 = tt.broadcast %33 : tensor<1x32xf32, #blocked> -> tensor<32x32xf32, #blocked>
      %35 = arith.addf %16, %34 : tensor<32x32xf32, #blocked>
      %36 = triton_gpu.convert_layout %16 : tensor<32x32xf32, #blocked> -> tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
      %37 = triton_gpu.convert_layout %35 : tensor<32x32xf32, #blocked> -> tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>
      %38 = tt.dot %36, %37, %cst : tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>> -> tensor<32x32xf32, #mma>
      %39 = triton_gpu.convert_layout %38 : tensor<32x32xf32, #mma> -> tensor<32x32xf32, #blocked>
      tt.store %18, %39 : tensor<32x32x!tt.ptr<f32>, #blocked>
      scf.yield %29, %32, %26, %22 : i32, i32, tensor<32xf32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>, tensor<32xi64, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    }
    tt.return
  }
}

// -----
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #triton_gpu.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = []}>
#shared = #triton_gpu.shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [1, 0], hasLeadingOffset = false}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, triton_gpu.target = "cuda:80"} {

// CHECK-LABEL:  tt.func @matmul_nested_ops
// CHECK:  %{{.*}}:5 = scf.for %[[ARG6:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ARG7:.*]] = %{{.*}}, %[[ARG8:.*]] = %{{.*}}-1_i32, %[[ARG9:.*]] = %{{.*}}, %[[ARG10:.*]] = %{{.*}}, %[[ARG11:.*]] = %{{.*}})

// CHECK:  %[[SUBI_19:.*]] = arith.subi %{{.*}}, %{{.*}}
// CHECK:  %[[CMPI_20:.*]] = arith.cmpi slt, %[[ARG6]], %[[SUBI_19]]
// CHECK:  %[[ADDI_21:.*]] = arith.addi %[[ARG6]], %{{.*}}
// CHECK:  %[[ADDPTR_22:.*]] = tt.addptr %[[ARG10]], %{{.*}}
// CHECK:  %[[CMPI_23:.*]] = arith.cmpi slt, %[[ADDI_21]], %{{.*}}
// CHECK:  %[[SPLAT_24:.*]] = tt.splat %[[CMPI_20]]
// CHECK:  %[[IF_25:.*]] = scf.if %[[CMPI_23]] -> (tensor<128x32x!tt.ptr<f16>, #blocked1>) {

// CHECK:  %[[ADDPTR_37:.*]] = tt.addptr %[[ADDPTR_22]], %{{.*}}
// CHECK:  scf.yield %[[ADDPTR_37]]
// CHECK:  } else {

// CHECK:  scf.yield %[[ADDPTR_22]]
// CHECK:  }

// CHECK:  %[[LOAD_26:.*]] = tt.load %[[IF_25]], %[[SPLAT_24]]
// CHECK:  %[[ADDI_27:.*]] = arith.addi %[[ARG8]], %{{.*}}
// CHECK:  %[[CMPI_28:.*]] = arith.cmpi slt, %[[ADDI_27]], %{{.*}}
// CHECK:  %[[SELECT_29:.*]] = arith.select %[[CMPI_28]], %[[ADDI_27]], %{{.*}}
// CHECK:  %[[LOCAL_LOAD_30:.*]] = triton_gpu.local_load %[[ARG11]]
// CHECK:  %[[CONVERT_LAYOUT_31:.*]] = triton_gpu.convert_layout %{{.*}}
// CHECK:  %[[DOT_32:.*]] = tt.dot %[[LOCAL_LOAD_30]], %[[CONVERT_LAYOUT_31]], %[[ARG7]]
// CHECK:  %[[ADDI_33:.*]] = arith.addi %[[ARG9]], %{{.*}}
// CHECK:  %[[CMPI_34:.*]] = arith.cmpi slt, %[[ADDI_33]], %{{.*}}
// CHECK:  %[[SELECT_35:.*]] = arith.select %[[CMPI_34]], %[[ADDI_33]], %{{.*}}
// CHECK:  %[[MEMDESC_SUBVIEW_36:.*]] = triton_gpu.memdesc_subview %{{.*}}[%[[SELECT_35]], %{{.*}}, %{{.*}}]
// CHECK:  triton_gpu.local_store %[[LOAD_26]], %[[MEMDESC_SUBVIEW_36]]
// CHECK:  scf.yield %[[DOT_32]], %[[SELECT_29]], %[[SELECT_35]], %[[IF_25]], %[[MEMDESC_SUBVIEW_36]]
// CHECK:  }

  tt.func @matmul_nested_ops(%arg0: index, %arg1: index, %arg2: index, %arg3: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg5: index) -> tensor<128x128xf32, #mma> {
    %c1_i32 = arith.constant 1 : i32
    %0 = arith.cmpi slt, %arg0, %arg1 : index
    %1 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %2 = tt.expand_dims %1 {axis = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi32, #blocked>
    %3 = tt.broadcast %2 : tensor<1x32xi32, #blocked> -> tensor<128x32xi32, #blocked>
    %4 = tt.splat %arg3 : !tt.ptr<f16> -> tensor<128x32x!tt.ptr<f16>, #blocked>
    %5 = tt.addptr %4, %3 : tensor<128x32x!tt.ptr<f16>, #blocked>, tensor<128x32xi32, #blocked>
    %cst = arith.constant dense<4> : tensor<128x32xi32, #blocked>
    %6 = arith.cmpi slt, %arg0, %arg5 : index
    %7 = tt.splat %0 : i1 -> tensor<128x32xi1, #blocked>
    %8 = scf.if %6 -> (tensor<128x32x!tt.ptr<f16>, #blocked>) {
      %19 = tt.addptr %5, %cst : tensor<128x32x!tt.ptr<f16>, #blocked>, tensor<128x32xi32, #blocked>
      scf.yield %19 : tensor<128x32x!tt.ptr<f16>, #blocked>
    } else {
      scf.yield %5 : tensor<128x32x!tt.ptr<f16>, #blocked>
    }
    %9 = tt.load %8, %7 : tensor<128x32x!tt.ptr<f16>, #blocked>
    %10 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %11 = tt.expand_dims %10 {axis = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x128xi32, #blocked1>
    %12 = tt.broadcast %11 : tensor<1x128xi32, #blocked1> -> tensor<32x128xi32, #blocked1>
    %13 = tt.splat %arg4 : !tt.ptr<f16> -> tensor<32x128x!tt.ptr<f16>, #blocked1>
    %14 = tt.addptr %13, %12 : tensor<32x128x!tt.ptr<f16>, #blocked1>, tensor<32x128xi32, #blocked1>
    %15 = tt.load %14 : tensor<32x128x!tt.ptr<f16>, #blocked1>
    %c0_i32 = arith.constant 0 : i32
    %c-1_i32 = arith.constant -1 : i32
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %16 = triton_gpu.local_alloc  : () -> !tt.memdesc<1x128x32xf16, #shared, #triton_gpu.shared_memory, mutable>
    %17 = triton_gpu.memdesc_subview %16[%c0_i32, %c0_i32, %c0_i32] : !tt.memdesc<1x128x32xf16, #shared, #triton_gpu.shared_memory, mutable> -> !tt.memdesc<128x32xf16, #shared, #triton_gpu.shared_memory, mutable>
    triton_gpu.local_store %9, %17 : tensor<128x32xf16, #blocked> -> !tt.memdesc<128x32xf16, #shared, #triton_gpu.shared_memory, mutable>
    %18:5 = scf.for %arg6 = %arg0 to %arg1 step %arg2 iter_args(%arg7 = %cst_0, %arg8 = %c-1_i32, %arg9 = %c0_i32, %arg10 = %8, %arg11 = %17) -> (tensor<128x128xf32, #mma>, i32, i32, tensor<128x32x!tt.ptr<f16>, #blocked>, !tt.memdesc<128x32xf16, #shared, #triton_gpu.shared_memory, mutable>) {
      %19 = arith.subi %arg1, %arg2 : index
      %20 = arith.cmpi slt, %arg6, %19 : index
      %21 = arith.addi %arg6, %arg2 : index
      %22 = tt.addptr %arg10, %cst : tensor<128x32x!tt.ptr<f16>, #blocked>, tensor<128x32xi32, #blocked>
      %23 = arith.cmpi slt, %21, %arg5 : index
      %24 = tt.splat %20 : i1 -> tensor<128x32xi1, #blocked>
      %25 = scf.if %23 -> (tensor<128x32x!tt.ptr<f16>, #blocked>) {
        %37 = tt.addptr %22, %cst : tensor<128x32x!tt.ptr<f16>, #blocked>, tensor<128x32xi32, #blocked>
        scf.yield %37 : tensor<128x32x!tt.ptr<f16>, #blocked>
      } else {
        scf.yield %22 : tensor<128x32x!tt.ptr<f16>, #blocked>
      }
      %26 = tt.load %25, %24 : tensor<128x32x!tt.ptr<f16>, #blocked>
      %27 = arith.addi %arg8, %c1_i32 : i32
      %28 = arith.cmpi slt, %27, %c1_i32 : i32
      %29 = arith.select %28, %27, %c0_i32 : i32
      %30 = triton_gpu.local_load %arg11 : !tt.memdesc<128x32xf16, #shared, #triton_gpu.shared_memory, mutable> -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %31 = triton_gpu.convert_layout %15 : tensor<32x128xf16, #blocked1> -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %32 = tt.dot %30, %31, %arg7 : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x128xf32, #mma>
      %33 = arith.addi %arg9, %c1_i32 : i32
      %34 = arith.cmpi slt, %33, %c1_i32 : i32
      %35 = arith.select %34, %33, %c0_i32 : i32
      %36 = triton_gpu.memdesc_subview %16[%35, %c0_i32, %c0_i32] : !tt.memdesc<1x128x32xf16, #shared, #triton_gpu.shared_memory, mutable> -> !tt.memdesc<128x32xf16, #shared, #triton_gpu.shared_memory, mutable>
      triton_gpu.local_store %26, %36 : tensor<128x32xf16, #blocked> -> !tt.memdesc<128x32xf16, #shared, #triton_gpu.shared_memory, mutable>
      scf.yield %32, %29, %35, %25, %36 : tensor<128x128xf32, #mma>, i32, i32, tensor<128x32x!tt.ptr<f16>, #blocked>, !tt.memdesc<128x32xf16, #shared, #triton_gpu.shared_memory, mutable>
    }
    triton_gpu.local_dealloc %16 : !tt.memdesc<1x128x32xf16, #shared, #triton_gpu.shared_memory, mutable>
    tt.return %18#0 : tensor<128x128xf32, #mma>
  }
}

// -----
#blocked = #triton_gpu.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #triton_gpu.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16, 16]}>
#shared = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1], hasLeadingOffset = true}>
#shared1 = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0], hasLeadingOffset = true}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, triton_gpu.target = "cuda:80", "triton_gpu.threads-per-warp" = 32 : i32} {

// CHECK-LABEL:  tt.func @dot_prologue_epilogue
// CHECK:  %{{.*}}:6 = scf.for %[[ARG4:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ARG5:.*]] = %{{.*}}, %[[ARG6:.*]] = %{{.*}}, %[[ARG7:.*]] = %{{.*}}, %[[ARG8:.*]] = %{{.*}}-1_i32, %[[ARG9:.*]] = %{{.*}}-1_i32, %[[ARG10:.*]] = %{{.*}})

// CHECK:  %[[CMPI_12:.*]] = arith.cmpi slt, %[[ARG4]], %{{.*}}
// CHECK:  %[[CMPI_13:.*]] = arith.cmpi slt, %[[ARG4]], %{{.*}}
// CHECK:  %[[IF_14:.*]] = scf.if %[[CMPI_13]] -> (tensor<64x16x!tt.ptr<f16>, #blocked>) {

// CHECK:  %[[ADDPTR_30:.*]] = tt.addptr %[[ARG6]], %{{.*}}
// CHECK:  scf.yield %[[ADDPTR_30]]
// CHECK:  } else {

// CHECK:  scf.yield %[[ARG6]]
// CHECK:  }

// CHECK:  %[[LOAD_15:.*]] = tt.load %[[IF_14]]
// CHECK:  %[[SPLAT_16:.*]] = tt.splat %[[CMPI_12]]
// CHECK:  %[[ADDPTR_17:.*]] = tt.addptr %[[ARG7]], %{{.*}}
// CHECK:  %[[LOAD_18:.*]] = tt.load %[[ADDPTR_17]], %[[SPLAT_16]]
// CHECK:  %[[LOCAL_ALLOC_19:.*]] = triton_gpu.local_alloc %[[LOAD_15]]
// CHECK:  %[[ADDI_20:.*]] = arith.addi %[[ARG8]], %{{.*}}
// CHECK:  %[[CMPI_21:.*]] = arith.cmpi slt, %[[ADDI_20]], %{{.*}}
// CHECK:  %[[SELECT_22:.*]] = arith.select %[[CMPI_21]], %[[ADDI_20]], %{{.*}}
// CHECK:  %[[ADDI_23:.*]] = arith.addi %[[ARG9]], %{{.*}}
// CHECK:  %[[CMPI_24:.*]] = arith.cmpi slt, %[[ADDI_23]], %{{.*}}
// CHECK:  %[[SELECT_25:.*]] = arith.select %[[CMPI_24]], %[[ADDI_23]], %{{.*}}
// CHECK:  %[[LOCAL_ALLOC_26:.*]] = triton_gpu.local_alloc %[[ARG10]]
// CHECK:  %[[WARP_GROUP_DOT_27:.*]] = triton_nvidia_gpu.warp_group_dot %[[LOCAL_ALLOC_26]], %[[LOCAL_ALLOC_19]], %[[ARG5]]
// CHECK:  %[[ADDPTR_28:.*]] = tt.addptr %[[ARG6]], %{{.*}}
// CHECK:  %[[IF_29:.*]] = scf.if %[[CMPI_13]] -> (tensor<128x16xf32, #mma>) {

// CHECK:  %[[MULF_30:.*]] = arith.mulf %[[WARP_GROUP_DOT_27]], %{{.*}}
// CHECK:  scf.yield %[[MULF_30]]
// CHECK:  } else {

// CHECK:  scf.yield %[[WARP_GROUP_DOT_27]]
// CHECK:  }

// CHECK:  scf.yield %[[IF_29]], %[[ADDPTR_28]], %[[ADDPTR_17]], %[[SELECT_22]], %[[SELECT_25]], %[[LOAD_18]]
// CHECK:  }

  tt.func @dot_prologue_epilogue(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: i32, %arg3: tensor<64x16xi32, #blocked> {tt.divisibility = 16 : i32}) -> tensor<128x16xf32, #mma> {
    %c7_i32 = arith.constant 7 : i32
    %0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %1 = tt.expand_dims %0 {axis = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
    %2 = tt.broadcast %1 : tensor<1x64xi32, #blocked1> -> tensor<128x64xi32, #blocked1>
    %3 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<128x64x!tt.ptr<f16>, #blocked1>
    %4 = tt.addptr %3, %2 : tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<128x64xi32, #blocked1>
    %5 = tt.load %4 : tensor<128x64x!tt.ptr<f16>, #blocked1>
    %c-1_i32 = arith.constant -1 : i32
    %cst = arith.constant dense<0> : tensor<64x16xi32, #blocked>
    %cst_0 = arith.constant dense<0> : tensor<128x64xi32, #blocked1>
    %c0_i32 = arith.constant 0 : i32
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<128x16xf32, #mma>
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32
    %6 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %7 = tt.expand_dims %6 {axis = 1 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %8 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<64x16x!tt.ptr<f16>, #blocked>
    %9 = tt.broadcast %7 : tensor<64x1xi32, #blocked> -> tensor<64x16xi32, #blocked>
    %10 = tt.addptr %8, %9 : tensor<64x16x!tt.ptr<f16>, #blocked>, tensor<64x16xi32, #blocked>
    %11:6 = scf.for %arg4 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg5 = %cst_1, %arg6 = %10, %arg7 = %4, %arg8 = %c-1_i32, %arg9 = %c-1_i32, %arg10 = %5) -> (tensor<128x16xf32, #mma>, tensor<64x16x!tt.ptr<f16>, #blocked>, tensor<128x64x!tt.ptr<f16>, #blocked1>, i32, i32, tensor<128x64xf16, #blocked1>)  : i32 {
      %12 = arith.cmpi slt, %arg4, %c7_i32 : i32
      %13 = tt.splat %12 : i1 -> tensor<128x64xi1, #blocked1>
      %14 = tt.addptr %arg7, %cst_0 : tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<128x64xi32, #blocked1>
      %15 = tt.load %14, %13 : tensor<128x64x!tt.ptr<f16>, #blocked1>
      %16 = arith.cmpi slt, %arg4, %arg2 : i32
      %17 = scf.if %16 -> (tensor<64x16x!tt.ptr<f16>, #blocked>) {
        %30 = tt.addptr %arg6, %arg3 : tensor<64x16x!tt.ptr<f16>, #blocked>, tensor<64x16xi32, #blocked>
        scf.yield %30 : tensor<64x16x!tt.ptr<f16>, #blocked>
      } else {
        scf.yield %arg6 : tensor<64x16x!tt.ptr<f16>, #blocked>
      }
      %18 = tt.load %17 : tensor<64x16x!tt.ptr<f16>, #blocked>
      %19 = arith.addi %arg8, %c1_i32 : i32
      %20 = arith.cmpi slt, %19, %c1_i32 : i32
      %21 = arith.select %20, %19, %c0_i32 : i32
      %22 = arith.addi %arg9, %c1_i32 : i32
      %23 = arith.cmpi slt, %22, %c1_i32 : i32
      %24 = arith.select %23, %22, %c0_i32 : i32
      %25 = triton_gpu.local_alloc %18 : (tensor<64x16xf16, #blocked>) -> !tt.memdesc<64x16xf16, #shared, #triton_gpu.shared_memory>
      %26 = triton_gpu.local_alloc %arg10 : (tensor<128x64xf16, #blocked1>) -> !tt.memdesc<128x64xf16, #shared1, #triton_gpu.shared_memory>
      %27 = triton_nvidia_gpu.warp_group_dot %26, %25, %arg5 : !tt.memdesc<128x64xf16, #shared1, #triton_gpu.shared_memory> * !tt.memdesc<64x16xf16, #shared, #triton_gpu.shared_memory> -> tensor<128x16xf32, #mma>
      %28 = tt.addptr %arg6, %cst : tensor<64x16x!tt.ptr<f16>, #blocked>, tensor<64x16xi32, #blocked>
      %29 = scf.if %16 -> (tensor<128x16xf32, #mma>) {
        %30 = arith.mulf %27, %cst_1 : tensor<128x16xf32, #mma>
        scf.yield %30 : tensor<128x16xf32, #mma>
      } else {
        scf.yield %27 : tensor<128x16xf32, #mma>
      }
      scf.yield %29, %28, %14, %21, %24, %15 : tensor<128x16xf32, #mma>, tensor<64x16x!tt.ptr<f16>, #blocked>, tensor<128x64x!tt.ptr<f16>, #blocked1>, i32, i32, tensor<128x64xf16, #blocked1>
    }
    tt.return %11#0 : tensor<128x16xf32, #mma>
  }
}

// -----
#blocked = #triton_gpu.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #triton_gpu.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16, 16]}>
#shared = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1], hasLeadingOffset = true}>
#shared1 = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0], hasLeadingOffset = true}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, triton_gpu.target = "cuda:80", "triton_gpu.threads-per-warp" = 32 : i32} {

// CHECK-LABEL:  tt.func @pipeline_downstream_dependencies
// CHECK:  %{{.*}}:6 = scf.for %[[ARG4:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ARG5:.*]] = %{{.*}}, %[[ARG6:.*]] = %{{.*}}, %[[ARG7:.*]] = %{{.*}}, %[[ARG8:.*]] = %{{.*}}-1_i32, %[[ARG9:.*]] = %{{.*}}-1_i32, %[[ARG10:.*]] = %{{.*}})

// CHECK:  %[[CMPI_12:.*]] = arith.cmpi slt, %[[ARG4]], %{{.*}}
// CHECK:  %[[LOAD_13:.*]] = tt.load %[[ARG6]]
// CHECK:  %[[SPLAT_14:.*]] = tt.splat %[[CMPI_12]]
// CHECK:  %[[ADDPTR_15:.*]] = tt.addptr %[[ARG7]], %{{.*}}
// CHECK:  %[[LOAD_16:.*]] = tt.load %[[ADDPTR_15]], %[[SPLAT_14]]
// CHECK:  %[[LOCAL_ALLOC_17:.*]] = triton_gpu.local_alloc %[[LOAD_13]]
// CHECK:  %[[ADDI_18:.*]] = arith.addi %[[ARG8]], %{{.*}}
// CHECK:  %[[CMPI_19:.*]] = arith.cmpi slt, %[[ADDI_18]], %{{.*}}
// CHECK:  %[[SELECT_20:.*]] = arith.select %[[CMPI_19]], %[[ADDI_18]], %{{.*}}
// CHECK:  %[[ADDI_21:.*]] = arith.addi %[[ARG9]], %{{.*}}
// CHECK:  %[[CMPI_22:.*]] = arith.cmpi slt, %[[ADDI_21]], %{{.*}}
// CHECK:  %[[SELECT_23:.*]] = arith.select %[[CMPI_22]], %[[ADDI_21]], %{{.*}}
// CHECK:  %[[LOCAL_ALLOC_24:.*]] = triton_gpu.local_alloc %[[ARG10]]
// CHECK:  %[[WARP_GROUP_DOT_25:.*]] = triton_nvidia_gpu.warp_group_dot %[[LOCAL_ALLOC_24]], %[[LOCAL_ALLOC_17]], %[[ARG5]]
// CHECK:  %[[CMPI_26:.*]] = arith.cmpi slt, %[[ARG4]], %{{.*}}
// CHECK:  %[[SELECT_27:.*]] = arith.select %[[CMPI_26]], %{{.*}}, %{{.*}}
// CHECK:  %[[IF_28:.*]] = scf.if %[[CMPI_26]] -> (tensor<128x16xf32, #mma>) {

// CHECK:  %[[MULF_30:.*]] = arith.mulf %[[WARP_GROUP_DOT_25]], %{{.*}}
// CHECK:  scf.yield %[[MULF_30]]
// CHECK:  } else {

// CHECK:  scf.yield %[[WARP_GROUP_DOT_25]]
// CHECK:  }

// CHECK:  %[[ADDPTR_29:.*]] = tt.addptr %[[ARG6]], %[[SELECT_27]]
// CHECK:  scf.yield %[[IF_28]], %[[ADDPTR_29]], %[[ADDPTR_15]], %[[SELECT_20]], %[[SELECT_23]], %[[LOAD_16]]
// CHECK:  }

  tt.func @pipeline_downstream_dependencies(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: i32, %arg3: tensor<64x16xi32, #blocked> {tt.divisibility = 16 : i32}) -> tensor<128x16xf32, #mma> {
    %c7_i32 = arith.constant 7 : i32
    %0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %1 = tt.expand_dims %0 {axis = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
    %2 = tt.broadcast %1 : tensor<1x64xi32, #blocked1> -> tensor<128x64xi32, #blocked1>
    %3 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<128x64x!tt.ptr<f16>, #blocked1>
    %4 = tt.addptr %3, %2 : tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<128x64xi32, #blocked1>
    %5 = tt.load %4 : tensor<128x64x!tt.ptr<f16>, #blocked1>
    %c-1_i32 = arith.constant -1 : i32
    %cst = arith.constant dense<0> : tensor<64x16xi32, #blocked>
    %cst_0 = arith.constant dense<1> : tensor<64x16xi32, #blocked>
    %cst_1 = arith.constant dense<0> : tensor<128x64xi32, #blocked1>
    %c0_i32 = arith.constant 0 : i32
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<128x16xf32, #mma>
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32
    %6 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %7 = tt.expand_dims %6 {axis = 1 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %8 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<64x16x!tt.ptr<f16>, #blocked>
    %9 = tt.broadcast %7 : tensor<64x1xi32, #blocked> -> tensor<64x16xi32, #blocked>
    %10 = tt.addptr %8, %9 : tensor<64x16x!tt.ptr<f16>, #blocked>, tensor<64x16xi32, #blocked>
    %11:6 = scf.for %arg4 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg5 = %cst_2, %arg6 = %10, %arg7 = %4, %arg8 = %c-1_i32, %arg9 = %c-1_i32, %arg10 = %5) -> (tensor<128x16xf32, #mma>, tensor<64x16x!tt.ptr<f16>, #blocked>, tensor<128x64x!tt.ptr<f16>, #blocked1>, i32, i32, tensor<128x64xf16, #blocked1>)  : i32 {
      %12 = arith.cmpi slt, %arg4, %c7_i32 : i32
      %13 = tt.splat %12 : i1 -> tensor<128x64xi1, #blocked1>
      %14 = tt.addptr %arg7, %cst_1 : tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<128x64xi32, #blocked1>
      %15 = tt.load %14, %13 : tensor<128x64x!tt.ptr<f16>, #blocked1>
      %16 = tt.load %arg6 : tensor<64x16x!tt.ptr<f16>, #blocked>
      %17 = arith.addi %arg8, %c1_i32 : i32
      %18 = arith.cmpi slt, %17, %c1_i32 : i32
      %19 = arith.select %18, %17, %c0_i32 : i32
      %20 = arith.addi %arg9, %c1_i32 : i32
      %21 = arith.cmpi slt, %20, %c1_i32 : i32
      %22 = arith.select %21, %20, %c0_i32 : i32
      %23 = triton_gpu.local_alloc %16 : (tensor<64x16xf16, #blocked>) -> !tt.memdesc<64x16xf16, #shared, #triton_gpu.shared_memory>
      %24 = triton_gpu.local_alloc %arg10 : (tensor<128x64xf16, #blocked1>) -> !tt.memdesc<128x64xf16, #shared1, #triton_gpu.shared_memory>
      %25 = triton_nvidia_gpu.warp_group_dot %24, %23, %arg5 : !tt.memdesc<128x64xf16, #shared1, #triton_gpu.shared_memory> * !tt.memdesc<64x16xf16, #shared, #triton_gpu.shared_memory> -> tensor<128x16xf32, #mma>
      %26 = arith.cmpi slt, %arg4, %arg2 : i32
      %27 = arith.select %26, %cst, %cst_0 : tensor<64x16xi32, #blocked>
      %28 = scf.if %26 -> (tensor<128x16xf32, #mma>) {
        %30 = arith.mulf %25, %cst_2 : tensor<128x16xf32, #mma>
        scf.yield %30 : tensor<128x16xf32, #mma>
      } else {
        scf.yield %25 : tensor<128x16xf32, #mma>
      }
      %29 = tt.addptr %arg6, %27 : tensor<64x16x!tt.ptr<f16>, #blocked>, tensor<64x16xi32, #blocked>
      scf.yield %28, %29, %14, %19, %22, %15 : tensor<128x16xf32, #mma>, tensor<64x16x!tt.ptr<f16>, #blocked>, tensor<128x64x!tt.ptr<f16>, #blocked1>, i32, i32, tensor<128x64xf16, #blocked1>
    }
    tt.return %11#0 : tensor<128x16xf32, #mma>
  }
}

// -----
#blocked = #triton_gpu.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, triton_gpu.target = "cuda:90", "triton_gpu.threads-per-warp" = 32 : i32} {

// CHECK-LABEL:  tt.func public @masked_add_kernel
// CHECK:  %{{.*}}:10 = scf.for %[[ARG4:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ARG5:.*]] = %{{.*}}-1_i32, %[[ARG6:.*]] = %{{.*}}-1_i32, %[[ARG7:.*]] = %{{.*}}, %[[ARG8:.*]] = %{{.*}}, %[[ARG9:.*]] = %{{.*}}, %[[ARG10:.*]] = %{{.*}}, %[[ARG11:.*]] = %{{.*}}, %[[ARG12:.*]] = %{{.*}}, %[[ARG13:.*]] = %{{.*}}, %[[ARG14:.*]] = %{{.*}})

// CHECK:  %[[CMPI_23:.*]] = arith.cmpi slt, %[[ARG4]], %{{.*}}
// CHECK:  %[[ADDI_24:.*]] = arith.addi %[[ARG4]], %{{.*}}
// CHECK:  %[[ADDI_25:.*]] = arith.addi %{{.*}}, %[[ADDI_24]]
// CHECK:  %[[SPLAT_26:.*]] = tt.splat %[[ADDI_25]]
// CHECK:  %[[ADDI_27:.*]] = arith.addi %[[SPLAT_26]], %{{.*}}
// CHECK:  %[[CMPI_28:.*]] = arith.cmpi slt, %[[ADDI_27]], %{{.*}}
// CHECK:  %[[SPLAT_29:.*]] = tt.splat %[[CMPI_23]]
// CHECK:  %[[ANDI_30:.*]] = arith.andi %[[SPLAT_29]], %[[CMPI_28]]
// CHECK:  %[[ADDPTR_31:.*]] = tt.addptr %{{.*}}, %[[ADDI_27]]
// CHECK:  %[[LOAD_32:.*]] = tt.load %[[ADDPTR_31]], %[[ANDI_30]], %{{.*}}
// CHECK:  %[[SPLAT_33:.*]] = tt.splat %[[CMPI_23]]
// CHECK:  %[[ANDI_34:.*]] = arith.andi %[[SPLAT_33]], %[[CMPI_28]]
// CHECK:  %[[ADDPTR_35:.*]] = tt.addptr %{{.*}}, %[[ADDI_27]]
// CHECK:  %[[LOAD_36:.*]] = tt.load %[[ADDPTR_35]], %[[ANDI_34]], %{{.*}}
// CHECK:  %[[ADDI_37:.*]] = arith.addi %[[ARG5]], %{{.*}}
// CHECK:  %[[CMPI_38:.*]] = arith.cmpi slt, %[[ADDI_37]], %{{.*}}
// CHECK:  %[[SELECT_39:.*]] = arith.select %[[CMPI_38]], %[[ADDI_37]], %{{.*}}
// CHECK:  %[[ADDI_40:.*]] = arith.addi %[[ARG6]], %{{.*}}
// CHECK:  %[[CMPI_41:.*]] = arith.cmpi slt, %[[ADDI_40]], %{{.*}}
// CHECK:  %[[SELECT_42:.*]] = arith.select %[[CMPI_41]], %[[ADDI_40]], %{{.*}}
// CHECK:  %[[ADDF_43:.*]] = arith.addf %[[ARG7]], %[[ARG9]]
// CHECK:  %[[ADDPTR_44:.*]] = tt.addptr %{{.*}}, %[[ARG11]]
// CHECK:  tt.store %[[ADDPTR_44]], %[[ADDF_43]], %[[ARG13]]
// CHECK:  scf.yield %[[SELECT_39]], %[[SELECT_42]], %[[ARG8]], %[[LOAD_32]], %[[ARG10]], %[[LOAD_36]], %[[ARG12]], %[[ADDI_27]], %[[ARG14]], %[[CMPI_28]]
// CHECK:  }

  tt.func public @masked_add_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 16 : i32}) attributes {noinline = false} {
    %c2048_i32 = arith.constant 2048 : i32
    %c1016800_i32 = arith.constant 1016800 : i32
    %0 = tt.get_program_id x : i32
    %c1024_i32 = arith.constant 1024 : i32
    %1 = arith.muli %0, %c1016800_i32 : i32
    %2 = arith.addi %1, %c1024_i32 : i32
    %3 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %4 = tt.splat %2 : i32 -> tensor<1024xi32, #blocked>
    %5 = tt.splat %arg3 : i32 -> tensor<1024xi32, #blocked>
    %6 = arith.addi %4, %3 : tensor<1024xi32, #blocked>
    %7 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %cst = arith.constant dense<0xFF800000> : tensor<1024xf32, #blocked>
    %8 = arith.cmpi slt, %6, %5 : tensor<1024xi32, #blocked>
    %9 = tt.addptr %7, %6 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    %10 = tt.load %9, %8, %cst : tensor<1024x!tt.ptr<f32>, #blocked>
    %11 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %12 = tt.addptr %11, %6 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    %13 = tt.load %12, %8, %cst : tensor<1024x!tt.ptr<f32>, #blocked>
    %14 = tt.splat %1 : i32 -> tensor<1024xi32, #blocked>
    %15 = arith.addi %14, %3 : tensor<1024xi32, #blocked>
    %16 = arith.cmpi slt, %15, %5 : tensor<1024xi32, #blocked>
    %17 = tt.addptr %7, %15 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    %18 = tt.load %17, %16, %cst : tensor<1024x!tt.ptr<f32>, #blocked>
    %19 = tt.addptr %11, %15 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    %20 = tt.load %19, %16, %cst : tensor<1024x!tt.ptr<f32>, #blocked>
    %c1014752_i32 = arith.constant 1014752 : i32
    %c2_i32 = arith.constant 2 : i32
    %c1_i32 = arith.constant 1 : i32
    %c-1_i32 = arith.constant -1 : i32
    %c0_i32 = arith.constant 0 : i32
    %21 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %22:10 = scf.for %arg4 = %c0_i32 to %c1016800_i32 step %c1024_i32 iter_args(%arg5 = %c-1_i32, %arg6 = %c-1_i32, %arg7 = %20, %arg8 = %13, %arg9 = %18, %arg10 = %10, %arg11 = %15, %arg12 = %6, %arg13 = %16, %arg14 = %8) -> (i32, i32, tensor<1024xf32, #blocked>, tensor<1024xf32, #blocked>, tensor<1024xf32, #blocked>, tensor<1024xf32, #blocked>, tensor<1024xi32, #blocked>, tensor<1024xi32, #blocked>, tensor<1024xi1, #blocked>, tensor<1024xi1, #blocked>)  : i32 {
      %23 = arith.cmpi slt, %arg4, %c1014752_i32 : i32
      %24 = arith.addi %arg4, %c2048_i32 : i32
      %25 = arith.addi %1, %24 : i32
      %26 = tt.splat %25 : i32 -> tensor<1024xi32, #blocked>
      %27 = arith.addi %26, %3 : tensor<1024xi32, #blocked>
      %28 = arith.cmpi slt, %27, %5 : tensor<1024xi32, #blocked>
      %29 = tt.splat %23 : i1 -> tensor<1024xi1, #blocked>
      %30 = arith.andi %29, %28 : tensor<1024xi1, #blocked>
      %31 = tt.addptr %7, %27 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
      %32 = tt.load %31, %30, %cst : tensor<1024x!tt.ptr<f32>, #blocked>
      %33 = tt.splat %23 : i1 -> tensor<1024xi1, #blocked>
      %34 = arith.andi %33, %28 : tensor<1024xi1, #blocked>
      %35 = tt.addptr %11, %27 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
      %36 = tt.load %35, %34, %cst : tensor<1024x!tt.ptr<f32>, #blocked>
      %37 = arith.addi %arg5, %c1_i32 : i32
      %38 = arith.cmpi slt, %37, %c2_i32 : i32
      %39 = arith.select %38, %37, %c0_i32 : i32
      %40 = arith.addi %arg6, %c1_i32 : i32
      %41 = arith.cmpi slt, %40, %c2_i32 : i32
      %42 = arith.select %41, %40, %c0_i32 : i32
      %43 = arith.addf %arg7, %arg9 : tensor<1024xf32, #blocked>
      %44 = tt.addptr %21, %arg11 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
      tt.store %44, %43, %arg13 : tensor<1024x!tt.ptr<f32>, #blocked>
      scf.yield %39, %42, %arg8, %36, %arg10, %32, %arg12, %27, %arg14, %28 : i32, i32, tensor<1024xf32, #blocked>, tensor<1024xf32, #blocked>, tensor<1024xf32, #blocked>, tensor<1024xf32, #blocked>, tensor<1024xi32, #blocked>, tensor<1024xi32, #blocked>, tensor<1024xi1, #blocked>, tensor<1024xi1, #blocked>
    }
    tt.return
  }
}
