#include "mlir/Support/LLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

using namespace mlir;
using namespace mlir::triton;

using ::mlir::triton::gpu::DotOperandEncodingAttr;
using ::mlir::triton::gpu::getShapePerCTA;
using ::mlir::triton::gpu::NvidiaMmaEncodingAttr;

using ValueTableFMA = std::map<std::tuple<int, int, int>, Value>;

static ValueTableFMA
getValueTableFromStructFMA(Value val, int batch, int nonK, int K,
                           ConversionPatternRewriter &rewriter, Location loc) {
  ValueTableFMA res;
  auto elems = unpackLLElements(loc, val, rewriter);
  assert(elems.size() == K * nonK * batch);
  int index = 0;
  for (unsigned b = 0; b < batch; ++b)
    for (unsigned k = 0; k < K; ++k)
      for (unsigned i = 0; i < nonK; ++i)
        res[{b, i, k}] = elems[index++];
  return res;
}

template <template <typename> typename Vec, typename T>
llvm::SmallVector<T> expandShapeTo3d(Vec<T> s) {
  int rank = s.size();
  assert(rank == 2 || rank == 3);
  llvm::SmallVector<T> expanded(3 - rank, 1);
  expanded.append(s.begin(), s.end());
  return expanded;
}

template <template <typename> typename Vec, typename T>
llvm::SmallVector<T> expandOrderTo3d(Vec<T> o) {
  int rank = o.size();
  if (rank == 3)
    return llvm::SmallVector<T>(o);
  assert(rank == 2);
  llvm::SmallVector<T> expanded;
  for (auto i : o)
    expanded.emplace_back(i + 1);
  expanded.emplace_back(0);
  return expanded;
}

LogicalResult convertFMADot(triton::DotOp op, triton::DotOp::Adaptor adaptor,
                            const LLVMTypeConverter *typeConverter,
                            ConversionPatternRewriter &rewriter) {
  auto *ctx = rewriter.getContext();
  auto loc = op.getLoc();

  auto A = op.getA();
  auto B = op.getB();
  auto C = op.getC();
  auto D = op.getResult();

  auto aTensorTy = cast<RankedTensorType>(A.getType());
  auto dTensorTy = cast<RankedTensorType>(D.getType());

  auto aShapePerCTA = expandShapeTo3d(getShapePerCTA(aTensorTy));
  auto dShapePerCTA = expandShapeTo3d(getShapePerCTA(dTensorTy));

  BlockedEncodingAttr dLayout =
      cast<BlockedEncodingAttr>(dTensorTy.getEncoding());
  auto order = expandOrderTo3d(dLayout.getOrder());
  auto cc = unpackLLElements(loc, adaptor.getC(), rewriter);

  Value llA = adaptor.getA();
  Value llB = adaptor.getB();

  auto sizePerThread = expandShapeTo3d(getSizePerThread(dLayout));
  auto shapePerCTATile = expandShapeTo3d(getShapePerCTATile(dLayout));

  int K = aShapePerCTA[2];

  // Dot produces 3d matrix of shape [Batch, M, N],
  // distributed between threads of the group.
  // retSize defines number of elements stored in one thread
  unsigned retSize[3];
  for (int i = 0; i < 3; ++i) {
    auto numRep = dShapePerCTA[i] / shapePerCTATile[i];
    if (numRep == 0)
      numRep = 1;
    retSize[i] = numRep * sizePerThread[i];
  }

  auto has =
      getValueTableFromStructFMA(llA, retSize[0], retSize[1], K, rewriter, loc);
  auto hbs =
      getValueTableFromStructFMA(llB, retSize[0], retSize[2], K, rewriter, loc);

  SmallVector<Value> ret = cc;

  for (unsigned b = 0; b < retSize[0]; ++b)
    for (unsigned m = 0; m < retSize[1]; ++m)
      for (unsigned n = 0; n < retSize[2]; ++n)
        for (unsigned k = 0; k < K; ++k) {
          unsigned idx[] = {b, m, n};

          unsigned linearIdx = 0;
          for (auto dim : llvm::reverse(order))
            linearIdx = linearIdx * retSize[dim] + idx[dim];

          ret[linearIdx] = rewriter.create<LLVM::FMulAddOp>(
              loc, has[{b, m, k}], hbs[{b, n, k}], ret[linearIdx]);
        }

  auto res = packLLElements(loc, typeConverter, ret, rewriter, dTensorTy);
  rewriter.replaceOp(op, res);

  return success();
}
