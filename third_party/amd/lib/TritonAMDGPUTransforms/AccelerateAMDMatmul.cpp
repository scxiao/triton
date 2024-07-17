#include "TritonAMDGPUTransforms/MfmaGroup.h"
#include "TritonAMDGPUTransforms/Passes.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include "llvm/Support/Debug.h"
#include <memory>

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace {
using tt::DotOp;
using ttg::AMDMfmaEncodingAttr;
using ttg::AMDWmmaEncodingAttr;
using ttg::BlockedEncodingAttr;
using ttg::ConvertLayoutOp;
using ttg::DotOperandEncodingAttr;
using ttg::SliceEncodingAttr;

enum class MatrixCoreVersion {
  CDNA_MFMA1,
  CDNA_MFMA2,
  CDNA_MFMA3,
  RDNA_WMMA,
  UNKNOWN
};

MatrixCoreVersion getMatrixCoreVersion(StringRef archGen) {
  if (archGen.contains("gfx11"))
    return MatrixCoreVersion::RDNA_WMMA;
  if (archGen.contains("gfx908"))
    return MatrixCoreVersion::CDNA_MFMA1;
  if (archGen.contains("gfx90a"))
    return MatrixCoreVersion::CDNA_MFMA2;
  if (archGen.contains("gfx940") || archGen.contains("gfx941") ||
      archGen.contains("gfx942"))
    return MatrixCoreVersion::CDNA_MFMA3;
  return MatrixCoreVersion::UNKNOWN;
}

int getMfmaVersion(MatrixCoreVersion matrixCoreVer) {
  if (MatrixCoreVersion::CDNA_MFMA1 == matrixCoreVer)
    return 1;
  if (MatrixCoreVersion::CDNA_MFMA2 == matrixCoreVer)
    return 2;
  if (MatrixCoreVersion::CDNA_MFMA3 == matrixCoreVer)
    return 3;
  return 0;
}

SmallVector<unsigned, 2> warpsPerTile(tt::DotOp dotOp,
                                      const ArrayRef<int64_t> shape,
                                      int numWarps,
                                      SmallVector<int64_t, 2> shapePerWarp) {
  auto rank = shape.size();
  // Early exit for batched matmul
  if (rank == 3)
    return {(unsigned)numWarps, 1, 1};

  auto filter = [&dotOp](Operation *op) {
    return op->getParentRegion() == dotOp->getParentRegion();
  };
  mlir::ForwardSliceOptions fwdOpt;
  fwdOpt.filter = filter;
  mlir::BackwardSliceOptions bwdOpt;
  bwdOpt.omitBlockArguments = true;
  bwdOpt.filter = filter;
  auto slices = mlir::getSlice(dotOp, bwdOpt, fwdOpt);
  for (Operation *op : slices)
    if (isa<tt::DotOp>(op) && (op != dotOp))
      return {(unsigned)numWarps, 1};

  SmallVector<int64_t, 2> tensorShape = {shape[0], shape[1]};
  SmallVector<unsigned, 2> ret = {1, 1};
  do {
    if (ret[0] * ret[1] >= numWarps)
      break;
    if (tensorShape[0] / (shapePerWarp[0] * 2) / ret[0] >=
        tensorShape[1] / shapePerWarp[1] / ret[1]) {
      if (ret[0] < tensorShape[0] / shapePerWarp[0]) {
        ret[0] *= 2;
      } else
        ret[1] *= 2;
    } else {
      ret[1] *= 2;
    }
  } while (true);

  if (ret[1] * shapePerWarp[1] > tensorShape[1]) {
    return {ret[1], ret[0]};
  }

  return ret;
}

SmallVector<unsigned, 2>
warpsPerTileMFMA(tt::DotOp dotOp, const ArrayRef<int64_t> shape, int numWarps,
                 SmallVector<int64_t, 2> shapePerWarp) {
  return warpsPerTile(dotOp, shape, numWarps, shapePerWarp);
}

SmallVector<unsigned, 2>
warpsPerTileWMMA(tt::DotOp dotOp, const ArrayRef<int64_t> shape, int numWarps) {
  return warpsPerTile(dotOp, shape, numWarps,
                      {AMDWmmaEncodingAttr::getMNKDimPerWMMAInstr()[0],
                       AMDWmmaEncodingAttr::getMNKDimPerWMMAInstr()[1]});
}

using OperandTypesVector = SmallVector<Type, 4>;
OperandTypesVector
selectMatrixCoreOperandTypes(tt::DotOp dot,
                             ArrayRef<OperandTypesVector> applicableTypes) {
  SmallVector<Value> dotOperands = {dot.getA(), dot.getB(), dot.getC(),
                                    dot.getD()};
  OperandTypesVector initElemTypes;
  llvm::transform(dotOperands, std::back_inserter(initElemTypes), [](Value v) {
    return cast<RankedTensorType>(v.getType()).getElementType();
  });

  // Use simple costmodel to define optimal set of the dot operands.
  // Most expensive - accuracy loss conversions:
  //   - any larger type -> any smaller type;
  //   - float -> int;
  //   - int -> float (not supported for now);
  //   - signed int -> unsigned int;
  //   - unsigned int -> signed int with same or less size.
  // They are never performed, better to use FMA.
  // Supported conversion for now costs `1`, no conversion costs `0`.
  // The model could be improved in the future. For example taken into account
  // chain dot could be detected and result conversion score is decreased.
  int maxConvertCost =
      std::numeric_limits<int32_t>::max() / applicableTypes.front().size();
  auto calcConvertCost = [&](Type fromTy, Type toTy) -> int32_t {
    if (fromTy == toTy)
      return 0;

    // Skip conversion between int and float. Int16/int32 cases are lowered to
    // FMA.
    if (fromTy.isIntOrIndex() != toTy.isIntOrIndex())
      return maxConvertCost;

    if (fromTy.isIntOrIndex() && toTy.isIntOrIndex() &&
        fromTy.isUnsignedInteger() != toTy.isUnsignedInteger())
      return fromTy.isUnsignedInteger() && fromTy.getIntOrFloatBitWidth() <
                                               toTy.getIntOrFloatBitWidth()
                 ? 1
                 : maxConvertCost;

    return fromTy.getIntOrFloatBitWidth() <= toTy.getIntOrFloatBitWidth()
               ? 1
               : maxConvertCost;
  };
  auto minCost = maxConvertCost;
  auto optTypes = OperandTypesVector();
  for (auto types : applicableTypes) {
    assert(types.size() == initElemTypes.size());
    int accumulatedConvertCost = 0;
    for (int i = 0; i < initElemTypes.size(); ++i) {
      accumulatedConvertCost += calcConvertCost(initElemTypes[i], types[i]);
    }
    if (accumulatedConvertCost < minCost) {
      minCost = accumulatedConvertCost;
      optTypes = types;
    }
  }
  return optTypes;
}

OperandTypesVector getOperandTypesForWmmaOp(mlir::PatternRewriter &rewriter,
                                            tt::DotOp dot) {
  Type f16 = rewriter.getF16Type();
  Type f32 = rewriter.getF32Type();
  Type bf16 = rewriter.getBF16Type();
  Type i8 = rewriter.getIntegerType(8);
  Type i32 = rewriter.getIntegerType(32);
  SmallVector<OperandTypesVector> applicableTypes = {
      // clang-format off
      {f16, f16, f32, f32},
      {f16, f16, f16, f16},
      {bf16, bf16, f32, f32},
      {bf16, bf16, bf16, bf16},
      {i8, i8, i32, i32},
      // i4, i4, i32, i32 - is supported configuration
      // by WMMA instruction, but not supported by triton
      // clang-format on
  };
  return selectMatrixCoreOperandTypes(dot, applicableTypes);
}

/**
 * @brief Convert layout and cast element type of a given tensor
 *
 * If old element type is different from new element type, this function
 * creates two new operations:
 * 1. %converted_value = layout_convert %value, newEncoding
 * 2. %casted_value = cast(fext, ftrunc, etc.) %value, newElemType
 *
 * If old element type is same as new element type, this function creates only
 * one operation: %converted_value = layout_convert %value, newEncoding
 *
 * @param rewriter
 * @param value original tensor value, which we need to convert and cast
 * @param newEncoding new encoding for the tenosr
 * @param newElemType new element type for the tensor
 * @return converted and optionaly casted tensor value
 */
Value convertAndCastTensor(mlir::PatternRewriter &rewriter, Value value,
                           ::mlir::Attribute newEncoding, Type newElemType) {
  assert(newElemType.isIntOrFloat());

  auto loc = value.getLoc();
  auto oldType = cast<RankedTensorType>(value.getType());
  auto oldElemType = oldType.getElementType();

  assert(oldElemType.isIntOrFloat());
  assert(oldElemType.isIntOrIndex() == newElemType.isIntOrIndex());

  auto convertedType =
      RankedTensorType::get(oldType.getShape(), oldElemType, newEncoding);

  Value convertedTensor =
      rewriter.create<ttg::ConvertLayoutOp>(loc, convertedType, value);

  if (newElemType == oldElemType)
    return convertedTensor;

  Type castedType = convertedType.cloneWith(std::nullopt, newElemType);

  Value castedTensor;

  if (newElemType.isIntOrIndex()) {
    unsigned oldWidth = oldElemType.getIntOrFloatBitWidth();
    unsigned newWidth = newElemType.getIntOrFloatBitWidth();
    if (oldWidth == newWidth)
      castedTensor = rewriter.create<mlir::arith::BitcastOp>(loc, convertedType,
                                                             convertedTensor);
    else if (oldWidth > newWidth)
      castedTensor = rewriter.create<mlir::arith::TruncIOp>(loc, castedType,
                                                            convertedTensor);
    else if (oldElemType.isSignedInteger())
      castedTensor = rewriter.create<mlir::arith::ExtSIOp>(loc, castedType,
                                                           convertedTensor);
    else
      castedTensor = rewriter.create<mlir::arith::ExtUIOp>(loc, castedType,
                                                           convertedTensor);
  } else {
    if (oldElemType.isF16() && newElemType.isF32())
      castedTensor = rewriter.create<mlir::arith::ExtFOp>(loc, castedType,
                                                          convertedTensor);
    else if (oldElemType.isF32() && newElemType.isF16())
      castedTensor = rewriter.create<mlir::arith::TruncFOp>(loc, castedType,
                                                            convertedTensor);
    else
      castedTensor =
          rewriter.create<tt::FpToFpOp>(loc, castedType, convertedTensor);
  }
  return castedTensor;
}

class BlockedToMFMA : public mlir::RewritePattern {
  int mfmaVersion;
  int enforcedNonKDim;
  int kPack;

public:
  BlockedToMFMA(mlir::MLIRContext *context, int mfmaVersion, int nonKDim,
                int kPack)
      : mlir::RewritePattern(tt::DotOp::getOperationName(), 2, context),
        mfmaVersion(mfmaVersion), enforcedNonKDim(nonKDim), kPack(kPack) {}

  bool isChainDot(tt::DotOp &dotOp) const {
    auto filter = [&dotOp](Operation *op) {
      return op->getParentRegion() == dotOp->getParentRegion();
    };
    mlir::ForwardSliceOptions fwdOpt;
    fwdOpt.filter = filter;
    mlir::BackwardSliceOptions bwdOpt;
    bwdOpt.omitBlockArguments = true;
    bwdOpt.filter = filter;
    auto slices = mlir::getSlice(dotOp, bwdOpt, fwdOpt);
    for (Operation *op : slices) {
      if (isa<tt::DotOp>(op) && (op != dotOp))
        return true;
    }
    return false;
  }

  bool isSecondDot(tt::DotOp &dotOp) const {
    auto filter = [&dotOp](Operation *op) {
      return op->getParentRegion() == dotOp->getParentRegion();
    };
    mlir::BackwardSliceOptions bwdOpt;
    bwdOpt.omitBlockArguments = true;
    bwdOpt.filter = filter;
    SetVector<Operation *> slices;
    mlir::getBackwardSlice(dotOp.getResult(), &slices, bwdOpt);
    if (llvm::find_if(slices, [](Operation *op) {
          return isa<tt::DotOp>(op);
        }) != slices.end())
      return true;
    return false;
  }

  /// @brief Choose MFMA instruction parameters
  /// @param dot target dot operation
  /// @return pair {mDim, nDim, kDim, kBase} sizes of one MFMA instruction
  /// arguments
  std::tuple<unsigned, unsigned, unsigned, unsigned>
  chooseMfmaDimensions(tt::DotOp dot) const {
    // number of matrix elements along k dim per one MFMA intruction
    unsigned kDim = 0;
    auto opType = cast<RankedTensorType>(dot.getA().getType());
    auto dataTypeA = opType.getElementType();
    auto dataTypeB =
        cast<RankedTensorType>(dot.getB().getType()).getElementType();

    auto resType = cast<RankedTensorType>(dot.getD().getType());
    auto resShape = resType.getShape();
    auto rank = resShape.size();
    auto M = resShape[rank - 2];
    auto N = resShape[rank - 1];

    unsigned mDim = 0;
    unsigned nDim = 0;
    if (enforcedNonKDim != 0) {
      mDim = enforcedNonKDim;
      nDim = enforcedNonKDim;
    } else {
      int minSize = std::min(M, N);
      if (minSize >= 32) {
        mDim = 32;
        nDim = 32;
      }
      if (minSize >= 16 && minSize < 32) {
        mDim = 16;
        nDim = 16;
      }
      if (minSize < 16) {
        if (M < 16 && N >= 64) {
          mDim = 4;
          nDim = 64;
        } else if (M >= 64 && N < 16) {
          mDim = 64;
          nDim = 4;
        } else {
          assert(opType.getShape()[rank - 1] >= 64 &&
                 "k should be at least 64 to use this layout");
          mDim = 4;
          nDim = 4;
        }
      }
    }
    assert(mDim != 0 && nDim != 0);

    auto maybeMfmaInsn =
        MfmaInsn::selectMfma(mDim, nDim, dataTypeA, dataTypeB, mfmaVersion);
    if (failed(maybeMfmaInsn))
      llvm::report_fatal_error("No match found in MFMA database\n");

    kDim = maybeMfmaInsn->getKDim();
    unsigned kBase = maybeMfmaInsn->getKBase();

    assert(kDim != 0);

    assert(M % mDim == 0 && N % nDim == 0);
    assert(opType.getShape()[rank - 1] % kDim == 0);
    return {mDim, nDim, kDim, kBase};
  }

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto dotOp = cast<tt::DotOp>(op);

    RankedTensorType oldRetType = dotOp.getType();
    if (!oldRetType.getEncoding() ||
        !isa<ttg::BlockedEncodingAttr>(oldRetType.getEncoding()))
      return failure();

    return failure();

    if (!supportMFMA(dotOp))
      return failure();

    auto CTALayout = ttg::getCTALayout(oldRetType.getEncoding());

    // get MFMA encoding for the given number of warps
    auto retShape = oldRetType.getShape();
    auto mod = op->getParentOfType<mlir::ModuleOp>();
    int numWarps = ttg::TritonGPUDialect::getNumWarps(mod);

    // operands
    Value a = dotOp.getA();
    Value b = dotOp.getB();
    auto oldAType = cast<RankedTensorType>(a.getType());
    auto oldBType = cast<RankedTensorType>(b.getType());
    auto ctx = oldAType.getContext();

    ttg::AMDMfmaEncodingAttr mfmaEnc;

    auto [mDim, nDim, kDim, kBase] = chooseMfmaDimensions(dotOp);

    auto warpsPerTile =
        warpsPerTileMFMA(dotOp, retShape, numWarps, {mDim, nDim});

    bool isTransposed = isChainDot(dotOp);
    mfmaEnc = ttg::AMDMfmaEncodingAttr::get(
        oldRetType.getContext(),
        /*versionMajor*/ mfmaVersion, /*versionMinor*/ 0, warpsPerTile,
        /*instrShape*/ mDim, nDim, isTransposed, CTALayout);

    Type mfmaAccType;
    if (oldRetType.getElementType().isIntOrIndex())
      mfmaAccType = rewriter.getIntegerType(32);
    else
      mfmaAccType = rewriter.getF32Type();

    // convert accumulator
    auto oldAcc = dotOp.getOperand(2);
    auto newAcc = convertAndCastTensor(rewriter, oldAcc, mfmaEnc, mfmaAccType);

    // Here is a brief explanation of kWidth, kBase, and kDim
    // 1. kWidth: the number of elements each thread loads from shared memory in
    //    preparation for mfma instructions. In theory each thread can issue one
    //    or more load instructions to load a total of kWidth elements, since
    //    those elements are not required to be in contiguous addresses in
    //    shared memory. But in practice, we make sure the kWidth elements can
    //    be loaded from shared memory by a single ds_read instruction by
    //    setting vecSize of the sharedLayout to be kWidth.
    // 2. kDim: the k dimension size of the mfma instruction. E.g. instruction
    //    mfma_32x32x16 has kDim = 16, meaning this mfma instruction can compute
    //    a matmul of operands with shape 32x16 and 16x32.
    // 3. kBase: the number of elements each thread holds for a single mfma
    //    instruction.
    // 4. relation between kBase and kDim:
    //    4.1 For mfma_32, kBase = kDim / 2
    //    4.2 For mfma_16, kBase = kDim / 4
    //    4.3 For mfma_4, it depends on how mfma_4 is used. We'll extend to
    //        mfma_4 later.
    // 5. relation between kWidth and kBase: For now it supports two cases
    //    5.1 kWidth = kBase, i.e. kPack = 1. In this case, each load from
    //        shared memory results in one mfma instruction.
    //    5.2 kWidth = 2 * kBase, i.e. kPack = 2. In this case, each load from
    //        shared memory results in two mfma instructions, since one mfma
    //        can only consume kBase elements from each thread.
    //    Note that we cannot have larger kPack since kPack = 2 means
    //    ds_read_b128, which is the largest vector size for shared memory load.
    auto kWidth = kBase;
    // in mfma 4x4 case argument matrix groups in 16 groups
    if (mDim == 4 && nDim == 4)
      kWidth = kDim / 16;
    if ((mDim == 4 && nDim == 64) || (mDim == 64 && nDim == 4))
      kWidth = kDim;

    // We want to extend kWidth by kPack (kPack=1 means no extension)
    // to increase ds_read vector size
    // However, in FA, the second dot can only use kWidth = kBase since it's
    // limited by the result of the first dot, which is of mfmaLayout.
    if (!isSecondDot(dotOp))
      kWidth *= kPack;

    auto newAType = RankedTensorType::get(
        oldAType.getShape(), oldAType.getElementType(),
        ttg::DotOperandEncodingAttr::get(ctx, 0, mfmaEnc, kWidth));
    auto newBType = RankedTensorType::get(
        oldBType.getShape(), oldBType.getElementType(),
        ttg::DotOperandEncodingAttr::get(ctx, 1, mfmaEnc, kWidth));
    a = rewriter.create<ttg::ConvertLayoutOp>(a.getLoc(), newAType, a);
    b = rewriter.create<ttg::ConvertLayoutOp>(b.getLoc(), newBType, b);
    auto newDot = rewriter.create<tt::DotOp>(
        dotOp.getLoc(), newAcc.getType(), a, b, newAcc,
        dotOp.getInputPrecision(), dotOp.getMaxNumImpreciseAcc());

    Value dotOutput =
        convertAndCastTensor(rewriter, newDot, oldRetType.getEncoding(),
                             oldRetType.getElementType());

    rewriter.replaceOp(op, dotOutput);

    return success();
  }
};

class AccelerateBlocked : public mlir::RewritePattern {
  StringRef arch;

public:
  AccelerateBlocked(mlir::MLIRContext *context, StringRef arch)
      : mlir::RewritePattern(tt::DotOp::getOperationName(), 1, context),
        arch(arch) {}

  bool isFloat(Type t) const { return t.isIntOrFloat() && !t.isIntOrIndex(); }

  Value castToElTy(mlir::PatternRewriter &rewriter, Value v, Type newTy) const {
    Location loc = v.getLoc();
    if (isFloat(v.getType()) && isFloat(newTy))
      return rewriter.create<triton::FpToFpOp>(loc, newTy, v);
    if (!isFloat(v.getType()) && isFloat(newTy))
      return rewriter.create<mlir::arith::SIToFPOp>(loc, newTy, v);
    if (isFloat(v.getType()) && !isFloat(newTy))
      return rewriter.create<mlir::arith::UIToFPOp>(loc, newTy, v);
    assert(false &&
           "int -> int cast is unexpected, FMA do not support this case");
    return Value();
  }

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto dotOp = cast<tt::DotOp>(op);
    auto a = dotOp.getA();
    auto b = dotOp.getB();
    auto c = dotOp.getC();
    auto d = dotOp.getD();

    Type aElTy = a.getType().getElementType();
    Type bElTy = b.getType().getElementType();
    Type cElTy = c.getType().getElementType();
    Type dElTy = d.getType().getElementType();

    int rank = a.getType().getShape().size();
    int k = a.getType().getShape()[rank - 1];

    bool dotAvailable = arch == "gfx908" || arch == "gfx90a" ||
                        arch.starts_with("gfx94") || arch.starts_with("gfx11");

    // Try Fp16 x Fp16 -> Fp32 dot
    if (dotAvailable && aElTy.isF16() && bElTy.isF16() && cElTy.isF32() &&
        dElTy.isF32()) {
      if (k >= 2)
        return success();
      // if k < 2: can not use DOT instruction, continue with FMA
    }
    // Try I8 x I8 -> I32 dot
    if (dotAvailable && aElTy.isSignedInteger(8) && bElTy.isSignedInteger(8) &&
        cElTy.isSignedInteger(32) && dElTy.isSignedInteger(32)) {
      if (k >= 4)
        return success();
      // if k < 4: can not use DOT instruction, continue with FMA
    }

    // Convert operands to one FP type to apply FMA case
    unsigned maxBitsize =
        std::max(aElTy.getIntOrFloatBitWidth(), bElTy.getIntOrFloatBitWidth());
    maxBitsize = std::max(maxBitsize, std::max(cElTy.getIntOrFloatBitWidth(),
                                               dElTy.getIntOrFloatBitWidth()));

    // Adjust to minimal available FMA bitwidth
    maxBitsize = std::max(maxBitsize, 16u);
    assert(maxBitsize <= 32);

    Type commonTy =
        maxBitsize == 16 ? rewriter.getF16Type() : rewriter.getF32Type();

    auto newA = castToElTy(rewriter, a, commonTy);
    auto newB = castToElTy(rewriter, b, commonTy);
    auto newC = castToElTy(rewriter, c, commonTy);

    auto newDot = rewriter.create<tt::DotOp>(
        dotOp.getLoc(), newC.getType(), newA, newB, newC,
        dotOp.getInputPrecision(), dotOp.getMaxNumImpreciseAcc());
    auto newD = castToElTy(rewriter, newDot.getResult(), dElTy);
    d.replaceAllUsesWith(newD);
    dotOp.erase();
    return success();
  }
};

static Value promoteOperand(OpBuilder &builder, Location loc, Value operand,
                            Type promotedType) {
  Type tensorPromotedType = cast<RankedTensorType>(operand.getType())
                                .cloneWith(std::nullopt, promotedType);
  return builder.create<triton::FpToFpOp>(loc, tensorPromotedType, operand);
}

// promote operands of dot op if the existing combination is not natively
// supported.
static void decomposeMixedModeDotOp(ModuleOp mod) {
  mod.walk([](triton::DotOp dotOp) -> void {
    auto D = dotOp.getD();
    OpBuilder builder(dotOp);
    Type AElType = dotOp.getA().getType().getElementType();
    Type promoteType;
    if (isa<AMDMfmaEncodingAttr>(D.getType().getEncoding())) {
      Type BElType = dotOp.getB().getType().getElementType();

      auto maxBitWidth = std::max(AElType.getIntOrFloatBitWidth(),
                                  BElType.getIntOrFloatBitWidth());

      // TODO check mfma tensor core version compatibility
      if (maxBitWidth == 8)
        return;

      if (AElType == BElType)
        return;

      if (maxBitWidth < 16)
        promoteType = builder.getF16Type();
      else if (maxBitWidth <= 32)
        promoteType = builder.getF32Type();
    } else if (isa<AMDWmmaEncodingAttr>(D.getType().getEncoding())) {
      Type BElType = dotOp.getB().getType().getElementType();

      if (AElType == BElType)
        return;

      // Other cases must be filtered earlier
      promoteType =
          AElType.getIntOrFloatBitWidth() > BElType.getIntOrFloatBitWidth()
              ? AElType
              : BElType;
    } else {
      // All needed transformations are done in AccelerateBlocked pattern
      return;
    }
    Location loc = dotOp.getLoc();
    Value promotedA = promoteOperand(builder, loc, dotOp.getA(), promoteType);
    Value promotedB = promoteOperand(builder, loc, dotOp.getB(), promoteType);
    dotOp.setOperand(0, promotedA);
    dotOp.setOperand(1, promotedB);
  });
}

class BlockedToWMMA : public mlir::RewritePattern {
public:
  BlockedToWMMA(mlir::MLIRContext *context)
      : mlir::RewritePattern(tt::DotOp::getOperationName(), 2, context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto ctx = op->getContext();
    auto dotOp = cast<tt::DotOp>(op);

    Value a = dotOp.getA();
    Value b = dotOp.getB();

    auto oldRetType = cast<RankedTensorType>(dotOp.getResult().getType());
    auto oldRetEncoding = oldRetType.getEncoding();
    if (!oldRetEncoding || !isa<ttg::BlockedEncodingAttr>(oldRetEncoding))
      return failure();

    auto oldAType = cast<RankedTensorType>(a.getType());
    auto oldBType = cast<RankedTensorType>(b.getType());
    auto retShape = oldRetType.getShape();
    auto aShape = oldAType.getShape();
    auto bShape = oldBType.getShape();

    // check shape
    auto mnkDim = AMDWmmaEncodingAttr::getMNKDimPerWMMAInstr();
    auto rank = aShape.size();
    if (aShape[rank - 2] % mnkDim[0] != 0 || // m
        bShape[rank - 1] % mnkDim[1] != 0 || // n
        aShape[rank - 1] % mnkDim[2] != 0)   // k
      return failure();

    // get operand types
    auto operandTypes = getOperandTypesForWmmaOp(rewriter, dotOp);
    if (operandTypes.empty())
      return failure();

    // get WMMA encoding for the given number of warps
    auto mod = op->getParentOfType<mlir::ModuleOp>();
    int numWarps = ttg::TritonGPUDialect::getNumWarps(mod);

    AMDWmmaEncodingAttr wmmaEnc;

    auto warpsPerTile = warpsPerTileWMMA(dotOp, retShape, numWarps);

    auto CTALayout = ttg::getCTALayout(oldRetEncoding);
    wmmaEnc = AMDWmmaEncodingAttr::get(ctx, warpsPerTile, CTALayout);

    auto newRetType = RankedTensorType::get(retShape, operandTypes[3], wmmaEnc);

    // convert accumulator
    auto oldAcc = dotOp.getOperand(2);
    auto newAcc =
        convertAndCastTensor(rewriter, oldAcc, wmmaEnc, operandTypes[2]);

    auto newAType = RankedTensorType::get(
        aShape, operandTypes[0],
        ttg::DotOperandEncodingAttr::get(ctx, 0, wmmaEnc, mnkDim[2]));
    auto newBType = RankedTensorType::get(
        bShape, operandTypes[1],
        ttg::DotOperandEncodingAttr::get(ctx, 1, wmmaEnc, mnkDim[2]));

    Value castedA = convertAndCastTensor(rewriter, a, newAType.getEncoding(),
                                         operandTypes[0]);
    Value castedB = convertAndCastTensor(rewriter, b, newBType.getEncoding(),
                                         operandTypes[1]);
    auto newDot = rewriter.create<tt::DotOp>(
        dotOp.getLoc(), newRetType, castedA, castedB, newAcc,
        dotOp.getInputPrecision(), dotOp.getMaxNumImpreciseAcc());

    Value dotOutput = convertAndCastTensor(rewriter, newDot, oldRetEncoding,
                                           oldRetType.getElementType());
    rewriter.replaceOp(op, dotOutput);
    return success();
  }
};
} // namespace

#define GEN_PASS_CLASSES
#include "TritonAMDGPUTransforms/Passes.h.inc"

class TritonAMDGPUAccelerateMatmulPass
    : public TritonAMDGPUAccelerateMatmulBase<
          TritonAMDGPUAccelerateMatmulPass> {
public:
  TritonAMDGPUAccelerateMatmulPass() = default;
  TritonAMDGPUAccelerateMatmulPass(StringRef archGen, int matrixInstructionSize,
                                   int kPack) {
    this->archGenerationName = archGen.data();
    this->matrixInstructionSize = matrixInstructionSize;
    this->kPack = kPack;
  }
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    mlir::RewritePatternSet patterns(context);
    auto matrixCoreVer = getMatrixCoreVersion(archGenerationName);
    if (MatrixCoreVersion::CDNA_MFMA1 == matrixCoreVer ||
        MatrixCoreVersion::CDNA_MFMA2 == matrixCoreVer ||
        MatrixCoreVersion::CDNA_MFMA3 == matrixCoreVer) {
      patterns.add<::BlockedToMFMA>(context, getMfmaVersion(matrixCoreVer),
                                    matrixInstructionSize, kPack);
    } else if (matrixCoreVer == MatrixCoreVersion::RDNA_WMMA) {
      patterns.add<::BlockedToWMMA>(context);
    }
    patterns.add<::AccelerateBlocked>(context, getAMDArch(m));
    if (applyPatternsAndFoldGreedily(m, std::move(patterns)).failed()) {
      signalPassFailure();
    }
    decomposeMixedModeDotOp(m);
  }
};

std::unique_ptr<Pass> mlir::createTritonAMDGPUAccelerateMatmulPass(
    std::string archGen, int matrixInstructionSize, int kPack) {
  return std::make_unique<TritonAMDGPUAccelerateMatmulPass>(
      archGen, matrixInstructionSize, kPack);
}
