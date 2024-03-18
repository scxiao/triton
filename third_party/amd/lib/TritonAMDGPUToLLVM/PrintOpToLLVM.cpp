#include "PatternTritonGPUOpToLLVM.h"
#include "Utility.h"

namespace {

using namespace mlir;
using namespace mlir::triton;

// The input print op contains:
//  - a "prefix" (string) specified by the user, and
//  - one or more "operands" (tensors).
//
// For each operand, we print all of the values contained in this GPU thread,
// one per line, along with the index of the value in its tensor.
struct PrintOpConversion : public ConvertOpToLLVMPattern<triton::PrintOp> {
  using ConvertOpToLLVMPattern<triton::PrintOp>::ConvertOpToLLVMPattern;

  struct PrintFormatting
  {
    Value formatStrValue;
    size_t formatStrSize;
  };

  static SmallString<16> getUniqueFormatGlobalName(mlir::ModuleOp moduleOp) {
    const char formatStringPrefix[] = "printfFormat_";
    // Get a unique global name.
    unsigned stringNumber = 0;
    SmallString<16> stringConstName;
    do {
      stringConstName.clear();
      (formatStringPrefix + Twine(stringNumber++)).toStringRef(stringConstName);
    } while (moduleOp.lookupSymbol(stringConstName));
    return stringConstName;
  }

  template <typename T>
  static LLVM::LLVMFuncOp
  getOrDefineFunction(T &moduleOp, const Location loc,
                      ConversionPatternRewriter &rewriter, StringRef name,
                      LLVM::LLVMFunctionType type) {
    LLVM::LLVMFuncOp ret;
    if (!(ret = moduleOp.template lookupSymbol<LLVM::LLVMFuncOp>(name))) {
      ConversionPatternRewriter::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      ret = rewriter.create<LLVM::LLVMFuncOp>(loc, name, type,
                                              LLVM::Linkage::External);
    }
    return ret;
  }

  // The code is borrowed from https://reviews.llvm.org/D110448
  // from GPUPrintfOpToHIPLowering::matchAndRewrite().
  void llPrintfHIP(mlir::Location loc, mlir::ModuleOp moduleOp, PrintFormatting formatting,
                   ValueRange args, ConversionPatternRewriter &rewriter,
                   bool stderr = false) const {

    auto typeConverter = getTypeConverter();
    mlir::Type llvmI8 = typeConverter->convertType(rewriter.getI8Type());
    auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    mlir::Type llvmI32 = typeConverter->convertType(rewriter.getI32Type());
    mlir::Type llvmI64 = typeConverter->convertType(rewriter.getI64Type());

    auto ocklBegin = getOrDefineFunction(
        moduleOp, loc, rewriter,
        (stderr ? "__ockl_fprintf_stderr_begin" : "__ockl_printf_begin"),
        (LLVM::LLVMFunctionType::get(llvmI64, stderr ? ArrayRef<mlir::Type>()
                                                     : llvmI64)));
    LLVM::LLVMFuncOp ocklAppendArgs;
    if (!args.empty()) {
      ocklAppendArgs = getOrDefineFunction(
          moduleOp, loc, rewriter, "__ockl_printf_append_args",
          LLVM::LLVMFunctionType::get(llvmI64,
                                      {llvmI64, /*numArgs*/ llvmI32, llvmI64,
                                       llvmI64, llvmI64, llvmI64, llvmI64,
                                       llvmI64, llvmI64, /*isLast*/ llvmI32}));
    }
    auto ocklAppendStringN = getOrDefineFunction(
        moduleOp, loc, rewriter, "__ockl_printf_append_string_n",
        LLVM::LLVMFunctionType::get(
            llvmI64,
            {llvmI64, {ptrType}, /*length (bytes)*/ llvmI64, /*isLast*/ llvmI32}));

    /// Start the printf hostcall
    Value zeroI64 = rewriter.create<LLVM::ConstantOp>(loc, llvmI64, 0);
    auto printfBeginCall = rewriter.create<LLVM::CallOp>(
        loc, ocklBegin, stderr ? ValueRange() : zeroI64);
    Value printfDesc = printfBeginCall.getResult();

    // Get a unique global name for the format.
    SmallString<16> stringConstName = getUniqueFormatGlobalName(moduleOp);

    auto prefixPtrType = ocklAppendStringN.getArgumentTypes()[1];
    Value prefixString = bitcast(formatting.formatStrValue, prefixPtrType);

    Value stringLen =
        rewriter.create<LLVM::ConstantOp>(loc, llvmI64, formatting.formatStrSize);

    Value oneI32 = rewriter.create<LLVM::ConstantOp>(loc, llvmI32, 1);
    Value zeroI32 = rewriter.create<LLVM::ConstantOp>(loc, llvmI32, 0);

    auto appendFormatCall = rewriter.create<LLVM::CallOp>(
        loc, ocklAppendStringN,
        ValueRange{printfDesc, prefixString, stringLen,
                   args.empty() ? oneI32 : zeroI32});
    printfDesc = appendFormatCall.getResult();

    // __ockl_printf_append_args takes 7 values per append call
    constexpr size_t argsPerAppend = 7;
    size_t nArgs = args.size();
    for (size_t group = 0; group < nArgs; group += argsPerAppend) {
      size_t bound = std::min(group + argsPerAppend, nArgs);
      size_t numArgsThisCall = bound - group;

      SmallVector<mlir::Value, 2 + argsPerAppend + 1> arguments;
      arguments.push_back(printfDesc);
      arguments.push_back(
          rewriter.create<LLVM::ConstantOp>(loc, llvmI32, numArgsThisCall));
      for (size_t i = group; i < bound; ++i) {
        Value arg = args[i];
        if (auto floatType = arg.getType().dyn_cast<FloatType>()) {
          if (!floatType.isF64())
            arg = fpext(typeConverter->convertType(rewriter.getF64Type()), arg);
          arg = bitcast(arg, llvmI64);
        }
        if (arg.getType().getIntOrFloatBitWidth() != 64)
           arg = zext(llvmI64, arg);

        arguments.push_back(arg);
      }
      // Pad out to 7 arguments since the hostcall always needs 7
      for (size_t extra = numArgsThisCall; extra < argsPerAppend; ++extra) {
        arguments.push_back(zeroI64);
      }

      auto isLast = (bound == nArgs) ? oneI32 : zeroI32;
      arguments.push_back(isLast);
      auto call = call(ocklAppendArgs, arguments);
      printfDesc = call.getResult();
    }
  }

  // Returns a Value for the format string, which you can reuse.
  PrintFormatting llPrintfHIP(mlir::Location loc, mlir::ModuleOp moduleOp, StringRef msg,
                   ValueRange args, ConversionPatternRewriter &rewriter,
                   bool stderr = false) const {
    assert(!msg.empty() && "printf with empty string not supported");
    PrintFormatting formatting;
    llvm::SmallString<32> msgNewline(msg);
    msgNewline.push_back('\n');
    msgNewline.push_back('\0');
    formatting.formatStrValue =
        LLVM::addStringToModule(loc, rewriter, "printfFormat_", msgNewline);
    formatting.formatStrSize = msgNewline.size_in_bytes();
    llPrintfHIP(loc, moduleOp, formatting, args, rewriter, stderr);
    return formatting;
  }

  LogicalResult
  matchAndRewrite(triton::PrintOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    Value prefixStr =
        LLVM::addStringToModule(loc, rewriter, "printfPrefix_", op.getPrefix());

    auto getPid = [&](int axis) {
      return LLVM::AMD::llGetPid(loc, rewriter, op->getParentOfType<ModuleOp>(), axis);
    };
    std::array<Value, 3> pid = {getPid(0), getPid(1), getPid(2)};

    // Simple printf of a string without any tensors.
    if (op.getNumOperands() == 0) {
      std::string formatStr;
      llvm::raw_string_ostream os(formatStr);
      os << "pid (" << getFormatSubstr(pid[0]) << ", "
         << getFormatSubstr(pid[1]) << ", " << getFormatSubstr(pid[2]) << ")" << op.getPrefix().str();
      llPrintfHIP(loc, op->getParentOfType<mlir::ModuleOp>(), formatStr, 
		      {pid[0], pid[1], pid[2]}, rewriter);
    } else {
      for (size_t i = 0; i < op.getNumOperands(); i++) {
        // Elements of the tensor that are resident in this GPU thread.
        auto elems = unpackLLElements(
            loc, adaptor.getOperands()[i], rewriter);

        // Get the indices of `elems` within the tensor.  Note that if `elems`
        // has an "interesting" layout, then these will not be in any
        // particularly nice order.

        // Extract the shape of the tensor being printed and use it to figure
        // out how many digits we need for each of the dimensions.
        SmallVector<int, 8> dimWidths;
        SmallVector<SmallVector<Value>> indices;
        if (auto rankedTy =
                op.getOperand(i).getType().dyn_cast<RankedTensorType>()) {
          indices =
              emitIndices(loc, rewriter, rankedTy.getEncoding(), rankedTy, false);
          for (int64_t dim : rankedTy.getShape()) {
            if (dim > 0) {
              dimWidths.push_back(static_cast<int>(std::ceil(std::log10(dim))));
            } else {
              dimWidths.push_back(0);
            }
          }
        } else {
          // We're printing a scalar.
          assert(elems.size() == 1);
          indices.push_back({});
        }

        if (!elems.empty()) {
          printTensor(op, prefixStr, /*operand=*/i,
                      /*numOperands=*/op.getNumOperands(), elems, pid, indices,
                      dimWidths, rewriter);
        }
      }
    }
    rewriter.eraseOp(op);
    return success();
  }

  void printTensor(triton::PrintOp op, Value prefixStr, size_t operand, size_t numOperands,
                   ArrayRef<Value> elems, std::array<Value, 3> pid,
                   ArrayRef<SmallVector<Value>> indices,
                   ArrayRef<int> dimWidths,
                   ConversionPatternRewriter &rewriter) const {
    assert(!elems.empty());
    assert(elems.size() == indices.size());
    assert(dimWidths.size() == indices.front().size());

    size_t rank = dimWidths.size();

    // Format is:
    //   pid (<x>, <y>, <z>) idx (<i1>, <i2>, ...)<prefix> (operand <n>) <elem>
    // where we leave off "(operand <n>)" if there's only one operand.
    //
    // The Python wrapper munges `prefix` so that it prints nicely (e.g. starts
    // with " " and ends with ": ").

    Value formatStrValue;
    PrintFormatting formatting;
    for (int i = 0; i < elems.size(); i++) {
      std::string formatStr;
      llvm::raw_string_ostream os(formatStr);

      // nvptx printf can only accept 32 args; if we pass more than that, it
      // will print garbage for the trailing args.
      constexpr int kMaxPrintfOperands = 32;
      SmallVector<Value, kMaxPrintfOperands> printfOperands;

      // TODO(jlebar): We really should pad the pid, but because the max pid is
      // not known at compile-time, this would require nontrivial device-side
      // work.
      os << "pid (";
      for (int j = 0; j < pid.size(); j++) {
        if (j != 0) {
          os << ", ";
        }
        os << getFormatSubstr(pid[j]);
        printfOperands.push_back(pid[j]);
      }
      os << ") ";

      // If `rank` is large enough, we could end up exceeding
      // kMaxPrintfOperands.  In that case, just truncate the index.
      // (Subtract 2 because we're going to add two operands after the index.)
      int maxAllowedRank = kMaxPrintfOperands - printfOperands.size() - 2;

      os << "idx (";
      const auto &index = indices[i];
      for (size_t dim = 0; dim < index.size(); dim++) {
        if (dim != 0) {
          os << ", ";
        }
        if (dim == maxAllowedRank) {
          os << "... (truncated)";
          break;
        }
        os << getFormatSubstr(index[dim], /*width=*/dimWidths[dim]);
        printfOperands.push_back(index[dim]);
      }
      os << ")";

      os << op.getPrefix().str();
      if (numOperands > 1) {
        os << "(operand " << operand << ") ";
      }

      auto elem = elems[i];
      os << getFormatSubstr(elem);
      printfOperands.push_back(elem);

      // It's the same format string each iteration, but it's a lot easier if we
      // construct the format string at the same time as we populate
      // printfOperands.  But we don't want to create BLOCK_SIZE duplicate
      // strings, so we cache the Value.
      if (i == 0) {
        formatting = llPrintfHIP(op->getLoc(), op->getParentOfType<mlir::ModuleOp>(), formatStr,
			printfOperands, rewriter);
      } else {
        llPrintfHIP(op->getLoc(), op->getParentOfType<mlir::ModuleOp>(), formatting,
			  printfOperands, rewriter);
      }
    }
  }

  std::string getFormatSubstr(Value value,
                              std::optional<int> width = std::nullopt) const {
    std::string prefix = "%";
    if (width.has_value()) {
      prefix += std::to_string(*width);
    }

    Type type = value.getType();
    if (type.isa<LLVM::LLVMPointerType>()) {
      return prefix + "p";
    } else if (type.isBF16() || type.isF16() || type.isF32() || type.isF64()) {
      return prefix + "f";
    } else if (type.isSignedInteger()) {
      if (type.getIntOrFloatBitWidth() == 64)
        return prefix + "lli";
      else
        return prefix + "i";
    } else if (type.isUnsignedInteger() || type.isSignlessInteger()) {
      if (type.getIntOrFloatBitWidth() == 64)
        return prefix + "llu";
      else
        return prefix + "u";
    }
    assert(false && "not supported type");
    return "";
  }

  // declare vprintf(i8*, i8*) as external function
  static LLVM::LLVMFuncOp
  getVprintfDeclaration(ConversionPatternRewriter &rewriter) {
    auto moduleOp =
        rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
    StringRef funcName("vprintf");
    Operation *funcOp = moduleOp.lookupSymbol(funcName);
    if (funcOp)
      return cast<LLVM::LLVMFuncOp>(*funcOp);

    auto *context = rewriter.getContext();

    SmallVector<Type> argsType{ptr_ty(context), ptr_ty(context)};
    auto funcType = LLVM::LLVMFunctionType::get(i32_ty, argsType);

    ConversionPatternRewriter::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(moduleOp.getBody());

    return rewriter.create<LLVM::LLVMFuncOp>(UnknownLoc::get(context), funcName,
                                             funcType);
  }

  // extend integer to int32, extend float to float64
  // this comes from vprintf alignment requirements.
  static std::pair<Type, Value>
  promoteValue(ConversionPatternRewriter &rewriter, Value value) {
    auto *context = rewriter.getContext();
    auto type = value.getType();
    Value newOp = value;
    Type newType = type;
    auto loc = UnknownLoc::get(context);

    bool bUnsigned = type.isUnsignedInteger();
    if (type.isIntOrIndex() && type.getIntOrFloatBitWidth() < 32) {
      if (bUnsigned) {
        newType = ui32_ty;
        newOp = zext(newType, value);
      } else {
        newType = i32_ty;
        newOp = sext(newType, value);
      }
    } else if (type.isBF16() || type.isF16() || type.isF32()) {
      newType = f64_ty;
      newOp = fpext(newType, value);
    }

    return {newType, newOp};
  }

  // Returns a Value for the format string, which you can reuse.
  static Value llPrintf(StringRef msg, ValueRange args,
                        ConversionPatternRewriter &rewriter) {
    assert(!msg.empty() && "printf with empty string not supported");
    llvm::SmallString<64> msgNewline(msg);
    msgNewline.push_back('\n');
    msgNewline.push_back('\0');
    Value msgValue =
        LLVM::addStringToModule(UnknownLoc::get(rewriter.getContext()),
                                rewriter, "printfFormat_", msgNewline);
    llPrintf(msgValue, args, rewriter);
    return msgValue;
  }

  static void llPrintf(Value msg, ValueRange args,
                       ConversionPatternRewriter &rewriter) {
    auto *ctx = rewriter.getContext();
    Type ptr = ptr_ty(ctx);
    auto moduleOp =
        rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
    auto funcOp = getVprintfDeclaration(rewriter);
    auto loc = UnknownLoc::get(ctx);

    Value one = i32_val(1);
    Value zero = i32_val(0);

    Value bufferPtr = null(ptr);

    SmallVector<Value, 16> newArgs;
    if (args.size() >= 1) {
      SmallVector<Type> argTypes;
      for (auto arg : args) {
        Type newType;
        Value newArg;
        std::tie(newType, newArg) = promoteValue(rewriter, arg);
        argTypes.push_back(newType);
        newArgs.push_back(newArg);
      }

      Type structTy = LLVM::LLVMStructType::getLiteral(ctx, argTypes);
      auto allocated =
          rewriter.create<LLVM::AllocaOp>(loc, ptr_ty(ctx), structTy, one,
                                          /*alignment=*/0);

      for (const auto &entry : llvm::enumerate(newArgs)) {
        auto index = i32_val(entry.index());
        auto fieldPtr =
            gep(ptr_ty(ctx), structTy, allocated, ArrayRef<Value>{zero, index});
        store(entry.value(), fieldPtr);
      }
      bufferPtr = bitcast(allocated, ptr);
    }

    SmallVector<Value> operands{msg, bufferPtr};
    call(funcOp, operands);
  }
};

} // namespace

void AMD::populatePrintOpToLLVMPattern(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<PrintOpConversion>(typeConverter, benefit);
}
