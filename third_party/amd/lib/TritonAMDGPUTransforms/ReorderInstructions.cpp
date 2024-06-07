#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/TritonGPUConversion.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#define GEN_PASS_CLASSES
#include "TritonAMDGPUTransforms/Passes.h"

using namespace mlir;

static bool willIncreaseRegisterPressure(Operation *op) {
  if (isa<triton::gpu::LocalLoadOp>(op))
    return true;
  if (auto cvt = dyn_cast<triton::gpu::ConvertLayoutOp>(op))
    return isa<triton::gpu::DotOperandEncodingAttr>(cvt.getType().getEncoding());
  return false;
}

static bool gatherDFG(Operation *op, SmallVector<Operation *> &dfg) {
  bool leadsToLoad = false;
  // BFS (filo)
  Block *block = op->getBlock();
  SmallVector<Operation *> oprs;
  for (auto operand : op->getOperands()) {
    if (Operation *pop = operand.getDefiningOp()) {
      if (pop->getBlock() == block) {
        // must reside in same block
        oprs.push_back(pop);
        dfg.push_back(pop);
        leadsToLoad |= isa<triton::LoadOp>(pop);
      }
    }
  }
  for (auto *op : oprs) {
    if (gatherDFG(op, dfg))
      leadsToLoad = true;
  }
  return leadsToLoad;
}

class TritonAMDGPUReorderInstructionsPass
    : public TritonAMDGPUReorderInstructionsBase<
          TritonAMDGPUReorderInstructionsPass> {
public:
  TritonAMDGPUReorderInstructionsPass() = default;

  void runOnOperation() override {
    ModuleOp m = getOperation();
    mlir::DominanceInfo dom(m);
    // Sink conversions into loops when they will increase
    // register pressure
    DenseMap<Operation *, Operation *> opToMove;
    auto moveAfter = [](Operation *lhs, Operation *rhs) {
      lhs->moveAfter(rhs);
    };
    m.walk([&](Operation *op) {
      if (!willIncreaseRegisterPressure(op))
        return;
      if (!op->hasOneUse())
        return;
      Operation *user = op->getUses().begin()->getOwner();
      if (user->getParentOfType<scf::ForOp>() ==
          op->getParentOfType<scf::ForOp>())
        return;
      opToMove.insert({op, user});
    });
    for (auto &kv : opToMove)
      kv.first->moveBefore(kv.second);
    opToMove.clear();
    // Move LocalLoadOp and LocalAllocOp immediately after their operands.
    m.walk([&](Operation *op) {
      if (!isa<triton::gpu::LocalLoadOp, triton::gpu::LocalAllocOp>(op) ||
          op->getNumOperands() < 1) {
        return;
      }
      if (Operation *argOp = op->getOperand(0).getDefiningOp())
        moveAfter(op, argOp);
    });
    // Move transpositions just after their definition
    m.walk([&](triton::TransOp op) {
      Operation *argOp = op.getSrc().getDefiningOp();
      if (!argOp)
        return;
      moveAfter(op, argOp);
    });
    // Move local stores early if it's global load is outside loop
    m.walk([&](triton::gpu::LocalStoreOp op) {
      // 0. gather DFG 
      SmallVector<Operation *> dfg{op};
      if (!gatherDFG(op, dfg)) {
        Block *block = op->getBlock();
        // 1. move to beginning of enclosing block
        for (auto *op : dfg)
          op->moveAfter(block, block->begin());
      }
    });
    // Move global loads early (prefetch)
    m.walk([&](triton::LoadOp op) {
      // 0. gather DFG
      SmallVector<Operation *> dfg{op};
      gatherDFG(op, dfg);
      Block *block = op->getBlock();
      // 1. move to beginning of enclosing block
      for (auto *op : dfg)
        op->moveAfter(block, block->begin());
    });
    return;
  }
};

std::unique_ptr<Pass> mlir::createTritonAMDGPUReorderInstructionsPass() {
  return std::make_unique<TritonAMDGPUReorderInstructionsPass>();
}
