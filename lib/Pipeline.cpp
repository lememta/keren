#include "keren/Pipeline.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassManager.h"
#include "stablehlo/conversions/linalg/transforms/Passes.h"
namespace mlir {
namespace stablehlo {
#define GEN_PASS_DECL
#include "stablehlo/reference/InterpreterPasses.h.inc"
} // namespace stablehlo
} // namespace mlir

namespace keren {

mlir::LogicalResult Pipeline::lowerStableHLOToLinalg(mlir::ModuleOp module) {
  mlir::PassManager pm(module.getContext());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::stablehlo::createStablehloLegalizeToLinalgPass());

  if (mlir::failed(pm.run(module))) {
    llvm::errs() << "error: StableHLO-to-Linalg lowering failed\n";
    return mlir::failure();
  }
  return mlir::success();
}

mlir::LogicalResult Pipeline::instrumentWithProbes(mlir::ModuleOp module) {
  mlir::PassManager pm(module.getContext());
  pm.addPass(mlir::stablehlo::createInterpreterInstrumentWithProbePass());

  if (mlir::failed(pm.run(module))) {
    llvm::errs() << "error: probe instrumentation failed\n";
    return mlir::failure();
  }
  return mlir::success();
}

} // namespace keren
