#include "keren/Pipeline.h"

#include "mlir/Pass/PassManager.h"
#include "stablehlo/conversions/linalg/StablehloToLinalg.h"

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

} // namespace keren
