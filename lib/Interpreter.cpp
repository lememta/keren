#include "keren/Interpreter.h"
#include "keren/Pipeline.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Parser/Parser.h"

#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/reference/Api.h"

namespace keren {

Interpreter::Interpreter(const InterpreterOptions &opts) : opts_(opts) {
  registerDialects();
}

void Interpreter::registerDialects() {
  mlir::DialectRegistry registry;
  registry.insert<mlir::stablehlo::StablehloDialect>();
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::arith::ArithDialect>();
  registry.insert<mlir::tensor::TensorDialect>();
  registry.insert<mlir::linalg::LinalgDialect>();
  context_.appendDialectRegistry(registry);
  context_.loadAllAvailableDialects();
}

void Interpreter::addHandler(std::unique_ptr<DialectHandler> handler) {
  handlers_.push_back(std::move(handler));
}

std::vector<TensorValue> Interpreter::runFile(llvm::StringRef filename) {
  // Parse the MLIR source file.
  auto module = mlir::parseSourceFile<mlir::ModuleOp>(filename, &context_);
  if (!module) {
    llvm::errs() << "error: failed to parse " << filename << "\n";
    return {};
  }

  // Optionally lower StableHLO to Linalg.
  if (opts_.lowerToLinalg) {
    if (mlir::failed(Pipeline::lowerStableHLOToLinalg(*module))) {
      return {};
    }
    // After lowering, Linalg interpretation would be needed.
    // For now, print the lowered IR and return empty.
    module->print(llvm::outs());
    llvm::outs() << "\n";
    return {};
  }

  // Use StableHLO's reference interpreter for whole-module evaluation.
  return evalViaStableHLO(*module);
}

std::vector<TensorValue>
Interpreter::evalViaStableHLO(mlir::ModuleOp module) {
  // evalModule expects no inputs for functions with no arguments,
  // or SmallVector<SmallVector<Tensor>> for multiple inputs.
  // We call with empty inputs assuming the entry function takes no args.
  auto errorOrResults = mlir::stablehlo::evalModule(module, {});
  if (mlir::failed(errorOrResults)) {
    llvm::errs() << "error: StableHLO evaluation failed\n";
    return {};
  }

  std::vector<TensorValue> results;
  for (auto &tensor : *errorOrResults) {
    results.emplace_back(std::move(tensor));
  }
  return results;
}

} // namespace keren
