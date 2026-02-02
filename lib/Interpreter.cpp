#include "keren/Interpreter.h"
#include "keren/Pipeline.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Parser/Parser.h"

#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/reference/Api.h"
#include "stablehlo/reference/Configuration.h"
#include "stablehlo/reference/InterpreterOps.h"

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
  registry.insert<mlir::stablehlo::interpreter::InterpreterDialect>();
  context_.appendDialectRegistry(registry);
  context_.loadAllAvailableDialects();
}

void Interpreter::addHandler(std::unique_ptr<DialectHandler> handler) {
  handlers_.push_back(std::move(handler));
}

std::vector<TensorValue>
Interpreter::runFile(llvm::StringRef filename,
                     llvm::ArrayRef<mlir::stablehlo::InterpreterValue> inputs) {
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

  // If tracing, insert probe ops before evaluation.
  if (!opts_.probeDir.empty()) {
    if (mlir::failed(Pipeline::instrumentWithProbes(*module)))
      return {};
  }

  // Use StableHLO's reference interpreter for whole-module evaluation.
  return evalViaStableHLO(*module, inputs);
}

std::vector<TensorValue>
Interpreter::evalViaStableHLO(
    mlir::ModuleOp module,
    llvm::ArrayRef<mlir::stablehlo::InterpreterValue> inputs) {
  mlir::stablehlo::InterpreterConfiguration config;
  config.mainFunction = opts_.entryFunction;
  if (!opts_.probeDir.empty())
    config.probeInstrumentationDir = opts_.probeDir;

  auto errorOrResults =
      mlir::stablehlo::evalModule(module, inputs, config);
  if (mlir::failed(errorOrResults)) {
    llvm::errs() << "error: StableHLO evaluation failed\n";
    return {};
  }

  std::vector<TensorValue> results;
  for (auto &val : *errorOrResults) {
    // InterpreterValue wraps a Tensor
    results.emplace_back(val.getTensor());
  }
  return results;
}

} // namespace keren
