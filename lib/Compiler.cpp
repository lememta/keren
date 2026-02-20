//===- Compiler.cpp - Keren compiler driver -------------------------------===//
//
// This file implements the main compiler driver that orchestrates
// StableHLO to Linalg compilation with optional fusion.
//
// SAGE Contract: @spec CompilerPipeline
//   @op compile(input: Module, options: CompilerOptions) -> Module
//   @req ∀ op ∈ all_ops(input): op.dialect ∈ {"stablehlo", "func", "arith"}
//   @ens ∀ op ∈ all_ops(result): op.dialect ∈ {"linalg", "tensor", "arith", "func"}
//
//===----------------------------------------------------------------------===//

#include "keren/Compiler.h"
#include "keren/Fusion.h"
#include "keren/OpLowering.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "stablehlo/dialect/StablehloOps.h"

#include "llvm/Support/raw_ostream.h"

namespace keren {

Compiler::Compiler(const CompilerOptions &opts) : opts_(opts) {
  initializeContext();
}

void Compiler::initializeContext() {
  mlir::DialectRegistry registry;

  // Input dialects
  registry.insert<mlir::stablehlo::StablehloDialect>();
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::arith::ArithDialect>();

  // Output dialects
  registry.insert<mlir::linalg::LinalgDialect>();
  registry.insert<mlir::tensor::TensorDialect>();
  registry.insert<mlir::math::MathDialect>();

  context_.appendDialectRegistry(registry);
  context_.loadAllAvailableDialects();
}

mlir::OwningOpRef<mlir::ModuleOp>
Compiler::compileFile(llvm::StringRef filename) {
  // Parse the input file
  auto module = mlir::parseSourceFile<mlir::ModuleOp>(filename, &context_);
  if (!module) {
    llvm::errs() << "error: failed to parse " << filename << "\n";
    return nullptr;
  }

  // Run compilation
  if (mlir::failed(compile(*module))) {
    llvm::errs() << "error: compilation failed\n";
    return nullptr;
  }

  return module;
}

mlir::LogicalResult Compiler::compile(mlir::ModuleOp module) {
  if (opts_.verbose) {
    llvm::outs() << "=== Input Module ===\n";
    module.print(llvm::outs());
    llvm::outs() << "\n";
  }

  // Step 1: Op-by-op lowering
  if (opts_.lowerOps) {
    if (mlir::failed(runOpLowering(module)))
      return mlir::failure();

    if (opts_.verbose) {
      llvm::outs() << "=== After Op Lowering ===\n";
      module.print(llvm::outs());
      llvm::outs() << "\n";
    }
  }

  // Step 2: Elementwise fusion
  if (opts_.fuse) {
    if (mlir::failed(runFusion(module)))
      return mlir::failure();

    if (opts_.verbose) {
      llvm::outs() << "=== After Fusion ===\n";
      module.print(llvm::outs());
      llvm::outs() << "\n";
    }
  }

  return mlir::success();
}

mlir::LogicalResult Compiler::runOpLowering(mlir::ModuleOp module) {
  // Create pattern set with all op lowerings
  mlir::RewritePatternSet patterns(&context_);
  OpLoweringRegistry::populatePatterns(patterns, &context_);

  // Apply patterns greedily
  mlir::GreedyRewriteConfig config;
  config.maxIterations = 10; // Limit iterations

  if (mlir::failed(mlir::applyPatternsAndFoldGreedily(module,
                                                       std::move(patterns),
                                                       config))) {
    llvm::errs() << "error: op lowering failed\n";
    return mlir::failure();
  }

  return mlir::success();
}

mlir::LogicalResult Compiler::runFusion(mlir::ModuleOp module) {
  // Run the fusion pass on each function
  mlir::PassManager pm(&context_);
  pm.addNestedPass<mlir::func::FuncOp>(createElementwiseFusionPass());

  if (mlir::failed(pm.run(module))) {
    llvm::errs() << "error: fusion pass failed\n";
    return mlir::failure();
  }

  return mlir::success();
}

void Compiler::printOutput(mlir::ModuleOp module) {
  if (opts_.outputFile.empty()) {
    module.print(llvm::outs());
    llvm::outs() << "\n";
  } else {
    std::error_code ec;
    llvm::raw_fd_ostream outFile(opts_.outputFile, ec);
    if (ec) {
      llvm::errs() << "error: cannot open output file: " << opts_.outputFile
                   << "\n";
      return;
    }
    module.print(outFile);
    outFile << "\n";
  }
}

} // namespace keren
