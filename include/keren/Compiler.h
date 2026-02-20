#ifndef KEREN_COMPILER_H
#define KEREN_COMPILER_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

#include <memory>
#include <string>

namespace keren {

/// Options for the Keren compiler.
struct CompilerOptions {
  /// Entry function name.
  std::string entryFunction = "main";

  /// Enable op-by-op lowering from StableHLO to Linalg.
  bool lowerOps = true;

  /// Enable elementwise fusion after lowering.
  bool fuse = false;

  /// Print verbose output showing each lowering step.
  bool verbose = false;

  /// Output file path (empty for stdout).
  std::string outputFile;
};

/// Main compiler driver for StableHLO to Linalg compilation.
///
/// SAGE Contract: @op compile
///   @req ∀ op ∈ all_ops(input): op.dialect ∈ {"stablehlo", "func", "arith"}
///   @ens ∀ op ∈ all_ops(result): op.dialect ∈ {"linalg", "tensor", "arith", "func"}
///   @ens ∀ f ∈ input.functions: semantics(f) = semantics(result.functions[f.name])
class Compiler {
public:
  Compiler(const CompilerOptions &opts);

  /// Initialize the MLIR context with required dialects.
  void initializeContext();

  /// Get the MLIR context.
  mlir::MLIRContext &getContext() { return context_; }

  /// Compile a file from StableHLO to Linalg.
  /// Returns the compiled module, or nullptr on failure.
  mlir::OwningOpRef<mlir::ModuleOp> compileFile(llvm::StringRef filename);

  /// Compile a module in-place.
  /// Returns success if compilation succeeded.
  mlir::LogicalResult compile(mlir::ModuleOp module);

  /// Run op-by-op lowering pass.
  mlir::LogicalResult runOpLowering(mlir::ModuleOp module);

  /// Run elementwise fusion pass.
  mlir::LogicalResult runFusion(mlir::ModuleOp module);

  /// Print the module to output.
  void printOutput(mlir::ModuleOp module);

private:
  CompilerOptions opts_;
  mlir::MLIRContext context_;
};

} // namespace keren

#endif // KEREN_COMPILER_H
