#ifndef KEREN_INTERPRETER_H
#define KEREN_INTERPRETER_H

#include "keren/DialectHandler.h"
#include "keren/TensorValue.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

#include "stablehlo/reference/Value.h"

#include <memory>
#include <string>
#include <vector>

namespace keren {

struct InterpreterOptions {
  std::string entryFunction = "main";
  bool lowerToLinalg = false;
  /// If non-empty, enable probe instrumentation and write per-op tensors here.
  std::string probeDir;
};

/// Main interpreter driver.
/// Parses .mlir files, optionally lowers dialects, and interprets the module.
class Interpreter {
public:
  explicit Interpreter(const InterpreterOptions &opts = {});

  /// Register a dialect handler for interpretation.
  void addHandler(std::unique_ptr<DialectHandler> handler);

  /// Parse and interpret an MLIR file. Returns the output tensors
  /// from the entry function's return values.
  /// On error, returns an empty vector and prints diagnostics to stderr.
  std::vector<TensorValue>
  runFile(llvm::StringRef filename,
          llvm::ArrayRef<mlir::stablehlo::InterpreterValue> inputs = {});

private:
  /// Set up all required dialects on the context.
  void registerDialects();

  /// Use StableHLO's evalModule for whole-module interpretation.
  std::vector<TensorValue>
  evalViaStableHLO(mlir::ModuleOp module,
                   llvm::ArrayRef<mlir::stablehlo::InterpreterValue> inputs);

  InterpreterOptions opts_;
public:
  mlir::MLIRContext &getContext() { return context_; }
private:
  mlir::MLIRContext context_;
  std::vector<std::unique_ptr<DialectHandler>> handlers_;
};

} // namespace keren

#endif // KEREN_INTERPRETER_H
