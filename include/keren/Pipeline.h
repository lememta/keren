#ifndef KEREN_PIPELINE_H
#define KEREN_PIPELINE_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"

namespace keren {

/// Manages lowering pipelines (e.g., StableHLO -> Linalg).
class Pipeline {
public:
  /// Lower StableHLO operations to Linalg dialect in-place.
  /// Returns failure if the pass pipeline fails.
  static mlir::LogicalResult lowerStableHLOToLinalg(mlir::ModuleOp module);
};

} // namespace keren

#endif // KEREN_PIPELINE_H
