#ifndef KEREN_DIALECTHANDLER_H
#define KEREN_DIALECTHANDLER_H

#include "keren/TensorValue.h"

#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace keren {

/// Map from SSA values to their computed tensor values.
using ValueMap = llvm::DenseMap<mlir::Value, TensorValue>;

/// Abstract base class for dialect-specific interpreters.
class DialectHandler {
public:
  virtual ~DialectHandler() = default;

  /// Return true if this handler can interpret the given operation.
  virtual bool canHandle(mlir::Operation *op) const = 0;

  /// Evaluate an operation, reading inputs from and writing outputs to the
  /// value map. Returns failure if the operation cannot be evaluated.
  virtual mlir::LogicalResult evaluate(mlir::Operation *op,
                                       ValueMap &valueMap) = 0;
};

} // namespace keren

#endif // KEREN_DIALECTHANDLER_H
