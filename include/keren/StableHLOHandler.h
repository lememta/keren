#ifndef KEREN_STABLEHLOHANDLER_H
#define KEREN_STABLEHLOHANDLER_H

#include "keren/DialectHandler.h"

namespace keren {

/// Dialect handler for StableHLO operations.
/// Delegates to stablehlo::evalModule for whole-module interpretation.
class StableHLOHandler : public DialectHandler {
public:
  bool canHandle(mlir::Operation *op) const override;
  mlir::LogicalResult evaluate(mlir::Operation *op, ValueMap &valueMap) override;
};

} // namespace keren

#endif // KEREN_STABLEHLOHANDLER_H
