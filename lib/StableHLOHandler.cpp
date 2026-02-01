#include "keren/StableHLOHandler.h"

#include "stablehlo/dialect/StablehloOps.h"

namespace keren {

bool StableHLOHandler::canHandle(mlir::Operation *op) const {
  return op->getDialect()->getNamespace() == "stablehlo";
}

mlir::LogicalResult StableHLOHandler::evaluate(mlir::Operation *op,
                                                ValueMap &valueMap) {
  // Per-op dispatch is deferred to a future version.
  // Whole-module evaluation via evalModule is handled by the Interpreter.
  return mlir::failure();
}

} // namespace keren
