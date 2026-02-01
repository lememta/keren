#include "keren/LinalgHandler.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"

namespace keren {

bool LinalgHandler::canHandle(mlir::Operation *op) const {
  return op->getDialect()->getNamespace() == "linalg";
}

mlir::LogicalResult LinalgHandler::evaluate(mlir::Operation *op,
                                             ValueMap &valueMap) {
  // Linalg interpretation is not yet implemented.
  op->emitError("Linalg interpretation is not yet implemented");
  return mlir::failure();
}

} // namespace keren
