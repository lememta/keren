#ifndef KEREN_LINALGHANDLER_H
#define KEREN_LINALGHANDLER_H

#include "keren/DialectHandler.h"

namespace keren {

/// Stub dialect handler for Linalg operations.
/// Will interpret linalg.generic and named ops once implemented.
class LinalgHandler : public DialectHandler {
public:
  bool canHandle(mlir::Operation *op) const override;
  mlir::LogicalResult evaluate(mlir::Operation *op, ValueMap &valueMap) override;
};

} // namespace keren

#endif // KEREN_LINALGHANDLER_H
