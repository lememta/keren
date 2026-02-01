#ifndef KEREN_TENSORVALUE_H
#define KEREN_TENSORVALUE_H

#include "stablehlo/reference/Tensor.h"

#include "mlir/IR/Types.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <variant>

namespace keren {

/// Wrapper around stablehlo::Tensor that can be extended later
/// with an OwnedBuffer variant for Linalg memref semantics.
class TensorValue {
public:
  TensorValue() = default;
  explicit TensorValue(mlir::stablehlo::Tensor tensor);

  /// Access the underlying StableHLO tensor.
  const mlir::stablehlo::Tensor &getTensor() const;

  /// Get the MLIR type of this tensor.
  mlir::Type getType() const;

  /// Print the tensor contents to the given stream.
  void print(llvm::raw_ostream &os) const;

private:
  mlir::stablehlo::Tensor tensor_;
};

} // namespace keren

#endif // KEREN_TENSORVALUE_H
