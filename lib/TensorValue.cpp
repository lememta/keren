#include "keren/TensorValue.h"

namespace keren {

TensorValue::TensorValue(mlir::stablehlo::Tensor tensor)
    : tensor_(std::move(tensor)) {}

const mlir::stablehlo::Tensor &TensorValue::getTensor() const {
  return tensor_;
}

mlir::Type TensorValue::getType() const { return tensor_.getType(); }

void TensorValue::print(llvm::raw_ostream &os) const { tensor_.print(os); }

} // namespace keren
