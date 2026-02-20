//===- ReductionOps.cpp - StableHLO reduce/broadcast lowering -------------===//
//
// This file implements op-by-op lowering for stablehlo.reduce and
// stablehlo.broadcast_in_dim to their Linalg equivalents.
//
// SAGE Contract: @op lower_reduce
//   @req op.name = "stablehlo.reduce"
//   @ens result[0].name = "linalg.reduce"
//   @ens output_rank = input_rank - |reduction_dims|
//
//===----------------------------------------------------------------------===//

#include "keren/Lowerings/ReductionOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include "stablehlo/dialect/StablehloOps.h"

namespace keren {
namespace lowerings {

//===----------------------------------------------------------------------===//
// ReduceOpLowering
//===----------------------------------------------------------------------===//

ReduceOpLowering::ReduceOpLowering(mlir::MLIRContext *ctx)
    : OpLoweringBase("stablehlo.reduce", ctx) {}

ReduceOpLowering::ReductionKind
ReduceOpLowering::detectReductionKind(mlir::Region &body) const {
  // Look at the single block's terminator's operand definition
  if (body.empty() || body.front().empty())
    return ReductionKind::Unknown;

  auto &block = body.front();

  // Find the operation that produces the result (before the return)
  for (auto &op : block) {
    if (mlir::isa<mlir::stablehlo::AddOp>(op) ||
        mlir::isa<mlir::arith::AddFOp>(op) ||
        mlir::isa<mlir::arith::AddIOp>(op)) {
      return ReductionKind::Add;
    }
    if (mlir::isa<mlir::stablehlo::MaxOp>(op) ||
        mlir::isa<mlir::arith::MaximumFOp>(op) ||
        mlir::isa<mlir::arith::MaxSIOp>(op)) {
      return ReductionKind::Max;
    }
    if (mlir::isa<mlir::stablehlo::MinOp>(op) ||
        mlir::isa<mlir::arith::MinimumFOp>(op) ||
        mlir::isa<mlir::arith::MinSIOp>(op)) {
      return ReductionKind::Min;
    }
    if (mlir::isa<mlir::stablehlo::MulOp>(op) ||
        mlir::isa<mlir::arith::MulFOp>(op) ||
        mlir::isa<mlir::arith::MulIOp>(op)) {
      return ReductionKind::Mul;
    }
  }
  return ReductionKind::Unknown;
}

void ReduceOpLowering::buildReduceBody(mlir::OpBuilder &builder,
                                       mlir::Location loc, ReductionKind kind,
                                       mlir::Type elemType,
                                       mlir::ValueRange args) const {
  mlir::Value lhs = args[0]; // input element
  mlir::Value rhs = args[1]; // accumulator

  mlir::Value result;
  bool isFloat = mlir::isa<mlir::FloatType>(elemType);

  switch (kind) {
  case ReductionKind::Add:
    result = isFloat ? builder.create<mlir::arith::AddFOp>(loc, lhs, rhs)
                           .getResult()
                     : builder.create<mlir::arith::AddIOp>(loc, lhs, rhs)
                           .getResult();
    break;
  case ReductionKind::Max:
    result = isFloat ? builder.create<mlir::arith::MaximumFOp>(loc, lhs, rhs)
                           .getResult()
                     : builder.create<mlir::arith::MaxSIOp>(loc, lhs, rhs)
                           .getResult();
    break;
  case ReductionKind::Min:
    result = isFloat ? builder.create<mlir::arith::MinimumFOp>(loc, lhs, rhs)
                           .getResult()
                     : builder.create<mlir::arith::MinSIOp>(loc, lhs, rhs)
                           .getResult();
    break;
  case ReductionKind::Mul:
    result = isFloat ? builder.create<mlir::arith::MulFOp>(loc, lhs, rhs)
                           .getResult()
                     : builder.create<mlir::arith::MulIOp>(loc, lhs, rhs)
                           .getResult();
    break;
  default:
    // Unknown reduction - just use add as fallback
    result = isFloat ? builder.create<mlir::arith::AddFOp>(loc, lhs, rhs)
                           .getResult()
                     : builder.create<mlir::arith::AddIOp>(loc, lhs, rhs)
                           .getResult();
    break;
  }

  builder.create<mlir::linalg::YieldOp>(loc, result);
}

mlir::LogicalResult
ReduceOpLowering::matchAndRewrite(mlir::Operation *op,
                                  mlir::PatternRewriter &rewriter) const {
  auto reduceOp = mlir::cast<mlir::stablehlo::ReduceOp>(op);
  auto loc = op->getLoc();

  // Get inputs and init values
  auto inputs = reduceOp.getInputs();
  auto initValues = reduceOp.getInitValues();

  // We only handle single-input reductions for now
  if (inputs.size() != 1 || initValues.size() != 1) {
    return rewriter.notifyMatchFailure(
        op, "only single-input reductions are supported");
  }

  auto input = inputs[0];
  auto initValue = initValues[0];

  auto inputType = mlir::cast<mlir::RankedTensorType>(input.getType());
  auto resultType =
      mlir::cast<mlir::RankedTensorType>(reduceOp.getResults()[0].getType());
  auto elemType = inputType.getElementType();

  // Get reduction dimensions
  auto reductionDims = reduceOp.getDimensions();
  llvm::SmallVector<int64_t> dims(reductionDims.begin(), reductionDims.end());

  // Detect reduction kind from body
  auto kind = detectReductionKind(reduceOp.getBody());

  // Create output tensor initialized with init value
  // First, get the scalar from init value (it's a 0-d tensor)
  auto initConstant = mlir::cast<mlir::stablehlo::ConstantOp>(
      initValue.getDefiningOp());
  auto initAttr = mlir::cast<mlir::DenseElementsAttr>(initConstant.getValue());

  mlir::Value initScalar;
  if (mlir::isa<mlir::FloatType>(elemType)) {
    auto val = initAttr.getSplatValue<mlir::APFloat>();
    initScalar = rewriter.create<mlir::arith::ConstantOp>(
        loc, rewriter.getFloatAttr(elemType, val));
  } else {
    auto val = initAttr.getSplatValue<mlir::APInt>();
    auto attr = rewriter.getIntegerAttr(elemType, val.getSExtValue());
    initScalar = rewriter.create<mlir::arith::ConstantOp>(loc, attr);
  }

  // Create empty tensor and fill with init value
  mlir::Value emptyTensor =
      rewriter.create<mlir::tensor::EmptyOp>(loc, resultType, mlir::ValueRange{});
  mlir::Value filledTensor =
      rewriter.create<mlir::linalg::FillOp>(loc, initScalar, emptyTensor)
          .getResult(0);

  // Create linalg.reduce
  auto linalgReduce = rewriter.create<mlir::linalg::ReduceOp>(
      loc, input, filledTensor, dims,
      [&](mlir::OpBuilder &b, mlir::Location loc, mlir::ValueRange args) {
        buildReduceBody(b, loc, kind, elemType, args);
      });

  rewriter.replaceOp(op, linalgReduce.getResults());
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// BroadcastInDimOpLowering
//===----------------------------------------------------------------------===//

BroadcastInDimOpLowering::BroadcastInDimOpLowering(mlir::MLIRContext *ctx)
    : OpLoweringBase("stablehlo.broadcast_in_dim", ctx) {}

mlir::LogicalResult
BroadcastInDimOpLowering::matchAndRewrite(mlir::Operation *op,
                                          mlir::PatternRewriter &rewriter) const {
  auto broadcastOp = mlir::cast<mlir::stablehlo::BroadcastInDimOp>(op);
  auto loc = op->getLoc();

  auto input = broadcastOp.getOperand();
  auto inputType = mlir::cast<mlir::RankedTensorType>(input.getType());
  auto resultType = mlir::cast<mlir::RankedTensorType>(broadcastOp.getType());

  auto broadcastDims = broadcastOp.getBroadcastDimensions();

  // Create empty output tensor
  mlir::Value emptyTensor =
      rewriter.create<mlir::tensor::EmptyOp>(loc, resultType, mlir::ValueRange{});

  // Convert broadcast dimensions to SmallVector<int64_t>
  llvm::SmallVector<int64_t> dims(broadcastDims.begin(), broadcastDims.end());

  // Create linalg.broadcast
  auto linalgBroadcast = rewriter.create<mlir::linalg::BroadcastOp>(
      loc, input, emptyTensor, dims);

  rewriter.replaceOp(op, linalgBroadcast.getResults());
  return mlir::success();
}

} // namespace lowerings
} // namespace keren
