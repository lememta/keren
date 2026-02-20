//===- ElementwiseOps.cpp - StableHLO elementwise op lowerings ------------===//
//
// This file implements op-by-op lowerings for StableHLO elementwise operations
// to Linalg generic operations.
//
// SAGE Contract: @op lower_elementwise_binary
//   @req op.operands[0].type.shape = op.operands[1].type.shape
//   @ens result[0].name = "linalg.generic"
//   @ens ∀ d ∈ result[0].iterator_types: d = "parallel"
//
//===----------------------------------------------------------------------===//

#include "keren/Lowerings/ElementwiseOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineMap.h"

#include "stablehlo/dialect/StablehloOps.h"

namespace keren {
namespace lowerings {

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

/// Create an empty tensor with the given type.
static mlir::Value createEmptyTensor(mlir::PatternRewriter &rewriter,
                                     mlir::Location loc,
                                     mlir::RankedTensorType type) {
  auto shape = type.getShape();
  llvm::SmallVector<mlir::Value> dynamicDims;
  // Handle dynamic dimensions if needed
  for (int64_t i = 0; i < type.getRank(); ++i) {
    if (mlir::ShapedType::isDynamic(shape[i])) {
      // Would need to extract from operand - for now assume static
    }
  }
  return rewriter.create<mlir::tensor::EmptyOp>(loc, type, dynamicDims);
}

/// Create identity indexing maps for elementwise ops.
/// For a rank-N tensor, creates N maps of the form (d0, d1, ..., dN-1) -> (d0, d1, ..., dN-1)
static llvm::SmallVector<mlir::AffineMap>
createIdentityMaps(mlir::MLIRContext *ctx, unsigned rank, unsigned numMaps) {
  llvm::SmallVector<mlir::AffineMap> maps;
  auto identityMap = mlir::AffineMap::getMultiDimIdentityMap(rank, ctx);
  for (unsigned i = 0; i < numMaps; ++i) {
    maps.push_back(identityMap);
  }
  return maps;
}

/// Create parallel iterator types for all dimensions.
static llvm::SmallVector<mlir::utils::IteratorType>
createParallelIterators(unsigned rank) {
  return llvm::SmallVector<mlir::utils::IteratorType>(
      rank, mlir::utils::IteratorType::parallel);
}

/// Generic binary elementwise lowering template.
/// Creates a linalg.generic with the specified body builder.
template <typename StableHLOOp>
static mlir::LogicalResult
lowerBinaryElementwise(StableHLOOp op, mlir::PatternRewriter &rewriter,
                       std::function<mlir::Value(mlir::OpBuilder &,
                                                  mlir::Location, mlir::Value,
                                                  mlir::Value)>
                           bodyBuilder) {
  auto loc = op.getLoc();
  auto resultType = mlir::cast<mlir::RankedTensorType>(op.getType());
  unsigned rank = resultType.getRank();

  // Create output tensor
  mlir::Value output = createEmptyTensor(rewriter, loc, resultType);

  // Create indexing maps (identity for elementwise)
  auto maps = createIdentityMaps(op.getContext(), rank, 3); // 2 inputs + 1 output

  // Create iterator types (all parallel for elementwise)
  auto iteratorTypes = createParallelIterators(rank);

  // Build linalg.generic
  auto genericOp = rewriter.create<mlir::linalg::GenericOp>(
      loc,
      /*resultTypes=*/resultType,
      /*inputs=*/mlir::ValueRange{op.getLhs(), op.getRhs()},
      /*outputs=*/mlir::ValueRange{output},
      /*indexingMaps=*/maps,
      /*iteratorTypes=*/iteratorTypes,
      /*bodyBuilder=*/
      [&](mlir::OpBuilder &b, mlir::Location loc, mlir::ValueRange args) {
        // args[0] = lhs element, args[1] = rhs element, args[2] = output element
        mlir::Value result = bodyBuilder(b, loc, args[0], args[1]);
        b.create<mlir::linalg::YieldOp>(loc, result);
      });

  rewriter.replaceOp(op, genericOp.getResults());
  return mlir::success();
}

/// Generic unary elementwise lowering template.
template <typename StableHLOOp>
static mlir::LogicalResult
lowerUnaryElementwise(StableHLOOp op, mlir::PatternRewriter &rewriter,
                      std::function<mlir::Value(mlir::OpBuilder &,
                                                 mlir::Location, mlir::Value)>
                          bodyBuilder) {
  auto loc = op.getLoc();
  auto resultType = mlir::cast<mlir::RankedTensorType>(op.getType());
  unsigned rank = resultType.getRank();

  mlir::Value output = createEmptyTensor(rewriter, loc, resultType);
  auto maps = createIdentityMaps(op.getContext(), rank, 2); // 1 input + 1 output
  auto iteratorTypes = createParallelIterators(rank);

  auto genericOp = rewriter.create<mlir::linalg::GenericOp>(
      loc, resultType,
      /*inputs=*/mlir::ValueRange{op.getOperand()},
      /*outputs=*/mlir::ValueRange{output}, maps, iteratorTypes,
      [&](mlir::OpBuilder &b, mlir::Location loc, mlir::ValueRange args) {
        mlir::Value result = bodyBuilder(b, loc, args[0]);
        b.create<mlir::linalg::YieldOp>(loc, result);
      });

  rewriter.replaceOp(op, genericOp.getResults());
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// Binary Elementwise Operations
//===----------------------------------------------------------------------===//

AddOpLowering::AddOpLowering(mlir::MLIRContext *ctx)
    : OpLoweringBase("stablehlo.add", ctx) {}

mlir::LogicalResult
AddOpLowering::matchAndRewrite(mlir::Operation *op,
                               mlir::PatternRewriter &rewriter) const {
  auto addOp = mlir::cast<mlir::stablehlo::AddOp>(op);
  auto elemType =
      mlir::cast<mlir::RankedTensorType>(addOp.getType()).getElementType();

  return lowerBinaryElementwise(
      addOp, rewriter,
      [&](mlir::OpBuilder &b, mlir::Location loc, mlir::Value lhs,
          mlir::Value rhs) -> mlir::Value {
        if (mlir::isa<mlir::FloatType>(elemType)) {
          return b.create<mlir::arith::AddFOp>(loc, lhs, rhs);
        } else {
          return b.create<mlir::arith::AddIOp>(loc, lhs, rhs);
        }
      });
}

MulOpLowering::MulOpLowering(mlir::MLIRContext *ctx)
    : OpLoweringBase("stablehlo.multiply", ctx) {}

mlir::LogicalResult
MulOpLowering::matchAndRewrite(mlir::Operation *op,
                               mlir::PatternRewriter &rewriter) const {
  auto mulOp = mlir::cast<mlir::stablehlo::MulOp>(op);
  auto elemType =
      mlir::cast<mlir::RankedTensorType>(mulOp.getType()).getElementType();

  return lowerBinaryElementwise(
      mulOp, rewriter,
      [&](mlir::OpBuilder &b, mlir::Location loc, mlir::Value lhs,
          mlir::Value rhs) -> mlir::Value {
        if (mlir::isa<mlir::FloatType>(elemType)) {
          return b.create<mlir::arith::MulFOp>(loc, lhs, rhs);
        } else {
          return b.create<mlir::arith::MulIOp>(loc, lhs, rhs);
        }
      });
}

SubOpLowering::SubOpLowering(mlir::MLIRContext *ctx)
    : OpLoweringBase("stablehlo.subtract", ctx) {}

mlir::LogicalResult
SubOpLowering::matchAndRewrite(mlir::Operation *op,
                               mlir::PatternRewriter &rewriter) const {
  auto subOp = mlir::cast<mlir::stablehlo::SubtractOp>(op);
  auto elemType =
      mlir::cast<mlir::RankedTensorType>(subOp.getType()).getElementType();

  return lowerBinaryElementwise(
      subOp, rewriter,
      [&](mlir::OpBuilder &b, mlir::Location loc, mlir::Value lhs,
          mlir::Value rhs) -> mlir::Value {
        if (mlir::isa<mlir::FloatType>(elemType)) {
          return b.create<mlir::arith::SubFOp>(loc, lhs, rhs);
        } else {
          return b.create<mlir::arith::SubIOp>(loc, lhs, rhs);
        }
      });
}

DivOpLowering::DivOpLowering(mlir::MLIRContext *ctx)
    : OpLoweringBase("stablehlo.divide", ctx) {}

mlir::LogicalResult
DivOpLowering::matchAndRewrite(mlir::Operation *op,
                               mlir::PatternRewriter &rewriter) const {
  auto divOp = mlir::cast<mlir::stablehlo::DivOp>(op);
  auto elemType =
      mlir::cast<mlir::RankedTensorType>(divOp.getType()).getElementType();

  return lowerBinaryElementwise(
      divOp, rewriter,
      [&](mlir::OpBuilder &b, mlir::Location loc, mlir::Value lhs,
          mlir::Value rhs) -> mlir::Value {
        if (mlir::isa<mlir::FloatType>(elemType)) {
          return b.create<mlir::arith::DivFOp>(loc, lhs, rhs);
        } else {
          // Signed integer division
          return b.create<mlir::arith::DivSIOp>(loc, lhs, rhs);
        }
      });
}

MaxOpLowering::MaxOpLowering(mlir::MLIRContext *ctx)
    : OpLoweringBase("stablehlo.maximum", ctx) {}

mlir::LogicalResult
MaxOpLowering::matchAndRewrite(mlir::Operation *op,
                               mlir::PatternRewriter &rewriter) const {
  auto maxOp = mlir::cast<mlir::stablehlo::MaxOp>(op);
  auto elemType =
      mlir::cast<mlir::RankedTensorType>(maxOp.getType()).getElementType();

  return lowerBinaryElementwise(
      maxOp, rewriter,
      [&](mlir::OpBuilder &b, mlir::Location loc, mlir::Value lhs,
          mlir::Value rhs) -> mlir::Value {
        if (mlir::isa<mlir::FloatType>(elemType)) {
          return b.create<mlir::arith::MaximumFOp>(loc, lhs, rhs);
        } else {
          return b.create<mlir::arith::MaxSIOp>(loc, lhs, rhs);
        }
      });
}

//===----------------------------------------------------------------------===//
// Unary Elementwise Operations
//===----------------------------------------------------------------------===//

NegateOpLowering::NegateOpLowering(mlir::MLIRContext *ctx)
    : OpLoweringBase("stablehlo.negate", ctx) {}

mlir::LogicalResult
NegateOpLowering::matchAndRewrite(mlir::Operation *op,
                                  mlir::PatternRewriter &rewriter) const {
  auto negOp = mlir::cast<mlir::stablehlo::NegOp>(op);
  auto elemType =
      mlir::cast<mlir::RankedTensorType>(negOp.getType()).getElementType();

  return lowerUnaryElementwise(
      negOp, rewriter,
      [&](mlir::OpBuilder &b, mlir::Location loc,
          mlir::Value input) -> mlir::Value {
        if (mlir::isa<mlir::FloatType>(elemType)) {
          return b.create<mlir::arith::NegFOp>(loc, input);
        } else {
          // Integer negation: 0 - x
          auto zero = b.create<mlir::arith::ConstantIntOp>(loc, 0, elemType);
          return b.create<mlir::arith::SubIOp>(loc, zero, input);
        }
      });
}

ExpOpLowering::ExpOpLowering(mlir::MLIRContext *ctx)
    : OpLoweringBase("stablehlo.exponential", ctx) {}

mlir::LogicalResult
ExpOpLowering::matchAndRewrite(mlir::Operation *op,
                               mlir::PatternRewriter &rewriter) const {
  auto expOp = mlir::cast<mlir::stablehlo::ExpOp>(op);

  return lowerUnaryElementwise(
      expOp, rewriter,
      [&](mlir::OpBuilder &b, mlir::Location loc,
          mlir::Value input) -> mlir::Value {
        return b.create<mlir::math::ExpOp>(loc, input);
      });
}

SqrtOpLowering::SqrtOpLowering(mlir::MLIRContext *ctx)
    : OpLoweringBase("stablehlo.sqrt", ctx) {}

mlir::LogicalResult
SqrtOpLowering::matchAndRewrite(mlir::Operation *op,
                                mlir::PatternRewriter &rewriter) const {
  auto sqrtOp = mlir::cast<mlir::stablehlo::SqrtOp>(op);

  return lowerUnaryElementwise(
      sqrtOp, rewriter,
      [&](mlir::OpBuilder &b, mlir::Location loc,
          mlir::Value input) -> mlir::Value {
        return b.create<mlir::math::SqrtOp>(loc, input);
      });
}

RsqrtOpLowering::RsqrtOpLowering(mlir::MLIRContext *ctx)
    : OpLoweringBase("stablehlo.rsqrt", ctx) {}

mlir::LogicalResult
RsqrtOpLowering::matchAndRewrite(mlir::Operation *op,
                                 mlir::PatternRewriter &rewriter) const {
  auto rsqrtOp = mlir::cast<mlir::stablehlo::RsqrtOp>(op);

  return lowerUnaryElementwise(
      rsqrtOp, rewriter,
      [&](mlir::OpBuilder &b, mlir::Location loc,
          mlir::Value input) -> mlir::Value {
        return b.create<mlir::math::RsqrtOp>(loc, input);
      });
}

LogOpLowering::LogOpLowering(mlir::MLIRContext *ctx)
    : OpLoweringBase("stablehlo.log", ctx) {}

mlir::LogicalResult
LogOpLowering::matchAndRewrite(mlir::Operation *op,
                               mlir::PatternRewriter &rewriter) const {
  auto logOp = mlir::cast<mlir::stablehlo::LogOp>(op);

  return lowerUnaryElementwise(
      logOp, rewriter,
      [&](mlir::OpBuilder &b, mlir::Location loc,
          mlir::Value input) -> mlir::Value {
        return b.create<mlir::math::LogOp>(loc, input);
      });
}

TanhOpLowering::TanhOpLowering(mlir::MLIRContext *ctx)
    : OpLoweringBase("stablehlo.tanh", ctx) {}

mlir::LogicalResult
TanhOpLowering::matchAndRewrite(mlir::Operation *op,
                                mlir::PatternRewriter &rewriter) const {
  auto tanhOp = mlir::cast<mlir::stablehlo::TanhOp>(op);

  return lowerUnaryElementwise(
      tanhOp, rewriter,
      [&](mlir::OpBuilder &b, mlir::Location loc,
          mlir::Value input) -> mlir::Value {
        return b.create<mlir::math::TanhOp>(loc, input);
      });
}

} // namespace lowerings
} // namespace keren
