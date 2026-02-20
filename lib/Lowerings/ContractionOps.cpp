//===- ContractionOps.cpp - StableHLO dot_general lowering ----------------===//
//
// This file implements op-by-op lowering for stablehlo.dot_general to
// linalg.matmul, linalg.batch_matmul, or linalg.generic.
//
// SAGE Contract: @op lower_dot_general
//   @req op.name = "stablehlo.dot_general"
//   @ens result[0].name ∈ {"linalg.matmul", "linalg.batch_matmul", "linalg.generic"}
//   @ens result[0].results[0].type = op.results[0].type
//
//===----------------------------------------------------------------------===//

#include "keren/Lowerings/ContractionOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineMap.h"

#include "stablehlo/dialect/StablehloOps.h"

namespace keren {
namespace lowerings {

DotGeneralOpLowering::DotGeneralOpLowering(mlir::MLIRContext *ctx)
    : OpLoweringBase("stablehlo.dot_general", ctx) {}

/// Check if this is a standard matmul: lhs_contracting=[1], rhs_contracting=[0]
/// with no batch dimensions.
bool DotGeneralOpLowering::isStandardMatmul(mlir::Operation *op) const {
  auto dotOp = mlir::cast<mlir::stablehlo::DotGeneralOp>(op);
  auto dims = dotOp.getDotDimensionNumbers();

  auto lhsContract = dims.getLhsContractingDimensions();
  auto rhsContract = dims.getRhsContractingDimensions();
  auto lhsBatch = dims.getLhsBatchingDimensions();
  auto rhsBatch = dims.getRhsBatchingDimensions();

  // Standard matmul: no batch dims, single contraction dim
  if (!lhsBatch.empty() || !rhsBatch.empty())
    return false;
  if (lhsContract.size() != 1 || rhsContract.size() != 1)
    return false;

  auto lhsType = mlir::cast<mlir::RankedTensorType>(dotOp.getLhs().getType());
  auto rhsType = mlir::cast<mlir::RankedTensorType>(dotOp.getRhs().getType());

  // Must be 2D tensors with contracting dims at [1] and [0]
  return lhsType.getRank() == 2 && rhsType.getRank() == 2 &&
         lhsContract[0] == 1 && rhsContract[0] == 0;
}

/// Check if this is a batch matmul: batch dims at [0], contracting at [2] and [1]
bool DotGeneralOpLowering::isBatchMatmul(mlir::Operation *op) const {
  auto dotOp = mlir::cast<mlir::stablehlo::DotGeneralOp>(op);
  auto dims = dotOp.getDotDimensionNumbers();

  auto lhsContract = dims.getLhsContractingDimensions();
  auto rhsContract = dims.getRhsContractingDimensions();
  auto lhsBatch = dims.getLhsBatchingDimensions();
  auto rhsBatch = dims.getRhsBatchingDimensions();

  // Batch matmul: single batch dim at [0], single contraction
  if (lhsBatch.size() != 1 || rhsBatch.size() != 1)
    return false;
  if (lhsContract.size() != 1 || rhsContract.size() != 1)
    return false;

  auto lhsType = mlir::cast<mlir::RankedTensorType>(dotOp.getLhs().getType());
  auto rhsType = mlir::cast<mlir::RankedTensorType>(dotOp.getRhs().getType());

  // Must be 3D tensors with batch at [0], contraction at [2] and [1]
  return lhsType.getRank() == 3 && rhsType.getRank() == 3 &&
         lhsBatch[0] == 0 && rhsBatch[0] == 0 && lhsContract[0] == 2 &&
         rhsContract[0] == 1;
}

mlir::LogicalResult
DotGeneralOpLowering::lowerToMatmul(mlir::Operation *op,
                                    mlir::PatternRewriter &rewriter) const {
  auto dotOp = mlir::cast<mlir::stablehlo::DotGeneralOp>(op);
  auto loc = op->getLoc();
  auto resultType = mlir::cast<mlir::RankedTensorType>(dotOp.getType());

  // Create zero-initialized output tensor
  mlir::Value zero;
  if (mlir::isa<mlir::FloatType>(resultType.getElementType())) {
    zero = rewriter.create<mlir::arith::ConstantOp>(
        loc, rewriter.getFloatAttr(resultType.getElementType(), 0.0));
  } else {
    auto attr = rewriter.getIntegerAttr(resultType.getElementType(), 0);
    zero = rewriter.create<mlir::arith::ConstantOp>(loc, attr);
  }

  mlir::Value emptyTensor =
      mlir::tensor::EmptyOp::create(rewriter, loc, resultType, mlir::ValueRange{});
  mlir::Value filledTensor =
      rewriter.create<mlir::linalg::FillOp>(loc, zero, emptyTensor).getResult(0);

  // Create linalg.matmul
  auto matmulOp = rewriter.create<mlir::linalg::MatmulOp>(
      loc, resultType, mlir::ValueRange{dotOp.getLhs(), dotOp.getRhs()},
      mlir::ValueRange{filledTensor});

  rewriter.replaceOp(op, matmulOp.getResults());
  return mlir::success();
}

mlir::LogicalResult
DotGeneralOpLowering::lowerToBatchMatmul(mlir::Operation *op,
                                         mlir::PatternRewriter &rewriter) const {
  auto dotOp = mlir::cast<mlir::stablehlo::DotGeneralOp>(op);
  auto loc = op->getLoc();
  auto resultType = mlir::cast<mlir::RankedTensorType>(dotOp.getType());

  // Create zero-initialized output tensor
  mlir::Value zero;
  if (mlir::isa<mlir::FloatType>(resultType.getElementType())) {
    zero = rewriter.create<mlir::arith::ConstantOp>(
        loc, rewriter.getFloatAttr(resultType.getElementType(), 0.0));
  } else {
    auto attr = rewriter.getIntegerAttr(resultType.getElementType(), 0);
    zero = rewriter.create<mlir::arith::ConstantOp>(loc, attr);
  }

  mlir::Value emptyTensor =
      mlir::tensor::EmptyOp::create(rewriter, loc, resultType, mlir::ValueRange{});
  mlir::Value filledTensor =
      rewriter.create<mlir::linalg::FillOp>(loc, zero, emptyTensor).getResult(0);

  // Create linalg.batch_matmul
  auto batchMatmulOp = rewriter.create<mlir::linalg::BatchMatmulOp>(
      loc, resultType, mlir::ValueRange{dotOp.getLhs(), dotOp.getRhs()},
      mlir::ValueRange{filledTensor});

  rewriter.replaceOp(op, batchMatmulOp.getResults());
  return mlir::success();
}

mlir::LogicalResult
DotGeneralOpLowering::lowerToGeneric(mlir::Operation *op,
                                     mlir::PatternRewriter &rewriter) const {
  auto dotOp = mlir::cast<mlir::stablehlo::DotGeneralOp>(op);
  auto loc = op->getLoc();
  auto dims = dotOp.getDotDimensionNumbers();

  auto lhsType = mlir::cast<mlir::RankedTensorType>(dotOp.getLhs().getType());
  auto rhsType = mlir::cast<mlir::RankedTensorType>(dotOp.getRhs().getType());
  auto resultType = mlir::cast<mlir::RankedTensorType>(dotOp.getType());

  auto lhsContract = dims.getLhsContractingDimensions();
  auto rhsContract = dims.getRhsContractingDimensions();
  auto lhsBatch = dims.getLhsBatchingDimensions();
  auto rhsBatch = dims.getRhsBatchingDimensions();

  // Build affine maps for the generic op
  // This is a simplified version - a full implementation would handle
  // all dimension permutations
  unsigned lhsRank = lhsType.getRank();
  unsigned rhsRank = rhsType.getRank();
  unsigned resultRank = resultType.getRank();

  // Calculate total number of dimensions in the iteration space
  unsigned numBatch = lhsBatch.size();
  unsigned numContract = lhsContract.size();
  unsigned numLhsFree = lhsRank - numBatch - numContract;
  unsigned numRhsFree = rhsRank - numBatch - numContract;
  unsigned numIterDims = numBatch + numLhsFree + numRhsFree + numContract;

  // Create iterator types: batch + lhs_free + rhs_free are parallel, contracting is reduction
  llvm::SmallVector<mlir::utils::IteratorType> iteratorTypes;
  for (unsigned i = 0; i < numBatch + numLhsFree + numRhsFree; ++i) {
    iteratorTypes.push_back(mlir::utils::IteratorType::parallel);
  }
  for (unsigned i = 0; i < numContract; ++i) {
    iteratorTypes.push_back(mlir::utils::IteratorType::reduction);
  }

  // Build affine expressions for indexing maps
  auto ctx = op->getContext();
  llvm::SmallVector<mlir::AffineExpr> lhsExprs, rhsExprs, resultExprs;

  // For simplicity, assume standard ordering: batch dims first, then free, then contracting
  unsigned dimIdx = 0;

  // Batch dimensions
  for (unsigned i = 0; i < numBatch; ++i) {
    lhsExprs.push_back(mlir::getAffineDimExpr(dimIdx, ctx));
    rhsExprs.push_back(mlir::getAffineDimExpr(dimIdx, ctx));
    resultExprs.push_back(mlir::getAffineDimExpr(dimIdx, ctx));
    ++dimIdx;
  }

  // LHS free dimensions
  for (unsigned i = 0; i < numLhsFree; ++i) {
    lhsExprs.push_back(mlir::getAffineDimExpr(dimIdx, ctx));
    resultExprs.push_back(mlir::getAffineDimExpr(dimIdx, ctx));
    ++dimIdx;
  }

  // RHS free dimensions
  for (unsigned i = 0; i < numRhsFree; ++i) {
    rhsExprs.push_back(mlir::getAffineDimExpr(dimIdx, ctx));
    resultExprs.push_back(mlir::getAffineDimExpr(dimIdx, ctx));
    ++dimIdx;
  }

  // Contracting dimensions (reduction)
  for (unsigned i = 0; i < numContract; ++i) {
    lhsExprs.push_back(mlir::getAffineDimExpr(dimIdx, ctx));
    rhsExprs.push_back(mlir::getAffineDimExpr(dimIdx, ctx));
    ++dimIdx;
  }

  // Create affine maps
  auto lhsMap = mlir::AffineMap::get(numIterDims, 0, lhsExprs, ctx);
  auto rhsMap = mlir::AffineMap::get(numIterDims, 0, rhsExprs, ctx);
  auto resultMap = mlir::AffineMap::get(numIterDims, 0, resultExprs, ctx);

  llvm::SmallVector<mlir::AffineMap> indexingMaps = {lhsMap, rhsMap, resultMap};

  // Create zero-initialized output tensor
  mlir::Value zero;
  auto elemType = resultType.getElementType();
  if (mlir::isa<mlir::FloatType>(elemType)) {
    zero = rewriter.create<mlir::arith::ConstantOp>(
        loc, rewriter.getFloatAttr(elemType, 0.0));
  } else {
    auto attr = rewriter.getIntegerAttr(elemType, 0);
    zero = rewriter.create<mlir::arith::ConstantOp>(loc, attr);
  }

  mlir::Value emptyTensor =
      mlir::tensor::EmptyOp::create(rewriter, loc, resultType, mlir::ValueRange{});
  mlir::Value filledTensor =
      rewriter.create<mlir::linalg::FillOp>(loc, zero, emptyTensor).getResult(0);

  // Create linalg.generic for general dot
  auto genericOp = rewriter.create<mlir::linalg::GenericOp>(
      loc, resultType,
      /*inputs=*/mlir::ValueRange{dotOp.getLhs(), dotOp.getRhs()},
      /*outputs=*/mlir::ValueRange{filledTensor}, indexingMaps, iteratorTypes,
      [&](mlir::OpBuilder &b, mlir::Location loc, mlir::ValueRange args) {
        mlir::Value lhs = args[0];
        mlir::Value rhs = args[1];
        mlir::Value acc = args[2];
        mlir::Value prod, sum;
        if (mlir::isa<mlir::FloatType>(elemType)) {
          prod = b.create<mlir::arith::MulFOp>(loc, lhs, rhs);
          sum = b.create<mlir::arith::AddFOp>(loc, acc, prod);
        } else {
          prod = b.create<mlir::arith::MulIOp>(loc, lhs, rhs);
          sum = b.create<mlir::arith::AddIOp>(loc, acc, prod);
        }
        b.create<mlir::linalg::YieldOp>(loc, sum);
      });

  rewriter.replaceOp(op, genericOp.getResults());
  return mlir::success();
}

mlir::LogicalResult
DotGeneralOpLowering::matchAndRewrite(mlir::Operation *op,
                                      mlir::PatternRewriter &rewriter) const {
  // Try specialized lowerings first, fall back to generic
  if (isStandardMatmul(op)) {
    return lowerToMatmul(op, rewriter);
  }
  if (isBatchMatmul(op)) {
    return lowerToBatchMatmul(op, rewriter);
  }
  return lowerToGeneric(op, rewriter);
}

} // namespace lowerings
} // namespace keren
