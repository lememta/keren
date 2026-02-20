//===- OpLowering.cpp - Op lowering pattern registry ----------------------===//
//
// This file implements the OpLoweringBase helper methods and the
// OpLoweringRegistry that collects all op lowering patterns.
//
//===----------------------------------------------------------------------===//

#include "keren/OpLowering.h"
#include "keren/Lowerings/ContractionOps.h"
#include "keren/Lowerings/ElementwiseOps.h"
#include "keren/Lowerings/ReductionOps.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineMap.h"

namespace keren {

//===----------------------------------------------------------------------===//
// OpLoweringBase Helpers
//===----------------------------------------------------------------------===//

mlir::Value OpLoweringBase::createEmptyTensor(mlir::PatternRewriter &rewriter,
                                               mlir::Location loc,
                                               mlir::RankedTensorType type) const {
  llvm::SmallVector<mlir::Value> dynamicDims;
  return mlir::tensor::EmptyOp::create(rewriter, loc, type, dynamicDims);
}

llvm::SmallVector<mlir::AffineMap>
OpLoweringBase::createIdentityMaps(mlir::MLIRContext *ctx, unsigned rank,
                                   unsigned numOperands) const {
  llvm::SmallVector<mlir::AffineMap> maps;
  auto identityMap = mlir::AffineMap::getMultiDimIdentityMap(rank, ctx);
  for (unsigned i = 0; i < numOperands; ++i) {
    maps.push_back(identityMap);
  }
  return maps;
}

llvm::SmallVector<mlir::utils::IteratorType>
OpLoweringBase::createParallelIterators(unsigned rank) const {
  return llvm::SmallVector<mlir::utils::IteratorType>(
      rank, mlir::utils::IteratorType::parallel);
}

//===----------------------------------------------------------------------===//
// OpLoweringRegistry
//===----------------------------------------------------------------------===//

void OpLoweringRegistry::populatePatterns(mlir::RewritePatternSet &patterns,
                                          mlir::MLIRContext *ctx) {
  registerElementwiseBinaryPatterns(patterns, ctx);
  registerElementwiseUnaryPatterns(patterns, ctx);
  registerContractionPatterns(patterns, ctx);
  registerReductionPatterns(patterns, ctx);
  registerShapePatterns(patterns, ctx);
}

void OpLoweringRegistry::registerElementwiseBinaryPatterns(
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *ctx) {
  patterns.add<lowerings::AddOpLowering>(ctx);
  patterns.add<lowerings::MulOpLowering>(ctx);
  patterns.add<lowerings::SubOpLowering>(ctx);
  patterns.add<lowerings::DivOpLowering>(ctx);
  patterns.add<lowerings::MaxOpLowering>(ctx);
}

void OpLoweringRegistry::registerElementwiseUnaryPatterns(
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *ctx) {
  patterns.add<lowerings::NegateOpLowering>(ctx);
  patterns.add<lowerings::ExpOpLowering>(ctx);
  patterns.add<lowerings::SqrtOpLowering>(ctx);
  patterns.add<lowerings::RsqrtOpLowering>(ctx);
  patterns.add<lowerings::LogOpLowering>(ctx);
  patterns.add<lowerings::TanhOpLowering>(ctx);
}

void OpLoweringRegistry::registerContractionPatterns(
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *ctx) {
  patterns.add<lowerings::DotGeneralOpLowering>(ctx);
}

void OpLoweringRegistry::registerReductionPatterns(
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *ctx) {
  patterns.add<lowerings::ReduceOpLowering>(ctx);
}

void OpLoweringRegistry::registerShapePatterns(
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *ctx) {
  patterns.add<lowerings::BroadcastInDimOpLowering>(ctx);
}

} // namespace keren
