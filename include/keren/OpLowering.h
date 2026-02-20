#ifndef KEREN_OPLOWERING_H
#define KEREN_OPLOWERING_H

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

namespace keren {

/// Base class for op-by-op lowerings from StableHLO to Linalg.
/// Each lowering handles a specific StableHLO operation.
class OpLoweringBase : public mlir::RewritePattern {
public:
  OpLoweringBase(mlir::StringRef opName, mlir::MLIRContext *ctx,
                 mlir::PatternBenefit benefit = 1)
      : mlir::RewritePattern(opName, benefit, ctx) {}

  /// Subclasses implement this to perform the actual lowering.
  virtual mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override = 0;

protected:
  /// Helper: create an empty tensor for the output.
  mlir::Value createEmptyTensor(mlir::PatternRewriter &rewriter,
                                mlir::Location loc,
                                mlir::RankedTensorType type) const;

  /// Helper: create identity indexing maps for elementwise ops.
  llvm::SmallVector<mlir::AffineMap>
  createIdentityMaps(mlir::MLIRContext *ctx, unsigned rank,
                     unsigned numOperands) const;

  /// Helper: create parallel iterator types.
  llvm::SmallVector<mlir::utils::IteratorType>
  createParallelIterators(unsigned rank) const;
};

/// Registry of all op lowerings.
class OpLoweringRegistry {
public:
  /// Populate the pattern set with all registered lowerings.
  static void populatePatterns(mlir::RewritePatternSet &patterns,
                               mlir::MLIRContext *ctx);

  /// Register elementwise binary ops (add, mul, sub, div).
  static void registerElementwiseBinaryPatterns(
      mlir::RewritePatternSet &patterns, mlir::MLIRContext *ctx);

  /// Register elementwise unary ops (neg, exp, sqrt, etc).
  static void registerElementwiseUnaryPatterns(
      mlir::RewritePatternSet &patterns, mlir::MLIRContext *ctx);

  /// Register contraction ops (dot_general).
  static void registerContractionPatterns(mlir::RewritePatternSet &patterns,
                                          mlir::MLIRContext *ctx);

  /// Register reduction ops (reduce).
  static void registerReductionPatterns(mlir::RewritePatternSet &patterns,
                                        mlir::MLIRContext *ctx);

  /// Register shape ops (broadcast_in_dim, reshape, transpose).
  static void registerShapePatterns(mlir::RewritePatternSet &patterns,
                                    mlir::MLIRContext *ctx);
};

} // namespace keren

#endif // KEREN_OPLOWERING_H
