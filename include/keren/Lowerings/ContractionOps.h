#ifndef KEREN_LOWERINGS_CONTRACTIONOPS_H
#define KEREN_LOWERINGS_CONTRACTIONOPS_H

#include "keren/OpLowering.h"

namespace keren {
namespace lowerings {

/// Lower stablehlo.dot_general to linalg.matmul / linalg.batch_matmul.
///
/// SAGE Contract: @op lower_dot_general
///   @req op.name = "stablehlo.dot_general"
///   @ens result[0].name ∈ {"linalg.matmul", "linalg.batch_matmul", "linalg.generic"}
///
/// The lowering handles:
/// - Standard 2D matmul: (M, K) x (K, N) -> (M, N)
/// - Batch matmul: (B, M, K) x (B, K, N) -> (B, M, N)
/// - General dot with arbitrary contracting/batch dimensions
class DotGeneralOpLowering : public OpLoweringBase {
public:
  DotGeneralOpLowering(mlir::MLIRContext *ctx);

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;

private:
  /// Check if this is a standard matmul pattern.
  bool isStandardMatmul(mlir::Operation *op) const;

  /// Check if this is a batch matmul pattern.
  bool isBatchMatmul(mlir::Operation *op) const;

  /// Lower to linalg.matmul for standard 2D case.
  mlir::LogicalResult lowerToMatmul(mlir::Operation *op,
                                    mlir::PatternRewriter &rewriter) const;

  /// Lower to linalg.batch_matmul for 3D batch case.
  mlir::LogicalResult lowerToBatchMatmul(mlir::Operation *op,
                                         mlir::PatternRewriter &rewriter) const;

  /// Lower to linalg.generic for general case.
  mlir::LogicalResult lowerToGeneric(mlir::Operation *op,
                                     mlir::PatternRewriter &rewriter) const;
};

} // namespace lowerings
} // namespace keren

#endif // KEREN_LOWERINGS_CONTRACTIONOPS_H
