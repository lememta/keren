#ifndef KEREN_LOWERINGS_REDUCTIONOPS_H
#define KEREN_LOWERINGS_REDUCTIONOPS_H

#include "keren/OpLowering.h"

namespace keren {
namespace lowerings {

/// Lower stablehlo.reduce to linalg.reduce.
///
/// SAGE Contract: @op lower_reduce
///   @req op.name = "stablehlo.reduce"
///   @ens result[0].name = "linalg.reduce"
///   @ens output_rank = input_rank - |reduction_dims|
///
/// Handles common reduction operations:
/// - Sum: body with arith.addf/addi
/// - Max: body with arith.maximumf/maxsi
/// - Min: body with arith.minimumf/minsi
/// - Product: body with arith.mulf/muli
class ReduceOpLowering : public OpLoweringBase {
public:
  ReduceOpLowering(mlir::MLIRContext *ctx);

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;

private:
  /// Detect the type of reduction from the StableHLO reduce body.
  enum class ReductionKind { Add, Max, Min, Mul, Unknown };
  ReductionKind detectReductionKind(mlir::Region &body) const;

  /// Build the body of the linalg.reduce op.
  void buildReduceBody(mlir::OpBuilder &builder, mlir::Location loc,
                       ReductionKind kind, mlir::Type elemType,
                       mlir::ValueRange args) const;
};

/// Lower stablehlo.broadcast_in_dim to linalg.broadcast.
///
/// SAGE Contract: @op lower_broadcast_in_dim
///   @req op.name = "stablehlo.broadcast_in_dim"
///   @ens result[0].name = "linalg.broadcast"
class BroadcastInDimOpLowering : public OpLoweringBase {
public:
  BroadcastInDimOpLowering(mlir::MLIRContext *ctx);

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};

} // namespace lowerings
} // namespace keren

#endif // KEREN_LOWERINGS_REDUCTIONOPS_H
