#ifndef KEREN_LOWERINGS_ELEMENTWISEOPS_H
#define KEREN_LOWERINGS_ELEMENTWISEOPS_H

#include "keren/OpLowering.h"

namespace keren {
namespace lowerings {

//===----------------------------------------------------------------------===//
// Binary Elementwise Operations
//===----------------------------------------------------------------------===//

/// Lower stablehlo.add to linalg.generic with arith.addf/addi
class AddOpLowering : public OpLoweringBase {
public:
  AddOpLowering(mlir::MLIRContext *ctx);

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};

/// Lower stablehlo.multiply to linalg.generic with arith.mulf/muli
class MulOpLowering : public OpLoweringBase {
public:
  MulOpLowering(mlir::MLIRContext *ctx);

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};

/// Lower stablehlo.subtract to linalg.generic with arith.subf/subi
class SubOpLowering : public OpLoweringBase {
public:
  SubOpLowering(mlir::MLIRContext *ctx);

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};

/// Lower stablehlo.divide to linalg.generic with arith.divf/divi
class DivOpLowering : public OpLoweringBase {
public:
  DivOpLowering(mlir::MLIRContext *ctx);

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};

/// Lower stablehlo.maximum to linalg.generic with arith.maximumf/maxsi
class MaxOpLowering : public OpLoweringBase {
public:
  MaxOpLowering(mlir::MLIRContext *ctx);

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};

//===----------------------------------------------------------------------===//
// Unary Elementwise Operations
//===----------------------------------------------------------------------===//

/// Lower stablehlo.negate to linalg.generic with arith.negf
class NegateOpLowering : public OpLoweringBase {
public:
  NegateOpLowering(mlir::MLIRContext *ctx);

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};

/// Lower stablehlo.exponential to linalg.generic with math.exp
class ExpOpLowering : public OpLoweringBase {
public:
  ExpOpLowering(mlir::MLIRContext *ctx);

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};

/// Lower stablehlo.sqrt to linalg.generic with math.sqrt
class SqrtOpLowering : public OpLoweringBase {
public:
  SqrtOpLowering(mlir::MLIRContext *ctx);

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};

/// Lower stablehlo.rsqrt to linalg.generic with math.rsqrt
class RsqrtOpLowering : public OpLoweringBase {
public:
  RsqrtOpLowering(mlir::MLIRContext *ctx);

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};

/// Lower stablehlo.log to linalg.generic with math.log
class LogOpLowering : public OpLoweringBase {
public:
  LogOpLowering(mlir::MLIRContext *ctx);

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};

/// Lower stablehlo.tanh to linalg.generic with math.tanh
class TanhOpLowering : public OpLoweringBase {
public:
  TanhOpLowering(mlir::MLIRContext *ctx);

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};

} // namespace lowerings
} // namespace keren

#endif // KEREN_LOWERINGS_ELEMENTWISEOPS_H
