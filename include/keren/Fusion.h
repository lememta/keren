#ifndef KEREN_FUSION_H
#define KEREN_FUSION_H

#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"

namespace keren {

//===----------------------------------------------------------------------===//
// Fusion Analysis
//===----------------------------------------------------------------------===//

/// Analyzes whether two operations can be fused.
///
/// SAGE Contract: @op can_fuse
///   @req producer.results ∩ consumer.operands ≠ ∅
///   @ens result = True ⟹ (is_elementwise(producer) ∧ is_elementwise(consumer))
///   @ens result = True ⟹ |users(producer.results[0])| = 1
class FusionAnalysis {
public:
  /// Check if an operation is elementwise (all parallel iterators).
  static bool isElementwiseOp(mlir::Operation *op);

  /// Check if producer and consumer can be fused.
  static bool canFuse(mlir::Operation *producer, mlir::Operation *consumer);

  /// Check if op is a fusion barrier (reduction, contraction).
  static bool isFusionBarrier(mlir::Operation *op);
};

//===----------------------------------------------------------------------===//
// Elementwise Fusion Pass
//===----------------------------------------------------------------------===//

/// Fuses consecutive elementwise linalg.generic operations.
///
/// SAGE Contract: @op fuse_elementwise
///   @req |ops| >= 2
///   @req ∀ op ∈ ops: is_elementwise(op)
///   @ens result.name = "linalg.generic"
///   @ens result.body contains ∀ op ∈ ops: op.body
///
/// Example:
///   Before:
///     %0 = linalg.generic {arith.addf}(%a, %b)
///     %1 = linalg.generic {arith.mulf}(%0, %c)
///
///   After:
///     %1 = linalg.generic {
///       %t = arith.addf %a, %b
///       %r = arith.mulf %t, %c
///       yield %r
///     }(%a, %b, %c)
class ElementwiseFusionPass
    : public mlir::PassWrapper<ElementwiseFusionPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ElementwiseFusionPass)

  llvm::StringRef getArgument() const override { return "keren-fuse-elementwise"; }
  llvm::StringRef getDescription() const override {
    return "Fuse consecutive elementwise linalg.generic operations";
  }

  void runOnOperation() override;

private:
  /// Find all fusable producer-consumer pairs.
  llvm::SmallVector<std::pair<mlir::Operation *, mlir::Operation *>>
  findFusionCandidates(mlir::func::FuncOp func);

  /// Fuse producer into consumer, returning the new fused op.
  mlir::Operation *fuseProducerIntoConsumer(mlir::Operation *producer,
                                            mlir::Operation *consumer,
                                            mlir::PatternRewriter &rewriter);
};

/// Create the elementwise fusion pass.
std::unique_ptr<mlir::Pass> createElementwiseFusionPass();

} // namespace keren

#endif // KEREN_FUSION_H
