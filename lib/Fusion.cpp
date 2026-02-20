//===- Fusion.cpp - Elementwise operation fusion --------------------------===//
//
// This file implements the ElementwiseFusionPass, which fuses consecutive
// elementwise linalg.generic operations to reduce memory bandwidth.
//
// SAGE Contract: @op fuse_elementwise
//   @req |ops| >= 2
//   @req ∀ op ∈ ops: is_elementwise(op)
//   @ens result.name = "linalg.generic"
//   @ens result.body contains ∀ op ∈ ops: op.body
//
// Example transformation:
//   %0 = linalg.generic {add}(%a, %b) -> tensor<NxMxf32>
//   %1 = linalg.generic {mul}(%0, %c) -> tensor<NxMxf32>
//
//   Becomes:
//   %1 = linalg.generic {
//     ^bb0(%a_elem, %b_elem, %c_elem, %out):
//       %t = arith.addf %a_elem, %b_elem
//       %r = arith.mulf %t, %c_elem
//       linalg.yield %r
//   }(%a, %b, %c)
//
//===----------------------------------------------------------------------===//

#include "keren/Fusion.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace keren {

//===----------------------------------------------------------------------===//
// FusionAnalysis
//===----------------------------------------------------------------------===//

bool FusionAnalysis::isElementwiseOp(mlir::Operation *op) {
  auto genericOp = mlir::dyn_cast<mlir::linalg::GenericOp>(op);
  if (!genericOp)
    return false;

  // Check that all iterator types are parallel
  auto iteratorTypes = genericOp.getIteratorTypesArray();
  for (auto iterType : iteratorTypes) {
    if (iterType != mlir::utils::IteratorType::parallel)
      return false;
  }

  // Check that all indexing maps are identity (or projections for broadcasts)
  // For strict elementwise, we require identity maps
  for (auto map : genericOp.getIndexingMapsArray()) {
    if (!map.isIdentity() && !map.isProjectedPermutation())
      return false;
  }

  return true;
}

bool FusionAnalysis::isFusionBarrier(mlir::Operation *op) {
  // Reductions and contractions are fusion barriers
  if (mlir::isa<mlir::linalg::ReduceOp>(op))
    return true;
  if (mlir::isa<mlir::linalg::MatmulOp>(op))
    return true;
  if (mlir::isa<mlir::linalg::BatchMatmulOp>(op))
    return true;

  // For generic ops, check if any iterator is a reduction
  if (auto genericOp = mlir::dyn_cast<mlir::linalg::GenericOp>(op)) {
    for (auto iterType : genericOp.getIteratorTypesArray()) {
      if (iterType == mlir::utils::IteratorType::reduction)
        return true;
    }
  }

  return false;
}

bool FusionAnalysis::canFuse(mlir::Operation *producer,
                             mlir::Operation *consumer) {
  // Both must be elementwise
  if (!isElementwiseOp(producer) || !isElementwiseOp(consumer))
    return false;

  // Producer must have exactly one result
  if (producer->getNumResults() != 1)
    return false;

  // Producer result must be used only by consumer (single consumer)
  mlir::Value producerResult = producer->getResult(0);
  if (!producerResult.hasOneUse())
    return false;

  // Check that consumer uses producer's result
  bool usesProducer = false;
  for (auto operand : consumer->getOperands()) {
    if (operand == producerResult) {
      usesProducer = true;
      break;
    }
  }
  if (!usesProducer)
    return false;

  // Check matching iteration domain
  auto producerGeneric = mlir::cast<mlir::linalg::GenericOp>(producer);
  auto consumerGeneric = mlir::cast<mlir::linalg::GenericOp>(consumer);

  // For simplicity, require same number of loops
  if (producerGeneric.getNumLoops() != consumerGeneric.getNumLoops())
    return false;

  return true;
}

//===----------------------------------------------------------------------===//
// ElementwiseFusionPass Implementation
//===----------------------------------------------------------------------===//

llvm::SmallVector<std::pair<mlir::Operation *, mlir::Operation *>>
ElementwiseFusionPass::findFusionCandidates(mlir::func::FuncOp func) {
  llvm::SmallVector<std::pair<mlir::Operation *, mlir::Operation *>> candidates;

  // Walk all linalg.generic ops
  func.walk([&](mlir::linalg::GenericOp genericOp) {
    // Check each operand to see if its producer can be fused
    for (auto operand : genericOp.getDpsInputs()) {
      if (auto definingOp = operand.getDefiningOp()) {
        if (FusionAnalysis::canFuse(definingOp, genericOp)) {
          candidates.push_back({definingOp, genericOp});
        }
      }
    }
  });

  return candidates;
}

mlir::Operation *
ElementwiseFusionPass::fuseProducerIntoConsumer(mlir::Operation *producer,
                                                mlir::Operation *consumer,
                                                mlir::PatternRewriter &rewriter) {
  auto producerGeneric = mlir::cast<mlir::linalg::GenericOp>(producer);
  auto consumerGeneric = mlir::cast<mlir::linalg::GenericOp>(consumer);
  auto loc = consumer->getLoc();

  // Find which consumer operand corresponds to producer's result
  mlir::Value producerResult = producerGeneric.getResult(0);
  unsigned producerOperandIdx = 0;
  for (auto [idx, operand] : llvm::enumerate(consumerGeneric.getDpsInputs())) {
    if (operand == producerResult) {
      producerOperandIdx = idx;
      break;
    }
  }

  // Collect new inputs: producer's inputs + consumer's inputs (excluding producer result)
  llvm::SmallVector<mlir::Value> newInputs;

  // Add producer's inputs
  for (auto input : producerGeneric.getDpsInputs()) {
    newInputs.push_back(input);
  }

  // Add consumer's inputs, excluding the one that comes from producer
  for (auto [idx, input] : llvm::enumerate(consumerGeneric.getDpsInputs())) {
    if (input != producerResult) {
      newInputs.push_back(input);
    }
  }

  // Use consumer's outputs
  auto outputs = consumerGeneric.getDpsInits();

  // Build new indexing maps
  llvm::SmallVector<mlir::AffineMap> newIndexingMaps;

  // Producer's input maps
  for (auto map : producerGeneric.getIndexingMapsArray()) {
    if (&map - &producerGeneric.getIndexingMapsArray()[0] <
        producerGeneric.getNumDpsInputs()) {
      newIndexingMaps.push_back(map);
    }
  }

  // Consumer's input maps (excluding the producer result's map)
  auto consumerMaps = consumerGeneric.getIndexingMapsArray();
  for (unsigned i = 0; i < consumerGeneric.getNumDpsInputs(); ++i) {
    if (i != producerOperandIdx) {
      newIndexingMaps.push_back(consumerMaps[i]);
    }
  }

  // Consumer's output maps
  for (unsigned i = consumerGeneric.getNumDpsInputs();
       i < consumerMaps.size(); ++i) {
    newIndexingMaps.push_back(consumerMaps[i]);
  }

  // Iterator types (same as consumer since domains match)
  auto iteratorTypes = consumerGeneric.getIteratorTypesArray();

  // Create the fused generic op
  auto fusedOp = rewriter.create<mlir::linalg::GenericOp>(
      loc, consumerGeneric.getResultTypes(), newInputs,
      llvm::SmallVector<mlir::Value>(outputs.begin(), outputs.end()),
      newIndexingMaps, iteratorTypes,
      [&](mlir::OpBuilder &b, mlir::Location loc, mlir::ValueRange args) {
        mlir::IRMapping mapper;

        // Map producer's block arguments to fused op's arguments
        mlir::Block &producerBlock = *producerGeneric.getBody();
        unsigned argIdx = 0;
        for (auto arg : producerBlock.getArguments()) {
          if (argIdx < producerGeneric.getNumDpsInputs()) {
            mapper.map(arg, args[argIdx]);
          }
          ++argIdx;
        }

        // Clone producer's body (except yield)
        mlir::Value producerBodyResult;
        for (auto &op : producerBlock.without_terminator()) {
          auto *cloned = b.clone(op, mapper);
          // The last non-yield op's result is what we pass to consumer
          if (cloned->getNumResults() > 0) {
            for (auto result : cloned->getResults()) {
              mapper.map(op.getResult(0), result);
              producerBodyResult = result;
            }
          }
        }

        // Get the actual yielded value from producer
        auto producerYield =
            mlir::cast<mlir::linalg::YieldOp>(producerBlock.getTerminator());
        producerBodyResult = mapper.lookupOrDefault(producerYield.getOperand(0));

        // Map consumer's block arguments
        mlir::Block &consumerBlock = *consumerGeneric.getBody();
        unsigned consumerArgIdx = 0;
        unsigned fusedArgIdx = producerGeneric.getNumDpsInputs();
        for (auto arg : consumerBlock.getArguments()) {
          if (consumerArgIdx == producerOperandIdx) {
            // This is where producer's result goes
            mapper.map(arg, producerBodyResult);
          } else if (consumerArgIdx < consumerGeneric.getNumDpsInputs()) {
            mapper.map(arg, args[fusedArgIdx]);
            ++fusedArgIdx;
          } else {
            // Output argument
            mapper.map(arg, args[fusedArgIdx]);
            ++fusedArgIdx;
          }
          ++consumerArgIdx;
        }

        // Clone consumer's body
        for (auto &op : consumerBlock.without_terminator()) {
          b.clone(op, mapper);
        }

        // Clone consumer's yield
        auto consumerYield =
            mlir::cast<mlir::linalg::YieldOp>(consumerBlock.getTerminator());
        llvm::SmallVector<mlir::Value> yieldOperands;
        for (auto operand : consumerYield.getOperands()) {
          yieldOperands.push_back(mapper.lookupOrDefault(operand));
        }
        b.create<mlir::linalg::YieldOp>(loc, yieldOperands);
      });

  return fusedOp;
}

void ElementwiseFusionPass::runOnOperation() {
  auto func = getOperation();

  // Iteratively find and apply fusions until no more candidates
  bool changed = true;
  while (changed) {
    changed = false;

    auto candidates = findFusionCandidates(func);
    if (candidates.empty())
      break;

    // Apply first valid fusion (greedy)
    for (auto [producer, consumer] : candidates) {
      // Verify fusion is still valid (ops might have changed)
      if (!FusionAnalysis::canFuse(producer, consumer))
        continue;

      mlir::PatternRewriter rewriter(func.getContext());
      rewriter.setInsertionPoint(consumer);

      auto fusedOp = fuseProducerIntoConsumer(producer, consumer, rewriter);
      if (fusedOp) {
        // Replace consumer with fused op
        rewriter.replaceOp(consumer, fusedOp->getResults());
        // Erase producer (no longer needed)
        rewriter.eraseOp(producer);
        changed = true;
        break; // Restart search
      }
    }
  }
}

std::unique_ptr<mlir::Pass> createElementwiseFusionPass() {
  return std::make_unique<ElementwiseFusionPass>();
}

} // namespace keren
