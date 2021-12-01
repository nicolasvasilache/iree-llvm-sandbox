//===- InParallelToSequentialFor.cpp.cpp - Rewrite InParallel as ForOp ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/IR/BuiltinOps.h>

#include "Dialects/LinalgExt/LinalgExtOps.h"
#include "Dialects/LinalgExt/PassDetail.h"
#include "Dialects/LinalgExt/Passes.h"
#include "Transforms/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Identifier.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::linalg_ext;

namespace {

SmallVector<Value> getValuesToYield(PerformConcurrentlyOp op) {
  return llvm::to_vector(llvm::map_range(
      op.yieldingOps(), [](ParallelInsertSliceOp op) { return op.dest(); }));
}

struct InParallelOpToSCFRewriter
    : public OpRewritePattern<linalg_ext::InParallelOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg_ext::InParallelOp inParallelOp,
                                PatternRewriter &rewriter) const override {
    // Construct the loop bounds based on the canonical arithmetic progression.
    Location loc = inParallelOp.getLoc();
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value numThreads = inParallelOp.num_threads();

    // Construct the op without a body builder: we need to clone the ops in the
    // body explicitly after having access to the new bbArgs.
    // As a consequence, `ensureTerminator` is not called and the `forOp` body
    // has no terminator.
    PerformConcurrentlyOp performConcurrentlyOp = inParallelOp.getTerminator();
    SmallVector<Value> valuesToYield = getValuesToYield(performConcurrentlyOp);
    scf::ForOp forOp =
        rewriter.create<scf::ForOp>(loc, zero, numThreads, one, valuesToYield);

    // Move the body while replacing the threadId by the forOp iv.
    SmallVector<Value> bbArgsTranslated{forOp.getInductionVar()};
    rewriter.mergeBlocks(&inParallelOp.region().front(), forOp.getBody(),
                         bbArgsTranslated);

    rewriter.setInsertionPointToStart(forOp.getBody());
    BlockAndValueMapping bvm;
    bvm.map(valuesToYield, forOp.getRegionIterArgs());

    // Create sequential insertSlice ops.
    SmallVector<Value> toYield;
    rewriter.setInsertionPoint(performConcurrentlyOp);
    for (ParallelInsertSliceOp op : performConcurrentlyOp.yieldingOps()) {
      toYield.push_back(rewriter.createOrFold<tensor::InsertSliceOp>(
          loc, op.source(), bvm.lookup(op.dest()), op.offsets(), op.sizes(),
          op.strides()));
    }

    // performConcurrentlyOp.yieldedValues come from above, not from bbArgs.
    // There is no rewriter method to make mergeBlocks update non-bbArgs.
    // Need to manually clone + bvm all uses that are now nested under forOp.
    SmallVector<Operation *> opsToReplace, cloned;
    for (Value toReplace : valuesToYield) {
      for (OpOperand &u : toReplace.getUses()) {
        Operation *op = u.getOwner();
        if (!forOp->isProperAncestor(op))
          continue;
        OpBuilder::InsertionGuard g(rewriter);
        rewriter.setInsertionPoint(op);
        Operation *cloned = rewriter.clone(*op, bvm);
        rewriter.replaceOp(op, cloned->getResults());
      }
    }

    // Insert terminator.
    rewriter.setInsertionPointToEnd(forOp.getBody());
    rewriter.create<scf::YieldOp>(loc, toYield);

    // Cleanup and replace.
    rewriter.eraseOp(performConcurrentlyOp);
    rewriter.replaceOp(inParallelOp, forOp.getResults());

    return success();
  }
};

struct InParallelToSequentialForPass
    : public InParallelToSequentialForBase<InParallelToSequentialForPass> {
  void runOnOperation() override;
};
} // namespace

void InParallelToSequentialForPass::runOnOperation() {
  FuncOp funcOp = getOperation();
  MLIRContext *context = funcOp.getContext();
  RewritePatternSet patterns(context);
  patterns.insert<InParallelOpToSCFRewriter>(context);
  (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
}

std::unique_ptr<OperationPass<FuncOp>>
mlir::linalg_ext::createInParallelToSequentialForPass() {
  return std::make_unique<InParallelToSequentialForPass>();
}