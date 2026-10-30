import bench.repairs.tritonbench_g.cross_entropy1.Kernels

namespace VeriTile.Bench.Repairs.CrossEntropyOps
open VeriTile.Triton

structure Parameters extends VeriTile.Bench.Repairs.CrossEntropy1.Parameters where
  zloss : RegionName
  zlossBase : Nat := 0
  logitScale : ℝ

/-- Complete selected FP32 forward source. Pointer bases are the actual
allocation-relative argument bindings. Widened integer arithmetic is retained
in this syntax and receives a separate fixed-width interpretation. -/
def forwardSyntax (p : Parameters) : ComputeKernel :=
  let z_loss_ptr := p.zloss
  let loss_ptr := p.loss
  let lse_ptr := p.lse
  let logits_ptr := p.logits
  let labels_ptr := p.labels
  let HAS_SMOOTHING := p.hasSmoothing
  let SPLIT := p.split
  triton {
    tl.ports inputs=[p.logits,Region.cast p.labels], outputs=[p.loss,p.lse,p.zloss]
    tl.param smoothing : tl.float32
    tl.param logit_scale : tl.float32
    tl.param lse_square_scale : tl.float32
    tl.param total_classes : tl.int64
    tl.region logits_ptr = tl.float32, loss_ptr = tl.float32, lse_ptr = tl.float32, z_loss_ptr = tl.float32
    row_idx = tl.program_id(0).to(tl.int64)
    col_block_idx = tl.program_id(1).to(tl.int64)
    logits_ptr = logits_ptr + $(p.logitsBase) + row_idx * ($(p.logitsStride)).to(tl.int64)
    col_offsets = col_block_idx * $(p.block) + tl.arange(0, $(p.block)).to(tl.int64)
    label_idx = tl.load(labels_ptr + $(p.labelsBase) + row_idx)
    logits = tl.load(logits_ptr + col_offsets, mask=col_offsets < $(p.cols), other=-float("inf")).to(tl.float32) * logit_scale
    logits = tl.where(col_offsets < $(p.cols), logits, -float("inf"))
    max_logits = tl.max(logits, 0)
    if HAS_SMOOTHING {
      sum_logits = tl.sum(tl.where(col_offsets < $(p.cols), logits, 0.0), 0)
    }
    lse = tl.log(tl.sum(tl.exp(logits - max_logits), 0)) + max_logits
    tl.store(lse_ptr + $(p.lseBase) + col_block_idx * $(p.rows) + row_idx, lse)
    if label_idx == $((p.ignored : Int)) {
      loss = 0.0
      z_loss = 0.0
    } else {
      label_idx -= $((p.classStart : Int))
      if (label_idx >= col_block_idx * $(p.block)) and
          (label_idx < min($(p.cols), (col_block_idx + $(1)) * $(p.block))) {
        logits_label = tl.load(logits_ptr + label_idx) * logit_scale
        if HAS_SMOOTHING {
          loss = (lse if not SPLIT else 0.0) -
            smoothing * sum_logits / total_classes -
            (1 - smoothing) * logits_label
        } else {
          loss = (lse if not SPLIT else 0.0) - logits_label
        }
      } else {
        if HAS_SMOOTHING {
          loss = smoothing *
            ((lse if not SPLIT else 0.0) - sum_logits / total_classes)
        } else {
          loss = 0.0
        }
      }
      if not SPLIT {
        z_loss = lse_square_scale * lse * lse
        loss += z_loss
      } else {
        z_loss = 0.0
      }
    }
    tl.store(loss_ptr + $(p.lossBase) + col_block_idx * $(p.rows) + row_idx, loss)
    if not SPLIT {
      tl.store(z_loss_ptr + $(p.zlossBase) + col_block_idx * $(p.rows) + row_idx, z_loss)
    }
  }

/-- Complete selected FP32 backward source, with explicit ignored-row zero
selection and safe ignored-label shift. All exp/correction/smoothing operations
remain in their actual source order. -/
def backwardSyntax (p : Parameters) : ComputeKernel :=
  let dlogits_ptr := p.gradient
  let dloss_ptr := p.upstream
  let logits_ptr := p.logits
  let lse_ptr := p.lse
  let labels_ptr := p.labels
  let HAS_SMOOTHING := p.hasSmoothing
  triton {
    tl.ports inputs=[p.upstream,p.logits,p.lse,Region.cast p.labels], outputs=[p.gradient]
    tl.param smoothing : tl.float32
    tl.param logit_scale : tl.float32
    tl.param lse_square_scale : tl.float32
    tl.param total_classes : tl.int64
    tl.region logits_ptr = tl.float32, dlogits_ptr = tl.float32,
      dloss_ptr = tl.float32, lse_ptr = tl.float32
    row_idx = tl.program_id(0).to(tl.int64)
    col_block_idx = tl.program_id(1).to(tl.int64)
    logits_ptr = logits_ptr + $(p.logitsBase) + row_idx * ($(p.logitsStride)).to(tl.int64)
    dlogits_ptr = dlogits_ptr + $(p.gradientBase) + row_idx * ($(p.gradientStride)).to(tl.int64)
    col_offsets = col_block_idx * $(p.block) + tl.arange(0, $(p.block)).to(tl.int64)
    label_idx = tl.load(labels_ptr + $(p.labelsBase) + row_idx)
    is_ignored = label_idx == $((p.ignored : Int))
    if not is_ignored {
      dloss = tl.load(dloss_ptr + $(p.upstreamBase) + row_idx * $(p.upstreamStride))
    } else {
      dloss = 0.0
    }
    logits = tl.load(logits_ptr + col_offsets, mask=col_offsets < $(p.cols), other=-float("inf")).to(tl.float32) * logit_scale
    logits = tl.where(col_offsets < $(p.cols), logits, -float("inf"))
    lse = tl.load(lse_ptr + $(p.lseBase) + row_idx)
    probs = tl.exp(logits - lse)
    probs += 2.0 * lse_square_scale * lse * probs
    label_idx = tl.where(is_ignored, $((p.classStart : Int)), label_idx) - $((p.classStart : Int))
    if HAS_SMOOTHING {
      smooth_positive = 1.0 - smoothing
      smooth_negative = smoothing / total_classes
      probs = tl.where(col_offsets == label_idx, probs - (1 - smoothing), probs) - smooth_negative
    } else {
      probs = tl.where(col_offsets == label_idx, probs - 1.0, probs)
    }
    tl.store(dlogits_ptr + col_offsets, tl.where(is_ignored, 0.0, (dloss * logit_scale) * probs), mask=col_offsets < $(p.cols))
  }

/-! Existing exact column-index candidates on the full selected runtime ABI. -/
/-- Complete selected FP32 forward source. Pointer bases are the actual
allocation-relative argument bindings. Widened integer arithmetic is retained
in this syntax and receives a separate fixed-width interpretation. -/
def forwardIndexCommutedSyntax (p : Parameters) : ComputeKernel :=
  let z_loss_ptr := p.zloss
  let loss_ptr := p.loss
  let lse_ptr := p.lse
  let logits_ptr := p.logits
  let labels_ptr := p.labels
  let HAS_SMOOTHING := p.hasSmoothing
  let SPLIT := p.split
  triton {
    tl.ports inputs=[p.logits,Region.cast p.labels], outputs=[p.loss,p.lse,p.zloss]
    tl.param smoothing : tl.float32
    tl.param logit_scale : tl.float32
    tl.param lse_square_scale : tl.float32
    tl.param total_classes : tl.int64
    tl.region logits_ptr = tl.float32, loss_ptr = tl.float32, lse_ptr = tl.float32, z_loss_ptr = tl.float32
    row_idx = tl.program_id(0).to(tl.int64)
    col_block_idx = tl.program_id(1).to(tl.int64)
    logits_ptr = logits_ptr + $(p.logitsBase) + row_idx * ($(p.logitsStride)).to(tl.int64)
    col_offsets = $(p.block) * col_block_idx + tl.arange(0, $(p.block)).to(tl.int64)
    label_idx = tl.load(labels_ptr + $(p.labelsBase) + row_idx)
    logits = tl.load(logits_ptr + col_offsets, mask=col_offsets < $(p.cols), other=-float("inf")).to(tl.float32) * logit_scale
    logits = tl.where(col_offsets < $(p.cols), logits, -float("inf"))
    max_logits = tl.max(logits, 0)
    if HAS_SMOOTHING {
      sum_logits = tl.sum(tl.where(col_offsets < $(p.cols), logits, 0.0), 0)
    }
    lse = tl.log(tl.sum(tl.exp(logits - max_logits), 0)) + max_logits
    tl.store(lse_ptr + $(p.lseBase) + col_block_idx * $(p.rows) + row_idx, lse)
    if label_idx == $((p.ignored : Int)) {
      loss = 0.0
      z_loss = 0.0
    } else {
      label_idx -= $((p.classStart : Int))
      if (label_idx >= col_block_idx * $(p.block)) and
          (label_idx < min($(p.cols), (col_block_idx + $(1)) * $(p.block))) {
        logits_label = tl.load(logits_ptr + label_idx) * logit_scale
        if HAS_SMOOTHING {
          loss = (lse if not SPLIT else 0.0) -
            smoothing * sum_logits / total_classes -
            (1 - smoothing) * logits_label
        } else {
          loss = (lse if not SPLIT else 0.0) - logits_label
        }
      } else {
        if HAS_SMOOTHING {
          loss = smoothing *
            ((lse if not SPLIT else 0.0) - sum_logits / total_classes)
        } else {
          loss = 0.0
        }
      }
      if not SPLIT {
        z_loss = lse_square_scale * lse * lse
        loss += z_loss
      } else {
        z_loss = 0.0
      }
    }
    tl.store(loss_ptr + $(p.lossBase) + col_block_idx * $(p.rows) + row_idx, loss)
    if not SPLIT {
      tl.store(z_loss_ptr + $(p.zlossBase) + col_block_idx * $(p.rows) + row_idx, z_loss)
    }
  }

/-- Complete selected FP32 backward source, with explicit ignored-row zero
selection and safe ignored-label shift. All exp/correction/smoothing operations
remain in their actual source order. -/
def backwardIndexCommutedSyntax (p : Parameters) : ComputeKernel :=
  let dlogits_ptr := p.gradient
  let dloss_ptr := p.upstream
  let logits_ptr := p.logits
  let lse_ptr := p.lse
  let labels_ptr := p.labels
  let HAS_SMOOTHING := p.hasSmoothing
  triton {
    tl.ports inputs=[p.upstream,p.logits,p.lse,Region.cast p.labels], outputs=[p.gradient]
    tl.param smoothing : tl.float32
    tl.param logit_scale : tl.float32
    tl.param lse_square_scale : tl.float32
    tl.param total_classes : tl.int64
    tl.region logits_ptr = tl.float32, dlogits_ptr = tl.float32,
      dloss_ptr = tl.float32, lse_ptr = tl.float32
    row_idx = tl.program_id(0).to(tl.int64)
    col_block_idx = tl.program_id(1).to(tl.int64)
    logits_ptr = logits_ptr + $(p.logitsBase) + row_idx * ($(p.logitsStride)).to(tl.int64)
    dlogits_ptr = dlogits_ptr + $(p.gradientBase) + row_idx * ($(p.gradientStride)).to(tl.int64)
    col_offsets = $(p.block) * col_block_idx + tl.arange(0, $(p.block)).to(tl.int64)
    label_idx = tl.load(labels_ptr + $(p.labelsBase) + row_idx)
    is_ignored = label_idx == $((p.ignored : Int))
    if not is_ignored {
      dloss = tl.load(dloss_ptr + $(p.upstreamBase) + row_idx * $(p.upstreamStride))
    } else {
      dloss = 0.0
    }
    logits = tl.load(logits_ptr + col_offsets, mask=col_offsets < $(p.cols), other=-float("inf")).to(tl.float32) * logit_scale
    logits = tl.where(col_offsets < $(p.cols), logits, -float("inf"))
    lse = tl.load(lse_ptr + $(p.lseBase) + row_idx)
    probs = tl.exp(logits - lse)
    probs += 2.0 * lse_square_scale * lse * probs
    label_idx = tl.where(is_ignored, $((p.classStart : Int)), label_idx) - $((p.classStart : Int))
    if HAS_SMOOTHING {
      smooth_positive = 1.0 - smoothing
      smooth_negative = smoothing / total_classes
      probs = tl.where(col_offsets == label_idx, probs - (1 - smoothing), probs) - smooth_negative
    } else {
      probs = tl.where(col_offsets == label_idx, probs - 1.0, probs)
    }
    tl.store(dlogits_ptr + col_offsets, tl.where(is_ignored, 0.0, (dloss * logit_scale) * probs), mask=col_offsets < $(p.cols))
  }

namespace RuntimeArgs
open FP.Structural
structure Values (α : Type) extends CrossEntropy1.RuntimeArgs.Values α where
  logitScale : α
structure Bound {α : Type} (p : Parameters) (initial : State α) (values : Values α) : Prop
    extends CrossEntropy1.RuntimeArgs.Bound p.toParameters initial values.toValues where
  logitScale : initial.regs .real [] "logit_scale"=some (fun _ => values.logitScale)
noncomputable def fp {α : Type} [Inhabited α] (initial : State α) (name : RegName) : α :=
  ((initial.regs .real [] name).getD (fun _ => default)) PUnit.unit
noncomputable def classes {α : Type} (initial : State α) : Int :=
  ((initial.regs .int [] "total_classes").getD (fun _ => 0)) PUnit.unit
end RuntimeArgs
end VeriTile.Bench.Repairs.CrossEntropyOps
