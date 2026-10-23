import bench.repairs.tritonbench_g.cross_entropy1.Kernels

/-! Historical expansion snapshot used only to prove complete-source preservation.
The authoritative source is the direct quotation in Kernels.lean, which does not
import this module. No execution or public source definition depends on this
legacy generator. -/

/-! External scalar declarations supply only ordinary ABI register types.
They emit no source operation and provide no numerical value. -/
open Lean VeriTile.Triton VeriTile.Triton.DSL
syntax "cross_entropy_legacy_source" "{" tritonStmt* "}" : term
macro_rules
  | `(cross_entropy_legacy_source { $stmts:tritonStmt* }) => do
    let body := stmts.toList
    let pinned := Inference.collectNatPinned body
    let intPinned := Inference.collectIntPinned body
    let regions := Inference.collectRegionDTypes body
    let pointers := Inference.collectPtrElems regions body
    let context : FPSourceContext := { regions := regions, legacyPointerCasts := Bool.false }
    let environment : Env := [("smoothing",.real,SInfo.scalar,some .fp32),
      ("lse_square_scale",.real,SInfo.scalar,some .fp32),
      ("total_classes",.int,SInfo.scalar,none)]
    let (_,code,_,_) ← expandStmts environment pinned intPinned regions pointers context [] [] body
    let (inputs,outputs) := Metadata.blockRegions body
    let inputs := inputs.toArray
    let outputs := outputs.toArray
    let result ← `(ComputeKernel.mk [$inputs,*] [$outputs,*] [$code,*])
    normalizePrecisionScopes none result.raw

namespace VeriTile.Bench.Repairs.CrossEntropy1.LegacySource
open VeriTile.Triton

/-- Complete selected FP32 forward source. Pointer bases are the actual
allocation-relative argument bindings. Widened integer arithmetic is retained
in this syntax and receives a separate fixed-width interpretation. -/
def forwardSyntax (p : Parameters) : ComputeKernel :=
  let loss_ptr := p.loss
  let lse_ptr := p.lse
  let logits_ptr := p.logits
  let labels_ptr := p.labels
  let HAS_SMOOTHING := p.hasSmoothing
  let SPLIT := p.split
  ComputeKernel.mk [p.logits,Region.cast p.labels] [p.loss,p.lse] (cross_entropy_legacy_source {
    tl.region logits_ptr = tl.float32, loss_ptr = tl.float32, lse_ptr = tl.float32
    row_idx = tl.program_id(0).to(tl.int64)
    col_block_idx = tl.program_id(1).to(tl.int64)
    logits_ptr = logits_ptr + $(p.logitsBase) + row_idx * ($(p.logitsStride)).to(tl.int64)
    col_offsets = col_block_idx * $(p.block) + tl.arange(0, $(p.block)).to(tl.int64)
    label_idx = tl.load(labels_ptr + $(p.labelsBase) + row_idx)
    logits = tl.load(logits_ptr + col_offsets, mask=col_offsets < $(p.cols), other=-float("inf")).to(tl.float32)
    max_logits = tl.max(logits, 0)
    if HAS_SMOOTHING {
      sum_logits = tl.sum(tl.where(col_offsets < $(p.cols), logits, 0.0), 0)
    }
    lse = tl.log(tl.sum(tl.exp(logits - max_logits), 0)) + max_logits
    tl.store(lse_ptr + $(p.lseBase) + col_block_idx * $(p.rows) + row_idx, lse)
    if label_idx == $((p.ignored : Int)) {
      loss = 0.0
    } else {
      label_idx -= $((p.classStart : Int))
      if (label_idx >= col_block_idx * $(p.block)) and
          (label_idx < min($(p.cols), (col_block_idx + $(1)) * $(p.block))) {
        logits_label = tl.load(logits_ptr + label_idx)
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
        loss += lse_square_scale * lse * lse
      }
    }
    tl.store(loss_ptr + $(p.lossBase) + col_block_idx * $(p.rows) + row_idx, loss)
  }).surfaceBody

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
  ComputeKernel.mk [p.upstream,p.logits,p.lse,Region.cast p.labels] [p.gradient] (cross_entropy_legacy_source {
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
    logits = tl.load(logits_ptr + col_offsets, mask=col_offsets < $(p.cols), other=-float("inf")).to(tl.float32)
    lse = tl.load(lse_ptr + $(p.lseBase) + row_idx)
    probs = tl.exp(logits - lse)
    probs += 2.0 * lse_square_scale * lse * probs
    label_idx = tl.where(is_ignored, $((p.classStart : Int)), label_idx) - $((p.classStart : Int))
    if HAS_SMOOTHING {
      smooth_negative = smoothing / total_classes
      probs = tl.where(col_offsets == label_idx, probs - (1 - smoothing), probs) - smooth_negative
    } else {
      probs = tl.where(col_offsets == label_idx, probs - 1.0, probs)
    }
    tl.store(dlogits_ptr + col_offsets, tl.where(is_ignored, 0.0, dloss * probs), mask=col_offsets < $(p.cols))
  }).surfaceBody

end VeriTile.Bench.Repairs.CrossEntropy1.LegacySource

namespace VeriTile.Bench.Repairs.CrossEntropy1.SourcePreservation
/-- The explicit parameter declarations emit no statements; the complete original source is retained. -/
theorem forward (p : Parameters) : forwardSyntax p = LegacySource.forwardSyntax p := by
  rfl
theorem backward (p : Parameters) : backwardSyntax p = LegacySource.backwardSyntax p := by
  rfl
#axiomsClean forward
#axiomsClean backward
end VeriTile.Bench.Repairs.CrossEntropy1.SourcePreservation
