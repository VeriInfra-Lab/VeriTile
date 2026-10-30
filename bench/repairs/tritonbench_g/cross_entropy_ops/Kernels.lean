import VeriTile.Triton.DSL.CallSignature
import bench.repairs.tritonbench_g.cross_entropy1.Kernels

namespace VeriTile.Bench.Repairs.CrossEntropyOps
open VeriTile.Triton

structure Parameters extends VeriTile.Bench.Repairs.CrossEntropy1.Parameters where
  zloss : RegionName
  zlossBase : Nat := 0
  logitScale : ℝ

/-! Complete original functions and existing column-index commutations.
Caller types, input values and allocation-relative pointer offsets are external
to the copied function bodies. -/

def forwardSignature : CallSignature where
  pointers := [("loss_ptr", .fp32), ("lse_ptr", .fp32), ("z_loss_ptr", .fp32), ("logits_ptr", .fp32), ("labels_ptr", .i64)]
  scalars := [("smoothing", .fp32), ("logit_scale", .fp32), ("lse_square_scale", .fp32), ("ignored_index", .integer), ("total_classes", .i64), ("class_start_idx", .i64), ("n_cols", .i64), ("n_rows", .i64), ("logits_row_stride", .i64)]
  constexpr := ["BLOCK_SIZE", "HAS_SMOOTHING", "SPLIT"]
  readPorts := ["logits_ptr", "labels_ptr"]
  writePorts := ["loss_ptr", "lse_ptr", "z_loss_ptr"]

def forwardSyntax (p : Parameters) : ComputeKernel :=
  let loss_ptr := p.loss
  let lse_ptr := p.lse
  let z_loss_ptr := p.zloss
  let logits_ptr := p.logits
  let labels_ptr := p.labels
  let BLOCK_SIZE := p.block
  let HAS_SMOOTHING := p.hasSmoothing
  let SPLIT := p.split
  with_triton_call forwardSignature call_args [
    ignored_index := p.ignored, total_classes := (p.totalClasses : Int), class_start_idx := p.classStart, n_cols := p.cols, n_rows := p.rows, logits_row_stride := p.logitsStride]
    pointer_offsets [loss_ptr := p.lossBase, lse_ptr := p.lseBase, z_loss_ptr := p.zlossBase, logits_ptr := p.logitsBase, labels_ptr := p.labelsBase] in triton {
    row_idx = tl.program_id(0).to(tl.int64)
    col_block_idx = tl.program_id(1).to(tl.int64)
    logits_ptr = logits_ptr + row_idx * logits_row_stride.to(tl.int64)
    col_offsets = col_block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    label_idx = tl.load(labels_ptr + row_idx)
    logits = tl.load(logits_ptr + col_offsets, mask=col_offsets < n_cols, other=-float("inf")).to(
        tl.float32
    ) * logit_scale
    logits = tl.where(col_offsets < n_cols, logits, -float("inf"))
    max_logits = tl.max(logits, 0)
    if HAS_SMOOTHING:
        sum_logits = tl.sum(tl.where(col_offsets < n_cols, logits, 0.0), 0)
    lse = tl.log(tl.sum(tl.exp(logits - max_logits), 0)) + max_logits
    tl.store(lse_ptr + col_block_idx * n_rows + row_idx, lse)
    if label_idx == ignored_index:
        loss = 0.0
        z_loss = 0.0
    else:
        label_idx -= class_start_idx
        if label_idx >= col_block_idx * BLOCK_SIZE and label_idx < min(
            n_cols, (col_block_idx + 1) * BLOCK_SIZE
        ):
            logits_label = tl.load(logits_ptr + label_idx) * logit_scale
            if HAS_SMOOTHING:
                loss = (
                    (lse if not SPLIT else 0.0)
                    - smoothing * sum_logits / total_classes
                    - (1 - smoothing) * logits_label
                )
            else:
                loss = (lse if not SPLIT else 0.0) - logits_label
        else:
            if HAS_SMOOTHING:
                loss = smoothing * ((lse if not SPLIT else 0.0) - sum_logits / total_classes)
            else:
                loss = 0.0
        if not SPLIT:
            z_loss = lse_square_scale * lse * lse
            loss += z_loss
        else:
            z_loss = 0.0
    tl.store(loss_ptr + col_block_idx * n_rows + row_idx, loss)
    if not SPLIT:
        tl.store(z_loss_ptr + col_block_idx * n_rows + row_idx, z_loss)
  }

def forwardIndexCommutedSyntax (p : Parameters) : ComputeKernel :=
  let loss_ptr := p.loss
  let lse_ptr := p.lse
  let z_loss_ptr := p.zloss
  let logits_ptr := p.logits
  let labels_ptr := p.labels
  let BLOCK_SIZE := p.block
  let HAS_SMOOTHING := p.hasSmoothing
  let SPLIT := p.split
  with_triton_call forwardSignature call_args [
    ignored_index := p.ignored, total_classes := (p.totalClasses : Int), class_start_idx := p.classStart, n_cols := p.cols, n_rows := p.rows, logits_row_stride := p.logitsStride]
    pointer_offsets [loss_ptr := p.lossBase, lse_ptr := p.lseBase, z_loss_ptr := p.zlossBase, logits_ptr := p.logitsBase, labels_ptr := p.labelsBase] in triton {
    row_idx = tl.program_id(0).to(tl.int64)
    col_block_idx = tl.program_id(1).to(tl.int64)
    logits_ptr = logits_ptr + row_idx * logits_row_stride.to(tl.int64)
    col_offsets = BLOCK_SIZE * col_block_idx + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    label_idx = tl.load(labels_ptr + row_idx)
    logits = tl.load(logits_ptr + col_offsets, mask=col_offsets < n_cols, other=-float("inf")).to(
        tl.float32
    ) * logit_scale
    logits = tl.where(col_offsets < n_cols, logits, -float("inf"))
    max_logits = tl.max(logits, 0)
    if HAS_SMOOTHING:
        sum_logits = tl.sum(tl.where(col_offsets < n_cols, logits, 0.0), 0)
    lse = tl.log(tl.sum(tl.exp(logits - max_logits), 0)) + max_logits
    tl.store(lse_ptr + col_block_idx * n_rows + row_idx, lse)
    if label_idx == ignored_index:
        loss = 0.0
        z_loss = 0.0
    else:
        label_idx -= class_start_idx
        if label_idx >= col_block_idx * BLOCK_SIZE and label_idx < min(
            n_cols, (col_block_idx + 1) * BLOCK_SIZE
        ):
            logits_label = tl.load(logits_ptr + label_idx) * logit_scale
            if HAS_SMOOTHING:
                loss = (
                    (lse if not SPLIT else 0.0)
                    - smoothing * sum_logits / total_classes
                    - (1 - smoothing) * logits_label
                )
            else:
                loss = (lse if not SPLIT else 0.0) - logits_label
        else:
            if HAS_SMOOTHING:
                loss = smoothing * ((lse if not SPLIT else 0.0) - sum_logits / total_classes)
            else:
                loss = 0.0
        if not SPLIT:
            z_loss = lse_square_scale * lse * lse
            loss += z_loss
        else:
            z_loss = 0.0
    tl.store(loss_ptr + col_block_idx * n_rows + row_idx, loss)
    if not SPLIT:
        tl.store(z_loss_ptr + col_block_idx * n_rows + row_idx, z_loss)
  }

def backwardSignature : CallSignature where
  pointers := [("dlogits_ptr", .fp32), ("dloss_ptr", .fp32), ("logits_ptr", .fp32), ("lse_ptr", .fp32), ("labels_ptr", .i64)]
  scalars := [("smoothing", .fp32), ("logit_scale", .fp32), ("lse_square_scale", .fp32), ("ignored_index", .integer), ("total_classes", .i64), ("class_start_idx", .i64), ("n_cols", .i64), ("logits_row_stride", .i64), ("dlogits_row_stride", .i64), ("dloss_row_stride", .i64)]
  constexpr := ["BLOCK_SIZE", "HAS_SMOOTHING"]
  readPorts := ["dloss_ptr", "logits_ptr", "lse_ptr", "labels_ptr"]
  writePorts := ["dlogits_ptr"]

def backwardSyntax (p : Parameters) : ComputeKernel :=
  let dlogits_ptr := p.gradient
  let dloss_ptr := p.upstream
  let logits_ptr := p.logits
  let lse_ptr := p.lse
  let labels_ptr := p.labels
  let BLOCK_SIZE := p.block
  let HAS_SMOOTHING := p.hasSmoothing
  with_triton_call backwardSignature call_args [
    ignored_index := p.ignored, total_classes := (p.totalClasses : Int), class_start_idx := p.classStart, n_cols := p.cols, logits_row_stride := p.logitsStride, dlogits_row_stride := p.gradientStride, dloss_row_stride := p.upstreamStride]
    pointer_offsets [dlogits_ptr := p.gradientBase, dloss_ptr := p.upstreamBase, logits_ptr := p.logitsBase, lse_ptr := p.lseBase, labels_ptr := p.labelsBase] in triton {
    row_idx = tl.program_id(0).to(tl.int64)
    col_block_idx = tl.program_id(1).to(tl.int64)
    logits_ptr = logits_ptr + row_idx * logits_row_stride.to(tl.int64)
    dlogits_ptr = dlogits_ptr + row_idx * dlogits_row_stride.to(tl.int64)
    col_offsets = col_block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    label_idx = tl.load(labels_ptr + row_idx)
    is_ignored = label_idx == ignored_index
    if not is_ignored:
        dloss = tl.load(dloss_ptr + row_idx * dloss_row_stride)
    else:
        dloss = 0.0
    logits = tl.load(logits_ptr + col_offsets, mask=col_offsets < n_cols, other=-float("inf")).to(
        tl.float32
    ) * logit_scale
    logits = tl.where(col_offsets < n_cols, logits, -float("inf"))
    lse = tl.load(lse_ptr + row_idx)
    probs = tl.exp(logits - lse)
    probs += 2.0 * lse_square_scale * lse * probs
    label_idx = tl.where(is_ignored, class_start_idx, label_idx) - class_start_idx
    if HAS_SMOOTHING:
        smooth_positive = 1.0 - smoothing
        smooth_negative = smoothing / total_classes
        probs = tl.where(col_offsets == label_idx, probs - (1 - smoothing), probs) - smooth_negative
    else:
        probs = tl.where(col_offsets == label_idx, probs - 1.0, probs)
    tl.store(dlogits_ptr + col_offsets, tl.where(is_ignored, 0.0, (dloss * logit_scale) * probs), mask=col_offsets < n_cols)
  }

def backwardIndexCommutedSyntax (p : Parameters) : ComputeKernel :=
  let dlogits_ptr := p.gradient
  let dloss_ptr := p.upstream
  let logits_ptr := p.logits
  let lse_ptr := p.lse
  let labels_ptr := p.labels
  let BLOCK_SIZE := p.block
  let HAS_SMOOTHING := p.hasSmoothing
  with_triton_call backwardSignature call_args [
    ignored_index := p.ignored, total_classes := (p.totalClasses : Int), class_start_idx := p.classStart, n_cols := p.cols, logits_row_stride := p.logitsStride, dlogits_row_stride := p.gradientStride, dloss_row_stride := p.upstreamStride]
    pointer_offsets [dlogits_ptr := p.gradientBase, dloss_ptr := p.upstreamBase, logits_ptr := p.logitsBase, lse_ptr := p.lseBase, labels_ptr := p.labelsBase] in triton {
    row_idx = tl.program_id(0).to(tl.int64)
    col_block_idx = tl.program_id(1).to(tl.int64)
    logits_ptr = logits_ptr + row_idx * logits_row_stride.to(tl.int64)
    dlogits_ptr = dlogits_ptr + row_idx * dlogits_row_stride.to(tl.int64)
    col_offsets = BLOCK_SIZE * col_block_idx + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    label_idx = tl.load(labels_ptr + row_idx)
    is_ignored = label_idx == ignored_index
    if not is_ignored:
        dloss = tl.load(dloss_ptr + row_idx * dloss_row_stride)
    else:
        dloss = 0.0
    logits = tl.load(logits_ptr + col_offsets, mask=col_offsets < n_cols, other=-float("inf")).to(
        tl.float32
    ) * logit_scale
    logits = tl.where(col_offsets < n_cols, logits, -float("inf"))
    lse = tl.load(lse_ptr + row_idx)
    probs = tl.exp(logits - lse)
    probs += 2.0 * lse_square_scale * lse * probs
    label_idx = tl.where(is_ignored, class_start_idx, label_idx) - class_start_idx
    if HAS_SMOOTHING:
        smooth_positive = 1.0 - smoothing
        smooth_negative = smoothing / total_classes
        probs = tl.where(col_offsets == label_idx, probs - (1 - smoothing), probs) - smooth_negative
    else:
        probs = tl.where(col_offsets == label_idx, probs - 1.0, probs)
    tl.store(dlogits_ptr + col_offsets, tl.where(is_ignored, 0.0, (dloss * logit_scale) * probs), mask=col_offsets < n_cols)
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
