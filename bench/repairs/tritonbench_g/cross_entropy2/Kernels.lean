import VeriTile.Triton.DSL.CallSignature
import bench.repairs.tritonbench_g.cross_entropy1.Kernels

namespace VeriTile.Bench.Repairs.CrossEntropy2
open VeriTile.Triton

structure Parameters extends VeriTile.Bench.Repairs.CrossEntropy1.Parameters where
  zLoss : RegionName
  zLossBase : Nat := 0
  logitScale : ℝ

/-! Complete original functions and existing column-index commutations.
Caller types, input values and allocation-relative pointer offsets are external
to the copied function bodies. -/

def forwardSignature : CallSignature where
  pointers := [("loss_ptr", .fp32), ("lse_ptr", .fp32), ("z_loss_ptr", .fp32), ("logits_ptr", .fp32), ("labels_ptr", .i32)]
  scalars := [("smoothing", .fp32), ("logit_scale", .fp32), ("lse_square_scale", .fp32), ("ignored_index", .i64), ("total_classes", .i64), ("class_start_idx", .i64), ("n_cols", .i64), ("n_rows", .i64), ("logits_row_stride", .i64)]
  constexpr := ["BLOCK_SIZE", "HAS_SMOOTHING", "SPLIT"]
  readPorts := ["logits_ptr", "labels_ptr"]
  writePorts := ["loss_ptr", "lse_ptr", "z_loss_ptr"]

def forwardSyntax (p : Parameters) : ComputeKernel :=
  let loss_ptr := p.loss
  let lse_ptr := p.lse
  let z_loss_ptr := p.zLoss
  let logits_ptr := p.logits
  let labels_ptr := p.labels
  let BLOCK_SIZE := p.block
  let HAS_SMOOTHING := p.hasSmoothing
  let SPLIT := p.split
  with_triton_call forwardSignature call_args [
    ignored_index := p.ignored, total_classes := (p.totalClasses : Int), class_start_idx := p.classStart, n_cols := p.cols, n_rows := p.rows, logits_row_stride := p.logitsStride]
    pointer_offsets [loss_ptr := p.lossBase, lse_ptr := p.lseBase, z_loss_ptr := p.zLossBase, logits_ptr := p.logitsBase, labels_ptr := p.labelsBase] in triton {
    row_idx = tl.program_id(0).to(tl.int64)
    col_block_idx = tl.program_id(1).to(tl.int64)
    logits_ptr = logits_ptr + row_idx * logits_row_stride.to(tl.int64)
    col_offsets = col_block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    label_idx = tl.load(labels_ptr + row_idx).to(tl.int64)
    owns_target = (label_idx >= class_start_idx) & (label_idx < class_start_idx + n_cols)
    logits = tl.load(logits_ptr + col_offsets, mask=col_offsets < n_cols, other=0.0).to(
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
        label_idx = tl.where(owns_target, label_idx, class_start_idx) - class_start_idx
        if owns_target and label_idx >= col_block_idx * BLOCK_SIZE and label_idx < min(
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
                if not SPLIT:
                    loss += (1 - smoothing) * lse
            else:
                loss = lse if not SPLIT else 0.0
        if not SPLIT:
            z_loss = lse_square_scale * lse * lse
            loss += z_loss
        else:
            z_loss = 0.0
    tl.store(loss_ptr + col_block_idx * n_rows + row_idx, loss)
    tl.store(z_loss_ptr + col_block_idx * n_rows + row_idx, z_loss)
  }

def forwardIndexCommutedSyntax (p : Parameters) : ComputeKernel :=
  let loss_ptr := p.loss
  let lse_ptr := p.lse
  let z_loss_ptr := p.zLoss
  let logits_ptr := p.logits
  let labels_ptr := p.labels
  let BLOCK_SIZE := p.block
  let HAS_SMOOTHING := p.hasSmoothing
  let SPLIT := p.split
  with_triton_call forwardSignature call_args [
    ignored_index := p.ignored, total_classes := (p.totalClasses : Int), class_start_idx := p.classStart, n_cols := p.cols, n_rows := p.rows, logits_row_stride := p.logitsStride]
    pointer_offsets [loss_ptr := p.lossBase, lse_ptr := p.lseBase, z_loss_ptr := p.zLossBase, logits_ptr := p.logitsBase, labels_ptr := p.labelsBase] in triton {
    row_idx = tl.program_id(0).to(tl.int64)
    col_block_idx = tl.program_id(1).to(tl.int64)
    logits_ptr = logits_ptr + row_idx * logits_row_stride.to(tl.int64)
    col_offsets = BLOCK_SIZE * col_block_idx + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    label_idx = tl.load(labels_ptr + row_idx).to(tl.int64)
    owns_target = (label_idx >= class_start_idx) & (label_idx < class_start_idx + n_cols)
    logits = tl.load(logits_ptr + col_offsets, mask=col_offsets < n_cols, other=0.0).to(
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
        label_idx = tl.where(owns_target, label_idx, class_start_idx) - class_start_idx
        if owns_target and label_idx >= col_block_idx * BLOCK_SIZE and label_idx < min(
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
                if not SPLIT:
                    loss += (1 - smoothing) * lse
            else:
                loss = lse if not SPLIT else 0.0
        if not SPLIT:
            z_loss = lse_square_scale * lse * lse
            loss += z_loss
        else:
            z_loss = 0.0
    tl.store(loss_ptr + col_block_idx * n_rows + row_idx, loss)
    tl.store(z_loss_ptr + col_block_idx * n_rows + row_idx, z_loss)
  }

def backwardSignature : CallSignature where
  pointers := [("dlogits_ptr", .fp32), ("dloss_ptr", .fp32), ("logits_ptr", .fp32), ("lse_ptr", .fp32), ("labels_ptr", .i32)]
  scalars := [("smoothing", .fp32), ("logit_scale", .fp32), ("lse_square_scale", .fp32), ("ignored_index", .i64), ("total_classes", .i64), ("class_start_idx", .i64), ("n_cols", .i64), ("logits_row_stride", .i64), ("dlogits_row_stride", .i64), ("dloss_row_stride", .i64)]
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
    label_idx = tl.load(labels_ptr + row_idx).to(tl.int64)
    owns_target = (label_idx >= class_start_idx) & (label_idx < class_start_idx + n_cols)
    is_ignored = label_idx == ignored_index
    if not is_ignored:
        dloss = tl.load(dloss_ptr + row_idx * dloss_row_stride)
    else:
        dloss = 0.0
    logits = tl.load(logits_ptr + col_offsets, mask=col_offsets < n_cols, other=0.0).to(
        tl.float32
    ) * logit_scale
    logits = tl.where(col_offsets < n_cols, logits, -float("inf"))
    lse = tl.load(lse_ptr + row_idx)
    probs = tl.exp(logits - lse)
    probs += 2.0 * lse_square_scale * lse * probs
    label_idx = tl.where(owns_target, label_idx, class_start_idx) - class_start_idx
    if HAS_SMOOTHING:
        smooth_negative = smoothing / total_classes
        probs = tl.where(owns_target & (col_offsets == label_idx), probs - (1 - smoothing), probs) - smooth_negative
    else:
        probs = tl.where(owns_target & (col_offsets == label_idx), probs - 1.0, probs)
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
    label_idx = tl.load(labels_ptr + row_idx).to(tl.int64)
    owns_target = (label_idx >= class_start_idx) & (label_idx < class_start_idx + n_cols)
    is_ignored = label_idx == ignored_index
    if not is_ignored:
        dloss = tl.load(dloss_ptr + row_idx * dloss_row_stride)
    else:
        dloss = 0.0
    logits = tl.load(logits_ptr + col_offsets, mask=col_offsets < n_cols, other=0.0).to(
        tl.float32
    ) * logit_scale
    logits = tl.where(col_offsets < n_cols, logits, -float("inf"))
    lse = tl.load(lse_ptr + row_idx)
    probs = tl.exp(logits - lse)
    probs += 2.0 * lse_square_scale * lse * probs
    label_idx = tl.where(owns_target, label_idx, class_start_idx) - class_start_idx
    if HAS_SMOOTHING:
        smooth_negative = smoothing / total_classes
        probs = tl.where(owns_target & (col_offsets == label_idx), probs - (1 - smoothing), probs) - smooth_negative
    else:
        probs = tl.where(owns_target & (col_offsets == label_idx), probs - 1.0, probs)
    tl.store(dlogits_ptr + col_offsets, tl.where(is_ignored, 0.0, (dloss * logit_scale) * probs), mask=col_offsets < n_cols)
  }

namespace RuntimeArgs
open FP.Structural
structure Values (α : Type) extends CrossEntropy1.RuntimeArgs.Values α where
  logitScale : α
structure Bound {α : Type} (p : Parameters) (initial : State α) (values : Values α) : Prop
    extends CrossEntropy1.RuntimeArgs.Bound p.toParameters initial values.toValues where
  logitScale : initial.regs .real [] "logit_scale"=some (fun _ => values.logitScale)
noncomputable abbrev fp := @CrossEntropy1.RuntimeArgs.fp
noncomputable abbrev classes := @CrossEntropy1.RuntimeArgs.classes
end RuntimeArgs

namespace Source
open VeriTile.Bench.Repairs.CrossEntropy1.Source
noncomputable section
abbrev FP32 := VeriTile.Bench.Repairs.CrossEntropy1.Source.FP32
abbrev Memory := VeriTile.Bench.Repairs.CrossEntropy1.Source.Memory
abbrev Cell := VeriTile.Bench.Repairs.CrossEntropy1.Source.Cell
abbrev Binary32 := VeriTile.Bench.Repairs.CrossEntropy1.Source.Binary32
abbrev Unary32 := VeriTile.Bench.Repairs.CrossEntropy1.Source.Unary32
abbrev write := @VeriTile.Bench.Repairs.CrossEntropy1.Source.write
abbrev load := @VeriTile.Bench.Repairs.CrossEntropy1.Source.load
abbrev fp := @VeriTile.Bench.Repairs.CrossEntropy1.Source.fp
abbrev nat := @VeriTile.Bench.Repairs.CrossEntropy1.Source.nat
abbrev binary := @VeriTile.Bench.Repairs.CrossEntropy1.Source.binary
abbrev unary := @VeriTile.Bench.Repairs.CrossEntropy1.Source.unary

/-- Only the logit load port is scaled. Mask padding remains negative infinity
for every scale, including zero and negative values. -/
def scaled {α : Type} (M : FP32 α) (a : ℝ) : FP32 α :=
  {M with load := fun x => M.mul (M.load x) (M.literal a)}

def values {α : Type} [Inhabited α] (M : FP32 α) (p : Parameters)
    (memory : Memory α) (row block : Nat) :=
  VeriTile.Bench.Repairs.CrossEntropy1.Source.forwardValues (scaled M p.logitScale) p.toParameters memory row block

def owns (p : Parameters) (target : Int) : Prop :=
  p.classStart≤target ∧ target<p.classStart+(p.cols : Int)

def blockOwns (p : Parameters) (target : Int) (block : Nat) : Prop :=
  (block*p.block : Int)≤target-p.classStart ∧
    target-p.classStart<(min p.cols ((block+1)*p.block) : Nat)

instance (p : Parameters) (target : Int) (block : Nat) : Decidable (blockOwns p target block) := by
  unfold blockOwns
  infer_instance

/-- The two returned scalar registers follow the original branch expression
order. The non-owning unsplit branch now includes the missing normalizer. -/
def lossZ {α : Type} [Inhabited α] (M : FP32 α) (p : Parameters)
    (memory : Memory α) (row block : Nat) (target : Int)
    (v : VeriTile.Bench.Repairs.CrossEntropy1.Source.ForwardValues α p.block) : α × α :=
  if target=p.ignored then (fp M 0,fp M 0) else
  let lseTerm := if p.split then fp M 0 else v.lse
  let smooth := fp M p.smoothing
  let sums := v.sumLogits.getD (fp M 0)
  let loss := if blockOwns p target block then
    let selected := binary M .mul
      (load M memory p.logits (VeriTile.Bench.Repairs.CrossEntropy1.Source.logitAddress p.toParameters row (target-p.classStart).toNat))
      (fp M p.logitScale)
    if p.hasSmoothing then
      binary M .sub (binary M .sub lseTerm (binary M .div (binary M .mul smooth sums) (nat M p.totalClasses)))
        (binary M .mul (binary M .sub (nat M 1) smooth) selected)
    else binary M .sub lseTerm selected
  else if p.hasSmoothing then
    let original := binary M .mul smooth (binary M .sub lseTerm (binary M .div sums (nat M p.totalClasses)))
    if p.split then original else
      binary M .add original (binary M .mul (binary M .sub (nat M 1) smooth) v.lse)
  else lseTerm
  if p.split then (loss,fp M 0) else
    let z := binary M .mul (binary M .mul (fp M p.squareScale) v.lse) v.lse
    (binary M .add loss z,z)

def forward {α : Type} [Inhabited α] (M : FP32 α) (p : Parameters)
    (memory : Memory α) (row block : Nat) : Memory α :=
  let target := VeriTile.Bench.Repairs.CrossEntropy1.Source.label p.toParameters memory row
  let v := values M p memory row block
  let afterLSE := write memory p.lse (p.lseBase+block*p.rows+row) (.mk .real v.lse)
  let pair := lossZ M p afterLSE row block target v
  let afterLoss := write afterLSE p.loss (p.lossBase+block*p.rows+row) (.mk .real pair.1)
  write afterLoss p.zLoss (p.zLossBase+block*p.rows+row) (.mk .real pair.2)

/-- All logit scaling, upstream scaling, exponential and LSE-square correction
operations retain their source order. Ignored outputs select a literal zero. -/
def backwardValues {α : Type} [Inhabited α] (M : FP32 α) (p : Parameters)
    (memory : Memory α) (row block : Nat) : Fin p.block → α :=
  let target := VeriTile.Bench.Repairs.CrossEntropy1.Source.label p.toParameters memory row
  let ignored := target=p.ignored
  let upstream := if ignored then fp M 0 else
    load M memory p.upstream (p.upstreamBase+row*p.upstreamStride)
  let logits := VeriTile.Bench.Repairs.CrossEntropy1.Source.loadedLogits (scaled M p.logitScale) p.toParameters memory row block
  let lse := load M memory p.lse (p.lseBase+row)
  fun lane =>
    let probability := unary M .exp (binary M .sub (logits lane) lse)
    let corrected := binary M .add probability
      (binary M .mul (binary M .mul (binary M .mul (fp M 2) (fp M p.squareScale)) lse) probability)
    let hit := (VeriTile.Bench.Repairs.CrossEntropy1.Source.column p.toParameters block lane.val : Int)=target-p.classStart
    let adjusted := if p.hasSmoothing then
      binary M .sub (if hit then binary M .sub corrected (binary M .sub (nat M 1) (fp M p.smoothing)) else corrected)
        (binary M .div (fp M p.smoothing) (nat M p.totalClasses))
      else if hit then binary M .sub corrected (fp M 1) else corrected
    if ignored then fp M 0 else binary M .mul (binary M .mul upstream (fp M p.logitScale)) adjusted

def backward {α : Type} [Inhabited α] (M : FP32 α) (p : Parameters)
    (memory : Memory α) (row block : Nat) : Memory α :=
  let vals := backwardValues M p memory row block
  (List.ofFn (fun lane : Fin p.block => lane)).foldl (fun current lane =>
    if VeriTile.Bench.Repairs.CrossEntropy1.Source.column p.toParameters block lane.val<p.cols then
      write current p.gradient (VeriTile.Bench.Repairs.CrossEntropy1.Source.gradientAddress p.toParameters row (VeriTile.Bench.Repairs.CrossEntropy1.Source.column p.toParameters block lane.val))
        (.mk .real (vals lane)) else current) memory

end
end Source
end VeriTile.Bench.Repairs.CrossEntropy2
