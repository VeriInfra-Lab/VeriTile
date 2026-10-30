import VeriTile.Triton.DSL.CallSignature
import VeriTile.Triton.DSL
import VeriTile.Triton.Source
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.CrossEntropy1
open VeriTile.Triton

/-- The source pointer arguments are represented as allocation region plus
element offset. Adding these bases binds actual Tensor.data_ptr(), including
nonzero storage offsets; it is not an extra floating source operation. -/
structure Parameters where
  loss : RegionName
  lse : RegionName
  logits : RegionName
  gradient : RegionName
  upstream : RegionName
  labels : Region .int
  lossBase : Nat := 0
  lseBase : Nat := 0
  logitsBase : Nat := 0
  gradientBase : Nat := 0
  upstreamBase : Nat := 0
  labelsBase : Nat := 0
  rows : Nat
  cols : Nat
  totalClasses : Nat
  classStart : Int
  ignored : Int
  logitsStride : Nat
  gradientStride : Nat
  upstreamStride : Nat
  block : Nat
  smoothing : ℝ
  squareScale : ℝ
  hasSmoothing : Bool
  split : Bool

/-! Complete original functions and existing column-index commutations.
Caller types, input values and allocation-relative pointer offsets are external
to the copied function bodies. -/

def forwardSignature : CallSignature where
  pointers := [("loss_ptr", .fp32), ("lse_ptr", .fp32), ("logits_ptr", .fp32), ("labels_ptr", .i64)]
  scalars := [("smoothing", .fp32), ("lse_square_scale", .fp32), ("ignored_index", .integer), ("total_classes", .i64), ("class_start_idx", .i64), ("n_cols", .i64), ("n_rows", .i64), ("logits_row_stride", .i64)]
  constexpr := ["BLOCK_SIZE", "HAS_SMOOTHING", "SPLIT"]
  readPorts := ["logits_ptr", "labels_ptr"]
  writePorts := ["loss_ptr", "lse_ptr"]

def forwardSyntax (p : Parameters) : ComputeKernel :=
  let loss_ptr := p.loss
  let lse_ptr := p.lse
  let logits_ptr := p.logits
  let labels_ptr := p.labels
  let BLOCK_SIZE := p.block
  let HAS_SMOOTHING := p.hasSmoothing
  let SPLIT := p.split
  with_triton_call forwardSignature call_args [
    ignored_index := p.ignored, total_classes := (p.totalClasses : Int), class_start_idx := p.classStart, n_cols := p.cols, n_rows := p.rows, logits_row_stride := p.logitsStride]
    pointer_offsets [loss_ptr := p.lossBase, lse_ptr := p.lseBase, logits_ptr := p.logitsBase, labels_ptr := p.labelsBase] in triton {
    row_idx = tl.program_id(0).to(tl.int64)
    col_block_idx = tl.program_id(1).to(tl.int64)
    logits_ptr = logits_ptr + row_idx * logits_row_stride.to(tl.int64)
    col_offsets = col_block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    label_idx = tl.load(labels_ptr + row_idx)
    logits = tl.load(logits_ptr + col_offsets, mask=col_offsets < n_cols, other=-float("inf")).to(tl.float32)
    max_logits = tl.max(logits, 0)
    if HAS_SMOOTHING:
        sum_logits = tl.sum(tl.where(col_offsets < n_cols, logits, 0.0), 0)
    lse = tl.log(tl.sum(tl.exp(logits - max_logits), 0)) + max_logits
    tl.store(lse_ptr + col_block_idx * n_rows + row_idx, lse)
    if label_idx == ignored_index:
        loss = 0.0
    else:
        label_idx -= class_start_idx
        if label_idx >= col_block_idx * BLOCK_SIZE and label_idx < min(n_cols, (col_block_idx + 1) * BLOCK_SIZE):
            logits_label = tl.load(logits_ptr + label_idx)
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
            loss += lse_square_scale * lse * lse
    tl.store(loss_ptr + col_block_idx * n_rows + row_idx, loss)
  }

def forwardIndexCommutedSyntax (p : Parameters) : ComputeKernel :=
  let loss_ptr := p.loss
  let lse_ptr := p.lse
  let logits_ptr := p.logits
  let labels_ptr := p.labels
  let BLOCK_SIZE := p.block
  let HAS_SMOOTHING := p.hasSmoothing
  let SPLIT := p.split
  with_triton_call forwardSignature call_args [
    ignored_index := p.ignored, total_classes := (p.totalClasses : Int), class_start_idx := p.classStart, n_cols := p.cols, n_rows := p.rows, logits_row_stride := p.logitsStride]
    pointer_offsets [loss_ptr := p.lossBase, lse_ptr := p.lseBase, logits_ptr := p.logitsBase, labels_ptr := p.labelsBase] in triton {
    row_idx = tl.program_id(0).to(tl.int64)
    col_block_idx = tl.program_id(1).to(tl.int64)
    logits_ptr = logits_ptr + row_idx * logits_row_stride.to(tl.int64)
    col_offsets = BLOCK_SIZE * col_block_idx + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    label_idx = tl.load(labels_ptr + row_idx)
    logits = tl.load(logits_ptr + col_offsets, mask=col_offsets < n_cols, other=-float("inf")).to(tl.float32)
    max_logits = tl.max(logits, 0)
    if HAS_SMOOTHING:
        sum_logits = tl.sum(tl.where(col_offsets < n_cols, logits, 0.0), 0)
    lse = tl.log(tl.sum(tl.exp(logits - max_logits), 0)) + max_logits
    tl.store(lse_ptr + col_block_idx * n_rows + row_idx, lse)
    if label_idx == ignored_index:
        loss = 0.0
    else:
        label_idx -= class_start_idx
        if label_idx >= col_block_idx * BLOCK_SIZE and label_idx < min(n_cols, (col_block_idx + 1) * BLOCK_SIZE):
            logits_label = tl.load(logits_ptr + label_idx)
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
            loss += lse_square_scale * lse * lse
    tl.store(loss_ptr + col_block_idx * n_rows + row_idx, loss)
  }

def backwardSignature : CallSignature where
  pointers := [("dlogits_ptr", .fp32), ("dloss_ptr", .fp32), ("logits_ptr", .fp32), ("lse_ptr", .fp32), ("labels_ptr", .i64)]
  scalars := [("smoothing", .fp32), ("lse_square_scale", .fp32), ("ignored_index", .integer), ("total_classes", .i64), ("class_start_idx", .i64), ("n_cols", .i64), ("logits_row_stride", .i64), ("dlogits_row_stride", .i64), ("dloss_row_stride", .i64)]
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
    logits = tl.load(logits_ptr + col_offsets, mask=col_offsets < n_cols, other=-float("inf")).to(tl.float32)
    lse = tl.load(lse_ptr + row_idx)
    probs = tl.exp(logits - lse)
    probs += 2.0 * lse_square_scale * lse * probs
    label_idx = tl.where(is_ignored, class_start_idx, label_idx) - class_start_idx
    if HAS_SMOOTHING:
        smooth_negative = smoothing / total_classes
        probs = tl.where(col_offsets == label_idx, probs - (1 - smoothing), probs) - smooth_negative
    else:
        probs = tl.where(col_offsets == label_idx, probs - 1.0, probs)
    tl.store(dlogits_ptr + col_offsets, tl.where(is_ignored, 0.0, dloss * probs), mask=col_offsets < n_cols)
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
    logits = tl.load(logits_ptr + col_offsets, mask=col_offsets < n_cols, other=-float("inf")).to(tl.float32)
    lse = tl.load(lse_ptr + row_idx)
    probs = tl.exp(logits - lse)
    probs += 2.0 * lse_square_scale * lse * probs
    label_idx = tl.where(is_ignored, class_start_idx, label_idx) - class_start_idx
    if HAS_SMOOTHING:
        smooth_negative = smoothing / total_classes
        probs = tl.where(col_offsets == label_idx, probs - (1 - smoothing), probs) - smooth_negative
    else:
        probs = tl.where(col_offsets == label_idx, probs - 1.0, probs)
    tl.store(dlogits_ptr + col_offsets, tl.where(is_ignored, 0.0, dloss * probs), mask=col_offsets < n_cols)
  }

namespace RuntimeArgs
open FP.Structural
structure Values (α : Type) where
  smoothing : α
  squareScale : α

/-- Ordinary ABI values retain FP32 words, including signed zeros. The
runtime int64 total-classes argument is bound before its floating conversion. -/
structure Bound {α : Type} (p : Parameters) (initial : State α) (values : Values α) : Prop where
  smoothing : initial.regs .real [] "smoothing"=some (fun _ => values.smoothing)
  squareScale : initial.regs .real [] "lse_square_scale"=some (fun _ => values.squareScale)
  totalClasses : initial.regs .int [] "total_classes"=some (fun _ => (p.totalClasses : Int))

noncomputable def fp {α : Type} [Inhabited α] (initial : State α) (name : RegName) : α :=
  ((initial.regs .real [] name).getD (fun _ => default)) PUnit.unit
noncomputable def classes {α : Type} (initial : State α) : Int :=
  ((initial.regs .int [] "total_classes").getD (fun _ => 0)) PUnit.unit
end RuntimeArgs

namespace Source
open FP.Structural
noncomputable section

inductive Binary32 where
  | add | sub | mul | div
inductive Unary32 where
  | exp | log

/-- Exact selected FP32 primitive interface. Reduction operands retain lane
order. A concrete word/expression interpretation binds loads and stores to
physical words; no field accepts a kernel, host call or final-output predicate. -/
structure FP32 (α : Type) where
  literal : ℝ → α
  fromNat : Nat → α
  negInf : α
  load : α → α
  add : α → α → α
  sub : α → α → α
  mul : α → α → α
  div : α → α → α
  exp : α → α
  log : α → α
  sum : {B : Nat} → (Fin B → α) → α
  maximum : {B : Nat} → (Fin B → α) → α

/-- A storage location can have several typed observations. In particular an
int64 snapshot may also be viewed as FP32 by a later caller. An exclusive
sum-tag cell cannot represent those legitimate overlapping views. The byte
refinement couples each active read channel to its actual element width;
kernel stores define the written channel and frame all untouched locations. -/
structure Cell (α : Type) where
  ofObservations ::
  observe : (dtype : TileDType) → FP.Structural.Value α dtype

namespace Cell

def read {α : Type} (dtype : TileDType) (cell : Cell α) : FP.Structural.Value α dtype := cell.observe dtype

def mk {α : Type} [Inhabited α] (dtype : TileDType) (value : FP.Structural.Value α dtype) : Cell α :=
  ⟨fun requested => if same : dtype=requested then same ▸ value else FP.Structural.defaultValue requested⟩

@[simp] theorem read_same {α : Type} [Inhabited α] (dtype : TileDType) (value : FP.Structural.Value α dtype) :
    (mk dtype value).read dtype=value := by simp [read,mk]

end Cell

/-- Task-local typed interpretation of the two straight-line JITs. Integer
channels retain exact Nat/Int carriers, FP32 cells retain the selected algebra,
and reductions retain their full ordered lane vectors. The syntax declarations
above are the source inventory; this interpreter exposes every source read,
arithmetic step, branch and store without an external execution/result oracle.
Register names are local bindings and do not cross kernel invocations. -/
abbrev Memory (α : Type) := RegionName → Nat → Cell α

def write {α : Type} (memory : Memory α) (region : RegionName) (offset : Nat)
    (cell : Cell α) : Memory α :=
  fun r i => if r=region ∧ i=offset then cell else memory r i

def fp {α : Type} (M : FP32 α) (value : ℝ) : α :=
  M.literal value
def nat {α : Type} (M : FP32 α) (value : Nat) : α :=
  M.fromNat value
def binary {α : Type} (M : FP32 α) (op : Binary32) (left right : α) : α :=
  match op with
  | .add => M.add left right
  | .sub => M.sub left right
  | .mul => M.mul left right
  | .div => M.div left right
def unary {α : Type} (M : FP32 α) (op : Unary32) (value : α) : α :=
  match op with
  | .exp => M.exp value
  | .log => M.log value
def load {α : Type} [Inhabited α] (M : FP32 α) (memory : Memory α)
    (region : RegionName) (offset : Nat) : α :=
  M.load ((memory region offset).read .real)
def label {α : Type} [Inhabited α] (p : Parameters) (memory : Memory α) (row : Nat) : Int :=
  (memory (Region.cast p.labels) (p.labelsBase+row)).read .int
def column (p : Parameters) (block lane : Nat) : Nat := block*p.block+lane
def logitAddress (p : Parameters) (row col : Nat) : Nat := p.logitsBase+row*p.logitsStride+col
def gradientAddress (p : Parameters) (row col : Nat) : Nat := p.gradientBase+row*p.gradientStride+col

def sum {α : Type} (M : FP32 α) {B : Nat} (values : Fin B → α) : α :=
  M.sum values
def maximum {α : Type} (M : FP32 α) {B : Nat} (values : Fin B → α) : α :=
  M.maximum values

def loadedLogits {α : Type} [Inhabited α] (M : FP32 α) (p : Parameters)
    (memory : Memory α) (row block : Nat) : Fin p.block → α := fun lane =>
  if column p block lane.val<p.cols then
    load M memory p.logits (logitAddress p row (column p block lane.val)) else M.negInf

structure ForwardValues (α : Type) (B : Nat) where
  logits : Fin B → α
  maxLogits : α
  sumLogits : Option α
  lse : α

def forwardValues {α : Type} [Inhabited α] (M : FP32 α) (p : Parameters)
    (memory : Memory α) (row block : Nat) : ForwardValues α p.block :=
  let logits := loadedLogits M p memory row block
  let maxLogits := maximum M logits
  let sumLogits := if p.hasSmoothing then
      some (sum M (fun lane => if column p block lane.val<p.cols then logits lane else fp M 0))
    else none
  let lse := binary M .add
    (unary M .log (sum M (fun lane => unary M .exp (binary M .sub (logits lane) maxLogits)))) maxLogits
  ⟨logits,maxLogits,sumLogits,lse⟩

/-- The target gather reads memory after the actual LSE store. Fresh output
allocation, rather than a numerical premise, later identifies it with caller
input memory. The smoothing expression tree follows each source branch. -/
def forwardLoss {α : Type} [Inhabited α] (M : FP32 α) (p : Parameters)
    (memoryAfterLSE : Memory α) (row block : Nat) (target : Int)
    (values : ForwardValues α p.block) : α :=
  if target=p.ignored then fp M 0 else
  let shifted := target-p.classStart
  let lseTerm := if p.split then fp M 0 else values.lse
  let smoothing := fp M p.smoothing
  let loss :=
    if (block*p.block : Int)≤shifted ∧ shifted<(min p.cols ((block+1)*p.block) : Nat) then
      let targetLogit := load M memoryAfterLSE p.logits (logitAddress p row shifted.toNat)
      if p.hasSmoothing then
        binary M .sub
          (binary M .sub lseTerm
            (binary M .div (binary M .mul smoothing (values.sumLogits.getD (fp M 0))) (nat M p.totalClasses)))
          (binary M .mul (binary M .sub (nat M 1) smoothing) targetLogit)
      else binary M .sub lseTerm targetLogit
    else if p.hasSmoothing then
      binary M .mul smoothing
        (binary M .sub lseTerm (binary M .div (values.sumLogits.getD (fp M 0)) (nat M p.totalClasses)))
    else fp M 0
  if p.split then loss else
    binary M .add loss (binary M .mul (binary M .mul (fp M p.squareScale) values.lse) values.lse)

def forward {α : Type} [Inhabited α] (M : FP32 α) (p : Parameters)
    (memory : Memory α) (row block : Nat) : Memory α :=
  let target := label p memory row
  let values := forwardValues M p memory row block
  let afterLSE := write memory p.lse (p.lseBase+block*p.rows+row) (.mk .real values.lse)
  let loss := forwardLoss M p afterLSE row block target values
  write afterLSE p.loss (p.lossBase+block*p.rows+row) (.mk .real loss)

def backwardValues {α : Type} [Inhabited α] (M : FP32 α) (p : Parameters)
    (memory : Memory α) (row block : Nat) : Fin p.block → α :=
  let target := label p memory row
  let isIgnored := target=p.ignored
  let upstream := if isIgnored then fp M 0 else
    load M memory p.upstream (p.upstreamBase+row*p.upstreamStride)
  let logits := loadedLogits M p memory row block
  let lse := load M memory p.lse (p.lseBase+row)
  let shifted := (if isIgnored then p.classStart else target)-p.classStart
  fun lane =>
    let probability := unary M .exp (binary M .sub (logits lane) lse)
    let corrected := binary M .add probability
      (binary M .mul (binary M .mul (binary M .mul (fp M 2) (fp M p.squareScale)) lse) probability)
    let adjusted := if p.hasSmoothing then
        let smoothNegative := binary M .div (fp M p.smoothing) (nat M p.totalClasses)
        binary M .sub
          (if (column p block lane.val : Int)=shifted then
            binary M .sub corrected (binary M .sub (nat M 1) (fp M p.smoothing)) else corrected)
          smoothNegative
      else if (column p block lane.val : Int)=shifted then binary M .sub corrected (fp M 1) else corrected
    if isIgnored then fp M 0 else binary M .mul upstream adjusted

def backward {α : Type} [Inhabited α] (M : FP32 α) (p : Parameters)
    (memory : Memory α) (row block : Nat) : Memory α :=
  let values := backwardValues M p memory row block
  (List.ofFn (fun lane : Fin p.block => lane)).foldl (fun current lane =>
    if column p block lane.val<p.cols then
      write current p.gradient (gradientAddress p row (column p block lane.val)) (.mk .real (values lane))
    else current) memory

end
end Source
end VeriTile.Bench.Repairs.CrossEntropy1
