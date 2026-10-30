import bench.repairs.tritonbench_g.cross_entropy2.Proofs.Grid
import bench.repairs.tritonbench_g.cross_entropy1.Proofs.HostTrace

namespace VeriTile.Bench.Repairs.CrossEntropy2.Host
open VeriTile.Triton Source
open VeriTile.Bench.Repairs.CrossEntropy1.Host (World Step bind)
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section

structure Options where
  smoothing : ℝ
  logitScale : ℝ
  squareScale : ℝ
  ignored : Int
  total : Nat
  start : Nat
  width : Nat
  hasSmoothing : Bool
  split : Bool

structure Runtime (α σ : Type) extends VeriTile.Bench.Repairs.CrossEntropy1.Host.Runtime α σ where
  setupForward2 : Parameters → Nat → World α σ → Step (World α σ) Unit
  setupBackward2 : Parameters → Nat → World α σ → Step (World α σ) Unit
  multiply : Tensor → Tensor → World α σ → Step (World α σ) Tensor
  toInt64 : Tensor → World α σ → Step (World α σ) Tensor

abbrev Return := Tensor × Tensor × Tensor

def valid (o : Options) (x y : Tensor) (rows cols : Nat) : Prop :=
  0<cols ∧ x.dtype="torch.float32" ∧ y.dtype="torch.int32" ∧ y.shape=[rows] ∧ y.device=x.device ∧
  0<o.width ∧ o.width≤65536 ∧ (∃ k, o.width=2^k) ∧
  -(2^63 : Int)≤o.ignored ∧ o.ignored<2^63 ∧ 0<o.total ∧ o.total<2^63 ∧ o.start+cols≤o.total ∧
  0≤o.smoothing ∧ o.smoothing≤1 ∧ 0≤o.squareScale

/-- Finite scalar admission belongs to the actual Python float ABI. The real
model erases scalar casts; the bit profile binds them to FP32 scalar ports. -/
def normalize {α σ : Type} (rt : Runtime α σ) (t : Tensor) (s : World α σ) : Step (World α σ) Tensor :=
  bind (rt.resolveNeg t s) fun resolved sr => rt.contiguous resolved sr

def parameters (o : Options) (x y loss lse z : Tensor) (rows cols : Nat) (split : Bool) : Parameters :=
  {loss:=loss.region,lse:=lse.region,zLoss:=z.region,logits:=x.region,gradient:=x.region,
   upstream:=loss.region,labels:=Region.cast y.region,
   lossBase:=loss.storageOffset,lseBase:=lse.storageOffset,zLossBase:=z.storageOffset,
   logitsBase:=x.storageOffset,labelsBase:=y.storageOffset,rows:=rows,cols:=cols,
   totalClasses:=o.total,classStart:=o.start,ignored:=o.ignored,
   logitsStride:=VeriTile.Bench.Repairs.CrossEntropy1.Host.rowStride x,
   gradientStride:=VeriTile.Bench.Repairs.CrossEntropy1.Host.rowStride x,
   upstreamStride:=VeriTile.Bench.Repairs.CrossEntropy1.Host.rowStride loss,
   block:=o.width,smoothing:=o.smoothing,logitScale:=o.logitScale,squareScale:=o.squareScale,
   hasSmoothing:=o.hasSmoothing,split:=split}

def forwardLaunch {α σ : Type} [Inhabited α] (M : FP32 α) (rt : Runtime α σ)
    (p : Parameters) (count : Nat) (s : World α σ) : Step (World α σ) Unit :=
  if p.rows=0 then .ok () s else
  bind (rt.setupForward2 p count s) fun _ ready =>
    .ok () {ready with memory:=Grid.forward M p ready.memory count}

def backwardGrid {α : Type} [Inhabited α] (M : FP32 α) (p : Parameters) (m : Memory α) (count : Nat) : Memory α :=
  (VeriTile.Bench.Repairs.CrossEntropy1.Grid.tasks p.rows count).foldl
    (fun current q => Source.backward M p current q.1 q.2) m

def backwardLaunch {α σ : Type} [Inhabited α] (M : FP32 α) (rt : Runtime α σ)
    (p : Parameters) (count : Nat) (s : World α σ) : Step (World α σ) Unit :=
  if p.rows=0 then .ok () s else
  bind (rt.setupBackward2 p count s) fun _ ready =>
    .ok () {ready with memory:=backwardGrid M p ready.memory count}

def aggregate {α σ : Type} (rt : Runtime α σ) (o : Options) (count : Nat)
    (labels loss lse : Tensor) (s : World α σ) : Step (World α σ) Return :=
  let localResult := if count>1 then
    bind (rt.logsumexp0 lse s) fun nextLSE logged =>
    bind (rt.sum0 loss logged) fun nextLoss summed => .ok (nextLoss,nextLSE) summed
    else .ok (loss,lse) s
  bind localResult fun pair reduced =>
  bind (rt.addInplace pair.1 pair.2 reduced) fun losses added =>
  bind (rt.scalarMul o.squareScale pair.2 added) fun scaled ss =>
  bind (rt.multiply scaled pair.2 ss) fun zLoss multiplied =>
  bind (rt.addInplace losses zLoss multiplied) fun losses corrected =>
  bind (rt.toInt64 labels corrected) fun wideLabels widened =>
  bind (rt.equalInt wideLabels o.ignored widened) fun mask compared =>
  bind (rt.maskedFillZero losses mask compared) fun losses masked =>
  bind (rt.maskedFillZero zLoss mask masked) fun zLoss final =>
    .ok (losses,pair.2,zLoss) final

def forward {α σ : Type} [Inhabited α] (M : FP32 α) (rt : Runtime α σ) (o : Options)
    (logits labels : Tensor) (s : World α σ) : Step (World α σ) Return := by
  classical
  exact match logits.shape with
  | [rows,cols] =>
    if valid o logits labels rows cols then
      bind (normalize rt logits s) fun logits xs =>
      bind (normalize rt labels xs) fun labels ys =>
      let count := VeriTile.Bench.Repairs.CrossEntropy1.Math.blocks cols o.width
      let split := o.split || decide (count>1 ∨ o.start≠0 ∨ cols≠o.total)
      let shape := if count>1 then [count,rows] else [rows]
      bind (rt.empty shape "torch.float32" logits.device ys) fun loss sl =>
      bind (rt.empty shape "torch.float32" logits.device sl) fun lse se =>
      bind (rt.empty shape "torch.float32" logits.device se) fun zLoss sz =>
      let p := parameters o logits labels loss lse zLoss rows cols split
      bind (forwardLaunch M rt p count sz) fun _ computed =>
      if split then aggregate rt o count labels loss lse computed else .ok (loss,lse,zLoss) computed
    else .error "invalid forward metadata" s
  | _ => .error "invalid logits rank" s

def backward {α σ : Type} [Inhabited α] (M : FP32 α) (rt : Runtime α σ) (o : Options)
    (upstream logits lse labels : Tensor) (s : World α σ) : Step (World α σ) Tensor := by
  classical
  exact match logits.shape with
  | [rows,cols] =>
    if valid o logits labels rows cols ∧
        upstream.dtype="torch.float32" ∧ lse.dtype="torch.float32" ∧
        upstream.shape=[rows] ∧ lse.shape=[rows] ∧ upstream.device=logits.device ∧ lse.device=logits.device then
      bind (normalize rt logits s) fun logits xs =>
      bind (normalize rt labels xs) fun labels ys =>
      bind (normalize rt upstream ys) fun upstream gs =>
      bind (normalize rt lse gs) fun lse ls =>
      bind (rt.emptyLike logits Bool.true ls) fun gradient allocated =>
      let p := {parameters o logits labels upstream lse lse rows cols Bool.false with
        gradient:=gradient.region,gradientBase:=gradient.storageOffset,
        gradientStride:=VeriTile.Bench.Repairs.CrossEntropy1.Host.rowStride gradient,
        upstream:=upstream.region,upstreamBase:=upstream.storageOffset,
        upstreamStride:=VeriTile.Bench.Repairs.CrossEntropy1.Host.rowStride upstream}
      bind (backwardLaunch M rt p (VeriTile.Bench.Repairs.CrossEntropy1.Math.blocks cols o.width) allocated)
        fun _ computed => .ok gradient computed
    else .error "invalid backward metadata" s
  | _ => .error "invalid logits rank" s

theorem bind_success {S A B : Type} (result : Step S A) (next : A → S → Step S B) (value : B) (state : S) :
    bind result next=.ok value state ↔ ∃ a s, result=.ok a s ∧ next a s=.ok value state := by
  cases result <;> simp [VeriTile.Bench.Repairs.CrossEntropy1.Host.bind]

/-- No later operation runs after an ordinary setup/library failure. -/
theorem failed_prefix {S A B : Type} (reason : String) (state : S) (next : A → S → Step S B) :
    bind (.error reason state) next=.error reason state := rfl

#axiomsClean forward
#axiomsClean backward
#axiomsClean failed_prefix
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy2.Host
