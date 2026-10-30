import bench.repairs.tritonbench_g.cross_entropy1.Proofs.Autograd

namespace VeriTile.Bench.Repairs.CrossEntropy1.Checked
open VeriTile.Triton Source Host
open TensorModel (Tensor)
noncomputable section

/-- Python classifications and actual pointer observations. Shapes, dtypes,
CUDA devices and negative bits remain in the ordinary Tensor descriptor. -/
structure ForwardMetadata where
  logitsTensor : Bool
  labelsTensor : Bool
  logitsStrided : Bool
  labelsStrided : Bool
  logitsContiguous : Bool
  labelsContiguous : Bool
  smoothingFloat : Bool
  scaleFloat : Bool
  smoothingFinite : Bool
  scaleFinite : Bool
  ignoredInt : Bool
  inplaceBool : Bool
  rows : Nat
  cols : Nat
  logitsPointer : Nat
  labelsPointer : Nat

/-- Exactly the successful branches of `_check_forward`, including the
value-dependent int64 label check. `None` selects the checked local entry. -/
def InputContract (m : ForwardMetadata) (o : Options) (logits labels : Tensor)
    (target : Nat → Int) : Prop :=
  m.logitsTensor=Bool.true ∧ m.labelsTensor=Bool.true ∧
  m.logitsStrided=Bool.true ∧ m.labelsStrided=Bool.true ∧
  logits.dtype="torch.float32" ∧ labels.dtype="torch.int64" ∧
  logits.isCUDA=Bool.true ∧ labels.isCUDA=Bool.true ∧ logits.device=labels.device ∧
  logits.shape=[m.rows,m.cols] ∧ labels.shape=[m.rows] ∧
  m.logitsContiguous=Bool.true ∧ lastStride logits=1 ∧ m.labelsContiguous=Bool.true ∧
  logits.isNeg=Bool.false ∧ labels.isNeg=Bool.false ∧
  m.smoothingFloat=Bool.true ∧ m.scaleFloat=Bool.true ∧ m.smoothingFinite=Bool.true ∧
  0≤o.smoothing ∧ o.smoothing≤1 ∧ m.scaleFinite=Bool.true ∧ 0≤o.squareScale ∧
  m.ignoredInt=Bool.true ∧ -(2^63 : Int)≤o.ignored ∧ o.ignored<2^63 ∧
  m.inplaceBool=Bool.true ∧ o.processGroup=none ∧
  0<m.cols ∧ m.rows≤2^31-1 ∧ Math.blocks m.cols (bwdWidth m.cols)≤65535 ∧
  (0<m.rows → m.logitsPointer%4=0 ∧ m.labelsPointer%8=0) ∧
  (∀ row, row<m.rows → target row=o.ignored ∨ (0≤target row ∧ target row<(m.cols : Int)))

def checkForward (m : ForwardMetadata) (o : Options) (logits labels : Tensor)
    (target : Nat → Int) : Bool := by
  classical
  exact decide (InputContract m o logits labels target)

structure BackwardMetadata where
  tensor : Bool
  strided : Bool
  pointer : Nat

/-- Upstream may have zero or nonunit strides: the existing snapshot protects
in-place aliases and accepts the broadcast gradient produced by `loss.sum()`. -/
def BackwardContract (m : BackwardMetadata) (rows : Nat) (logits upstream : Tensor) : Prop :=
  m.tensor=Bool.true ∧ m.strided=Bool.true ∧ upstream.dtype="torch.float32" ∧
  upstream.isCUDA=Bool.true ∧ upstream.device=logits.device ∧ upstream.shape=[rows] ∧
  upstream.isNeg=Bool.false ∧ (0<rows → m.pointer%4=0)

def checkBackward (m : BackwardMetadata) (rows : Nat) (logits upstream : Tensor) : Bool :=
  by classical exact decide (BackwardContract m rows logits upstream)

def forward {α σ : Type} [Inhabited α] (M : FP32 α) (rt : Runtime α σ)
    (m : ForwardMetadata) (o : Options) (logits labels : Tensor) (target : Nat → Int)
    (s : World α σ) : Step (World α σ) ForwardReturn :=
  if checkForward m o logits labels target then Host.forward M rt o logits labels s
  else .error "checked forward contract" s


/-- Saved tensor retrieval precedes the upstream guard, exactly as in Python. -/
def backward {α σ : Type} [Inhabited α] (M : FP32 α) (rt : Runtime α σ)
    (versions : σ → Nat → Nat) (saved : Autograd.Saved) (m : BackwardMetadata)
    (rows : Nat) (upstream : Tensor) (s : World α σ) : Step (World α σ) Tensor :=
  bind (Autograd.retrieve versions saved s) fun context ready =>
    if checkBackward m rows context.logits upstream then Host.backward M rt context upstream ready
    else .error "checked backward contract" ready

def sourceSHA256 : String := "e6932518ddf889508a922c49c256fd521eee4041d9115ba2d34954647ba24957"

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy1.Checked
