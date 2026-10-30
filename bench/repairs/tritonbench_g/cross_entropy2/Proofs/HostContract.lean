import bench.repairs.tritonbench_g.cross_entropy2.Proofs.Host

namespace VeriTile.Bench.Repairs.CrossEntropy2.Checked
open VeriTile.Triton Source Host
open VeriTile.Bench.Repairs.CrossEntropy1.Host (World Step)
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section

structure TensorMetadata where
  tensor : Bool
  strided : Bool
  contiguous : Bool
  pointer : Nat

/-- Direct checked tensors; the accepted resolve/contiguous calls do not
expand the admitted layout, device or dtype domain. -/
def Direct (m : TensorMetadata) (t : Tensor) : Prop :=
  m.tensor=Bool.true ∧ m.strided=Bool.true ∧ t.isCUDA=Bool.true ∧
  m.contiguous=Bool.true ∧ t.isNeg=Bool.false ∧ (0<t.numel → m.pointer%4=0)

structure ScalarMetadata where
  smoothingFloat : Bool
  scaleFloat : Bool
  squareFloat : Bool
  smoothingFinite : Bool
  scaleFinite : Bool
  squareFinite : Bool
  ignoredInt : Bool
  totalInt : Bool
  startInt : Bool
  startNonnegative : Bool
  widthInt : Bool
  smoothingBool : Bool
  currentDevice : Nat

def Scalars (m : ScalarMetadata) : Prop :=
  m.smoothingFloat=Bool.true ∧ m.scaleFloat=Bool.true ∧ m.squareFloat=Bool.true ∧
  m.smoothingFinite=Bool.true ∧ m.scaleFinite=Bool.true ∧ m.squareFinite=Bool.true ∧
  m.ignoredInt=Bool.true ∧ m.totalInt=Bool.true ∧ m.startInt=Bool.true ∧
  m.startNonnegative=Bool.true ∧ m.widthInt=Bool.true ∧ m.smoothingBool=Bool.true

/-- The common successful branches of `_checked_validate`, including the
int64-widened label predicate. Labels outside this shard remain allowed when
they are valid global labels. No absent remote logits are inferred. -/
def CommonContract (mx my : TensorMetadata) (ms : ScalarMetadata) (o : Options)
    (x y : Tensor) (rows cols : Nat) (target : Nat → Int) : Prop :=
  Direct mx x ∧ Direct my y ∧ (Scalars ms ∧ x.device=ms.currentDevice) ∧ x.shape=[rows,cols] ∧
  Host.valid o x y rows cols ∧
  (0<rows → rows≤2^31-1 ∧ CrossEntropy1.Math.blocks cols o.width≤65535) ∧
  (∀ row, row<rows → target row=o.ignored ∨ (0≤target row ∧ target row<(o.total : Int)))

def ForwardContract (mx my : TensorMetadata) (ms : ScalarMetadata) (splitBool : Bool)
    (o : Options) (x y : Tensor) (rows cols : Nat) (target : Nat → Int) : Prop :=
  CommonContract mx my ms o x y rows cols target ∧ splitBool=Bool.true

def RowContract (m : TensorMetadata) (rows : Nat) (x t : Tensor) : Prop :=
  Direct m t ∧ t.dtype="torch.float32" ∧ t.shape=[rows] ∧ t.device=x.device

def BackwardContract (mx my mg ml : TensorMetadata) (ms : ScalarMetadata)
    (o : Options) (g x l y : Tensor) (rows cols : Nat) (target : Nat → Int) : Prop :=
  CommonContract mx my ms o x y rows cols target ∧ RowContract mg rows x g ∧ RowContract ml rows x l

/-- Guard adapters wrap the unchanged complete host semantics. -/
def forward {α σ : Type} [Inhabited α] (M : FP32 α) (rt : Runtime α σ)
    (mx my : TensorMetadata) (ms : ScalarMetadata) (splitBool : Bool)
    (o : Options) (x y : Tensor) (rows cols : Nat) (target : Nat → Int) (s : World α σ) :
    Step (World α σ) Return := by
  classical
  exact if ForwardContract mx my ms splitBool o x y rows cols target then Host.forward M rt o x y s
    else .error "checked forward contract" s

def backward {α σ : Type} [Inhabited α] (M : FP32 α) (rt : Runtime α σ)
    (mx my mg ml : TensorMetadata) (ms : ScalarMetadata)
    (o : Options) (g x l y : Tensor) (rows cols : Nat) (target : Nat → Int) (s : World α σ) :
    Step (World α σ) Tensor := by
  classical
  exact if BackwardContract mx my mg ml ms o g x l y rows cols target then Host.backward M rt o g x l y s
    else .error "checked backward contract" s

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy2.Checked
