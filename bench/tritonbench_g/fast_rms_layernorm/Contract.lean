import VeriTile.Torch.Tensor
import bench.tritonbench_g.fast_rms_layernorm.Proofs.HostMetadata

namespace VeriTile.Bench.Repairs.FastRmsLayernorm.Contract
open Host

/-- Properties queried from actual tensor handles before any view or snapshot.
Shapes and device IDs are the corresponding fields of the source Request. -/
structure TensorMetadata where
  tensor : Bool
  strided : Bool
  fp32 : Bool
  cuda : Bool
  current : Bool
  contiguous : Bool
  negative : Bool
  storageId : Nat
  address : Nat
  deriving DecidableEq

structure Metadata where
  input : TensorMetadata
  weight : TensorMetadata
  epsilonFloat : Bool
  epsilonFinite : Bool
  gemmaBool : Bool

/-- Adapter for actual public PyTorch queries. Storage identity and the current
CUDA device are separate runtime observations; dataPtr includes storage offsets. -/
def observeTensor (t : VeriTile.Torch.TensorMetadata) (currentDevice storageId : Nat) : TensorMetadata :=
  ⟨Bool.true,t.isStrided,t.isFP32,t.isCuda,decide (t.device.index=currentDevice),
    t.isContiguous,t.isNeg,storageId,t.dataPtr.toNat⟩

def tensorChecks (t : TensorMetadata) : List Bool :=
  [t.tensor,t.strided,t.fp32,t.cuda,t.current,t.contiguous,!t.negative]
def TensorContract (t : TensorMetadata) : Prop :=
  t.tensor=Bool.true ∧ t.strided=Bool.true ∧ t.fp32=Bool.true ∧
  t.cuda=Bool.true ∧ t.current=Bool.true ∧ t.contiguous=Bool.true ∧ t.negative=Bool.false

def ranked : InputShape → Bool | .scalar => Bool.false | .axes _ _ => Bool.true
def cols : InputShape → Nat | .scalar => 0 | .axes _ n => n
def rows : InputShape → Nat | .scalar => 0 | .axes leading _ => leading.prod

def pythonChecks {α : Type} (r : Request α) (m : Metadata) : List Bool :=
  tensorChecks m.input ++ tensorChecks m.weight ++
  [decide (r.inputDevice=r.weightDevice),ranked r.shape,decide (0<cols r.shape),
   decide (cols r.shape≤65536),decide (r.weightShape.prod=cols r.shape),
   decide (rows r.shape≤2147483647),decide (m.input.storageId≠m.weight.storageId),
   decide (r.shape.dims.prod=0 ∨ m.input.address%16=0),decide (m.weight.address%16=0),
   m.epsilonFloat,m.epsilonFinite,m.gemmaBool]

def InputContract {α : Type} (r : Request α) (m : Metadata) : Prop :=
  TensorContract m.input ∧ TensorContract m.weight ∧
  r.inputDevice=r.weightDevice ∧ ranked r.shape=Bool.true ∧
  0<cols r.shape ∧ cols r.shape≤65536 ∧ r.weightShape.prod=cols r.shape ∧
  rows r.shape≤2147483647 ∧ m.input.storageId≠m.weight.storageId ∧
  (r.shape.dims.prod=0 ∨ m.input.address%16=0) ∧ m.weight.address%16=0 ∧
  m.epsilonFloat=Bool.true ∧ m.epsilonFinite=Bool.true ∧ m.gemmaBool=Bool.true

def Admitted {α : Type} (r : Request α) (m : Metadata) : Prop :=
  (pythonChecks r m).all id=Bool.true

theorem python_checks_iff {α : Type} (r : Request α) (m : Metadata) :
    Admitted r m ↔ InputContract r m := by
  simp [Admitted,pythonChecks,tensorChecks,InputContract,TensorContract,List.all_cons,
    Bool.and_eq_true,Bool.not_eq_true, and_assoc]

theorem block_bound (n : Nat) (bound : n≤65536) : Settings.block n≤65536 := by
  have clog : Nat.clog 2 n≤16 := (Nat.clog_le_iff_le_pow (by decide : 1<2)).mpr bound
  exact (Nat.pow_le_pow_right (by decide : 1≤2) clog)

theorem validates {α : Type} (r : Request α) (m : Metadata) (ok : Admitted r m) :
    ∃ h,validate r=.ok h ∧ h.rows≤2147483647 := by
  have hc:=(python_checks_iff r m).mp ok
  rcases hc with ⟨_,_,device,rank,pos,lim,weight,grid,_⟩
  cases shape : r.shape with
  | scalar => simp [shape,ranked] at rank
  | axes leading n =>
    simp only [shape,cols,rows] at pos lim weight grid
    have block:=block_bound n lim
    exact ⟨⟨leading.prod,n,Settings.block n,pos,Settings.covered n,block,r.gemma,r.eps⟩,
      by simp [validate,shape,pos,block,device,weight],grid⟩

noncomputable def checkedCall {α : Type} (r : Request α) (m : Metadata) (ok : Admitted r m) : Call :=
  (validates r m ok).choose

theorem checked_valid {α : Type} (r : Request α) (m : Metadata) (ok : Admitted r m) :
    validate r=.ok (checkedCall r m ok) := (validates r m ok).choose_spec.1

theorem checked_grid {α : Type} (r : Request α) (m : Metadata) (ok : Admitted r m) :
    (checkedCall r m ok).rows≤2147483647 := (validates r m ok).choose_spec.2

def backwardChecks {α : Type} (ctx : Context α) (g : Upstream α) (m : TensorMetadata) : List Bool :=
  tensorChecks m ++ [decide (g.device=ctx.device),decide (g.shape.prod=ctx.call.cells)]
def BackwardContract {α : Type} (ctx : Context α) (g : Upstream α) (m : TensorMetadata) : Prop :=
  TensorContract m ∧ g.device=ctx.device ∧ g.shape.prod=ctx.call.cells
def BackwardAdmitted {α : Type} (ctx : Context α) (g : Upstream α) (m : TensorMetadata) : Prop :=
  (backwardChecks ctx g m).all id=Bool.true

theorem backward_checks_iff {α : Type} (ctx : Context α) (g : Upstream α) (m : TensorMetadata) :
    BackwardAdmitted ctx g m ↔ BackwardContract ctx g m := by
  simp [BackwardAdmitted,BackwardContract,backwardChecks,tensorChecks,TensorContract,and_assoc]

theorem backward_validated {α : Type} (ctx : Context α) (g : Upstream α) (m : TensorMetadata)
    (ok : BackwardAdmitted ctx g m) : backwardValidate ctx g=.ok () := by
  obtain ⟨_,device,shape⟩:=(backward_checks_iff ctx g m).mp ok
  simp [backwardValidate,device,shape]

#axiomsClean python_checks_iff
#axiomsClean backward_checks_iff
#axiomsClean checked_valid
#axiomsClean checked_grid
theorem observed_tensor_checks (t : VeriTile.Torch.TensorMetadata) (currentDevice storageId : Nat) :
    TensorContract (observeTensor t currentDevice storageId) ↔
      t.layout=.strided ∧ t.dtype=.floating .fp32 ∧ t.device.kind="cuda" ∧
      t.device.index=currentDevice ∧ t.isContiguous=Bool.true ∧ t.isNeg=Bool.false := by
  simp [TensorContract,observeTensor,VeriTile.Torch.TensorMetadata.isStrided,
    VeriTile.Torch.TensorMetadata.isFP32,VeriTile.Torch.TensorMetadata.isCuda]

#axiomsClean observed_tensor_checks
end VeriTile.Bench.Repairs.FastRmsLayernorm.Contract
