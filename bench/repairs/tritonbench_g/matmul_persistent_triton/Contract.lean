import bench.repairs.tritonbench_g.matmul_persistent_triton.Proofs.Host

namespace VeriTile.Bench.Repairs.MatmulPersistentTriton.Checked
open Host

/-- The dynamic tensor/layout/rank/dtype/CUDA/stride checks establish the
ordinary descriptor. Supported Storage has exactly FP16 and E4M3 constructors. -/
def MetadataContract (args : Arguments) : Prop :=
  args.A.columns = args.B.rows ∧ args.A.dtype = args.B.dtype ∧
  args.A.device = args.B.device ∧ args.A.isNeg = Bool.false ∧ args.A.isConj = Bool.false ∧
  args.B.isNeg = Bool.false ∧ args.B.isConj = Bool.false ∧
    (args.A.rows*args.A.columns=0 ∨ args.A.pointer.toNat%args.A.dtype.bytes=0) ∧
    (args.B.rows*args.B.columns=0 ∨ args.B.pointer.toNat%args.B.dtype.bytes=0)

def metadataChecks (args : Arguments) : Bool :=
  decide (args.A.columns = args.B.rows ∧ args.A.dtype = args.B.dtype ∧
    args.A.device = args.B.device ∧ args.A.isNeg = Bool.false ∧ args.A.isConj = Bool.false ∧
    args.B.isNeg = Bool.false ∧ args.B.isConj = Bool.false ∧
    (args.A.rows*args.A.columns=0 ∨ args.A.pointer.toNat%args.A.dtype.bytes=0) ∧
    (args.B.rows*args.B.columns=0 ∨ args.B.pointer.toNat%args.B.dtype.bytes=0))

inductive Error where
  | tensors | layout | rank | dtype | cuda | negativeStride | metadata
  deriving DecidableEq, Repr

def checkInputs (args : Arguments) : Except Error Unit :=
  if metadataChecks args then .ok () else .error .metadata

inductive DynamicArguments where
  | nonTensors | nonStrided | nonMatrix | unsupportedDtype | nonCuda | negativeStride
  | tensors (args : Arguments)

def checkArguments : DynamicArguments → Except Error Unit
  | .nonTensors => .error .tensors
  | .nonStrided => .error .layout
  | .nonMatrix => .error .rank
  | .unsupportedDtype => .error .dtype
  | .nonCuda => .error .cuda
  | .negativeStride => .error .negativeStride
  | .tensors args => checkInputs args

def Admitted : DynamicArguments → Prop
  | .tensors args => MetadataContract args
  | _ => False

theorem checkInputs_iff (args : Arguments) : checkInputs args = .ok () ↔ MetadataContract args := by
  have hm : metadataChecks args=Bool.true ↔ MetadataContract args := by
    simp only [metadataChecks, MetadataContract, decide_eq_true_eq]
  rw [←hm]
  simp [checkInputs,ite_eq_iff]

theorem checkArguments_iff (args : DynamicArguments) : checkArguments args = .ok () ↔ Admitted args := by
  cases args <;> simp [checkArguments,Admitted,checkInputs_iff]

/-- Remaining premises are ordinary representable valid-storage/live-object
contracts. All source i64 bounds follow from these in the accepted Bounds and
ControlBounds proofs; no task-specific small-size envelope is assumed. -/
theorem admitted_input {σ : Type} (live : σ → Nat → Prop) (args : Arguments) (initial : σ)
    (guard : checkInputs args = .ok ())
    (aBound : args.A.Bounded) (bBound : args.B.Bounded)
    (aLive : live initial args.A.storage) (bLive : live initial args.B.storage) :
    Host.InputContract live args initial := by
  obtain ⟨shape,dtype,device,an,ac,bn,bc,_,_⟩ := (checkInputs_iff args).1 guard
  exact ⟨shape,dtype,device,⟨an,ac⟩,⟨bn,bc⟩,aBound,bBound,aLive,bLive⟩

#axiomsClean checkArguments_iff
#axiomsClean admitted_input
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.MatmulPersistentTriton.Checked
