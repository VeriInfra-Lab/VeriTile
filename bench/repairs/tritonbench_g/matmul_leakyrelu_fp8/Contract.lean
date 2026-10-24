import bench.repairs.tritonbench_g.matmul_leakyrelu_fp8.Proofs.Host

namespace VeriTile.Bench.Repairs.MatmulLeakyreluFp8.Checked
open Host
noncomputable section
open Classical

/-- These are ordinary already-queried Tensor observations. Unsupported raw
objects/layouts/ranks/dtypes/devices are classified before descriptor queries. -/
inductive DynamicArguments where
  | nonTensor | nonStrided | nonMatrix | nonFP32 | nonCuda | negativeStride | nonStringActivation
  | tensors (args : Arguments)

def MetadataContract (args : Arguments) : Prop :=
  args.A.columns = args.B.rows ∧ contiguous args.A ∧
  args.A.dtype = .fp32 ∧ args.B.dtype = .fp32 ∧ args.A.device = args.B.device ∧
  args.A.isNeg = Bool.false ∧ args.A.isConj = Bool.false ∧
  args.B.isNeg = Bool.false ∧ args.B.isConj = Bool.false ∧
  (args.A.rows*args.A.columns ≠ 0 → args.A.pointer.toNat%4 = 0) ∧
  (args.B.rows*args.B.columns ≠ 0 → args.B.pointer.toNat%4 = 0)

def checkInputs (args : Arguments) : Except String Unit :=
  if args.A.columns ≠ args.B.rows then .error "shape"
  else if ¬contiguous args.A then .error "contiguous A"
  else if args.A.dtype ≠ .fp32 ∨ args.B.dtype ≠ .fp32 then .error "FP32"
  else if args.A.device ≠ args.B.device then .error "device"
  else if args.A.isNeg ∨ args.A.isConj ∨ args.B.isNeg ∨ args.B.isConj then .error "resolved"
  else if ¬((args.A.rows*args.A.columns ≠ 0 → args.A.pointer.toNat%4 = 0) ∧
    (args.B.rows*args.B.columns ≠ 0 → args.B.pointer.toNat%4 = 0)) then .error "alignment"
  else .ok ()

def check : DynamicArguments → Except String Unit
  | .tensors args => checkInputs args
  | .nonTensor => .error "Tensor"
  | .nonStrided => .error "layout"
  | .nonMatrix => .error "rank"
  | .nonFP32 => .error "FP32"
  | .nonCuda => .error "CUDA"
  | .negativeStride => .error "stride"
  | .nonStringActivation => .error "activation"

def Admitted : DynamicArguments → Prop
  | .tensors args => MetadataContract args
  | _ => False

theorem checks_iff (args : Arguments) : checkInputs args = .ok () ↔ MetadataContract args := by
  unfold checkInputs MetadataContract
  simp only [ite_eq_iff]
  simp
  tauto

theorem dynamic_checks_iff (args : DynamicArguments) : check args = .ok () ↔ Admitted args := by
  cases args <;> simp [check,Admitted,checks_iff]

/-- The storage capacities and signed native descriptor range are ordinary
Tensor-library facts; all kernel source bounds are derived from them. -/
theorem checked_input {σ : Type} (live : σ → Nat → Prop) (args : Arguments) (state : σ)
    (accepted : checkInputs args = .ok ()) (aBound : args.A.Bounded) (bBound : args.B.Bounded)
    (aLive : live state args.A.storage) (bLive : live state args.B.storage) : InputContract live args state := by
  obtain ⟨shape,contiguous,aType,bType,device,an,ac,bn,bc,_,_⟩ := (checks_iff args).mp accepted
  exact ⟨shape,contiguous,aType,bType,device,⟨an,ac⟩,⟨bn,bc⟩,aBound,bBound,aLive,bLive⟩

#axiomsClean dynamic_checks_iff
#axiomsClean checked_input
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MatmulLeakyreluFp8.Checked
