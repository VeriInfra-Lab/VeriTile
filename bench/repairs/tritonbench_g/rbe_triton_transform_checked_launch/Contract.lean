import bench.repairs.tritonbench_g.rbe_triton_transform.Proofs.KernelIndex
import bench.repairs.tritonbench_g.rbe_triton_transform_checked_launch.Kernels

namespace VeriTile.Bench.Repairs.RbeTritonTransformCheckedLaunch
open VeriTile Triton Torch
open VeriTile.Bench.Repairs
noncomputable section

namespace Contract
abbrev Position := RbeTritonTransform.Contract.Position

def batch (x : TensorMetadata) : Nat := x.shape[0]?.getD 0
def launchGrid (x : TensorMetadata) : Nat :=
  ((x.shape[1]?.getD 0+1)/2)*((x.shape[2]?.getD 0+1023)/1024)

/-- The existing input contract plus exactly the two nonempty launch guards. -/
structure InputContract (x : TensorMetadata) (pos : Position) : Prop where
  existing : RbeTritonTransform.Contract.InputContract x pos
  batchLimit : 0<x.numel → batch x≤2^31-1
  gridLimit : 0<x.numel → launchGrid x≤65535

inductive Error where
  | existing : RbeTritonTransform.Contract.Error → Error
  | batch | grid
  deriving DecidableEq, Repr

def checkInputs (x : TensorMetadata) (pos : Position) : Except Error Unit :=
  match RbeTritonTransform.Contract.checkInputs x pos with
  | .error error => .error (.existing error)
  | .ok _ =>
    if 0<x.numel ∧ 2^31-1<batch x then .error .batch
    else if 0<x.numel ∧ 65535<launchGrid x then .error .grid
    else .ok ()

theorem checkInputs_iff (x : TensorMetadata) (pos : Position) :
    checkInputs x pos=.ok () ↔ InputContract x pos := by
  have unpack : InputContract x pos ↔
      RbeTritonTransform.Contract.InputContract x pos ∧
      (0<x.numel → batch x≤2^31-1) ∧ (0<x.numel → launchGrid x≤65535) :=
    ⟨fun h => ⟨h.existing,h.batchLimit,h.gridLimit⟩,
      fun ⟨a,b,c⟩ => ⟨a,b,c⟩⟩
  rw [unpack,←RbeTritonTransform.Contract.checkInputs_iff]
  cases h : RbeTritonTransform.Contract.checkInputs x pos with
  | error error => simp [checkInputs,h]
  | ok value =>
    cases value
    simp [checkInputs,h,ite_eq_iff]

def checkArguments (x : ArgumentMetadata) (pos : Position) : Except Error Unit :=
  match x with
  | .nonTensor => .error (.existing .tensor)
  | .tensor x => checkInputs x pos

def AdmittedArguments (x : ArgumentMetadata) (pos : Position) : Prop :=
  match x with
  | .nonTensor => False
  | .tensor x => InputContract x pos

theorem checkArguments_iff (x : ArgumentMetadata) (pos : Position) :
    checkArguments x pos=.ok () ↔ AdmittedArguments x pos := by
  cases x <;> simp [checkArguments,AdmittedArguments,checkInputs_iff]

/-- Exact domain delta. The original domain is preserved for every empty shape. -/
theorem domain_exact (x : TensorMetadata) (pos : Position) :
    InputContract x pos ↔ RbeTritonTransform.Contract.InputContract x pos ∧
      (x.numel=0 ∨ batch x≤2^31-1 ∧ launchGrid x≤65535) := by
  constructor
  · intro h
    refine ⟨h.existing,?_⟩
    by_cases empty : x.numel=0
    · exact .inl empty
    · exact .inr ⟨h.batchLimit (by omega),h.gridLimit (by omega)⟩
  · rintro ⟨existing,empty|limits⟩
    · exact ⟨existing,by omega,by omega⟩
    · exact ⟨existing,fun _ => limits.1,fun _ => limits.2⟩

theorem empty_domain_unchanged (x : TensorMetadata) (pos : Position) (empty : x.numel=0) :
    checkInputs x pos=.ok () ↔ RbeTritonTransform.Contract.checkInputs x pos=.ok () := by
  rw [checkInputs_iff,RbeTritonTransform.Contract.checkInputs_iff,domain_exact]
  simp [empty]
end Contract

/-- The same actual Tensor and shape binding, with admission from the new guards. -/
structure Input (x : RbeTritonTransform.Binding.Tensor) (r : RbeTritonTransform.Request) : Prop where
  shape : x.actual.metadata.shape=[r.batches,r.rows,r.columns]
  admitted : Contract.InputContract x.actual.metadata ⟨Bool.true,r.position⟩

def Input.original (input : Input x r) : RbeTritonTransform.Binding.Input x r :=
  ⟨input.shape,input.admitted.existing⟩

/-- The extra native-launch premise is discharged from the actual new guard contract. -/
theorem Input.supported (input : Input x r) : RbeTritonTransform.Index.SupportedLaunch r := by
  have nonempty (positive : 0<r.size) : 0<x.actual.metadata.numel := by
    change 0<x.size
    rw [input.original.size]
    exact positive
  constructor
  · intro positive
    simpa [Contract.batch,input.shape] using input.admitted.batchLimit (nonempty positive)
  · intro positive
    simpa [Contract.launchGrid,input.shape,RbeTritonTransform.grid,
      RbeTritonTransform.rowParts,RbeTritonTransform.parts] using
        input.admitted.gridLimit (nonempty positive)

#guard_msgs (drop info) in
#specNonCircular Contract.InputContract avoiding [VeriTile.Bench.Repairs.RbeTritonTransformCheckedLaunch.source,
  RbeTritonTransform.Runtime.compose,FP.Structural.exec]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RbeTritonTransformCheckedLaunch
