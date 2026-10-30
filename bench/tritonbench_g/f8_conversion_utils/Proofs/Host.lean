import bench.tritonbench_g.f8_conversion_utils.Proofs.Contract
import bench.tritonbench_g.f8_conversion_utils.Proofs.NativeGrid

namespace VeriTile.Bench.Repairs.F8ConversionUtils.Checked
open VeriTile.Torch
noncomputable section

def outputDType : Direction → DType
  | .decode => .other "float16"
  | .encode => .other "int8"

structure TensorResult where
  identity : Nat
  metadata : TensorMetadata

/-- Ordinary live Tensor/empty_like API facts. The input pointer includes its
storage offset; neither allocation requires padded tail capacity. Output
identity, shape/dtype/device and disjointness are the fresh allocator contract. -/
structure Allocated (identity : Nat) (d : Direction) (f : FormatArgument) (x : TensorMetadata) where
  result : TensorResult
  source : Native.Allocation x.numel (profile d f x).inputWidth
  target : Native.Allocation x.numel (profile d f x).outputWidth
  inputPointer : source.base = x.dataPtr.toNat
  outputPointer : target.base = result.metadata.dataPtr.toNat
  outputAligned : target.base % (profile d f x).outputWidth = 0
  fresh : result.identity ≠ identity
  sameShape : result.metadata.shape = x.shape
  sameDevice : result.metadata.device = x.device
  dtype : result.metadata.dtype = outputDType d
  contiguous : result.metadata.isContiguous = Bool.true
  /-- Fresh empty_like storage has no lazy negative view bit. -/
  noNeg : result.metadata.isNeg = Bool.false
  separate : Native.Separate source target

inductive Failure where
  | input (reason : InputError)
  | allocation | launch

structure Returned (M : Type) where
  tensor : TensorResult
  memory : M
  currentDevice : Nat

/-- Observation after successful completion of all submitted chunks. False
launchReady represents a propagated compile/setup/launch error and has no
successful-return claim. The empty branch submits no chunk. CUDA device scope
restoration is part of its ordinary context-manager API semantics. -/
def execute {M A B : Type} (ops : Memory.Primitives M A B
    (profile d f x).inputWidth (profile d f x).outputWidth)
    (compute : A → B) (before : M) (savedDevice : Nat)
    (allocated : Except Unit (Allocated identity d f x)) (launchReady : Bool) :
    Except Failure (Returned M) :=
  match allocated with
  | .error _ => .error .allocation
  | .ok a =>
    if x.numel ≠ 0 ∧ launchReady = Bool.false then .error .launch
    else .ok ⟨a.result,Lowering.allChunks ops (profile d f x).twice compute
      a.source.base a.target.base x.numel (launchChunks x.numel) before,savedDevice⟩

def runChecked {M A B : Type} (ops : Memory.Primitives M A B
    (profile d f x).inputWidth (profile d f x).outputWidth)
    (compute : A → B) (before : M) (savedDevice : Nat)
    (allocated : Except Unit (Allocated identity d f x)) (launchReady : Bool) :
    Except Failure (Returned M) :=
  match checkInputs d f (.tensor x) with
  | .error reason => .error (.input reason)
  | .ok () => execute ops compute before savedDevice allocated launchReady

theorem checked_success {M A B : Type} (ops : Memory.Primitives M A B
    (profile d f x).inputWidth (profile d f x).outputWidth)
    (compute : A → B) (before : M) (savedDevice : Nat)
    (checked : checkInputs d f (.tensor x) = .ok ())
    (a : Allocated identity d f x) (ready : Bool) (launch : x.numel = 0 ∨ ready = Bool.true) :
    runChecked ops compute before savedDevice (.ok a) ready =
      .ok ⟨a.result,Lowering.allChunks ops (profile d f x).twice compute
        a.source.base a.target.base x.numel (launchChunks x.numel) before,savedDevice⟩ := by
  have noError : ¬ (x.numel ≠ 0 ∧ ready = Bool.false) := by rcases launch with h | h <;> simp [h]
  simp [runChecked,checked,execute,noError]

theorem rejects {M A B : Type} (ops : Memory.Primitives M A B
    (profile d f x).inputWidth (profile d f x).outputWidth)
    (compute : A → B) (before : M) (savedDevice : Nat)
    (bad : checkInputs d f (.tensor x) = .error reason)
    (allocated : Except Unit (Allocated identity d f x)) (ready : Bool) :
    runChecked ops compute before savedDevice allocated ready = .error (.input reason) := by
  simp [runChecked,bad]

/-- No unclassified failure or failed source run is hidden in a numerical
assumption. A successful checked host observation comes from the actual fresh
allocation and the complete chunk execution. -/
theorem outcomes {M A B : Type} (ops : Memory.Primitives M A B
    (profile d f x).inputWidth (profile d f x).outputWidth)
    (compute : A → B) (before : M) (savedDevice : Nat)
    (checked : checkInputs d f (.tensor x) = .ok ())
    (allocated : Except Unit (Allocated identity d f x)) (ready : Bool) :
    (allocated = .error () ∧ runChecked ops compute before savedDevice allocated ready = .error .allocation) ∨
    (∃ a,allocated = .ok a ∧ x.numel ≠ 0 ∧ ready = Bool.false ∧
      runChecked ops compute before savedDevice allocated ready = .error .launch) ∨
    (∃ a,allocated = .ok a ∧ (x.numel = 0 ∨ ready = Bool.true) ∧
      runChecked ops compute before savedDevice allocated ready =
        .ok ⟨a.result,Lowering.allChunks ops (profile d f x).twice compute
          a.source.base a.target.base x.numel (launchChunks x.numel) before,savedDevice⟩) := by
  cases allocated with
  | error e => cases e; exact Or.inl ⟨rfl,by simp [runChecked,checked,execute]⟩
  | ok a =>
    by_cases fail : x.numel ≠ 0 ∧ ready = Bool.false
    · exact Or.inr (Or.inl ⟨a,rfl,fail.1,fail.2,by simp [runChecked,checked,execute,fail]⟩)
    · have launch : x.numel = 0 ∨ ready = Bool.true := by cases ready <;> simp_all
      exact Or.inr (Or.inr ⟨a,rfl,launch,checked_success ops compute before savedDevice checked a ready launch⟩)

#axiomsClean checked_success
#axiomsClean rejects
#axiomsClean outcomes
end
end VeriTile.Bench.Repairs.F8ConversionUtils.Checked
