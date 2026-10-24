import bench.repairs.tritonbench_g.cross_entropy_ops.Proofs.ResidentContract
import bench.repairs.tritonbench_g.cross_entropy_ops.Proofs.StorageOrigins
import bench.repairs.tritonbench_g.cross_entropy_ops.Proofs.SignedValues
import bench.repairs.tritonbench_g.cross_entropy_ops.Proofs.SourceAST
import bench.repairs.tritonbench_g.cross_entropy_ops.Proofs.NativeWhole

namespace VeriTile.Bench.Repairs.CrossEntropyOps.Resident
open CrossEntropy1.Host (World Step)
open CrossEntropy1.Source (FP32)
open CrossEntropy1.Floating (Expr)
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section

inductive Interpretation where
  | numeric
  | origins

def Scalar : Interpretation → Type
  | .numeric => SignedExpr
  | .origins => Marker

instance (mode : Interpretation) : Inhabited (Scalar mode) := by
  cases mode <;> dsimp [Scalar] <;> infer_instance

def scalarInterpreter : (mode : Interpretation) → FP32 (Scalar mode)
  | .numeric => signedFP
  | .origins => markerFP

/-- Only ordinary runtime primitive interpretations vary. Both source
scalar interpretations and the complete wrapper control flow are fixed. -/
structure Model (σ : Type) where
  mode : Interpretation
  runtime : Host.Runtime (Scalar mode) σ
  versions : σ → Nat → Nat
  integer : World (Scalar mode) σ → Tensor → Nat → Int
  groupSize : Nat → Nat
  groupRank : Nat → Nat

def scalarNeg : (mode : Interpretation) → Scalar mode → Scalar mode
  | .numeric => SignedExpr.neg
  | .origins => Marker.neg

def hostInterpreter : (mode : Interpretation) → CrossEntropy1.Host.TensorArithmetic (Scalar mode)
  | .numeric => signedHost
  | .origins => markerHost

def Model.observations {σ : Type} (model : Model σ) : CrossEntropy1.Host.Observations (Scalar model.mode) σ :=
  ⟨scalarNeg model.mode,model.integer⟩

/-- The source entry with its complete ordinary arguments. Backward retains
the exact saved context, original native saved boundary, and ignored z input. -/
inductive Program (σ : Type) where
  | forward (options : Host.Options) (logits labels : Tensor)
  | backward (context : Host.Context) (saved : σ) (upstream : Tensor) (zUpstream : Option Tensor)

def Result {σ : Type} : Program σ → Type
  | .forward _ _ _ => Host.ForwardReturn
  | .backward _ _ _ _ => List (Option Tensor)

/-- Ordinary tensors exposed by a returned Python value or saved context.
This contains no expected arithmetic value or shape-specific output formula. -/
def exports {σ : Type} : (program : Program σ) → Result program → List Tensor
  | .forward _ _ _, returned => [returned.losses,returned.zLosses,returned.context.logits,
      returned.context.lse,returned.context.labels,returned.context.inplaceLogits]
  | .backward _ _ _ _, returned => returned.filterMap id

/-- Metadata extracted from the one actual source kernel invocation. It
contains the complete logical views and parameters, including the internal
work allocation. It contains no arithmetic output or expected byte. -/
structure KernelABI (σ : Type) where
  before : σ
  after : σ
  parameters : Parameters
  count : Nat
  warps : Nat
  inputs : List Tensor
  outputs : List Tensor

/-- Each dispatch request retains actual pointers, allocation identities,
scalar words, shape/strides, grid, compiler/target, stream/rank/group/work.
The joint resident key also contains the complete entry arguments above. -/
structure CompleteABI {σ : Type} (program : Program σ) where
  entry : Program σ
  entryMatches : entry=program
  initial : σ
  kernel : KernelABI σ
  calls : List (InvocationGraphs.Request σ)

/-- This is the fixed interpreter, not an arbitrary callback and not Spec.
Its primitive kernel path is Host.launch → Grid → Source.forward/backward. -/
def run {σ : Type} (model : Model σ) : (program : Program σ) →
    World (Scalar model.mode) σ → Step (World (Scalar model.mode) σ) (Result program)
  | .forward options logits labels =>
      Host.forward (scalarInterpreter model.mode) model.runtime options logits labels
  | .backward context saved upstream zUpstream =>
      Host.pythonBackward (scalarInterpreter model.mode) model.runtime model.versions
        context saved upstream zUpstream

theorem python_success {α σ : Type} [Inhabited α] (M : FP32 α) (rt : Host.Runtime α σ)
    (versions : σ → Nat → Nat) (c : Host.Context) (saved : σ) (upstream : Tensor) (z : Option Tensor)
    (initial final : World α σ) (result : List (Option Tensor))
    (success : Host.pythonBackward M rt versions c saved upstream z initial=.ok result final) :
    ∃ output, Host.backwardEntry M rt versions c saved upstream z initial=.ok output final ∧
      result=Host.backwardTuple output := by
  cases h : Host.backwardEntry M rt versions c saved upstream z initial with
  | error message state =>
    simp only [Host.pythonBackward,h,CrossEntropy1.Host.bind] at success
    cases success
  | ok output state =>
    simp only [Host.pythonBackward,h,CrossEntropy1.Host.bind,Step.ok.injEq] at success
    exact ⟨output,by simpa only [success.2] using h,success.1.symm⟩

def kernelMetadata {σ : Type} (model : Model σ) : (program : Program σ) →
    (initial : World (Scalar model.mode) σ) → (result : Result program) →
    (final : World (Scalar model.mode) σ) → run model program initial=.ok result final → KernelABI σ
  | .forward options logits labels, initial, result, final, success =>
    let f := Classical.choice (Host.forward_success (scalarInterpreter model.mode) model.runtime
      options logits labels initial result final success)
    ⟨f.zAllocated.native,f.computed.native,f.parameters,f.count,
      CrossEntropy1.Host.warps f.parameters.block,[f.workingLogits,f.labels],[f.losses,f.lse,f.zlosses]⟩
  | .backward context saved upstream z, initial, result, final, success =>
    let h := python_success (scalarInterpreter model.mode) model.runtime model.versions context saved upstream z
      initial final result success
    let output := Classical.choose h
    let call := (Classical.choose_spec h).1
    let b := Classical.choice (Host.backward_success (scalarInterpreter model.mode) model.runtime context
      upstream output z initial final
      (Host.backward_entry_success (scalarInterpreter model.mode) model.runtime model.versions context saved
        upstream output z initial final call).2)
    ⟨b.prepared.native,b.computed.native,b.parameters,b.count,
      CrossEntropy1.Host.warps b.parameters.block,[context.logits,context.labels,context.lse,b.upstream],[b.work]⟩

def language (σ : Type) : SourceLanguage σ where
  Program := Program σ
  Syntax := ModuleAST
  Model := Model σ
  State model := World (Scalar model.mode) σ
  Result := Result
  ABI := CompleteABI
  Metadata _ := KernelABI σ
  ast _ := moduleAST
  sourceDigest _ := pythonDigest
  native _ state := state.native
  run := run
  coherent _ _ abi initial := abi.initial=initial.native
  abiMetadata _ abi := abi.kernel
  metadata := kernelMetadata

theorem forward_interpreter {σ : Type} (model : Model σ) (o : Host.Options) (x y : Tensor)
    (initial : World (Scalar model.mode) σ) :
    (language σ).run model (.forward o x y) initial=
      Host.forward (scalarInterpreter model.mode) model.runtime o x y initial := rfl

theorem backward_interpreter {σ : Type} (model : Model σ) (c : Host.Context)
    (saved : σ) (u : Tensor) (z : Option Tensor) (initial : World (Scalar model.mode) σ) :
    (language σ).run model (.backward c saved u z) initial=
      Host.pythonBackward (scalarInterpreter model.mode) model.runtime model.versions c saved u z initial := rfl

#axiomsClean forward_interpreter
#axiomsClean backward_interpreter
#axiomsClean python_success
#axiomsClean kernelMetadata
#specNonCircular run avoiding [Spec.loss,Spec.zLoss,Spec.gradient]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropyOps.Resident
