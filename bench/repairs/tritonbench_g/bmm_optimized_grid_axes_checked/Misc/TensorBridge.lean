import bench.repairs.tritonbench_g.bmm_optimized_grid_axes_checked.Misc.RawCompatibility
import bench.repairs.tritonbench_g.bmm_optimized_grid_axes_checked.Proofs.Tensor

namespace VeriTile.Bench.Repairs.BmmOptimizedGridAxesChecked.TensorBridge
open VeriTile.Triton FP.Structural
open BmmOptimized TensorAPI
noncomputable section

alias kernel_real := RawCompatibility.kernel_real
alias kernel_native := RawCompatibility.kernel_native

structure Outcome (α : Type) where
  tensor : Tensor
  memory : Physical α
  restoredDevice : Nat
  sourceState : State α

def outcome {α : Type} [Inhabited α] (a : Invocation) (memory : Physical α)
    (final : State α) : Outcome α :=
  ⟨returned a,publish a memory final,a.callerDevice,final⟩

/-- Checked wrapper path followed by interpretation of the actual source grid.
The empty output has an empty CTA list; no arithmetic launch is introduced. -/
def execute {α : Type} [Inhabited α] (a : Invocation) (c : Config)
    (o : Scalar α) (rest : Algebra α) (ps : List Program) (memory : Physical α)
    (exterior : RegionName → Nat → FP.Structural.Cell α) : Except String (Outcome α) := do
  checkInputs a.metadata
  match Bridge.grid a.shape c o rest ps (initial a memory exterior) with
  | none=>.error "source execution"
  | some final=>.ok (outcome a memory final)

structure TensorFacts (a : Invocation) (t : Tensor) : Prop where
  object : t.object=a.outputObject
  storage : t.storageId=a.outputStorage
  freshLeft : t.storageId≠a.left.storageId
  freshRight : t.storageId≠a.right.storageId
  shape : t.metadata.shape=[a.shape.batch,a.shape.m,a.shape.n]
  dtype : t.metadata.dtype=.floating .fp32
  layout : t.metadata.layout=.strided
  contiguous : t.metadata.isContiguous=Bool.true
  resolved : t.metadata.isNeg=Bool.false
  device : t.metadata.device=⟨"cuda",a.left.metadata.device.index⟩
  strides : (t.batchStride,t.rowStride,t.columnStride)=Host.outputStrides a.shape
  offset : t.storageOffset=0
  pointer : t.metadata.dataPtr.toNat=a.outputBase

theorem tensorFacts (a : Invocation) (ordinary : TensorAPI.Ordinary a) : TensorFacts a (returned a) := by
  constructor
  all_goals try rfl
  · exact ordinary.freshLeft
  · exact ordinary.freshRight
  · simpa [returned,BitVec.toNat_ofNat] using Nat.mod_eq_of_lt ordinary.outputPointer

structure RealClaim (a : Invocation) (before : Physical ℝ)
    (exterior : RegionName → Nat → FP.Structural.Cell ℝ) (result : Outcome ℝ) : Prop where
  tensor : TensorFacts a result.tensor
  device : result.restoredDevice=a.callerDevice
  values : ∀ x,a.shape.Active x →
    read result.memory result.tensor x.batch x.row x.column=product a before x
  frame : ∀ allocation off,allocation≠a.outputStorage ∨ a.shape.size≤off →
    result.memory allocation off=before allocation off
  sourceFrame : Bridge.Frame a.shape (initial a before exterior) result.sourceState

theorem checked_real (a : Invocation) (admitted : InputContract a.metadata)
    (ordinary : TensorAPI.Ordinary a) (c : Config) (rest : Algebra ℝ)
    (ps : List Program) (schedule : ps.Perm (programs a.shape c)) (before : Physical ℝ)
    (exterior : RegionName → Nat → FP.Structural.Cell ℝ) :
    ∃ result,execute a c realScalar rest ps before exterior=.ok result ∧
      RealClaim a before exterior result ∧ IndexABI a.shape ∧
      (Live a.shape → a.shape.batch≤65535 ∧ a.shape.gy c≤65535) := by
  obtain ⟨final,run,values,frame⟩:=LegacyIO.complete a.shape c realScalar rest ps schedule
    (Host.prepare id a.call (caller a before)) (initial a before exterior)
    (initial_related a before exterior)
  refine ⟨outcome a before final,?_,?_,admitted.original.indices,?_⟩
  · simp [execute,(checkInputs_iff a.metadata).2 admitted,run]
    rfl
  · refine ⟨tensorFacts a ordinary,rfl,?_,publish_frame a before final,frame⟩
    intro x hx
    change read (publish a before final) (returned a) x.batch x.row x.column=product a before x
    rw [read_returned a before final x hx,values x hx,Cell.read_same,real_target a admitted before x hx]
  · exact fun live=>launch_axes a.metadata admitted live c

#axiomsClean checked_real
#specNonCircular RealClaim avoiding [execute,Bridge.grid,Execution.exec]
end
end VeriTile.Bench.Repairs.BmmOptimizedGridAxesChecked.TensorBridge

namespace VeriTile.Bench.Repairs.BmmOptimizedGridAxesChecked.TensorBridge
open VeriTile.Triton FP.Structural
open BmmOptimized TensorAPI
noncomputable section

structure ByteOutcome where
  tensor : Tensor
  memory : PhysicalBytes
  restoredDevice : Nat
  sourceState : State Native.Word

def byteOutcome (a : Invocation) (before : PhysicalBytes) (final : State Native.Word) : ByteOutcome :=
  ⟨returned a,publishBytes a before final,a.callerDevice,final⟩

def executeNative (a : Invocation) (c : Config) (ops : Native.Instructions)
    (rest : Algebra Native.Word) (ps : List Program) (before : PhysicalBytes)
    (exterior : RegionName → Nat → FP.Structural.Cell Native.Word) : Except String ByteOutcome := do
  checkInputs a.metadata
  match Bridge.grid a.shape c (Native.scalar ops) rest ps (initial a (loadWords before) exterior) with
  | none=>.error "source execution"
  | some final=>.ok (byteOutcome a before final)

structure NativeClaim (a : Invocation) (target : Native.Instructions) (before : PhysicalBytes)
    (exterior : RegionName → Nat → FP.Structural.Cell Native.Word) (result : ByteOutcome) : Prop where
  tensor : TensorFacts a result.tensor
  device : result.restoredDevice=a.callerDevice
  bytes : ∀ x,a.shape.Active x → ∀ i,
    tensorByte result.memory result.tensor x.batch x.row x.column i=
      chain target a (loadWords before) x (a.shape.iterations*32) i
  frame : ∀ allocation off,allocation≠a.outputStorage ∨ 4*a.shape.size≤off →
    result.memory allocation off=before allocation off
  sourceFrame : Bridge.Frame a.shape (initial a (loadWords before) exterior) result.sourceState

/-- Every output byte is the independent padded scalar fma.rn.f32 chain.
The only numerical comparison premise is equality of the scalar instruction;
no dot, block, tensor, successful-run or final-result equality is assumed. -/
theorem checked_native (a : Invocation) (admitted : InputContract a.metadata)
    (ordinary : TensorAPI.Ordinary a) (c : Config) (actual target : Native.Instructions)
    (fma : ∀ l r carry,actual.fmaRN l r carry=target.fmaRN l r carry)
    (rest : Algebra Native.Word) (ps : List Program) (schedule : ps.Perm (programs a.shape c))
    (before : PhysicalBytes) (exterior : RegionName → Nat → FP.Structural.Cell Native.Word) :
    ∃ result,executeNative a c actual rest ps before exterior=.ok result ∧
      NativeClaim a target before exterior result ∧ IndexABI a.shape ∧
      (Live a.shape → a.shape.batch≤65535 ∧ a.shape.gy c≤65535) := by
  obtain ⟨final,run,values,frame⟩:=LegacyIO.complete a.shape c (Native.scalar actual) rest ps schedule
    (Host.prepare id a.call (caller a (loadWords before))) (initial a (loadWords before) exterior)
    (initial_related a (loadWords before) exterior)
  refine ⟨byteOutcome a before final,?_,?_,admitted.original.indices,?_⟩
  · simp [executeNative,(checkInputs_iff a.metadata).2 admitted,run]
    rfl
  · refine ⟨tensorFacts a ordinary,rfl,?_,bytes_frame a before final,frame⟩
    intro x hx i
    change tensorByte (publishBytes a before final) (returned a) x.batch x.row x.column i=_
    rw [bytes_output a before final x hx i,values x hx,Cell.read_same,
      native_target actual a admitted (loadWords before) x hx,
      chain_equiv actual target fma a (loadWords before) x]
  · exact fun live=>launch_axes a.metadata admitted live c

#axiomsClean checked_native
#specNonCircular NativeClaim avoiding [executeNative,Bridge.grid,Execution.exec]
end
end VeriTile.Bench.Repairs.BmmOptimizedGridAxesChecked.TensorBridge
