import bench.repairs.tritonbench_g.fused_rwkv6_kernel_scalar_abi_checked.CandidateKernels
import VeriTile.Triton.Float.LogicalProgramIO
import VeriTile.Triton.Float.ProgramEquivalenceSyntax
import bench.repairs.tritonbench_g.fused_rwkv6_kernel_scalar_abi_checked.Proofs.Observations
import VeriTile.Triton.Float.ObservedInputCellsIO

namespace VeriTile.Bench.Repairs.FusedRwkv6KernelScalarABIChecked.Interface
open VeriTile Triton FP.Structural FusedRwkv6Kernel
noncomputable section
attribute [local instance] Classical.propDecidable

/-- Logical tensors keep the original head pitches; no flattened capacity or
extra pitch inequality is substituted for the original mathematical domain. -/
def inputPorts (c : Shape) : List IOChannel :=
  [⟨.real,[c.Z,c.T,c.K],"q"⟩,⟨.real,[c.Z,c.T,c.K],"k"⟩,
   ⟨.real,[c.Z,c.T,c.V],"v"⟩,⟨.real,[c.Z,c.T,c.K],"w"⟩,
   ⟨.real,[c.H,c.K],"u"⟩,⟨.real,[c.Z,c.K,c.V],"h0"⟩]
def outputPorts (c : Shape) : List IOChannel :=
  [⟨.real,[c.NK,c.Z,c.T,c.V],"o"⟩,⟨.real,[c.Z,c.K,c.V],"ht"⟩]
abbrev Inputs (α : Type) (c : Shape) := ChannelValues α (inputPorts c)

def inputRead (c : Shape) : (i : Fin (inputPorts c).length) → TileIndex (inputPorts c)[i].shape → Nat
  | ⟨0,_⟩,j => inputKey c j.1.val j.2.1.val j.2.2.1.val
  | ⟨1,_⟩,j => inputKey c j.1.val j.2.1.val j.2.2.1.val
  | ⟨2,_⟩,j => inputValue c j.1.val j.2.1.val j.2.2.1.val
  | ⟨3,_⟩,j => inputKey c j.1.val j.2.1.val j.2.2.1.val
  | ⟨4,_⟩,j => j.1.val*c.K+j.2.1.val
  | ⟨5,_⟩,j => stateAddress c j.1.val j.2.1.val j.2.2.1.val

def outputWrite (c : Shape) : (o : Fin (outputPorts c).length) → TileIndex (outputPorts c)[o].shape → Nat
  | ⟨0,_⟩,j => partialAddress c j.2.1.val j.1.val j.2.2.1.val j.2.2.2.1.val
  | ⟨1,_⟩,j => stateAddress c j.1.val j.2.1.val j.2.2.1.val

def invocation {α : Type} (M : Algebra α) (c : Shape) (scale : α)
    (p : FusedRwkv6Kernel.Program) : Invocation α where
  kernel := kernel c
  backend := FP.ProfiledRewrite.engine M
  arguments := [⟨"scale",.real,[],fun _ => scale⟩]
  clearRegisters := Bool.true
  pids := fun ax => if ax=0 then p.iv else if ax=1 then p.ik else if ax=2 then p.z else 0
  numPids := some (fun ax => if ax=0 then c.NV else if ax=1 then c.NK else if ax=2 then c.Z else 0)

def programIO {α : Type} (M : Algebra α) (c : Shape) (scale : α) : ProgramIO α α where
  inputs := inputPorts c
  outputs := outputPorts c
  scratch := []
  buffers := ["q","k","v","w","u","h0","o","ht"]
  read := fun i _ => inputRead c i
  inputMask := fun i _ _ => Launches c ∧ (i.val≠5 ∨ c.initial=Bool.true)
  write := fun o _ => outputWrite c o
  outputMask := fun o _ _ => o.val=0 ∨ c.final=Bool.true
  scratchWrite := fun i => Fin.elim0 i
  scratchMask := fun i => Fin.elim0 i
  schedule := fun _ => if Launches c then (programs c).map (invocation M c scale) else []
  encode := id
  observe := fun _ _ => id

/-- Logical input arrays only; out-of-shape entries are never active operands. -/
def logicalInputs {α : Type} [Inhabited α] (c : Shape) (xs : Inputs α c) : Input α where
  q := fun z t e => if h : z<c.Z ∧ t<c.T ∧ e<c.K then xs (0 : Fin 6) (⟨z,h.1⟩,⟨t,h.2.1⟩,⟨e,h.2.2⟩,()) else default
  k := fun z t e => if h : z<c.Z ∧ t<c.T ∧ e<c.K then xs (1 : Fin 6) (⟨z,h.1⟩,⟨t,h.2.1⟩,⟨e,h.2.2⟩,()) else default
  v := fun z t f => if h : z<c.Z ∧ t<c.T ∧ f<c.V then xs (2 : Fin 6) (⟨z,h.1⟩,⟨t,h.2.1⟩,⟨f,h.2.2⟩,()) else default
  w := fun z t e => if h : z<c.Z ∧ t<c.T ∧ e<c.K then xs (3 : Fin 6) (⟨z,h.1⟩,⟨t,h.2.1⟩,⟨e,h.2.2⟩,()) else default
  u := fun z e => if h : z%c.H<c.H ∧ e<c.K then xs (4 : Fin 6) (⟨z%c.H,h.1⟩,⟨e,h.2⟩,()) else default
  h0 := fun z e f => if h : z<c.Z ∧ e<c.K ∧ f<c.V then xs (5 : Fin 6) (⟨z,h.1⟩,⟨e,h.2.1⟩,⟨f,h.2.2⟩,()) else default

/-- Independent scalar recurrence, retaining source order and padded lanes. -/
def formula {α : Type} [Inhabited α] (ops : Ops α) (c : Shape) (plan : Plans c) (scale : α)
    (xs : Inputs α c) : ChannelValues α (outputPorts c)
  | ⟨0,_⟩,j => ops.storeCast (scalarPartial ops c plan (logicalInputs c xs) scale
      j.2.1.val j.1.val (visit c j.2.2.1.val) j.2.2.2.1.val)
  | ⟨1,_⟩,j => ops.storeCast (scalarState ops c (logicalInputs c xs)
      j.1.val j.2.1.val j.2.2.1.val c.T)

end
end VeriTile.Bench.Repairs.FusedRwkv6KernelScalarABIChecked.Interface

namespace VeriTile.Bench.Repairs.FusedRwkv6KernelScalarABIChecked.Pair
open VeriTile Triton FP.Structural FusedRwkv6Kernel
noncomputable section
attribute [local instance] Classical.propDecidable
def invocation {α : Type} (M : Algebra α) (c : Shape) (scale : α)
    (p : FusedRwkv6Kernel.Program) : Invocation α :=
  {Interface.invocation M c scale p with kernel := earlyUKernel c}

def variant {α : Type} (M : Algebra α) (c : Shape) (scale : α) :
    ProgramIO.Variant (Interface.programIO M c scale) where
  schedule := fun _ => if Launches c then (programs c).map (invocation M c scale) else []
  observe := (Interface.programIO M c scale).observe

def original (plan : Native.ReductionPlan) (c : Shape) (scale : Native.Term) :=
  ProgramIO.Implementation.original (Interface.programIO (Native.algebra plan) c scale) (fun _ => True)

def earlyU (plan : Native.ReductionPlan) (c : Shape) (scale : Native.Term) :=
  ProgramIO.Implementation.ofVariant (variant (Native.algebra plan) c scale) (fun _ => True)

end
end VeriTile.Bench.Repairs.FusedRwkv6KernelScalarABIChecked.Pair

namespace VeriTile.Bench.Repairs.FusedRwkv6KernelScalarABIChecked.Interface
open VeriTile Triton FP.Structural FusedRwkv6Kernel
noncomputable section

structure TensorInputs (α : Type) (c : Shape) where
  q : Fin (c.Z) → Fin (c.T) → Fin (c.K) → α
  k : Fin (c.Z) → Fin (c.T) → Fin (c.K) → α
  v : Fin (c.Z) → Fin (c.T) → Fin (c.V) → α
  w : Fin (c.Z) → Fin (c.T) → Fin (c.K) → α
  u : Fin (c.H) → Fin (c.K) → α
  initialState : Fin (c.Z) → Fin (c.K) → Fin (c.V) → α

def inputCoordinates (α : Type) (c : Shape) : ChannelValues α (inputPorts c) ≃ TensorInputs α c where
  toFun := fun xs => ⟨Values.curryEquiv α .real [c.Z, c.T, c.K] (xs (0 : Fin 6)),
    Values.curryEquiv α .real [c.Z, c.T, c.K] (xs (1 : Fin 6)),
    Values.curryEquiv α .real [c.Z, c.T, c.V] (xs (2 : Fin 6)),
    Values.curryEquiv α .real [c.Z, c.T, c.K] (xs (3 : Fin 6)),
    Values.curryEquiv α .real [c.H, c.K] (xs (4 : Fin 6)),
    Values.curryEquiv α .real [c.Z, c.K, c.V] (xs (5 : Fin 6))⟩
  invFun := fun xs => (Fin.cases ((Values.curryEquiv α .real [c.Z, c.T, c.K]).symm xs.q) (Fin.cases ((Values.curryEquiv α .real [c.Z, c.T, c.K]).symm xs.k) (Fin.cases ((Values.curryEquiv α .real [c.Z, c.T, c.V]).symm xs.v) (Fin.cases ((Values.curryEquiv α .real [c.Z, c.T, c.K]).symm xs.w) (Fin.cases ((Values.curryEquiv α .real [c.H, c.K]).symm xs.u) (Fin.cases ((Values.curryEquiv α .real [c.Z, c.K, c.V]).symm xs.initialState) (fun i => Fin.elim0 i)))))))
  left_inv := by
    intro xs
    funext i
    fin_cases i
    · exact (Values.curryEquiv α .real [c.Z, c.T, c.K]).symm_apply_apply (xs (0 : Fin 6))
    · exact (Values.curryEquiv α .real [c.Z, c.T, c.K]).symm_apply_apply (xs (1 : Fin 6))
    · exact (Values.curryEquiv α .real [c.Z, c.T, c.V]).symm_apply_apply (xs (2 : Fin 6))
    · exact (Values.curryEquiv α .real [c.Z, c.T, c.K]).symm_apply_apply (xs (3 : Fin 6))
    · exact (Values.curryEquiv α .real [c.H, c.K]).symm_apply_apply (xs (4 : Fin 6))
    · exact (Values.curryEquiv α .real [c.Z, c.K, c.V]).symm_apply_apply (xs (5 : Fin 6))
  right_inv := by intro xs; cases xs; rfl

structure TensorOutputs (α : Type) (c : Shape) where
  partials : Fin (c.NK) → Fin (c.Z) → Fin (c.T) → Fin (c.V) → α
  finalState : Fin (c.Z) → Fin (c.K) → Fin (c.V) → α

def outputCoordinates (α : Type) (c : Shape) : ChannelValues α (outputPorts c) ≃ TensorOutputs α c where
  toFun := fun xs => ⟨Values.curryEquiv α .real [c.NK, c.Z, c.T, c.V] (xs (0 : Fin 2)),
    Values.curryEquiv α .real [c.Z, c.K, c.V] (xs (1 : Fin 2))⟩
  invFun := fun xs => (Fin.cases ((Values.curryEquiv α .real [c.NK, c.Z, c.T, c.V]).symm xs.partials) (Fin.cases ((Values.curryEquiv α .real [c.Z, c.K, c.V]).symm xs.finalState) (fun i => Fin.elim0 i)))
  left_inv := by
    intro xs
    funext i
    fin_cases i
    · exact (Values.curryEquiv α .real [c.NK, c.Z, c.T, c.V]).symm_apply_apply (xs (0 : Fin 2))
    · exact (Values.curryEquiv α .real [c.Z, c.K, c.V]).symm_apply_apply (xs (1 : Fin 2))
  right_inv := by intro xs; cases xs; rfl

/-- Total mathematical extension of the named finite inputs. The bijection
retains optional and inactive entries; source masks are unchanged. -/
def tensorData [Inhabited α] (c : Shape) (xs : TensorInputs α c) : Input α :=
  logicalInputs c ((inputCoordinates α c).symm xs)

def logicalIO (c : Shape) (scale : ℝ) :
    LogicalProgramIO ℝ ℝ (TensorInputs ℝ c) (TensorOutputs ℝ c) where
  core := programIO Real.algebra c scale
  inputs := inputCoordinates ℝ c
  outputs := outputCoordinates ℝ c
end
end VeriTile.Bench.Repairs.FusedRwkv6KernelScalarABIChecked.Interface

namespace VeriTile.Bench.Repairs.FusedRwkv6KernelScalarABIChecked.Reference
open VeriTile Triton FP.Structural FusedRwkv6Kernel
noncomputable section

def real (c : Shape) (scale : ℝ) (xs : Interface.Inputs ℝ c) :
    ChannelValues ℝ (Interface.outputPorts c)
  | ⟨0,_⟩,j => Real.partialFormula c (Interface.logicalInputs c xs) scale
      j.2.1.val j.1.val (visit c j.2.2.1.val) j.2.2.2.1.val
  | ⟨1,_⟩,j => realState c (Interface.logicalInputs c xs) j.1.val j.2.1.val j.2.2.1.val c.T

#specNonCircular real avoiding [kernel,Source.launch,FP.Structural.exec,FusedRwkv6Kernel.runProgram]
end
end VeriTile.Bench.Repairs.FusedRwkv6KernelScalarABIChecked.Reference

namespace VeriTile.Bench.Repairs.FusedRwkv6KernelScalarABIChecked.Runtime
open VeriTile Triton FP.Structural FusedRwkv6Kernel
noncomputable section
attribute [local instance] Classical.propDecidable

/-- The caller's scalar ABI selects a complete, unchanged Python function body.
Integer 1 has Triton's native constexpr specialization; floating 1.0 does not. -/
inductive Scalar (α : Type) where
  | fp32 (value : α)
  | i32 (value : Int)
  | i64 (value : Int)
  | u64 (value : Nat)
  | one

def Scalar.value {α : Type} (M : Algebra α) : Scalar α → α
  | .fp32 value => value
  | .i32 value | .i64 value => M.fromInt (some .fp32) value
  | .u64 value => M.fromNat (some .fp32) value
  | .one => M.literal (some .fp32) .real 1

def Scalar.Fits {α : Type} : Scalar α → Prop
  | .fp32 _ | .one => True
  | .i32 value => -(2^31 : Int) ≤ value ∧ value < 2^31
  | .i64 value => -(2^63 : Int) ≤ value ∧ value < 2^63
  | .u64 value => value < 2^64

/-- Scalar classification in Triton 3.7.1's `specialize.cc`. The existing
branch-local ScalarABI contract supplies representability when a launch occurs. -/
def classify {α : Type} (floatValue : BitVec 64 → α) : PythonNumber → Scalar α
  | .float bits => .fp32 (floatValue bits)
  | .integer value =>
    if value=1 then .one
    else if -(2^31 : Int) ≤ value ∧ value < 2^31 then .i32 value
    else if value < (2^63 : Int) then .i64 value
    else .u64 value.toNat

def source {α : Type} (c : Shape) : Scalar α → ComputeKernel
  | .fp32 _ => kernel c
  | .i32 value => kernelI32 c value
  | .i64 value => kernelI64 c value
  | .u64 value => kernelU64 c value
  | .one => kernelOne c

def candidate {α : Type} (c : Shape) : Scalar α → ComputeKernel
  | .fp32 _ => earlyUKernel c
  | .i32 value => earlyUKernelI32 c value
  | .i64 value => earlyUKernelI64 c value
  | .u64 value => earlyUKernelU64 c value
  | .one => earlyUKernelOne c

/-- The auxiliary real register makes the arithmetic operand available to the
common proof interface. Integer bodies use their own typed call argument and
perform the conversion in the source expression; they do not read this register. -/
def invocation {α : Type} (M : Algebra α) (c : Shape) (scalar : Scalar α)
    (early : Bool) (p : FusedRwkv6Kernel.Program) : Invocation α :=
  {Interface.invocation M c (scalar.value M) p with
    kernel := if early then candidate c scalar else source c scalar}

def schedule {α : Type} (M : Algebra α) (c : Shape) (scalar : Scalar α)
    (early : Bool) : List (Invocation α) :=
  if Launches c then (programs c).map (invocation M c scalar early) else []

/-- The complete original IO layout and masks are shared without alteration;
only the actual scalar-specialized source invocations change. -/
def programIO {α : Type} (M : Algebra α) (c : Shape) (scalar : Scalar α) : ProgramIO α α :=
  {Interface.programIO M c (scalar.value M) with schedule := fun _ => schedule M c scalar Bool.false}

def variant {α : Type} (M : Algebra α) (c : Shape) (scalar : Scalar α) :
    ProgramIO.Variant (programIO M c scalar) where
  schedule := fun _ => schedule M c scalar Bool.true
  observe := (programIO M c scalar).observe

def nativeScalar (binding : Native.ScalarBinding) (raw : PythonNumber) : Scalar Native.Term :=
  classify (fun bits => Native.scalar binding (.float bits)) raw

def original (plan : Native.ReductionPlan) (binding : Native.ScalarBinding)
    (raw : PythonNumber) (c : Shape) :=
  ProgramIO.Implementation.original
    (programIO (Native.scalarAlgebra plan binding) c (nativeScalar binding raw)) (fun _ => True)

def earlyU (plan : Native.ReductionPlan) (binding : Native.ScalarBinding)
    (raw : PythonNumber) (c : Shape) :=
  ProgramIO.Implementation.ofVariant
    (variant (Native.scalarAlgebra plan binding) c (nativeScalar binding raw)) (fun _ => True)

def logicalIO (c : Shape) (scalar : Scalar ℝ) :
    LogicalProgramIO ℝ ℝ (Interface.TensorInputs ℝ c) (Interface.TensorOutputs ℝ c) :=
  {Interface.logicalIO c (scalar.value Real.algebra) with
    core := programIO Real.algebra c scalar}
end
end VeriTile.Bench.Repairs.FusedRwkv6KernelScalarABIChecked.Runtime
