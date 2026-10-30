import Mathlib.Tactic.FinCases
import bench.tritonbench_g.triton_linear_activation.Proofs.KernelNativeModel
import VeriTile.Triton.Float.ProgramIO
import VeriTile.Triton.Float.LogicalProgramIO

/-! Public typed operand windows of the complete selected linear kernel.
The windows describe one program, matching the existing kernel theorem's scope.
Rows and columns wrap for input reads; output coordinates remain unwrapped.
No nonempty-shape or program-ID restriction is added. -/
namespace VeriTile.Bench.Repairs.TritonLinearActivation.LinearIO
open VeriTile.Triton FP.Structural KernelSource
noncomputable section
set_option autoImplicit false

def group (p : Parameters) (pid : Nat) := pid / KernelValues.width p
def groupRows (p : Parameters) (pid : Nat) :=
  min (KernelValues.gridRows p - group p pid * p.groupRows) p.groupRows
def rowIndex (p : Parameters) (pid : Nat) (i : Fin p.blockRows) :=
  (group p pid * p.groupRows + pid % groupRows p pid) * p.blockRows + i.val
def colIndex (p : Parameters) (pid : Nat) (j : Fin p.blockColumns) :=
  (pid % KernelValues.width p / groupRows p pid) * p.blockColumns + j.val
def active (p : Parameters) (pid : Nat) (i : TileIndex [p.blockRows,p.blockColumns]) : Prop :=
  rowIndex p pid i.1 < p.rows ∧ colIndex p pid i.2.1 < p.columns

def inputs (mode : Mode) (p : Parameters) : List IOChannel :=
  [⟨(dtype mode).toTileDType,[p.blockRows,p.inner],p.input⟩,
   ⟨(dtype mode).toTileDType,[p.blockColumns,p.inner],p.weight⟩,
   ⟨(dtype mode).toTileDType,[p.blockColumns],p.bias⟩]
def outputs (mode : Mode) (p : Parameters) : List IOChannel :=
  [⟨(dtype mode).toTileDType,[p.blockRows,p.blockColumns],p.output⟩,
   ⟨(dtype mode).toTileDType,[p.blockRows,p.blockColumns],p.saved⟩]
abbrev Inputs (α : Type) (mode : Mode) (p : Parameters) := ChannelValues α (inputs mode p)
abbrev Outputs (α : Type) (mode : Mode) (p : Parameters) := ChannelValues α (outputs mode p)

/-- Pointer dtypes remain FP32 or FP16 as selected by the source. Aliases
between inputs and outputs are admitted; the optional saved output has its
own source mask. There is no private scratch. -/
def memory (α : Type) (mode : Mode) (p : Parameters) (pid : Nat) : MemoryIO α where
  inputs := inputs mode p
  outputs := outputs mode p
  scratch := []
  buffers := [p.input,p.weight,p.bias,p.output,p.saved]
  read
    | ⟨0,_⟩, _, i => (rowIndex p pid i.1 % p.rows)*p.inputRowStride+i.2.1.val*p.inputInnerStride
    | ⟨1,_⟩, _, i => (colIndex p pid i.1 % p.columns)*p.weightColStride+i.2.1.val*p.weightInnerStride
    | ⟨2,_⟩, _, i => colIndex p pid i.1 % p.columns
  inputMask
    | ⟨0,_⟩, _, _ => True
    | ⟨1,_⟩, _, _ => True
    | ⟨2,_⟩, _, i => p.hasBias = Bool.true ∧ colIndex p pid i.1 % p.columns < p.columns
  write
    | ⟨0,_⟩, _, i => rowIndex p pid i.1*p.outputRowStride+colIndex p pid i.2.1*p.outputColStride
    | ⟨1,_⟩, _, i => rowIndex p pid i.1*p.savedRowStride+colIndex p pid i.2.1*p.savedColStride
  outputMask
    | ⟨0,_⟩, _, i => active p pid i
    | ⟨1,_⟩, _, i => p.saveInputs = Bool.true ∧ active p pid i
  scratchWrite := fun k => Fin.elim0 k
  scratchMask := fun k => Fin.elim0 k

def invocation {α : Type} (M : Algebra α) (D : KernelValues.Dot α)
    (mode : Mode) (p : Parameters) (pid : Nat) : Invocation α where
  kernel := source mode p
  backend := KernelValues.engine M D
  pids := fun _ => pid
  setPids := fun ax => ax == 0

def programIO {α β : Type} (M : Algebra β) (D : KernelValues.Dot β)
    (encode : α → β) (observe : β → α) (mode : Mode) (p : Parameters) (pid : Nat) : ProgramIO α β where
  toMemoryIO := memory α mode p pid
  schedule := fun _ => [invocation M D mode p pid]
  encode := encode
  observe := fun _ _ => observe

def realIO (mode : Mode) (p : Parameters) (pid : Nat) : ProgramIO ℝ ℝ :=
  programIO KernelRealArithmetic.algebra KernelRealArithmetic.dot id id mode p pid

/-- Independent dense mathematical formula over the declared operand tiles.
Only active outputs are observed. Every K coordinate participates exactly once. -/
def preactivation (mode : Mode) (p : Parameters) (values : Inputs ℝ mode p)
    (i : TileIndex [p.blockRows,p.blockColumns]) : ℝ :=
  (if p.hasBias then toFloat (dtype mode) (values ⟨2,by simp [inputs]⟩ (i.2.1,PUnit.unit)) else 0) +
    ∑ k : Fin p.inner,
      toFloat (dtype mode) (values ⟨0,by simp [inputs]⟩ (i.1,k,PUnit.unit)) *
      toFloat (dtype mode) (values ⟨1,by simp [inputs]⟩ (i.2.1,k,PUnit.unit))

def realTarget (mode : Mode) (p : Parameters) (values : Inputs ℝ mode p) : Outputs ℝ mode p
  | ⟨0,_⟩ => fun i => ofFloat (dtype mode) (realActivation p.activation (preactivation mode p values i))
  | ⟨1,_⟩ => fun i => ofFloat (dtype mode) (preactivation mode p values i)

/-- Source builds its exact expression graph. Observing that graph applies
the explicitly selected compiler policy and fixed instruction/libdevice model.
This does not assert equality between different contraction policies. -/
def nativeIO (profile : LocalProfile) (policy : KernelNativeModel.CompilerPolicy) (c : Fin 98)
    (mode : Mode) (p : Parameters) (pid : Nat) : ProgramIO Word KernelNativeModel.Term :=
  programIO KernelNativeModel.algebra (KernelNativeModel.dot c mode) KernelNativeModel.Term.raw
    (KernelNativeModel.evaluate profile policy) mode p pid

def nativeLeft (mode : Mode) (p : Parameters) (values : Inputs Word mode p)
    (row : Fin p.blockRows) (k : Nat) : KernelNativeModel.Term :=
  if hk : k < p.inner then
    .raw (toFloat (dtype mode) (values ⟨0,by simp [inputs]⟩ (row,⟨k,hk⟩,PUnit.unit)))
  else KernelValues.paddedZero KernelNativeModel.algebra mode

def nativeRight (mode : Mode) (p : Parameters) (values : Inputs Word mode p)
    (col : Fin p.blockColumns) (k : Nat) : KernelNativeModel.Term :=
  if hk : k < p.inner then
    .raw (toFloat (dtype mode) (values ⟨1,by simp [inputs]⟩ (col,⟨k,hk⟩,PUnit.unit)))
  else KernelValues.paddedZero KernelNativeModel.algebra mode

/-- Independent source-order native recurrence on logical operands. Matrix
instruction width, padding, literal words, casts, and contraction policy retain
the existing model; no Real-to-native numerical equality is asserted. -/
def nativePre (c : Fin 98) (mode : Mode) (p : Parameters) (pid : Nat)
    (values : Inputs Word mode p) (i : TileIndex [p.blockRows,p.blockColumns]) : KernelNativeModel.Term :=
  foldN (fun cur acc => .binary (some .fp32) .real .add acc
    (.cast (some .fp32) .fp32 .real
      (.dot (some .fp32) (dtype mode) (dtype mode) .fp32 {} (selectedInstruction c mode) p.blockInner
        (fun k => nativeLeft mode p values i.1 (cur*p.blockInner+k.val))
        (fun k => nativeRight mode p values i.2.1 (cur*p.blockInner+k.val)))))
    (ceilDiv p.inner p.blockInner)
    (if p.hasBias then .binary (some .fp32) .real .add (KernelValues.zero KernelNativeModel.algebra)
      (.cast (some .fp32) (dtype mode) .real
        (if colIndex p pid i.2.1 % p.columns < p.columns then
          .raw (toFloat (dtype mode) (values ⟨2,by simp [inputs]⟩ (i.2.1,PUnit.unit)))
        else KernelValues.paddedZero KernelNativeModel.algebra mode))
      else KernelValues.zero KernelNativeModel.algebra)

def nativeTarget (profile : LocalProfile) (policy : KernelNativeModel.CompilerPolicy) (c : Fin 98)
    (mode : Mode) (p : Parameters) (pid : Nat) (values : Inputs Word mode p) : Outputs Word mode p
  | ⟨0,_⟩ => fun i => ofFloat (dtype mode) (KernelNativeModel.castWord .real (dtype mode)
      (KernelNativeModel.evaluate profile policy
        (KernelValues.activation KernelNativeModel.algebra p.activation (nativePre c mode p pid values i))))
  | ⟨1,_⟩ => fun i => ofFloat (dtype mode) (KernelNativeModel.castWord .real (dtype mode)
      (KernelNativeModel.evaluate profile policy (nativePre c mode p pid values i)))

structure LogicalInputs (p : Parameters) where
  left : Fin p.blockRows → Fin p.inner → ℝ
  right : Fin p.blockColumns → Fin p.inner → ℝ
  bias : Fin p.blockColumns → ℝ

structure LogicalOutputs (p : Parameters) where
  result : Fin p.blockRows → Fin p.blockColumns → ℝ
  savedPreactivation : Fin p.blockRows → Fin p.blockColumns → ℝ

def logicalInputs (mode : Mode) (p : Parameters) : Inputs ℝ mode p ≃ LogicalInputs p where
  toFun xs := ⟨fun i k => toFloat (dtype mode) (xs ⟨0,by simp [inputs]⟩ (i,k,())),
    fun j k => toFloat (dtype mode) (xs ⟨1,by simp [inputs]⟩ (j,k,())),
    fun j => toFloat (dtype mode) (xs ⟨2,by simp [inputs]⟩ (j,()))⟩
  invFun xs := fun port => match port with
    | ⟨0,_⟩ => fun i => ofFloat (dtype mode) (xs.left i.1 i.2.1)
    | ⟨1,_⟩ => fun i => ofFloat (dtype mode) (xs.right i.1 i.2.1)
    | ⟨2,_⟩ => fun i => ofFloat (dtype mode) (xs.bias i.1)
  left_inv := by
    intro xs; funext port i; fin_cases port <;> cases mode <;> cases i with | mk a rest =>
      cases rest <;> rfl
  right_inv := by intro xs; cases xs; cases mode <;> rfl

def logicalOutputs (mode : Mode) (p : Parameters) : Outputs ℝ mode p ≃ LogicalOutputs p where
  toFun xs := ⟨fun i j => toFloat (dtype mode) (xs ⟨0,by simp [outputs]⟩ (i,j,())),
    fun i j => toFloat (dtype mode) (xs ⟨1,by simp [outputs]⟩ (i,j,()))⟩
  invFun xs := fun port => match port with
    | ⟨0,_⟩ => fun i => ofFloat (dtype mode) (xs.result i.1 i.2.1)
    | ⟨1,_⟩ => fun i => ofFloat (dtype mode) (xs.savedPreactivation i.1 i.2.1)
  left_inv := by
    intro xs; funext port i; fin_cases port <;> rcases i with ⟨a,b,u⟩ <;> cases u <;> cases mode <;> rfl
  right_inv := by intro xs; cases xs; cases mode <;> rfl

def logicalRealIO (mode : Mode) (p : Parameters) (pid : Nat) :
    LogicalProgramIO ℝ ℝ (LogicalInputs p) (LogicalOutputs p) where
  core := realIO mode p pid
  inputs := logicalInputs mode p
  outputs := logicalOutputs mode p

end
end VeriTile.Bench.Repairs.TritonLinearActivation.LinearIO
