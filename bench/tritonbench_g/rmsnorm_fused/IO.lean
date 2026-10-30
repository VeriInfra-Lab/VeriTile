import bench.tritonbench_g.rmsnorm_fused.Kernels
import bench.tritonbench_g.rms_norm_triton.Proofs.Mathematical_WholeMath
import VeriTile.Triton.Float.LogicalProgramIO
import Mathlib.Logic.Equiv.Fin.Basic

/-! Shared finite IO windows and runtime ABI for the complete row grid.
Both public entries execute the canonical two-loop FP32 source. The Structural
cell carrier is Real; FP32 precision remains explicit in the source AST. -/
namespace VeriTile.Bench.Repairs.RmsnormFused.Interface
open VeriTile Triton FP.Structural
noncomputable section

def inputs (shape : Shape) : List IOChannel :=
  [⟨.real, [shape.rows*shape.width], "X"⟩, ⟨.real, [shape.width], "W"⟩]
def outputs (shape : Shape) : List IOChannel :=
  [⟨.real, [shape.rows*shape.width], "Y"⟩]
abbrev Inputs (α : Type) (shape : Shape) := ChannelValues α (inputs shape)
abbrev Outputs (α : Type) (shape : Shape) := ChannelValues α (outputs shape)

def inputOffset (shape : Shape) : (i : Fin (inputs shape).length) →
    TileIndex (inputs shape)[i].shape → Nat
  | ⟨0, _⟩ => fun j => j.1.val
  | ⟨1, _⟩ => fun j => j.1.val
  | ⟨i+2, h⟩ => False.elim (by simp [inputs] at h)
def outputIndex (shape : Shape) : (o : Fin (outputs shape).length) →
    TileIndex (outputs shape)[o].shape → Fin (shape.rows*shape.width)
  | ⟨0, _⟩ => fun j => j.1
  | ⟨i+1, h⟩ => False.elim (by simp [outputs] at h)

def memory (α : Type) (shape : Shape) : MemoryIO α where
  inputs := inputs shape
  outputs := outputs shape
  scratch := []
  buffers := ["X","W","Y"]
  read := fun i _ j => inputOffset shape i j
  inputMask := fun _ _ _ => True
  write := fun o _ j => (outputIndex shape o j).val
  outputMask := fun _ _ _ => True
  scratchWrite := fun k => Fin.elim0 k
  scratchMask := fun k => Fin.elim0 k

/-- Every launch installs the actual runtime epsilon argument, fresh local
registers, and its row PID. No epsilon constant specialization is introduced. -/
def invocation {β : Type} (A : Algebra β) (shape : Shape) (epsilon : β)
    (row : Nat) : Invocation β where
  kernel := source "X" "Y" "W" shape.width shape.block
  backend := A
  arguments := [⟨"eps", .real, [], fun _ => epsilon⟩]
  clearRegisters := Bool.true
  pids := fun ax => if ax=0 then row else 0
  numPids := some (fun ax => if ax=0 then shape.rows else 1)

def programIO {α β : Type} (A : Algebra β) (shape : Shape) (epsilon : β)
    (rows : List Nat) (encode : α → β) (observe : Nat → β → α) : ProgramIO α β :=
  { memory α shape with
    schedule := fun _ => rows.map (invocation A shape epsilon)
    encode := encode
    observe := fun o j => observe (outputIndex shape o j).val }

def input {α : Type} {shape : Shape} (values : Inputs α shape)
    (i : Fin (shape.rows*shape.width)) : α :=
  values ⟨0, by simp [inputs]⟩ (i,PUnit.unit)
def weight {α : Type} {shape : Shape} (values : Inputs α shape)
    (i : Fin shape.width) : α := values ⟨1, by simp [inputs]⟩ (i,PUnit.unit)
def inputAt {α : Type} [Inhabited α] {shape : Shape} (values : Inputs α shape) (i : Nat) : α :=
  if h : i<shape.rows*shape.width then input values ⟨i,h⟩ else default
def weightAt {α : Type} [Inhabited α] {shape : Shape} (values : Inputs α shape) (i : Nat) : α :=
  if h : i<shape.width then weight values ⟨i,h⟩ else default

def result {α : Type} {shape : Shape} (ys : Fin (shape.rows*shape.width) → α) : Outputs α shape
  | ⟨0, _⟩ => fun j => ys j.1
  | ⟨i+1, h⟩ => False.elim (by simp [outputs] at h)

/-- Row-major coordinates are a bijection even when a dimension is empty. -/
def matrixValues (rows width : Nat) :
    (Fin (rows*width) → ℝ) ≃ (Fin rows → Fin width → ℝ) :=
  (Equiv.arrowCongr finProdFinEquiv.symm (Equiv.refl ℝ)).trans (Equiv.curry _ _ _)

/-- Complete X matrix and W vector, without losing either channel. -/
def logicalInputs (shape : Shape) : Inputs ℝ shape ≃
    ((Fin shape.rows → Fin shape.width → ℝ) × (Fin shape.width → ℝ)) :=
  (ChannelValues.consEquiv ℝ _ _).trans (Equiv.prodCongr
    ((Values.vectorEquiv ℝ .real _).trans (matrixValues shape.rows shape.width))
    ((ChannelValues.singleEquiv ℝ _).trans (Values.vectorEquiv ℝ .real _)))

/-- Every result cell of Y is one logical matrix coordinate. -/
def logicalOutputs (shape : Shape) : Outputs ℝ shape ≃
    (Fin shape.rows → Fin shape.width → ℝ) :=
  (ChannelValues.singleEquiv ℝ _).trans
    ((Values.vectorEquiv ℝ .real _).trans (matrixValues shape.rows shape.width))

/-- The complete original source and row schedule under logical coordinates. -/
def logicalIO (shape : Shape) (epsilon : ℝ) (rows : List Nat) :
    LogicalProgramIO ℝ ℝ ((Fin shape.rows → Fin shape.width → ℝ) × (Fin shape.width → ℝ))
      (Fin shape.rows → Fin shape.width → ℝ) where
  core := programIO VeriTile.Bench.TritonBenchG.RmsNormTriton.Whole.realAlgebra
    shape epsilon rows id (fun _ => id)
  inputs := logicalInputs shape
  outputs := logicalOutputs shape

end
end VeriTile.Bench.Repairs.RmsnormFused.Interface
