import VeriTile.Triton.Float.LogicalProgramIO
import bench.tritonbench_g.iv_dependent_matmul.Proofs.Interface
import VeriTile.Triton.Float.InputBindingIO

namespace VeriTile.Bench.Repairs.IvDependentMatmulDotShape.IO
open VeriTile.Triton FP.Structural
open TritonBenchG.IvDependentMatmul FPValue Whole
noncomputable section

/-- Original raw allocation geometry. Native exact FP32 input-cell tags are
expressed by the shared `ImplementsCells` judgment, not hidden in this layout. -/
structure Layout (g : Kernel.Geometry) where
  freshA : g.C≠g.A
  freshB : g.C≠g.B
  (baseA baseB baseC : Nat)
  physicalFreshA : g.rows*g.inner=0 ∨ g.rows*g.columns=0 ∨
    baseA+4*(g.rows*g.inner)≤baseC ∨ baseC+4*(g.rows*g.columns)≤baseA
  physicalFreshB : g.inner*g.columns=0 ∨ g.rows*g.columns=0 ∨
    baseB+4*(g.inner*g.columns)≤baseC ∨ baseC+4*(g.rows*g.columns)≤baseB
  alignA : baseA%4=0
  alignB : baseB%4=0
  alignC : baseC%4=0
  inputAddressable : baseA+4*(g.rows*g.inner)≤2^64
  weightAddressable : baseB+4*(g.inner*g.columns)≤2^64
  outputAddressable : baseC+4*(g.rows*g.columns)≤2^64

abbrev inputs (g : Kernel.Geometry) : List IOChannel :=
  [⟨.fp32,[g.rows,g.inner],g.A⟩,⟨.fp32,[g.inner,g.columns],g.B⟩]
abbrev outputs (g : Kernel.Geometry) : List IOChannel :=
  [⟨.fp32,[g.rows,g.columns],g.C⟩]

def invocation (M : Algebra α) (graph : Graph) (g : Kernel.Geometry) (pid : Nat) : Invocation α where
  kernel := Kernel.quotation g.schedule graph g.request.parameters
  backend := M
  pids := fun ax => if ax=0 then pid else 0

/-- Actual complete selected grid. Both inputs and the output use contiguous
FP32 cells; the source retains its intermediate FP16 narrowing and FP32 store.
All five original pointer schedules remain selectable by `g.schedule`. -/
abbrev matmulIO (M : Algebra α) (graph : Graph) (g : Kernel.Geometry) : ProgramIO α α where
  inputs := inputs g
  outputs := outputs g
  scratch := []
  buffers := [g.A,g.B,g.C]
  read := fun i _ => match i with
    | ⟨0,_⟩ => fun j => j.1.val*max g.inner 1+j.2.1.val
    | ⟨1,_⟩ => fun j => j.1.val*max g.columns 1+j.2.1.val
    | ⟨_+2,h⟩ => False.elim (by simp at h)
  inputMask := fun _ _ _ => True
  write := fun o _ => match o with
    | ⟨0,_⟩ => fun j => j.1.val*max g.columns 1+j.2.1.val
    | ⟨_+1,h⟩ => False.elim (by simp at h)
  outputMask := fun _ _ _ => True
  scratchWrite := fun k => k.elim0
  scratchMask := fun k => k.elim0
  schedule := fun _ => (List.range (programs g.request)).map (invocation M graph g)
  encode := id
  observe := fun _ _ => id

/-- Finite logical matrices extended only to state independent formulas. -/
def matrixA [Inhabited α] (g : Kernel.Geometry) (vals : ChannelValues α (inputs g))
    (i k : Nat) : α :=
  if hi : i<g.rows then if hk : k<g.inner then vals 0 (⟨i,hi⟩,⟨k,hk⟩,())
  else default else default

def matrixB [Inhabited α] (g : Kernel.Geometry) (vals : ChannelValues α (inputs g))
    (k j : Nat) : α :=
  if hk : k<g.inner then if hj : j<g.columns then vals 1 (⟨k,hk⟩,⟨j,hj⟩,())
  else default else default

def result (g : Kernel.Geometry) (f : Nat → Nat → α) : ChannelValues α (outputs g) :=
  fun o => match o with
    | ⟨0,_⟩ => fun j => f j.1.val j.2.1.val
    | ⟨_+1,h⟩ => False.elim (by simp at h)

/-- Select another of the five existing source pointer schedules. Geometry,
operand precision, arithmetic graph and complete launch grid stay fixed. -/
def reschedule (g : Kernel.Geometry) (schedule : Kernels.Schedule) : Kernel.Geometry :=
  { g with schedule := schedule }

def scheduledVariant (M : Algebra α) (graph : Graph) (g : Kernel.Geometry)
    (schedule : Kernels.Schedule) : ProgramIO.Variant (matmulIO M graph g) where
  schedule := (matmulIO M graph (reschedule g schedule)).schedule
  observe := (matmulIO M graph (reschedule g schedule)).observe

end
end VeriTile.Bench.Repairs.IvDependentMatmulDotShape.IO

namespace VeriTile.Bench.Repairs.IvDependentMatmulDotShape.IO
open VeriTile.Triton FP.Structural
open IvDependentMatmul
noncomputable section
structure LogicalInputs (α : Type) (g : Kernel.Geometry) where
  a : Fin g.rows → Fin g.inner → α
  b : Fin g.inner → Fin g.columns → α

def logicalInputs (α : Type) (g : Kernel.Geometry) : ChannelValues α (inputs g) ≃ LogicalInputs α g where
  toFun xs := ⟨fun i k => xs 0 (i,k,()),fun k j => xs 1 (k,j,())⟩
  invFun xs := fun o => match o with
    | ⟨0,_⟩ => fun j => xs.a j.1 j.2.1
    | ⟨1,_⟩ => fun j => xs.b j.1 j.2.1
  left_inv := by intro xs; funext o; fin_cases o <;> funext j <;> rcases j with ⟨r,c,⟨⟩⟩ <;> rfl
  right_inv := by intro xs; cases xs; rfl

def logicalOutputs (α : Type) (g : Kernel.Geometry) : ChannelValues α (outputs g) ≃ (Fin g.rows → Fin g.columns → α) where
  toFun xs := fun i j => xs 0 (i,j,())
  invFun xs := fun o => match o with | ⟨0,_⟩ => fun j => xs j.1 j.2.1
  left_inv := by intro xs; funext o; fin_cases o; funext j; rcases j with ⟨r,c,⟨⟩⟩; rfl
  right_inv := by intro xs; rfl
end
end VeriTile.Bench.Repairs.IvDependentMatmulDotShape.IO

namespace VeriTile.Bench.Repairs.IvDependentMatmulDotShape.IO
open VeriTile.Triton FP.Structural
noncomputable section
open IvDependentMatmul
def logicalIO (M : Algebra α) (graph : TritonBenchG.IvDependentMatmul.FPValue.Graph) (g : Kernel.Geometry) : LogicalProgramIO α α (LogicalInputs α g) (Fin g.rows → Fin g.columns → α) where
  core := matmulIO M graph g
  inputs := logicalInputs α g
  outputs := logicalOutputs α g
end
end VeriTile.Bench.Repairs.IvDependentMatmulDotShape.IO
