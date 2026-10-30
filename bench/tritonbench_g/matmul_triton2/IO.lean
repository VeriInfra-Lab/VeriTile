import VeriTile.Triton.Float.LogicalProgramIO
import bench.tritonbench_g.matmul_triton2.Proofs.Grid
import VeriTile.Triton.Float.ProgramIO

/-! Shared public matrix channels and the exact selected grouped launch.
The source fixes FP32 storage and its accumulator-fed TF32 dot graph;
`.real` is its Structural payload channel, with Real or FP32-word carriers. -/
namespace VeriTile.Bench.Repairs.MatmulTriton2.KernelOnly.IO
open VeriTile.Triton FP.Structural Revision001
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
noncomputable section

def inputs (p : Core.Parameters) : List IOChannel :=
  [⟨.real,[p.rows,p.innerSize],p.input⟩,⟨.real,[p.innerSize,p.columns],p.weight⟩]
def outputs (p : Core.Parameters) : List IOChannel :=
  [⟨.real,[p.rows,p.columns],p.output⟩]

def launch {α : Type} (M : Algebra α) (P : Core.Primitives α)
    (p : Core.Parameters) (pid : Nat) : Invocation α where
  kernel := Core.source p
  backend := Core.backend M P
  pids := fun ax => if ax=0 then pid else 0

/-- All logical matrix coordinates are active, with the original three
independent stride pairs. Inputs may alias each other; output separation and
injectivity are explicit premises of the public specifications. -/
def matmul {α : Type} (M : Algebra α) (P : Core.Primitives α)
    (p : Core.Parameters) : ProgramIO α α where
  inputs := inputs p
  outputs := outputs p
  buffers := [p.input,p.weight,p.output]
  scratch := []
  read := fun port _ => Fin.cases
    (fun j => j.1.val*p.inputRow+j.2.1.val*p.inputInner)
    (fun next => Fin.cases
      (fun j => j.1.val*p.weightInner+j.2.1.val*p.weightColumn)
      (fun impossible => Fin.elim0 impossible) next) port
  inputMask := fun _ _ _ => True
  write := fun port _ => Fin.cases
    (fun j => j.1.val*p.outputRow+j.2.1.val*p.outputColumn)
    (fun impossible => Fin.elim0 impossible) port
  outputMask := fun _ _ _ => True
  scratchWrite := fun impossible => Fin.elim0 impossible
  scratchMask := fun impossible => Fin.elim0 impossible
  schedule := fun _ => (List.range (Geometry.count (Core.geometry p))).map (launch M P p)
  encode := id
  observe := fun _ _ => id

/-- Irrelevant default extension outside a finite logical matrix. -/
def matrix {α : Type} [Inhabited α] {rows columns : Nat}
    (xs : Values α .real [rows,columns]) (r c : Nat) : α :=
  if hr : r<rows then if hc : c<columns then
    xs (⟨r,hr⟩,⟨c,hc⟩,PUnit.unit) else default else default

def formula {α : Type} [Inhabited α] (p : Core.Parameters)
    (f : (Nat → Nat → α) → (Nat → Nat → α) → Nat → Nat → α)
    (vals : ChannelValues α (inputs p)) : ChannelValues α (outputs p) :=
  Fin.cases (fun j => f (matrix (vals (0 : Fin 2))) (matrix (vals (1 : Fin 2)))
    j.1.val j.2.1.val) (fun impossible => Fin.elim0 impossible)

end
end VeriTile.Bench.Repairs.MatmulTriton2.KernelOnly.IO

namespace VeriTile.Bench.Repairs.MatmulTriton2.KernelOnly.IO
open VeriTile.Triton FP.Structural
noncomputable section

/-- Complete input channels reindexed as two ordinary logical matrices. -/
def matrixInputs (p : Revision001.Core.Parameters) :
    ChannelValues ℝ (inputs p) ≃ ((Fin p.rows → Fin p.innerSize → ℝ) × (Fin p.innerSize → Fin p.columns → ℝ)) where
  toFun := fun xs => (fun row col => id (xs (0 : Fin 2) (row,col,PUnit.unit)),
    fun row col => id (xs (1 : Fin 2) (row,col,PUnit.unit)))
  invFun := fun ab => Fin.cases (fun j => id (ab.1 j.1 j.2.1))
    (Fin.cases (fun j => id (ab.2 j.1 j.2.1)) (fun port => Fin.elim0 port))
  left_inv := by
    intro xs
    funext port j
    fin_cases port <;> rcases j with ⟨row,col,⟨⟩⟩ <;> rfl
  right_inv := by
    intro ab
    apply Prod.ext <;> funext row col <;> rfl

/-- Complete sole output channel reindexed as an ordinary logical matrix. -/
def matrixOutputs (p : Revision001.Core.Parameters) :
    ChannelValues ℝ (outputs p) ≃ (Fin p.rows → Fin p.columns → ℝ) where
  toFun := fun ys row col => id (ys (0 : Fin 1) (row,col,PUnit.unit))
  invFun := fun f => Fin.cases (fun j => id (f j.1 j.2.1)) (fun port => Fin.elim0 port)
  left_inv := by
    intro ys
    funext port j
    fin_cases port
    rcases j with ⟨row,col,⟨⟩⟩
    rfl
  right_inv := by
    intro f
    funext row col
    rfl

end
end VeriTile.Bench.Repairs.MatmulTriton2.KernelOnly.IO

namespace VeriTile.Bench.Repairs.MatmulTriton2.KernelOnly.IO
open VeriTile.Triton FP.Structural

noncomputable section

/-- A bijective logical view of the unchanged complete source IO. -/
def logicalIO (rest : Algebra ℝ) (P : Revision001.Core.Primitives ℝ) (p : Revision001.Core.Parameters) : LogicalProgramIO ℝ ℝ
    ((Fin p.rows → Fin p.innerSize → ℝ) × (Fin p.innerSize → Fin p.columns → ℝ)) (Fin p.rows → Fin p.columns → ℝ) where
  core := matmul rest P p
  inputs := matrixInputs p
  outputs := matrixOutputs p

end
end VeriTile.Bench.Repairs.MatmulTriton2.KernelOnly.IO
