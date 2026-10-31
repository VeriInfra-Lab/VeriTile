import VeriTile.Triton.Float.LogicalProgramIO
import VeriTile.Triton.Float.ProgramEquivalence
import bench.tritonbench_g.matmul_triton1.IO
import bench.tritonbench_g.matmul_triton2.Kernels
import bench.tritonbench_g.matmul_triton2.Proofs.Grid

namespace VeriTile.Bench.Repairs.MatmulTriton2.KernelOnly.IO
open VeriTile.Triton FP.Structural Revision001
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
noncomputable section
abbrev inputs := VeriTile.Bench.Repairs.MatmulTriton1.MatmulIO.inputs
abbrev outputs := VeriTile.Bench.Repairs.MatmulTriton1.MatmulIO.outputs
abbrev left := @VeriTile.Bench.Repairs.MatmulTriton1.MatmulIO.left
abbrev right := @VeriTile.Bench.Repairs.MatmulTriton1.MatmulIO.right
abbrev realTarget := VeriTile.Bench.Repairs.MatmulTriton1.MatmulIO.realTarget
abbrev precisionTarget := @VeriTile.Bench.Repairs.MatmulTriton1.MatmulIO.precisionTarget

def launch {α : Type} (rest : Algebra α) (P : Core.Primitives α)
    (p : Core.Parameters) (pid : Nat) : Invocation (VeriTile.Bench.Repairs.MatmulTriton1.DotProfile.Value α) where
  kernel := source p
  backend := VeriTile.Bench.Repairs.MatmulTriton1.DotProfile.algebra rest P {}
  pids := fun ax => if ax=0 then pid else 0

/-- Same three independent stride pairs, with the actual repaired source and
its original default dot options. Integer casts have an explicit public bound. -/
def matmul {α : Type} (rest : Algebra α) (P : Core.Primitives α)
    (p : Core.Parameters) : ProgramIO α (VeriTile.Bench.Repairs.MatmulTriton1.DotProfile.Value α) where
  toMemoryIO := VeriTile.Bench.Repairs.MatmulTriton1.MatmulIO.memoryIO α p
  schedule := fun _ => (List.range (Geometry.count (Core.geometry p))).map (launch rest P p)
  encode := VeriTile.Bench.Repairs.MatmulTriton1.DotProfile.Value.input
  observe := fun _ _ => VeriTile.Bench.Repairs.MatmulTriton1.DotProfile.observe P

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
def logicalIO (rest : Algebra ℝ) (P : Revision001.Core.Primitives ℝ) (p : Revision001.Core.Parameters) : LogicalProgramIO ℝ (VeriTile.Bench.Repairs.MatmulTriton1.DotProfile.Value ℝ)
    ((Fin p.rows → Fin p.innerSize → ℝ) × (Fin p.innerSize → Fin p.columns → ℝ)) (Fin p.rows → Fin p.columns → ℝ) where
  core := matmul rest P p
  inputs := matrixInputs p
  outputs := matrixOutputs p

end
end VeriTile.Bench.Repairs.MatmulTriton2.KernelOnly.IO

namespace VeriTile.Bench.Repairs.MatmulTriton2.KernelOnly.Pair
open VeriTile.Triton FP.Structural Revision001
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
noncomputable section
variable {α : Type} [Inhabited α]

def launch (rest : Algebra α) (P : Core.Primitives α) (p : Core.Parameters) (pid : Nat) : Invocation (VeriTile.Bench.Repairs.MatmulTriton1.DotProfile.Value α) :=
  { IO.launch rest P p pid with kernel := maskHoisted p }
def candidateIO (rest : Algebra α) (P : Core.Primitives α) (p : Core.Parameters) : ProgramIO α (VeriTile.Bench.Repairs.MatmulTriton1.DotProfile.Value α) :=
  { IO.matmul rest P p with schedule := fun _ =>
      (List.range (Geometry.count (Core.geometry p))).map (launch rest P p) }

def variant (rest : Algebra α) (P : Core.Primitives α) (p : Core.Parameters) :
    ProgramIO.Variant (IO.matmul rest P p) where
  schedule := (candidateIO rest P p).schedule
  observe := (IO.matmul rest P p).observe

end
end VeriTile.Bench.Repairs.MatmulTriton2.KernelOnly.Pair
