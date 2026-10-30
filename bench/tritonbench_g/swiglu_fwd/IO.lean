import VeriTile.Triton.Float.LogicalProgramIO
import bench.tritonbench_g.swiglu_fwd.Proofs.Mathematical_WholePublic
import bench.tritonbench_g.swiglu_fwd.Kernels
import bench.tritonbench_g.swiglu_fwd.Proofs.Mathematical_WholeTile
import VeriTile.Triton.Float.ProgramIO

namespace VeriTile.Bench.Repairs.SwigluFwd.KernelPublic
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.SwigluFwd.Whole
noncomputable section

/-- Two simultaneous input snapshots and one masked strided output tile. -/
abbrev io {α β : Type} (M : Algebra β) (encode : α → β) (observe : β → α)
    (c : Layout) (B row tile : Nat) : ProgramIO α β where
  inputs := [⟨.real,[B],c.X⟩,⟨.real,[B],c.Y⟩]
  outputs := [⟨.real,[B],c.OUT⟩]
  scratch := []
  buffers := [c.X,c.Y,c.OUT]
  read := fun i => Fin.cases (fun _ j => row*c.sx+(tile*B+j.1.val))
    (fun k => Fin.cases (fun _ j => row*c.sy+(tile*B+j.1.val)) (fun z => Fin.elim0 z) k) i
  inputMask := fun i => Fin.cases (fun _ j => tile*B+j.1.val<c.cols)
    (fun k => Fin.cases (fun _ j => tile*B+j.1.val<c.cols) (fun z => Fin.elim0 z) k) i
  write := fun o => Fin.cases (fun _ j => row*c.so+(tile*B+j.1.val)) (fun z => Fin.elim0 z) o
  outputMask := fun o => Fin.cases (fun _ j => tile*B+j.1.val<c.cols) (fun z => Fin.elim0 z) o
  scratchWrite := fun k => Fin.elim0 k
  scratchMask := fun k => Fin.elim0 k
  schedule := fun _ => [{
    kernel := kernel c.X c.Y c.OUT c.sx c.sy c.so c.cols B
    backend := M
    pids := fun ax => if ax=0 then row else if ax=1 then tile else 0 }]
  encode := encode
  observe := fun _ _ => observe

def target {α β : Type} (M : Algebra β) (encode : α → β) (observe : β → α)
    (c : Layout) (B row tile : Nat) (f : α → α → α)
    (xs : (io M encode observe c B row tile).Inputs) :
    (io M encode observe c B row tile).Outputs :=
  fun o => Fin.cases (fun j => f (xs 0 j) (xs 1 j)) (fun z => Fin.elim0 z) o

end
end VeriTile.Bench.Repairs.SwigluFwd.KernelPublic

namespace VeriTile.Bench.Repairs.SwigluFwd.Logical
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.SwigluFwd.Whole
noncomputable section

/-- Named logical snapshots; every original input channel is retained. -/
structure Inputs (B : Nat) where
  x : Fin B → ℝ
  y : Fin B → ℝ

/-- Exact input coordinate change, including inactive lanes. -/
def inputs (c : Layout) (B row tile : Nat) : (KernelPublic.io (α := ℝ) realAlgebra id id c B row tile).Inputs ≃ Inputs B where
  toFun := fun values => ⟨fun i => values 0 (i,()), fun i => values 1 (i,())⟩
  invFun := fun xs port => match port with
    | ⟨0,_⟩ => fun j => xs.x j.1
    | ⟨1,_⟩ => fun j => xs.y j.1
    | ⟨n+2,h⟩ => False.elim (by simp [KernelPublic.io] at h)
  left_inv := by
    intro values
    funext port
    fin_cases port <;> funext j <;> rcases j with ⟨i,⟨⟩⟩ <;> rfl
  right_inv := by intro xs; cases xs; rfl

/-- Every declared output cell is retained, even when its source mask is inactive. -/
def outputs (c : Layout) (B row tile : Nat) : (KernelPublic.io (α := ℝ) realAlgebra id id c B row tile).Outputs ≃ (Fin B → ℝ) where
  toFun := fun values i => values 0 (i,())
  invFun := fun f port => match port with
    | ⟨0,_⟩ => fun j => f j.1
    | ⟨n+1,h⟩ => False.elim (by simp [KernelPublic.io] at h)
  left_inv := by
    intro values
    funext port
    fin_cases port <;> funext j <;> rcases j with ⟨i,⟨⟩⟩ <;> rfl
  right_inv := by
    intro f
    rfl

def io (c : Layout) (B row tile : Nat) : LogicalProgramIO ℝ ℝ (Inputs B) (Fin B → ℝ) where
  core := KernelPublic.io realAlgebra id id c B row tile
  inputs := inputs c B row tile
  outputs := outputs c B row tile

end
end VeriTile.Bench.Repairs.SwigluFwd.Logical
