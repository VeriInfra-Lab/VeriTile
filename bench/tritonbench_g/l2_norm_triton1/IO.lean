import bench.tritonbench_g.l2_norm_triton1.Kernels
import VeriTile.Triton.Float.ProgramEquivalence
import Mathlib.Analysis.SpecialFunctions.Sqrt
import VeriTile.Triton.Float.LogicalProgramIO

/-! Public full-grid channels, actual runtime-argument invocations and independent
logical L2 formula. Source precision stays in the shared FP32 source; `.real`
is its Structural interpreter's floating payload channel. -/
namespace VeriTile.Bench.Repairs.L2NormTriton1.GridIO
open VeriTile.Triton FP.Structural
noncomputable section

def launch {α : Type} (A : Algebra α) (N B row : Nat) (eps : α) : Invocation α where
  kernel := source N B
  backend := A
  arguments := [⟨"eps", .real, [], fun _ => eps⟩]
  pids := fun ax => if ax = 0 then row else 0

def candidateLaunch {α : Type} (A : Algebra α) (N B row : Nat) (eps : α) : Invocation α :=
  { launch A N B row eps with kernel := indexCommuted N B }

def schedule {α : Type} (A : Algebra α) (rows N B : Nat) (eps : α) : List (Invocation α) :=
  (List.range rows).map (fun row => launch A N B row eps)

def candidateSchedule {α : Type} (A : Algebra α) (rows N B : Nat) (eps : α) : List (Invocation α) :=
  (List.range rows).map (fun row => candidateLaunch A N B row eps)

abbrev channels (rows N : Nat) : List IOChannel := [⟨.real, [rows, N], "X"⟩]
abbrev Inputs (α : Type) (rows N : Nat) := ChannelValues α (channels rows N)

/-- Complete logical row grid. Epsilon remains a runtime scalar argument on
every invocation; no literal specialization or one-row replacement is used. -/
abbrev gridIO {α : Type} (A : Algebra α) (rows N B : Nat) (eps : α) : ProgramIO α α where
  inputs := channels rows N
  outputs := [⟨.real, [rows, N], "Y"⟩]
  scratch := []
  buffers := ["X", "Y"]
  read := Fin.cases (fun _ j => j.1.val * N + j.2.1.val) (fun i => i.elim0)
  inputMask := fun _ _ _ => True
  write := Fin.cases (fun _ j => j.1.val * N + j.2.1.val) (fun i => i.elim0)
  outputMask := fun _ _ _ => True
  scratchWrite := fun i => i.elim0
  scratchMask := fun i => i.elim0
  schedule := fun _ => schedule A rows N B eps
  encode := id
  observe := fun _ _ => id

def candidate {α : Type} (A : Algebra α) (rows N B : Nat) (eps : α) :
    ProgramIO.Variant (gridIO A rows N B eps) where
  schedule := fun _ => candidateSchedule A rows N B eps
  observe := fun _ _ => id

def rowValues {α : Type} [Inhabited α] {rows N : Nat}
    (vals : Inputs α rows N) (row : Fin rows) (col : Nat) : α :=
  if hc : col < N then vals 0 (row, ⟨col, hc⟩, PUnit.unit) else default

def outputValues {α : Type} {rows N : Nat} (f : Fin rows → Fin N → α) :
    ChannelValues α [⟨.real, [rows, N], "Y"⟩] :=
  Fin.cases (fun j => f j.1 j.2.1) (fun i => i.elim0)

/-- Mathematical L2 normalization, independent of source/block/reduction order.
Real sqrt/division are total, so epsilon remains unrestricted. -/
def formula (N : Nat) (eps : ℝ) (xs : Nat → ℝ) (i : Nat) : ℝ :=
  xs i / Real.sqrt ((∑ j∈Finset.range N, (xs j)^2) + eps)

def mathematical (rows N : Nat) (eps : ℝ) (vals : Inputs ℝ rows N) :=
  outputValues (N := N) (fun row col => formula N eps (rowValues vals row) col.val)

#specNonCircular formula avoiding [source, indexCommuted]
end
end VeriTile.Bench.Repairs.L2NormTriton1.GridIO

namespace VeriTile.Bench.Repairs.L2NormTriton1.LogicalIO
open VeriTile.Triton FP.Structural
noncomputable section

def inputs (A : Algebra ℝ) (rows N B : Nat) (eps : ℝ) : (GridIO.gridIO A rows N B eps).Inputs ≃ (Fin rows → Fin N → ℝ) where
  toFun := fun v row col => v 0 (row,col,())
  invFun := fun x ⟨0,_⟩ j => x j.1 j.2.1
  left_inv := by
    intro v
    funext i j
    fin_cases i <;> rfl
  right_inv := by
    intro x
    rfl

def outputs (A : Algebra ℝ) (rows N B : Nat) (eps : ℝ) : (GridIO.gridIO A rows N B eps).Outputs ≃ (Fin rows → Fin N → ℝ) where
  toFun := fun v row col => v 0 (row,col,())
  invFun := fun y ⟨0,_⟩ j => y j.1 j.2.1
  left_inv := by
    intro v
    funext i j
    fin_cases i
    rfl
  right_inv := fun _ => rfl

def io (A : Algebra ℝ) (rows N B : Nat) (eps : ℝ) : LogicalProgramIO ℝ ℝ (Fin rows → Fin N → ℝ) (Fin rows → Fin N → ℝ) where
  core := GridIO.gridIO A rows N B eps
  inputs := inputs A rows N B eps
  outputs := outputs A rows N B eps

end
end VeriTile.Bench.Repairs.L2NormTriton1.LogicalIO
