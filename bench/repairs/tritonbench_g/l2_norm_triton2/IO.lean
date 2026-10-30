import bench.repairs.tritonbench_g.l2_norm_triton2.Kernels
import bench.repairs.tritonbench_g.l2_norm_triton1.IO
import VeriTile.Triton.Float.ProgramEquivalence
import VeriTile.Triton.Float.ProgramEquivalenceSyntax
import Mathlib.Analysis.SpecialFunctions.Sqrt
import VeriTile.Triton.Float.LogicalProgramIO

/-! Public full-grid channels, actual runtime-argument invocations and independent
logical L2 formula. Source precision stays in the shared FP32 source; `.real`
is its Structural interpreter's floating payload channel. -/
namespace VeriTile.Bench.Repairs.L2NormTriton2.GridIO
open VeriTile.Triton FP.Structural
noncomputable section

/-- The forward source is definitionally the shared L2NormTriton1 source;
reuse its complete public channel layout, scalar ABI and row schedule. -/
abbrev forwardIO {α : Type} (A : Algebra α) (rows N B : Nat) (eps : α) :=
  L2NormTriton1.GridIO.gridIO A rows N B eps

abbrev forwardCandidate {α : Type} (A : Algebra α) (rows N B : Nat) (eps : α) :=
  L2NormTriton1.GridIO.candidate A rows N B eps

def launch {α : Type} (A : Algebra α) (N B row : Nat) (eps : α) : Invocation α where
  kernel := backward N B
  backend := A
  arguments := [⟨"eps", .real, [], fun _ => eps⟩]
  pids := fun ax => if ax = 0 then row else 0

def candidateLaunch {α : Type} (A : Algebra α) (N B row : Nat) (eps : α) : Invocation α :=
  { launch A N B row eps with kernel := backwardCommuted N B }

def schedule {α : Type} (A : Algebra α) (rows N B : Nat) (eps : α) : List (Invocation α) :=
  (List.range rows).map (fun row => launch A N B row eps)

def candidateSchedule {α : Type} (A : Algebra α) (rows N B : Nat) (eps : α) : List (Invocation α) :=
  (List.range rows).map (fun row => candidateLaunch A N B row eps)

abbrev channels (rows N : Nat) : List IOChannel := [⟨.real, [rows, N], "X"⟩, ⟨.real, [rows, N], "DY"⟩]
abbrev Inputs (α : Type) (rows N : Nat) := ChannelValues α (channels rows N)

/-- Complete logical row grid. Epsilon remains a runtime scalar argument on
every invocation; no literal specialization or one-row replacement is used. -/
abbrev gridIO {α : Type} (A : Algebra α) (rows N B : Nat) (eps : α) : ProgramIO α α where
  inputs := channels rows N
  outputs := [⟨.real, [rows, N], "DX"⟩]
  scratch := []
  buffers := ["X", "DY", "DX"]
  read := Fin.cases (fun _ j => j.1.val * N + j.2.1.val)
    (Fin.cases (fun _ j => j.1.val * N + j.2.1.val) (fun i => i.elim0))
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
    (vals : Inputs α rows N) (gradient : Bool) (row : Fin rows) (col : Nat) : α :=
  if hc : col < N then
    if gradient then vals 1 (row, ⟨col, hc⟩, PUnit.unit)
    else vals 0 (row, ⟨col, hc⟩, PUnit.unit)
  else default

def outputValues {α : Type} {rows N : Nat} (f : Fin rows → Fin N → α) :
    ChannelValues α [⟨.real, [rows, N], "DX"⟩] :=
  Fin.cases (fun j => f j.1 j.2.1) (fun i => i.elim0)

/-- Mathematical L2 gradient, independent of source/block/reduction order.
Real sqrt/division are total, so epsilon remains unrestricted. -/
def formula (N : Nat) (eps : ℝ) (xs dys : Nat → ℝ) (i : Nat) : ℝ :=
  let squared := ∑ j∈Finset.range N, (xs j)^2
  let cross := ∑ j∈Finset.range N, dys j*xs j
  dys i / Real.sqrt (squared+eps) - cross/(squared+eps)/Real.sqrt (squared+eps)*xs i

def mathematical (rows N : Nat) (eps : ℝ) (vals : Inputs ℝ rows N) :=
  outputValues (N := N) (fun row col => formula N eps
    (rowValues vals Bool.false row) (rowValues vals Bool.true row) col.val)

#specNonCircular formula avoiding [backward, backwardCommuted]
#specNonCircular mathematical avoiding [backward, backwardCommuted, gridIO]
end
end VeriTile.Bench.Repairs.L2NormTriton2.GridIO

namespace VeriTile.Bench.Repairs.L2NormTriton2.FloatingIO
open VeriTile.Triton FP.Structural
noncomputable section

/-- The actual original and existing index-commuted complete forward grids. -/
def forwardOriginal {α : Type} (A : Algebra α) (rows N B : Nat) (eps : α) :=
  ProgramIO.Implementation.original (GridIO.forwardIO A rows N B eps) (fun _ => True)

def forwardCandidate {α : Type} (A : Algebra α) (rows N B : Nat) (eps : α) :=
  ProgramIO.Implementation.ofVariant (GridIO.forwardCandidate A rows N B eps) (fun _ => True)

/-- The actual original and existing index-commuted complete backward grids. -/
def backwardOriginal {α : Type} (A : Algebra α) (rows N B : Nat) (eps : α) :=
  ProgramIO.Implementation.original (GridIO.gridIO A rows N B eps) (fun _ => True)

def backwardCandidate {α : Type} (A : Algebra α) (rows N B : Nat) (eps : α) :=
  ProgramIO.Implementation.ofVariant (GridIO.candidate A rows N B eps) (fun _ => True)

end
end VeriTile.Bench.Repairs.L2NormTriton2.FloatingIO

namespace VeriTile.Bench.Repairs.L2NormTriton2.LogicalIO
open VeriTile.Triton FP.Structural
noncomputable section

structure Inputs (rows N : Nat) where
  x : Fin rows → Fin N → ℝ
  dy : Fin rows → Fin N → ℝ

def inputs (A : Algebra ℝ) (rows N B : Nat) (eps : ℝ) : (GridIO.gridIO A rows N B eps).Inputs ≃ (Inputs rows N) where
  toFun := fun v => ⟨fun row col => v 0 (row,col,()), fun row col => v 1 (row,col,())⟩
  invFun := fun x i => match i with
    | ⟨0,_⟩ => fun j => x.x j.1 j.2.1
    | ⟨1,_⟩ => fun j => x.dy j.1 j.2.1
  left_inv := by
    intro v
    funext i j
    fin_cases i <;> rfl
  right_inv := by
    intro x
    cases x; rfl

def outputs (A : Algebra ℝ) (rows N B : Nat) (eps : ℝ) : (GridIO.gridIO A rows N B eps).Outputs ≃ (Fin rows → Fin N → ℝ) where
  toFun := fun v row col => v 0 (row,col,())
  invFun := fun y ⟨0,_⟩ j => y j.1 j.2.1
  left_inv := by
    intro v
    funext i j
    fin_cases i
    rfl
  right_inv := fun _ => rfl

def io (A : Algebra ℝ) (rows N B : Nat) (eps : ℝ) : LogicalProgramIO ℝ ℝ (Inputs rows N) (Fin rows → Fin N → ℝ) where
  core := GridIO.gridIO A rows N B eps
  inputs := inputs A rows N B eps
  outputs := outputs A rows N B eps

end
end VeriTile.Bench.Repairs.L2NormTriton2.LogicalIO
