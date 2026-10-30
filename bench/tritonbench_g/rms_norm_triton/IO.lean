import bench.tritonbench_g.rms_norm_triton.Proofs.Mathematical_WholeGrid
import bench.tritonbench_g.rms_norm_triton.Proofs.Mathematical_WholeMath
import bench.tritonbench_g.rms_norm_triton.Proofs.Mathematical_Observed
import VeriTile.Triton.Float.ProgramIO

import bench.tritonbench_g.rms_norm_triton.Kernels
import VeriTile.Triton.Float.ProgramEquivalenceSyntax
import VeriTile.Triton.Float.LogicalProgramIO


/-! Canonical launch, channel layout and logical adapters shared by Real and native readers. -/
namespace VeriTile.Bench.Repairs.RmsNormTriton.KernelSafety
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.RmsNormTriton Whole
noncomputable section

abbrev invocation {α : Type} (M : Algebra α) (c : Layout) (eps : ℝ) (pid : Nat) : Invocation α where
  kernel := c.source eps
  backend := engine M
  pids := fun ax => if ax=0 then pid else 0

end
end VeriTile.Bench.Repairs.RmsNormTriton.KernelSafety

namespace VeriTile.Bench.Repairs.RmsNormTriton.GridPublic
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.RmsNormTriton Whole Observed
noncomputable section

abbrev io {α β : Type} (M : Algebra β) (encode : α → β) (observe : Nat → β → α)
    (c : Layout) (eps : ℝ) (rows : List Nat) : ProgramIO α β where
  inputs := [⟨.real,[c.M,c.N],c.X⟩,⟨.real,[c.N],c.W⟩]
  outputs := [⟨.real,[c.M,c.N],c.Y⟩]
  scratch := []
  buffers := [c.X,c.W,c.Y]
  read := fun i => Fin.cases (fun _ j => j.1.val*c.N+j.2.1.val)
    (fun k => Fin.cases (fun _ j => j.1.val) (fun z => Fin.elim0 z) k) i
  inputMask := fun _ _ _ => True
  write := fun o => Fin.cases (fun _ j => j.1.val*c.N+j.2.1.val) (fun z => Fin.elim0 z) o
  outputMask := fun _ _ _ => True
  scratchWrite := fun z => Fin.elim0 z
  scratchMask := fun z => Fin.elim0 z
  schedule := fun _ => rows.map (KernelSafety.invocation M c eps)
  encode := encode
  observe := fun o => Fin.cases (fun j => observe j.2.1.val) (fun z => Fin.elim0 z) o

def inputRow {α β : Type} [Inhabited α] (M : Algebra β) (encode : α → β) (observe : Nat → β → α)
    (c : Layout) (eps : ℝ) (rows : List Nat)
    (xs : (io M encode observe c eps rows).Inputs) (row : Fin c.M) (j : Nat) : α :=
  if h : j<c.N then xs 0 (row,⟨j,h⟩,()) else default

def weights {α β : Type} [Inhabited α] (M : Algebra β) (encode : α → β) (observe : Nat → β → α)
    (c : Layout) (eps : ℝ) (rows : List Nat)
    (xs : (io M encode observe c eps rows).Inputs) (j : Nat) : α :=
  if h : j<c.N then xs 1 (⟨j,h⟩,()) else default

def target {α β : Type} [Inhabited α] (M : Algebra β) (encode : α → β)
    (observe : Nat → β → α) (c : Layout) (eps : ℝ) (rows : List Nat)
    (f : (Nat → α) → (Nat → α) → Nat → α)
    (xs : (io M encode observe c eps rows).Inputs) : (io M encode observe c eps rows).Outputs :=
  fun o => Fin.cases (fun j => f (inputRow M encode observe c eps rows xs j.1)
    (weights M encode observe c eps rows xs) j.2.1.val) (fun z => Fin.elim0 z) o

end
end VeriTile.Bench.Repairs.RmsNormTriton.GridPublic

namespace VeriTile.Bench.Repairs.RmsNormTriton.FloatingIO
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.RmsNormTriton Whole Observed
noncomputable section

/-- The original full row schedule and lane-specific native word observer. -/
abbrev nativeIO (c : Layout) (eps : ℝ) (M : NativeModel (BitVec 32))
    (profile : ReductionFamily) (rows : List Nat) :=
  GridPublic.io graphAlgebra Term.input (interpret M profile) c eps rows

def candidateInvocation (c : Layout) (eps : ℝ) (pid : Nat) : Invocation (Term (BitVec 32)) :=
  { KernelSafety.invocation graphAlgebra c eps pid with
    kernel := indexCommutedKernel c.Y c.X c.W c.N c.B eps }

/-- Same public channels, inputs, runtime epsilon and native observer; the
existing address-commuted kernel replaces every invocation in the full grid. -/
def candidate (c : Layout) (eps : ℝ) (M : NativeModel (BitVec 32))
    (profile : ReductionFamily) (rows : List Nat) :
    ProgramIO.Variant (nativeIO c eps M profile rows) where
  schedule := fun _ => rows.map (candidateInvocation c eps)
  observe := (nativeIO c eps M profile rows).observe

def original (c : Layout) (eps : ℝ) (M : NativeModel (BitVec 32))
    (profile : ReductionFamily) (rows : List Nat) :=
  ProgramIO.Implementation.original (nativeIO c eps M profile rows) (fun _ => True)

def indexCommuted (c : Layout) (eps : ℝ) (M : NativeModel (BitVec 32))
    (profile : ReductionFamily) (rows : List Nat) :=
  ProgramIO.Implementation.ofVariant (candidate c eps M profile rows) (fun _ => True)

end
end VeriTile.Bench.Repairs.RmsNormTriton.FloatingIO

namespace VeriTile.Bench.Repairs.RmsNormTriton.LogicalIO
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.RmsNormTriton Whole
noncomputable section

/-- Lossless matrix/vector input coordinates for the original complete grid. -/
def inputs (c : Layout) (eps : ℝ) (rows : List Nat) :
    (GridPublic.io realAlgebra id (fun _ => id) c eps rows).Inputs ≃
      ((Fin c.M → Fin c.N → ℝ) × (Fin c.N → ℝ)) where
  toFun := fun values => (fun row col => values 0 (row,col,()), fun col => values 1 (col,()))
  invFun := fun values => Fin.cases (fun j => values.1 j.1 j.2.1)
    (Fin.cases (fun j => values.2 j.1) (fun o => o.elim0))
  left_inv := by
    intro values
    funext channel
    fin_cases channel <;> funext j
    · rcases j with ⟨row,col,⟨⟩⟩; rfl
    · rcases j with ⟨col,⟨⟩⟩; rfl
  right_inv := by intro values; cases values; rfl

/-- The one output channel is exactly a matrix, with empty cases preserved. -/
def outputs (c : Layout) (eps : ℝ) (rows : List Nat) :
    (GridPublic.io realAlgebra id (fun _ => id) c eps rows).Outputs ≃
      (Fin c.M → Fin c.N → ℝ) where
  toFun := fun values row col => values 0 (row,col,())
  invFun := fun values => Fin.cases (fun j => values j.1 j.2.1) (fun o => o.elim0)
  left_inv := by
    intro values
    funext channel
    fin_cases channel
    funext j
    rcases j with ⟨row,col,⟨⟩⟩; rfl
  right_inv := fun _ => rfl

/-- Original source, arbitrary complete row order, and complete memory contract. -/
def io (c : Layout) (eps : ℝ) (rows : List Nat) :
    LogicalProgramIO ℝ ℝ ((Fin c.M → Fin c.N → ℝ) × (Fin c.N → ℝ))
      (Fin c.M → Fin c.N → ℝ) where
  core := GridPublic.io realAlgebra id (fun _ => id) c eps rows
  inputs := inputs c eps rows
  outputs := outputs c eps rows

end
end VeriTile.Bench.Repairs.RmsNormTriton.LogicalIO

namespace VeriTile.Bench.Repairs.RmsNormTriton.Correct
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.RmsNormTriton Whole
noncomputable section
/-- Complete row grid of the actual FP32 source. X and Y have shape [M,N],
weights have shape [N], and row-major windows retain the empty-grid case. -/
abbrev rmsIO (c : Layout) (eps : ℝ) (rows : List Nat) : ProgramIO ℝ ℝ :=
  GridPublic.io realAlgebra id (fun _ => id) c eps rows

/-- Independent RMSNorm formula on logical input and weight arrays. -/
def formula (N : Nat) (eps : ℝ) (xs ws : Nat → ℝ) (i : Nat) : ℝ :=
  realFormula N eps xs ws i

def expected (c : Layout) (eps : ℝ) (rows : List Nat)
    (xs : (rmsIO c eps rows).Inputs) : (rmsIO c eps rows).Outputs :=
  fun o => Fin.cases (fun j => formula c.N eps
    (GridPublic.inputRow realAlgebra id (fun _ => id) c eps rows xs j.1)
    (GridPublic.weights realAlgebra id (fun _ => id) c eps rows xs) j.2.1.val)
    (fun z => Fin.elim0 z) o

end
end VeriTile.Bench.Repairs.RmsNormTriton.Correct
