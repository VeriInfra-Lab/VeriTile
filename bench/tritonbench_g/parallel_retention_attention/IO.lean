import bench.tritonbench_g.parallel_retention_attention.Kernels
import VeriTile.Triton.Float.ProgramIO
import Mathlib.Tactic.FinCases
import VeriTile.Triton.Float.LogicalProgramIO

namespace VeriTile.Bench.Repairs.ParallelRetentionAttention.PublicIO
open VeriTile.Triton FP.Structural Kernel Kernel.Native
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (withDot)
noncomputable section

def forwardInputs (s : Shape) : List IOChannel :=
  [⟨.real,[s.B*s.H,s.T,s.DK],"q"⟩,⟨.real,[s.B*s.H,s.T,s.DK],"k"⟩,
    ⟨.real,[s.B*s.H,s.T,s.DV],"v"⟩]
def backwardInputs (s : Shape) : List IOChannel :=
  [⟨.real,[s.B*s.H,s.T,s.DK],"q"⟩,⟨.real,[s.B*s.H,s.T,s.DK],"k"⟩,
    ⟨.real,[s.B*s.H,s.T,s.DV],"v"⟩,⟨.real,[s.B*s.H,s.T,s.DV],"g"⟩]
def forwardOutputs (s : Shape) : List IOChannel := [⟨.real,[128,s.BV],"o"⟩]
def backwardOutputs (s : Shape) : List IOChannel :=
  [⟨.real,[64,s.BK],"dq"⟩,⟨.real,[64,s.BK],"dk"⟩,⟨.real,[64,s.BV],"dv"⟩]

def forwardRead {α : Type} (s : Shape) :
    (i : Fin (forwardInputs s).length) → ChannelValues α (forwardInputs s) →
      TileIndex ((forwardInputs s)[i]).shape → Nat := by
  refine Fin.cases ?_ (Fin.cases ?_ (Fin.cases ?_ (fun k => k.elim0)))
  · intro _ j; exact dense s.T s.DK j.1.val j.2.1.val j.2.2.1.val
  · intro _ j; exact dense s.T s.DK j.1.val j.2.1.val j.2.2.1.val
  · intro _ j; exact dense s.T s.DV j.1.val j.2.1.val j.2.2.1.val

def backwardRead {α : Type} (s : Shape) :
    (i : Fin (backwardInputs s).length) → ChannelValues α (backwardInputs s) →
      TileIndex ((backwardInputs s)[i]).shape → Nat := by
  refine Fin.cases ?_ (Fin.cases ?_ (Fin.cases ?_ (Fin.cases ?_ (fun k => k.elim0))))
  · intro _ j; exact dense s.T s.DK j.1.val j.2.1.val j.2.2.1.val
  · intro _ j; exact dense s.T s.DK j.1.val j.2.1.val j.2.2.1.val
  · intro _ j; exact dense s.T s.DV j.1.val j.2.1.val j.2.2.1.val
  · intro _ j; exact dense s.T s.DV j.1.val j.2.1.val j.2.2.1.val

def backwardWrite {α : Type} (s : Shape) (tile : Tile) :
    (o : Fin (backwardOutputs s).length) → ChannelValues α (backwardInputs s) →
      TileIndex ((backwardOutputs s)[o]).shape → Nat := by
  refine Fin.cases ?_ (Fin.cases ?_ (Fin.cases ?_ (fun k => k.elim0)))
  · intro _ j; exact keyAddress s tile j
  · intro _ j; exact keyAddress s tile j
  · intro _ j; exact valueAddress s tile j

def backwardMask {α : Type} (s : Shape) (tile : Tile) :
    (o : Fin (backwardOutputs s).length) → ChannelValues α (backwardInputs s) →
      TileIndex ((backwardOutputs s)[o]).shape → Prop := by
  refine Fin.cases ?_ (Fin.cases ?_ (Fin.cases ?_ (fun k => k.elim0)))
  · intro _ j; exact activeTile s.T s.DK (tile.chunk*64) (tile.key*s.BK) j
  · intro _ j; exact activeTile s.T s.DK (tile.chunk*64) (tile.key*s.BK) j
  · intro _ j; exact activeTile s.T s.DV (tile.chunk*64) (tile.value*s.BV) j

def programIds (s : Shape) (tile : Tile) (ax : Nat) : Nat :=
  match ax with
  | 0 => ProgramDomain.gridX s tile
  | 1 => tile.chunk
  | 2 => tile.bh
  | _ => 0

def forwardInvocation (α : Type) (s : Shape) (tile : Tile) : Invocation (Expr α) where
  kernel := forwardKernel s
  backend := withDot algebra dots
  pids := programIds s tile
  clearRegisters := Bool.true
  numPids := some (fun _ => 0)

def backwardInvocation (α : Type) (s : Shape) (tile : Tile) : Invocation (Expr α) where
  kernel := backwardKernel s (programIds s tile)
  backend := withDot algebra dots
  pids := programIds s tile
  clearRegisters := Bool.true
  numPids := some (fun _ => 0)

/-- The .real tags are the existing Structural cells. The authoritative source
retains FP32 loads, dot/cast contexts, and FP32 stores. No private scratch occurs
inside these per-program kernel endpoints; output shards stay explicit. -/
def forwardIO {α : Type} (a : Scalar α) (p : DotTrees) (s : Shape) (tile : Tile) :
    ProgramIO α (Expr α) where
  inputs := forwardInputs s
  outputs := forwardOutputs s
  scratch := []
  buffers := ["q","k","v","o"]
  read := forwardRead s
  inputMask := fun _ _ _ => True
  write := Fin.cases (fun _ j => forwardAddress s tile j) (fun k => k.elim0)
  outputMask := Fin.cases (fun _ j => activeTile s.T s.DV (tile.chunk*128) (tile.value*s.BV) j)
    (fun k => k.elim0)
  scratchWrite := fun k => k.elim0
  scratchMask := fun k => k.elim0
  schedule := fun _ => [forwardInvocation α s tile]
  encode := Expr.input
  observe := fun _ _ => evaluate a p

def backwardIO {α : Type} (a : Scalar α) (p : DotTrees) (s : Shape) (tile : Tile) :
    ProgramIO α (Expr α) where
  inputs := backwardInputs s
  outputs := backwardOutputs s
  scratch := []
  buffers := ["q","k","v","g","dq","dk","dv"]
  read := backwardRead s
  inputMask := fun _ _ _ => True
  write := backwardWrite s tile
  outputMask := backwardMask s tile
  scratchWrite := fun k => k.elim0
  scratchMask := fun k => k.elim0
  schedule := fun _ => [backwardInvocation α s tile]
  encode := Expr.input
  observe := fun _ _ => evaluate a p

end
end VeriTile.Bench.Repairs.ParallelRetentionAttention.PublicIO

namespace VeriTile.Bench.Repairs.ParallelRetentionAttention.PublicIO
open VeriTile.Triton FP.Structural Kernel Kernel.Native
noncomputable section

/-- Logical tensor indexing; zero outside a channel is the source's explicit
two-axis block-pointer padding, not an extra input-domain assumption. -/
def tensor {α : Type} (zero : α) (heads time width : Nat)
    (xs : TileIndex [heads,time,width] → α) (bh t e : Nat) : α :=
  if h : bh<heads ∧ t<time ∧ e<width then
    xs (⟨bh,h.1⟩,⟨t,h.2.1⟩,⟨e,h.2.2⟩,PUnit.unit)
  else zero

def forwardLogical {α : Type} (zero : α) (s : Shape)
    (xs : ChannelValues α (forwardInputs s)) : Inputs α where
  q := tensor zero (s.B*s.H) s.T s.DK (xs (0 : Fin 3))
  k := tensor zero (s.B*s.H) s.T s.DK (xs (1 : Fin 3))
  v := tensor zero (s.B*s.H) s.T s.DV (xs (2 : Fin 3))
  g := fun _ _ _ => zero

def backwardLogical {α : Type} (zero : α) (s : Shape)
    (xs : ChannelValues α (backwardInputs s)) : Inputs α where
  q := tensor zero (s.B*s.H) s.T s.DK (xs (0 : Fin 4))
  k := tensor zero (s.B*s.H) s.T s.DK (xs (1 : Fin 4))
  v := tensor zero (s.B*s.H) s.T s.DV (xs (2 : Fin 4))
  g := tensor zero (s.B*s.H) s.T s.DV (xs (3 : Fin 4))

end
end VeriTile.Bench.Repairs.ParallelRetentionAttention.PublicIO

namespace VeriTile.Bench.Repairs.ParallelRetentionAttention.LogicalView
open VeriTile.Triton FP.Structural Kernel Kernel.Native
noncomputable section
structure ForwardInputs (s : Shape) where
  q : Fin (s.B*s.H) → Fin s.T → Fin s.DK → ℝ
  k : Fin (s.B*s.H) → Fin s.T → Fin s.DK → ℝ
  v : Fin (s.B*s.H) → Fin s.T → Fin s.DV → ℝ
def forwardInputs (s : Shape) : ChannelValues ℝ (PublicIO.forwardInputs s) ≃ ForwardInputs s where
  toFun := fun xs => ⟨(fun h t e => xs ⟨0,by simp [PublicIO.forwardInputs]⟩ (h,t,e,())),(fun h t e => xs ⟨1,by simp [PublicIO.forwardInputs]⟩ (h,t,e,())),(fun h t e => xs ⟨2,by simp [PublicIO.forwardInputs]⟩ (h,t,e,()))⟩
  invFun := fun xs => (Fin.cases (fun j => xs.q j.1 j.2.1 j.2.2.1) (Fin.cases (fun j => xs.k j.1 j.2.1 j.2.2.1) (Fin.cases (fun j => xs.v j.1 j.2.1 j.2.2.1) (fun k => k.elim0))))
  left_inv := by intro xs; funext o j; fin_cases o <;> rfl
  right_inv := by intro xs; cases xs; rfl
/-- Natural-coordinate lookup preserves the old independent zero padding. -/
def forwardTensor (s : Shape) (xs : ForwardInputs s) : Inputs ℝ :=
  PublicIO.forwardLogical 0 s ((forwardInputs s).symm xs)
structure BackwardInputs (s : Shape) where
  q : Fin (s.B*s.H) → Fin s.T → Fin s.DK → ℝ
  k : Fin (s.B*s.H) → Fin s.T → Fin s.DK → ℝ
  v : Fin (s.B*s.H) → Fin s.T → Fin s.DV → ℝ
  g : Fin (s.B*s.H) → Fin s.T → Fin s.DV → ℝ
def backwardInputs (s : Shape) : ChannelValues ℝ (PublicIO.backwardInputs s) ≃ BackwardInputs s where
  toFun := fun xs => ⟨(fun h t e => xs ⟨0,by simp [PublicIO.backwardInputs]⟩ (h,t,e,())),(fun h t e => xs ⟨1,by simp [PublicIO.backwardInputs]⟩ (h,t,e,())),(fun h t e => xs ⟨2,by simp [PublicIO.backwardInputs]⟩ (h,t,e,())),(fun h t e => xs ⟨3,by simp [PublicIO.backwardInputs]⟩ (h,t,e,()))⟩
  invFun := fun xs => (Fin.cases (fun j => xs.q j.1 j.2.1 j.2.2.1) (Fin.cases (fun j => xs.k j.1 j.2.1 j.2.2.1) (Fin.cases (fun j => xs.v j.1 j.2.1 j.2.2.1) (Fin.cases (fun j => xs.g j.1 j.2.1 j.2.2.1) (fun k => k.elim0)))))
  left_inv := by intro xs; funext o j; fin_cases o <;> rfl
  right_inv := by intro xs; cases xs; rfl
/-- Natural-coordinate lookup preserves the old independent zero padding. -/
def backwardTensor (s : Shape) (xs : BackwardInputs s) : Inputs ℝ :=
  PublicIO.backwardLogical 0 s ((backwardInputs s).symm xs)

def forwardOutputs (s : Shape) : ChannelValues ℝ (PublicIO.forwardOutputs s) ≃ (Fin 128 → Fin s.BV → ℝ) where
  toFun := fun ys r c => ys ⟨0,by simp [PublicIO.forwardOutputs]⟩ (r,c,())
  invFun := fun ys => Fin.cases (fun j => ys j.1 j.2.1) (fun k => k.elim0)
  left_inv := by intro ys; funext o j; fin_cases o; rfl
  right_inv := fun _ => rfl

structure BackwardOutputs (s : Shape) where
  query : Fin 64 → Fin s.BK → ℝ
  key : Fin 64 → Fin s.BK → ℝ
  value : Fin 64 → Fin s.BV → ℝ

def backwardOutputs (s : Shape) : ChannelValues ℝ (PublicIO.backwardOutputs s) ≃ BackwardOutputs s where
  toFun := fun ys => ⟨(fun r c => ys ⟨0,by simp [PublicIO.backwardOutputs]⟩ (r,c,())),
    (fun r c => ys ⟨1,by simp [PublicIO.backwardOutputs]⟩ (r,c,())),
    (fun r c => ys ⟨2,by simp [PublicIO.backwardOutputs]⟩ (r,c,()))⟩
  invFun := fun ys => Fin.cases (fun j => ys.query j.1 j.2.1)
    (Fin.cases (fun j => ys.key j.1 j.2.1)
      (Fin.cases (fun j => ys.value j.1 j.2.1) (fun k => k.elim0)))
  left_inv := by intro ys; funext o j; fin_cases o <;> rfl
  right_inv := by intro ys; cases ys; rfl

/-- The exact per-program feature shards, without wrapper summation. -/
def forwardIO (p : DotTrees) (s : Shape) (tile : Tile) :
    LogicalProgramIO ℝ (Expr ℝ) (ForwardInputs s) (Fin 128 → Fin s.BV → ℝ) where
  core := PublicIO.forwardIO realScalar p s tile
  inputs := forwardInputs s
  outputs := forwardOutputs s

def backwardIO (p : DotTrees) (s : Shape) (tile : Tile) :
    LogicalProgramIO ℝ (Expr ℝ) (BackwardInputs s) (BackwardOutputs s) where
  core := PublicIO.backwardIO realScalar p s tile
  inputs := backwardInputs s
  outputs := backwardOutputs s
end
end VeriTile.Bench.Repairs.ParallelRetentionAttention.LogicalView
