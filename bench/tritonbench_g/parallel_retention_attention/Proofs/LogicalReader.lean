import bench.tritonbench_g.parallel_retention_attention.Proofs.ReaderBefore
import bench.tritonbench_g.parallel_retention_attention.IO
namespace VeriTile.Bench.Repairs.ParallelRetentionAttention.LogicalReader
open VeriTile.Triton FP.Structural Kernel Kernel.Native
open scoped BigOperators
noncomputable section
/-- Logical zero padding in time and feature coordinates, and head decay. -/
def padded (T D : Nat) (x : Nat → Nat → Nat → ℝ) (bh t e : Nat) : ℝ :=
  if t<T ∧ e<D then x bh t e else 0

def decay (h : Nat) : ℝ := 1-(2:ℝ)^(-(5+(h:ℝ)))

/-- Forward and derivative products use exactly this feature shard. -/
def forwardTerm (s : Shape) (x : Inputs ℝ) (tile : Tile) (r v n : Nat) : ℝ :=
  (∑ e∈Finset.range s.BK,
      (padded s.T s.DK x.q tile.bh (tile.chunk*128+r) (tile.key*s.BK+e)*(Real.sqrt (s.DK:ℝ))⁻¹)*
      padded s.T s.DK x.k tile.bh n (tile.key*s.BK+e))*
    padded s.T s.DV x.v tile.bh n (tile.value*s.BV+v)

def queryTerm (s : Shape) (x : Inputs ℝ) (tile : Tile) (r e n : Nat) : ℝ :=
  (∑ v∈Finset.range s.BV,
      padded s.T s.DV x.g tile.bh (tile.chunk*64+r) (tile.value*s.BV+v)*
      padded s.T s.DV x.v tile.bh n (tile.value*s.BV+v))*
    padded s.T s.DK x.k tile.bh n (tile.key*s.BK+e)

def keyTerm (s : Shape) (x : Inputs ℝ) (tile : Tile) (r e t : Nat) : ℝ :=
  (∑ v∈Finset.range s.BV,
      padded s.T s.DV x.v tile.bh (tile.chunk*64+r) (tile.value*s.BV+v)*
      padded s.T s.DV x.g tile.bh t (tile.value*s.BV+v))*
    padded s.T s.DK x.q tile.bh t (tile.key*s.BK+e)

def valueTerm (s : Shape) (x : Inputs ℝ) (tile : Tile) (r v t : Nat) : ℝ :=
  (∑ e∈Finset.range s.BK,
      padded s.T s.DK x.k tile.bh (tile.chunk*64+r) (tile.key*s.BK+e)*
      padded s.T s.DK x.q tile.bh t (tile.key*s.BK+e))*
    padded s.T s.DV x.g tile.bh t (tile.value*s.BV+v)

def forwardFormula (s : Shape) (tile : Tile) (xs : LogicalView.ForwardInputs s)
    (row : Fin 128) (column : Fin s.BV) : ℝ :=
  let x := LogicalView.forwardTensor s xs
  ∑ n∈Finset.range (tile.chunk*128+row.val+1),
    decay (tile.bh%s.H)^(tile.chunk*128+row.val-n)*forwardTerm s x tile row.val column.val n

def backwardFormula (s : Shape) (tile : Tile) (xs : LogicalView.BackwardInputs s) :
    LogicalView.BackwardOutputs s :=
  let x := LogicalView.backwardTensor s xs
  { query := fun row column => (Real.sqrt (s.DK:ℝ))⁻¹*∑ n∈Finset.range (tile.chunk*64+row.val+1),
      decay (tile.bh%s.H)^(tile.chunk*64+row.val-n)*queryTerm s x tile row.val column.val n
    key := fun row column => (Real.sqrt (s.DK:ℝ))⁻¹*∑ n∈Finset.range s.T,if tile.chunk*64+row.val≤n then
      decay (tile.bh%s.H)^(n-(tile.chunk*64+row.val))*keyTerm s x tile row.val column.val n else 0
    value := fun row column => (Real.sqrt (s.DK:ℝ))⁻¹*∑ n∈Finset.range s.T,if tile.chunk*64+row.val≤n then
      decay (tile.bh%s.H)^(n-(tile.chunk*64+row.val))*valueTerm s x tile row.val column.val n else 0 }

theorem forward_iff (p : DotTrees) (s : Shape) (tile : Tile) :
    (LogicalView.forwardIO p s tile).Implements (forwardFormula s tile) ↔
    (ReaderBefore.forwardIO p s tile).Implements (ReaderBefore.forwardFormula s tile) := Iff.rfl
theorem backward_iff (p : DotTrees) (s : Shape) (tile : Tile) :
    (LogicalView.backwardIO p s tile).Implements (backwardFormula s tile) ↔
    (ReaderBefore.backwardIO p s tile).Implements (ReaderBefore.backwardFormula s tile) := Iff.rfl
end
end VeriTile.Bench.Repairs.ParallelRetentionAttention.LogicalReader
