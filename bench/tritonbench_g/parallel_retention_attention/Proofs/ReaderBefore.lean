import bench.tritonbench_g.parallel_retention_attention.Proofs.PublicRealIO
import VeriTile.Meta.Specification

namespace VeriTile.Bench.Repairs.ParallelRetentionAttention.ReaderBefore
open VeriTile.Triton FP.Structural Kernel Kernel.Native
open scoped BigOperators VeriTile.Triton.FP.Structural.ProgramIO
noncomputable section

/- Complete per-program forward and combined backward kernels. Real is the
mathematical interpretation of the unchanged FP32 source. The channels expose
logical input tensors and the actual output shards; no private scratch is used
within these endpoints. The wrapper's later shard sum is outside their scope. -/

def forwardIO (p : DotTrees) (s : Shape) (tile : Tile) :
    ProgramIO ℝ (Expr ℝ) :=
  PublicIO.forwardIO realScalar p s tile

def backwardIO (p : DotTrees) (s : Shape) (tile : Tile) :
    ProgramIO ℝ (Expr ℝ) :=
  PublicIO.backwardIO realScalar p s tile


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

/-- Independent causal attention sum for this feature shard. -/
def forwardFormula (s : Shape) (tile : Tile) (xs : ChannelValues ℝ (PublicIO.forwardInputs s)) :
    ChannelValues ℝ (PublicIO.forwardOutputs s) :=
  let x := PublicIO.forwardLogical 0 s xs
  Fin.cases (fun j => (show ℝ from ∑ n∈Finset.range (tile.chunk*128+j.1.val+1),
    decay (tile.bh%s.H)^(tile.chunk*128+j.1.val-n)*forwardTerm s x tile j.1.val j.2.1.val n))
    (fun k => k.elim0)

/-- Independent differentiated causal sums for the three backward shards. -/
def backwardFormula (s : Shape) (tile : Tile) (xs : ChannelValues ℝ (PublicIO.backwardInputs s)) :
    ChannelValues ℝ (PublicIO.backwardOutputs s) := by
  let x := PublicIO.backwardLogical 0 s xs
  refine Fin.cases ?_ (Fin.cases ?_ (Fin.cases ?_ (fun k => k.elim0)))
  · intro j; exact (Real.sqrt (s.DK:ℝ))⁻¹*∑ n∈Finset.range (tile.chunk*64+j.1.val+1),
      decay (tile.bh%s.H)^(tile.chunk*64+j.1.val-n)*queryTerm s x tile j.1.val j.2.1.val n
  · intro j; exact (Real.sqrt (s.DK:ℝ))⁻¹*∑ n∈Finset.range s.T,if tile.chunk*64+j.1.val≤n then
      decay (tile.bh%s.H)^(n-(tile.chunk*64+j.1.val))*keyTerm s x tile j.1.val j.2.1.val n else 0
  · intro j; exact (Real.sqrt (s.DK:ℝ))⁻¹*∑ n∈Finset.range s.T,if tile.chunk*64+j.1.val≤n then
      decay (tile.bh%s.H)^(n-(tile.chunk*64+j.1.val))*valueTerm s x tile j.1.val j.2.1.val n else 0



theorem forward (p : DotTrees) (s : Shape) (tile : Tile)
    (valid : ProgramDomain s 128 tile) :
    Spec.Real (forwardIO p s tile ⊨ forwardFormula s tile) :=
  PublicIO.forward_real p s tile valid

theorem backward (p : DotTrees) (s : Shape) (tile : Tile)
    (valid : ProgramDomain s 64 tile) (keyPositive : 0<s.BK) (valuePositive : 0<s.BV) :
    Spec.Real (backwardIO p s tile ⊨ backwardFormula s tile) :=
  PublicIO.backward_real p s tile valid keyPositive valuePositive

end
end VeriTile.Bench.Repairs.ParallelRetentionAttention.ReaderBefore
