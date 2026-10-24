import bench.repairs.tritonbench_g.chunk_gla_simple_min_dot_tile.Proofs.LogicalIO
import bench.repairs.tritonbench_g.chunk_gla_simple_min_dot_tile.Proofs.KernelRealResults

namespace VeriTile.Bench.Repairs.ChunkGlaSimpleMinDotTile.Correct
open VeriTile Triton
open Host
open scoped VeriTile.Triton.LogicalScheduledKernelIO
open scoped BigOperators
noncomputable section

/- Independent scalar mathematics and the complete logical output channels.
These definitions are definitionally connected to the canonical target used below. -/
namespace Mathematics
open VeriTile.Bench.TritonBenchG.ChunkGlaSimple.Math
def qhSum (p : Layout) (l : Launch) (a : Inputs) (n i j : Nat) : ℝ :=
  ∑ kb ∈ Finset.range n, ∑ e : Fin p.BK,
    query p l a i (kb*p.BK+e.val)*history p l a (kb*p.BK+e.val) j

def qkSum (p : Layout) (l : Launch) (a : Inputs) (n i j : Nat) : ℝ :=
  ∑ kb ∈ Finset.range n, ∑ e : Fin p.BK,
    query p l a i (kb*p.BK+e.val)*key p l a (kb*p.BK+e.val) j

def score (p : Layout) (l : Launch) (a : Inputs) (i j : Nat) : ℝ :=
  if j ≤ i then qkSum p l a (numBlocks p) i j*Real.exp (gate p l a i-gate p l a j) else 0

def expected (p : Layout) (l : Launch) (a : Inputs) (scale : ℝ) (i j : Nat) : ℝ :=
  (qhSum p l a (numBlocks p) i j*Real.exp (gate p l a i)+
    ∑ t : Fin p.BT, score p l a i t.val*value p l a t.val j)*scale

def result (c : Shape) (bt : 0<c.BT) (scale : ℝ) (xs : PublicIO.Logical.Inputs c)
    (coordinate : Fin c.heads × Fin c.T × Fin c.V) : ℝ :=
  expected (Host.layout c)
    (PublicIO.Logical.tileLaunch c bt coordinate.1 coordinate.2.1 coordinate.2.2)
    (PublicIO.Logical.arrays c scale xs) scale
    (coordinate.2.1.val%c.BT) (coordinate.2.2.val%c.BV)
end Mathematics


/-- Complete packed Q/K/V/history/gate inputs, FP32 source interpreted over
Real, complete output, and the actual three-dimensional source schedule. -/
def chunkIO (c : Shape) (scale : ℝ) := PublicIO.Logical.io c scale

/-- Every output is the independent gated-history plus causal QK/V formula.
The shared judgment includes actual source safety, successful complete grid
execution and exact exterior memory preservation. -/
specification whole_kernel (c : Shape) (bt : 0<c.BT) (scale : ℝ) :
    Spec.Real (chunkIO c scale ⊨ Mathematics.result c bt scale) :=
  PublicIO.Logical.correct c bt scale

#axiomsClean whole_kernel
#stmtSurfaceSubset whole_kernel ⊆
  [Spec.Real,LogicalScheduledKernelIO.Implements,chunkIO,
   Mathematics.result,PublicIO.Logical.Inputs,Shape,Shape.BT,Shape.heads,Shape.T,Shape.V]
#specNonCircular Mathematics.result avoiding
  [kernel,DirectReal.result,runAlgorithmSchedule,FPGrid.run,FP.Structural.exec]
end
end VeriTile.Bench.Repairs.ChunkGlaSimpleMinDotTile.Correct
