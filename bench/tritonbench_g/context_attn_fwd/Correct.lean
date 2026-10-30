import bench.tritonbench_g.context_attn_fwd.Proofs.LogicalIO

namespace VeriTile.Bench.Repairs.ContextAttnFwd.PublicKernel
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ContextAttnFwd
open scoped VeriTile.Triton.LogicalDomainIO
open scoped BigOperators
noncomputable section

/- Independent scalar mathematics and the complete logical output channels.
These definitions are definitionally connected to the canonical target used below. -/
namespace Mathematics
open VeriTile.Bench.TritonBenchG.ContextAttnFwd.Formula
def score (p : Layout) (l : Launch) (a : RealInterface.Logical.Arrays) (scale : ℝ) (i j : Nat) : ℝ :=
  if (j : Int) ≤ queryRow p l i+a.prompt (batch p l) then
    (∑ e : Fin p.D, a.query l i e.val * a.key l j e.val) * scale
  else -100000000
end Mathematics

def mathematical (p : Layout) (g : Grid) (scale : ℝ)
    (a : RealInterface.Logical.Arrays) (e : RealInterface.Event p g) : ℝ :=
  let l := RealInterface.launch e.1
  (∑ k : Fin (a.blocks l*p.BN),
    pow2 (Mathematics.score p l a scale e.2.1.val k.val) *
      a.value l k.val e.2.2.1.val) /
    ∑ k : Fin (a.blocks l*p.BN),
      pow2 (Mathematics.score p l a scale e.2.1.val k.val)


/-- The complete original FP32 source in Real semantics: finite typed inputs,
the independent base-2 weighted sum, actual access safety and the full frame.
All ordinary-grid CTAs read the same initial arrays. -/
specification correct_real (p : Layout) (r : Regions) (extent : RegionBounds)
    (g : Grid) (scale : ℝ) (positive : 0 < p.BN) :
    Spec.Real (RealInterface.Logical.io p r extent g scale ⊨{RealInterface.Logical.domain p r extent g scale}
      (fun values => mathematical p g scale values.arrays)) :=
  RealInterface.Logical.correct_arrays p r extent g scale positive (mathematical p g scale) (by intros; rfl)

#stmtSurfaceSubset correct_real ⊆ [Spec.Real,RealInterface.Logical.io,RealInterface.Logical.domain,
  mathematical,LogicalDomainIO.ImplementsOn,Layout,Regions,RegionBounds,Grid,
  VeriTile.Bench.TritonBenchG.ContextAttnFwd.Formula.Layout.BN,
  RealInterface.Values,RealInterface.Event,GridIndex,TileIndex,
  VeriTile.Bench.TritonBenchG.ContextAttnFwd.Formula.Layout.BM,
  VeriTile.Bench.TritonBenchG.ContextAttnFwd.Formula.Layout.D,GridIndex.fintype,
  RealInterface.Logical.Inputs,RealInterface.Logical.Inputs.arrays,RealInterface.Logical.Arrays,AlgorithmProgramIO.ImplementsOn,RealInterface.io]
#axiomsClean correct_real
end
end VeriTile.Bench.Repairs.ContextAttnFwd.PublicKernel
