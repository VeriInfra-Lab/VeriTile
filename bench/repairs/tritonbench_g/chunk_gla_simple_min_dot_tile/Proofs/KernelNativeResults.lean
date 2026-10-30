import bench.repairs.tritonbench_g.chunk_gla_simple_min_dot_tile.Proofs.ScalarGraph

namespace VeriTile.Bench.Repairs.ChunkGlaSimpleMinDotTile.KernelNativeResults
open VeriTile Triton FP.Structural
open Host
noncomputable section

/-- Complete source execution and independent ordered scalar-FMA target.
Fixed scalar primitives delimit numerical realization, without output ports. -/
theorem whole_kernel (c : Shape) (bt : 0<c.BT) (s : State NativeGraph.Expr) (scale : BitVec 32) :
    ∃ final,FPGrid.run ScalarGraph.algebra c ScalarGraph.scaleTag (FPGrid.programs c) s=some final ∧
      FPGrid.Result ScalarGraph.algebra c ScalarGraph.scaleTag (ScalarGraph.dots c) s final ∧
      (∀ h t v,ScalarGraph.lower scale ((final.mem "o" (h.val*(c.T*c.V)+t.val*c.V+v.val)).read .fp32)=
        ScalarGraph.value c s scale h t v bt) :=
  ScalarGraph.complete c bt s scale

#axiomsClean whole_kernel
end
end VeriTile.Bench.Repairs.ChunkGlaSimpleMinDotTile.KernelNativeResults
