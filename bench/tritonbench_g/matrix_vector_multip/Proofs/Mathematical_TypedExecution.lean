import bench.tritonbench_g.matrix_vector_multip.Proofs.Mathematical_TypedSource
import VeriTile.Triton.Float.Control
import bench.tritonbench_g.isfinite_kernel.Proofs.Mathematical_MemoryProjection
namespace VeriTile.Bench.TritonBenchG.MatrixVectorMultip.Typed
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
structure Parameters where
  (a b out : RegionName)
  (rows columns strideRow strideColumn strideVector strideOut blockRows blockColumns : Nat)
def kernel (d : FloatDType) (p : Parameters) := source d p.a p.b p.out p.rows p.columns
  p.strideRow p.strideColumn p.strideVector p.strideOut p.blockRows p.blockColumns

def execution {α : Type} [Inhabited α] (M : Algebra α) (d : FloatDType)
    (p : Parameters) (s : State α) (hw : 0 < p.blockColumns) (hcols : p.columns ≤ p.blockColumns) :
    {t : State α // FP.Structural.exec M (kernel d p) s = some t} := by
  rcases p with ⟨a,b,o,n,m,ar,ac,bc,oc,bn,bm⟩
  have hnot : ¬ bm < m := Nat.not_lt.mpr hcols
  have hneq : bm ≠ 0 := Nat.ne_of_gt hw
  by_cases hm : 0 < m
  all_goals cases d
  all_goals
    refine ⟨?_, ?_⟩; swap
    simp [kernel, source, FP.Structural.exec, ComputeKernel.surfaceBody, run, step,
      evalExpr, evalComputeOp, evalOp_unfold, evalOffsets, store, numeric, bop,
      numericLt, natLt, ofFloat, toFloat, Region.cast, Option.bind,
      FloatDType.toTileDType, FloatDType.numericDType, ComputeDType.eraseDType,
      FP.Structural.range, hnot, hneq, hm, hw]
    try rfl
#axiomsClean execution
end
end VeriTile.Bench.TritonBenchG.MatrixVectorMultip.Typed
