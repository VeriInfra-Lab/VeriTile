import bench.repairs.tritonbench_g.rbe_triton_transform_checked_launch.Proofs.LogicalReader
namespace VeriTile.Bench.Repairs.RbeTritonTransformCheckedLaunch.Correct
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.RbeTritonTransform
open scoped VeriTile.Triton.FP.Structural.LogicalProgramIO
noncomputable section
/-- Independent dense rotation on [batch,row,column], including odd tails. -/
def result (r : Request) (xs : LogicalView.Tensor r) (b : Fin r.batches)
    (m : Fin r.rows) (c : Fin r.columns) : ℝ :=
  let n := c.val/2*2
  let u := LogicalView.column r xs b m n
  let v := LogicalView.column r xs b m (n+1)
  let a := ((r.position : ℝ)+(m.val : ℝ))/Real.rpow 10000 ((n%128 : Nat)/(128 : ℝ))
  if decide (c.val%2=1) then u*Real.sin a+v*Real.cos a else u*Real.cos a-v*Real.sin a

/-- The complete original two-axis grid, both barriers, all odd/empty shapes,
source safety, successful execution and exact exterior frame are preserved. -/
specification kernel_correct (r : Request) (X Y : RegionName) (different : X≠Y)
    (qs : List Runtime.Program) (all : qs.Perm (Runtime.programs r)) :
    VeriTile.Spec.Real (LogicalView.rotationIO r X Y qs ⊨ result r) :=
  (LogicalReader.judgment_iff r X Y qs).mpr (ReaderBefore.kernel_correct r X Y different qs all)
#axiomsClean kernel_correct
end
end VeriTile.Bench.Repairs.RbeTritonTransformCheckedLaunch.Correct
