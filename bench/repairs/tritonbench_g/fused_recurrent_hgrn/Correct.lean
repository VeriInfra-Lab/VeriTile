import bench.repairs.tritonbench_g.fused_recurrent_hgrn.Proofs.LogicalCoordinates

namespace VeriTile.Bench.Repairs.FusedRecurrentHgrn
open VeriTile.Triton FP.Structural
open scoped VeriTile.Triton.FP.Structural.LogicalProgramIO
noncomputable section

/-- Independent recurrence: h[0] = h0 (or zero),
h[t+1] = g[t]*h[t]+x[t]; dx[t] = a[t+1]+do[t],
dg[t] = dx[t]*h[t], a[t] = dx[t]*g[t], a[T] = dht (or zero).
The output tuple is saved h[1..T], dx, dg, optional h[T]. -/
def formula (BH T D : Nat) (U V : Bool) (vals : IO.TensorInputs ℝ BH T D) :
    IO.TensorOutputs ℝ BH T D :=
  let input := IO.tensorData BH T D vals
  { savedState := fun bh t d => state realOps U input bh.val d.val (t.val+1)
    dx := fun bh t d => FusedRecurrentHgrn.dx realOps V input T bh.val d.val t.val
    dg := fun bh t d => FusedRecurrentHgrn.dg realOps U V input T bh.val d.val t.val
    finalState := fun bh d => state realOps U input bh.val d.val T }

/-- Scalar loss along x+e*vx and g+e*vg, with h0 held detached. -/
def objective (BH T D : Nat) (U V : Bool) (vals : IO.TensorInputs ℝ BH T D)
    (bh d : Nat) (vx vg : Nat → ℝ) (e : ℝ) : ℝ :=
  let input := IO.tensorData BH T D vals
  (∑ t∈Finset.range T, input "do" bh t d*pathState input U bh d vx vg e (t+1))+
    terminal realOps V input bh d*pathState input U bh d vx vg e T

abbrev kernelIO (BH T D BDF BDB : Nat) (U S V : Bool) :=
  IO.logicalIO BH T D BDF BDB U S V

/-- Complete forward/backward grids and their mathematical gradient. The same
dx/dg functions in formula give the directional derivative of objective. -/
specification kernel_correct (BH T D BDF BDB : Nat) (hF : 0<BDF) (hB : 0<BDB)
    (U S V : Bool) :
    VeriTile.Spec.Real ((kernelIO BH T D BDF BDB U S V ⊨ formula BH T D U V) ∧
      ∀ vals (bh : Fin BH) (d : Fin D) (vx vg : Nat → ℝ),
        HasDerivAt (objective BH T D U V vals bh.val d.val vx vg)
          (∑ t∈Finset.range T,
            (dx realOps V (IO.tensorData BH T D vals) T bh.val d.val t*vx t+
             dg realOps U V (IO.tensorData BH T D vals) T bh.val d.val t*vg t)) 0) :=
  IO.logical_correct BH T D BDF BDB hF hB U S V

#specNonCircular formula avoiding [forward,backward,Execution.gridRun,FP.Structural.exec]
#specNonCircular objective avoiding [forward,backward,Execution.gridRun,FP.Structural.exec]
#axiomsClean kernel_correct
#stmtSurfaceSubset kernel_correct ⊆
  [VeriTile.Spec.Real,kernelIO,LogicalProgramIO.Implements,formula,objective,
   IO.TensorInputs,IO.TensorOutputs,IO.tensorData,realOps,dx,dg,HasDerivAt,Region.mk,TileDType.real]
end
end VeriTile.Bench.Repairs.FusedRecurrentHgrn
