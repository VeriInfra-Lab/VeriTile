import bench.repairs.tritonbench_g.layer_norm_fwd_nonzero_tile.Proofs.NativeMathematics
import bench.repairs.tritonbench_g.layer_norm_fwd_nonzero_tile.Proofs.NativeImplementation
import bench.repairs.tritonbench_g.layer_norm_fwd_nonzero_tile.Proofs.LogicalIO

namespace VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile.ActualLogical
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open VeriTile.Bench.Repairs.LayerNormFwd
open scoped VeriTile.Triton.LogicalIO
noncomputable section

theorem lift_formula (f : Flags) (r : Regions) (stride : Strides)
    (M N : Nat) (eps p : ℝ) (width : N ≤ tile N) :
    (io f r stride M N eps p).lift (LogicalIO.mathematical f M N eps p) =
      NativeInterface.target ActualMathematics.algebra ActualMathematics.comparisons
        f r stride M N eps p width := by
  rw [ActualMathematics.target]
  funext a
  change (outputArrays r M N).symm
      ((LogicalIO.outputs M N).symm (LogicalIO.mathematical f M N eps p
        (LogicalIO.inputs M N (inputArrays r M N a)))) = _
  have old := congrFun (LogicalIO.lift_formula f r stride M N eps p) (inputArrays r M N a)
  exact congrArg (outputArrays r M N).symm old

/-- The actual copied body implements the independent formula with successful
execution, actual access safety, every enabled output and the entire exterior
memory frame. No execution or whole-kernel premise is supplied. -/
theorem implements (f : Flags) (r : Regions) (stride : Strides)
    (M N : Nat) (eps p : ℝ) (width : widthCheck N = Bool.true)
    (sep : Domain.ReadWriteSeparated f r) (geometry : NativeGeometry.Geometry f r stride M N) :
    io f r stride M N eps p ⊨ LogicalIO.mathematical f M N eps p := by
  change (NativeInterface.io ActualMathematics.algebra f r stride M N eps p).Implements _
  rw [lift_formula f r stride M N eps p ((widthCheck_iff N).mp width)]
  exact NativeInterface.implements ActualMathematics.algebra ActualMathematics.comparisons
    f r stride M N eps p width sep geometry

#axiomsClean implements
end
end VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile.ActualLogical
