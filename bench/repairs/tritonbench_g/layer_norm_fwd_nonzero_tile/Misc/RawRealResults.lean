import bench.repairs.tritonbench_g.layer_norm_fwd_nonzero_tile.Source
import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.RealRevision.ActiveGrid

namespace VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open VeriTile.Bench.Repairs.LayerNormFwd
noncomputable section

/-- Every row executes the complete shared quotation. The target uses only
input reads, logical-column sums, canonical Philox and total Real operations. -/
theorem raw_correct (f : Flags) (r : Regions) (stride : Strides) (M N : Nat)
    (eps p : ℝ) (width : widthCheck N=Bool.true)
    (sep : Domain.ReadWriteSeparated f r) (geometry : NativeGeometry.Geometry f r stride M N)
    (initial : BlockState) :
    ∃ final, Nonempty (Kernel.GridLaunchedOrdinary
      ((quoted f r stride M N eps p).eraseDType.toAlgKernel)
      (WholeGrid.grid M) initial final) ∧
      WholeGrid.Post f r stride M N eps p initial final ∧
      HostReturn.observe f r stride M N final=HostReturn.oracle f r stride M N eps p initial := by
  obtain ⟨final,run,post⟩ := RealRevision.ActiveGrid.source_correct f r stride M N eps p
    (tile N) (tile_arange N) ((widthCheck_iff N).mp width) sep geometry initial
  exact ⟨final,run,post,RealRevision.ActiveGrid.observe_eq_oracle f r stride M N eps p
    initial final geometry sep post⟩

/-- The Real empty-row convention is explicit; it is not a native FP claim. -/
theorem real_zero_stats (f : Flags) (inputs : Inputs) (M row : Nat) (eps p : ℝ) :
    Formula.mean f inputs M 0 p row=0 ∧
    Formula.variance f inputs M 0 p row=0 ∧
    Formula.rstd f inputs M 0 eps p row=1/Real.sqrt eps := by
  simp [Formula.mean,Formula.variance,Formula.rstd]

#axiomsClean raw_correct
#axiomsClean real_zero_stats
end
end VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile
