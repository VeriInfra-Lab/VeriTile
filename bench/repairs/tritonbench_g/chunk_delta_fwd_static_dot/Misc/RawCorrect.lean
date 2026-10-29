import bench.repairs.tritonbench_g.chunk_delta_fwd.RealOutputs

namespace VeriTile.Bench.Repairs.ChunkDeltaFwdStaticDot.RawCorrect
open VeriTile Triton FP.Structural ChunkDeltaFwd
noncomputable section

/-- Complete raw grid with the independent mathematical recurrence, every
history/corrected/final output, and the full exterior Cell frame. -/
theorem chunk_delta_kernel_correct (p : Parameters) (shape : Geometry.ShapeValid p)
    (schedule : List PID) (order : schedule.Perm (grid p)) (before : Memory ℝ) :
    Spec.Real (∃ final,runGrid Real.algebra Real.dot p schedule before=some final ∧
      Real.Spec p before before final) :=
  Real.whole p shape schedule order before before (SameInputs.refl before)

abbrev arbitrary_grid_order := @chunk_delta_kernel_correct
#axiomsClean chunk_delta_kernel_correct
end
end VeriTile.Bench.Repairs.ChunkDeltaFwdStaticDot.RawCorrect
