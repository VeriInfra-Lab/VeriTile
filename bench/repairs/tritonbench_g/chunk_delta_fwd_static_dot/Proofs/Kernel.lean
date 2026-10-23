import bench.repairs.tritonbench_g.chunk_delta_fwd_static_dot.Contract
import bench.repairs.tritonbench_g.chunk_delta_fwd.RealOutputs

namespace VeriTile.Bench.Repairs.ChunkDeltaFwdStaticDot.Kernel
open VeriTile Triton FP.Structural ChunkDeltaFwd
noncomputable section

/-- Any complete CTA order executes the same shared quoted source and satisfies
the tensor coverage/ownership specification, including its exterior frame. -/
theorem whole_grid {α : Type} [Inhabited α] (M : Algebra α)
    (D : VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput.DotPrimitive α)
    (r : Request) (h : InputContract r) (schedule : List PID)
    (order : schedule.Perm (grid r.params)) (input : Memory α) :
    ∃ final,runGrid M D r.params schedule input=some final ∧
      GridSpec M D r.params (r.shape (base_checked r h)) input input final :=
  ChunkDeltaFwd.whole_grid M D r.params (r.shape (base_checked r h))
    schedule order input input (SameInputs.refl input)

theorem real_grid (r : Request) (h : InputContract r) (schedule : List PID)
    (order : schedule.Perm (grid r.params)) (input : Memory ℝ) :
    ∃ final,runGrid Real.algebra Real.dot r.params schedule input=some final ∧
      Real.Spec r.params input input final := by
  obtain ⟨final,execution,result⟩ := whole_grid Real.algebra Real.dot r h schedule order input
  exact ⟨final,execution,Real.grid_to_real _ _ _ _ _ result⟩

#axiomsClean whole_grid
#axiomsClean real_grid
end
end VeriTile.Bench.Repairs.ChunkDeltaFwdStaticDot.Kernel
