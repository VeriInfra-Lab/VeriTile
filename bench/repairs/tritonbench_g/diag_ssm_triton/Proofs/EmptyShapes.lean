import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.SourceActions

/-! Empty Torch views can report data_ptr=0 while retaining a backing/offset.
The affine View coordinate is then unobserved: source access bounds prohibit
an empty-view word access, and the following exact zero-axis/time executions
show which source operations remain. -/
namespace VeriTile.Bench.Repairs.DiagSsmTriton.EmptyShapes
open Kernels ByteMemory Execution
set_option autoImplicit false

theorem schedule_zero_width (d : Dimensions) (schedule : SourceActions.Schedule d) (zero : d.width = 0) :
    schedule.order = [] := by
  apply List.eq_nil_iff_forall_not_mem.mpr
  intro pid member
  have bound := schedule.valid pid member
  simp [zero, NativeIndex.grid] at bound

theorem zero_width_grids (codec : Codec Word) (route : Route) (graph : Graph) (ops : Operations Word)
    (d : Dimensions) (fv : ForwardViews) (bv : BackwardViews) (schedule : SourceActions.Schedule d)
    (zero : d.width = 0) (m : Heap) :
    forwardGrid codec route graph ops d fv schedule.order m = m ∧
      backwardGrid codec route graph ops d bv schedule.order m = m := by
  rw [schedule_zero_width d schedule zero]
  exact ⟨rfl, rfl⟩

theorem zero_time_forward (codec : Codec Word) (route : Route) (graph : Graph) (ops : Operations Word)
    (d : Dimensions) (views : ForwardViews) (pid : Nat) (m : Heap) (zero : d.length = 0) :
    forwardProgram codec route graph ops d views pid m = forwardInitial codec route ops d views pid m := by
  simp [forwardProgram, zero, forwardLoop]

theorem zero_time_backward (codec : Codec Word) (route : Route) (graph : Graph) (ops : Operations Word)
    (d : Dimensions) (views : BackwardViews) (pid : Nat) (m : Heap) (zero : d.length = 0) :
    backwardProgram codec route graph ops d views pid m =
      backwardTerminal codec route ops d views pid (backwardInitial route ops d views pid m) := by
  simp [backwardProgram, zero, backwardLoop]

theorem empty_view_no_word (view : View) (empty : view.words = 0) : ¬ ∃ word, word < view.words := by
  simp [empty]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.DiagSsmTriton.EmptyShapes
