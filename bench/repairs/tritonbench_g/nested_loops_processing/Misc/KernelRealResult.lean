import bench.repairs.tritonbench_g.nested_loops_processing.Proofs.KernelState
import bench.repairs.tritonbench_g.nested_loops_processing.Proofs.Synchronization

namespace VeriTile.Bench.Repairs.NestedLoopsProcessing.KernelRealResult
open VeriTile.Triton FP.Structural
open KernelResult TypedExecution
open VeriTile.Bench.Repairs.LayernormFwdTriton.Memory
noncomputable section

/-- Complete source execution and the independent signed-int32 Real target,
under ordinary pre-launch input and output memory contents. -/
theorem kernel_correctness {α : Type} [Inhabited α]
    (M : Algebra α) (rows cols : Nat) (positive : 0<rows*cols) (s : State α)
    (inputs : ∀ i, i<rows*cols → s.mem "input" i = .mk .int (signedValue (arangeWord i)))
    (outputs : ∀ i, i<rows*cols → s.mem "output" i = .mk .int (0 : Int)) :
    ∃ final : State α,
      exec M (Kernels.source (rows*cols) cols) s = some final ∧
      (∀ i, i<rows*cols → ((final.mem "output" i).read .int : ℝ) = realTarget rows cols i) ∧
      (∀ (r : RegionName) i, r ≠ "output" ∨ rows*cols ≤ i → final.mem r i = s.mem r i) := by
  refine ⟨finalState (rows*cols) cols s, source_run M _ _ _, ?_, final_frame _ _ _⟩
  intro i hi
  rw [initialized_cell s rows cols positive inputs outputs i hi, Cell.read_same]
  exact int32_real_exact rows cols i

/-- Empty extents take the wrapper's skip-launch branch and query no pointers. -/
theorem empty_output (rows cols : Nat) (empty : rows*cols=0) :
    ∀ i, i<rows*cols → realTarget rows cols i=0 := by
  intro i hi
  omega

abbrev empty_kernel_skip := @empty_launch

abbrev source_barriers := @TypedExecution.inner_barriers
abbrev native_store_order := @CtaBarrierContract.cross_store_order
abbrev source_final_offsets := @TypedExecution.final_offsets
abbrev source_integer_offsets := @offset_intermediates
abbrev physical_frame := @execute_frame

#axiomsClean kernel_correctness
#axiomsClean empty_output
#specNonCircular realTarget avoiding [Kernels.source,TypedExecution.finalState,sourceStores,execute]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.NestedLoopsProcessing.KernelRealResult
