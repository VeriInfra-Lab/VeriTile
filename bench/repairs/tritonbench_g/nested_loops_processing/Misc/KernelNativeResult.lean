import bench.repairs.tritonbench_g.nested_loops_processing.Proofs.SignedOffsets
import bench.repairs.tritonbench_g.nested_loops_processing.Proofs.KernelState
import bench.repairs.tritonbench_g.nested_loops_processing.Proofs.TypedFP
import bench.repairs.tritonbench_g.nested_loops_processing.Proofs.Synchronization

namespace VeriTile.Bench.Repairs.NestedLoopsProcessing.KernelNativeResult
open VeriTile.Triton FP.Structural
open KernelResult TypedExecution
open VeriTile.Bench.Repairs.LayernormFwdTriton.Memory
noncomputable section

/-- Both complete typed sources succeed with exact signed-int32 outputs and
full exterior memory frames under raw input/output initialization. -/
theorem kernel_equiv {α : Type} [Inhabited α]
    (M : Algebra α) (rows cols : Nat) (positive : 0<rows*cols) (s : State α)
    (inputs : ∀ i, i<rows*cols → s.mem "input" i = .mk .int (signedValue (arangeWord i)))
    (outputs : ∀ i, i<rows*cols → s.mem "output" i = .mk .int (0 : Int)) :
    ∃ final : State α,
      exec M (Kernels.source (rows*cols) cols) s = some final ∧
      exec M (Kernels.commuted (rows*cols) cols) s = some final ∧
      (∀ i, i<rows*cols → final.mem "output" i = .mk .int (signedValue (wordTarget rows cols i))) ∧
      (∀ i, i<rows*cols → BitVec.ofInt 32 ((final.mem "output" i).read .int) = wordTarget rows cols i) ∧
      (∀ (r : RegionName) i, r ≠ "output" ∨ rows*cols ≤ i → final.mem r i = s.mem r i) := by
  refine ⟨actualFinalState (rows*cols) cols s, source_run M _ _ _, candidate_run M _ _ _, ?_, ?_, ?_⟩
  · rw [source_memory]
    exact initialized_cell s rows cols positive inputs outputs
  · intro i hi
    rw [source_memory,initialized_cell s rows cols positive inputs outputs i hi, Cell.read_same]
    simp [signedValue]
  · rw [source_memory]
    exact final_frame _ _ _

abbrev empty_kernel_skip := @empty_launch

abbrev all_states_equivalent := @source_equivalence
abbrev physical_frame := @execute_frame
abbrev source_barriers := @SignedExecution.inner_barriers
abbrev native_store_order := @CtaBarrierContract.cross_store_order
abbrev source_final_offsets := @SignedExecution.final_offsets

#axiomsClean kernel_equiv
#specNonCircular wordTarget avoiding [Kernels.source,Kernels.commuted,TypedExecution.finalState,sourceStores,execute]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.NestedLoopsProcessing.KernelNativeResult
