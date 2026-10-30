import bench.repairs.tritonbench_g.nested_loops_processing.Proofs.SignedFinish
import bench.repairs.tritonbench_g.nested_loops_processing.Proofs.TypedMemory
import bench.tritonbench_g.kcache_copy_triton.Proofs.TypedCopy

/-! The actual signed-register execution and the historical natural-address
skeleton perform exactly the same memory transactions. Each lemma covers all
cells and allows arbitrary initial memory, including arbitrary payload words.
The decomposition is proof-only; `Kernels` contains the pasted authority. -/
namespace VeriTile.Bench.Repairs.NestedLoopsProcessing.SignedExecution
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy (masked_scatter_memory)
set_option maxHeartbeats 2000000
set_option maxRecDepth 10000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem loaded_memory (count : Nat) (s t : State α) (a : OffsetTile)
    (same : s.mem = t.mem) : loaded count s a = TypedExecution.loaded count t a := by
  unfold loaded TypedExecution.loaded
  rw [same]

theorem innerState_memory (count : Nat) (u : Unique) (s t : State α)
    (a c : OffsetTile) (v1 v2 : Payload) (delta : Nat) (same : s.mem = t.mem) :
    (innerState count u s a c v1 v2 delta).mem =
      (TypedExecution.innerState count u t a c v1 v2 delta).mem := by
  simp only [innerState,TypedExecution.innerState,scatter,TypedExecution.scatter,
    loaded,TypedExecution.loaded,bump,TypedExecution.bump,State.setReg_mem,
    masked_scatter_memory,same]
  rfl

theorem innerLoopState_memory (count : Nat) (u : Unique) (s t : State α)
    (a c : OffsetTile) (v1 v2 : Payload) (delta : Nat) (same : s.mem = t.mem) :
    (innerLoopState count u s a c v1 v2 delta).mem =
      (TypedExecution.innerLoopState count u t a c v1 v2 delta).mem := by
  unfold innerLoopState TypedExecution.innerLoopState
  apply innerState_memory
  exact innerState_memory count u _ _ a c v1 v2 delta same

theorem middleState_memory (count : Nat) (u : Unique) (s t : State α)
    (a c : OffsetTile) (v1 : Payload) (delta : Nat) (same : s.mem = t.mem) :
    (middleState count u s a c v1 delta).mem =
      (TypedExecution.middleState count u t a c v1 delta).mem := by
  unfold middleState TypedExecution.middleState
  rw [loaded_memory count s t (bump a delta) same]
  apply innerLoopState_memory
  exact same

theorem middleLoopState_memory (count : Nat) (u : Unique) (s t : State α)
    (a c : OffsetTile) (v1 : Payload) (delta : Nat) (same : s.mem = t.mem) :
    (middleLoopState count u s a c v1 delta).mem =
      (TypedExecution.middleLoopState count u t a c v1 delta).mem := by
  unfold middleLoopState TypedExecution.middleLoopState
  apply middleState_memory
  exact middleState_memory count u _ _ a c v1 delta same

theorem outerState_memory (count : Nat) (u : Unique) (s t : State α)
    (a c : OffsetTile) (delta : Nat) (same : s.mem = t.mem) :
    (outerState count u s a c delta).mem =
      (TypedExecution.outerState count u t a c delta).mem := by
  unfold outerState TypedExecution.outerState
  simp only [State.setReg_mem]
  rw [loaded_memory count s t a same]
  apply middleLoopState_memory
  exact same

theorem outerLoopState_memory (count : Nat) (u : Unique) (s t : State α)
    (a c : OffsetTile) (delta : Nat) (same : s.mem = t.mem) :
    (outerLoopState count u s a c delta).mem =
      (TypedExecution.outerLoopState count u t a c delta).mem := by
  unfold outerLoopState TypedExecution.outerLoopState
  apply outerState_memory
  exact outerState_memory count u _ _ a c delta same

/-- Every final memory cell agrees. Intermediate registers retain their actual
signed source types and are deliberately not equated to the old Nat registers. -/
theorem finalState_memory (count cols : Nat) (s : State α) :
    (finalState count cols s).mem = (TypedExecution.finalState count cols s).mem := by
  apply outerLoopState_memory
  rfl

#axiomsClean finalState_memory
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.NestedLoopsProcessing.SignedExecution
