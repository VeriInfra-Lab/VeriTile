import bench.repairs.tritonbench_g.nested_loops_processing.Proofs.SignedOuter
import bench.repairs.tritonbench_g.nested_loops_processing.Proofs.SignedInitialize

namespace VeriTile.Bench.Repairs.NestedLoopsProcessing.SignedExecution
open VeriTile.Triton FP.Structural Kernels
set_option maxHeartbeats 1000000
set_option maxRecDepth 10000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem outer_registers (count : Nat) (u : Unique) (s : State α)
    (a c : OffsetTile) (delta : Nat) :
    (outerState count u s a c delta).regs .int [2,2] "a_offsets" = some (signed (advances 7 a delta)) ∧
    (outerState count u s a c delta).regs .int [2,2] "c_offsets" = some (signed (advances 12 c delta)) := by
  have h := middleLoop_registers count u
    (s.setReg "a1" .int [2,2] (loaded count s a)) a c
    (loaded count s a) delta
  constructor
  · simp [outerState]
  · simpa [outerState] using h.2

def outerLoopState (count : Nat) (u : Unique) (s : State α)
    (a c : OffsetTile) (delta : Nat) : State α :=
  let t := outerState count u (s.setReg "i" .nat [] (fun _ ↦ 0)) a c delta
  outerState count u (t.setReg "i" .nat [] (fun _ ↦ 1))
    (advances 7 a delta) (advances 12 c delta) delta

theorem outer_loop (M : Algebra α) (count : Nat) (u : Unique)
    (cols : Nat) (s : State α) (a c : OffsetTile)
    (ha : s.regs .int [2,2] "a_offsets" = some (signed a))
    (hc : s.regs .int [2,2] "c_offsets" = some (signed c))
    (hu : s.regs .bool [2,2] "unique" = some u)
    (context : Context count s) :
    range M "i" 0 2 1 (outer count cols) s =
      some (outerLoopState count u s a c 2) := by
  rw [range_two]
  have e0 := outer_step M count u cols (s.setReg "i" .nat [] (fun _ ↦ 0)) a c
    (by simpa using ha) (by simpa using hc) (by simpa using hu) (by simpa [Context] using context)
  rw [e0]
  have regs := outer_registers count u (s.setReg "i" .nat [] (fun _ ↦ 0)) a c 2
  exact outer_step M count u cols _ _ _ (by simpa using regs.1) (by simpa using regs.2)
    (by simpa [outerState, middleLoopState, middleState, middlePrefixState, innerLoopState, innerState] using hu) (by simpa [Context,innerState,innerLoopState,middlePrefixState,middleState,middleLoopState,outerState] using context)

def finalState (count cols : Nat) (s : State α) : State α :=
  outerLoopState count (unique cols) (prefixState s count cols) (initialA cols) (initialC cols) 2

/-- Successful execution of the actual typed source, including every load,
masked store, barrier, cached tile and all eight inner-loop iterations. -/
theorem source_run (M : Algebra α) (count cols : Nat) (s : State α) :
    exec M (source count cols) s = some (finalState count cols s) := by
  change run M (source count cols).surfaceBody s = _
  rw [source_decomposition, run_append, prefix_step]
  have h := outer_loop M count (unique cols) cols (prefixState s count cols)
    (initialA cols) (initialC cols)
    (by simp [prefixState]) (by simp [prefixState]) (by simp [prefixState]) (by simp [prefixState,Context])
  simpa only [run, step, evalOp_unfold, h, finalState, Option.bind_eq_bind, Option.bind_some] using h

end VeriTile.Bench.Repairs.NestedLoopsProcessing.SignedExecution

#guard_msgs (drop info) in
#auditModuleAxioms
