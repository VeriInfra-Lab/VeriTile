import bench.repairs.tritonbench_g.nested_loops_processing.Proofs.TypedMiddle

namespace VeriTile.Bench.Repairs.NestedLoopsProcessing.TypedExecution
open VeriTile.Triton FP.Structural Kernels.LegacyPastedSurface
set_option maxHeartbeats 1000000
set_option maxRecDepth 10000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def advances : Nat → OffsetTile → Nat → OffsetTile
  | 0, p, _ => p
  | n + 1, p, delta => bump (advances n p delta) delta

theorem middleLoop_registers (count : Nat) (u : Unique) (s : State α)
    (a c : OffsetTile) (v1 : Payload) (delta : Nat) :
    (middleLoopState count u s a c v1 delta).regs .nat [2,2] "a_offsets" = some (advances 6 a delta) ∧
    (middleLoopState count u s a c v1 delta).regs .nat [2,2] "c_offsets" = some (advances 12 c delta) := by
  simp [middleLoopState, middleState, middlePrefixState, innerLoopState, innerState,
    scatter_regs, advances]

def outerPrefix (count : Nat) (cols : Nat) : List ComputeStmt :=
  (outer count cols).take 1

def outerSuffix (count : Nat) (cols : Nat) : List ComputeStmt :=
  (outer count cols).drop 2

def outerState (count : Nat) (u : Unique) (s : State α)
    (a c : OffsetTile) (delta : Nat) : State α :=
  let t := s.setReg "a1" .int [2,2] (loaded count s a)
  let t := middleLoopState count u t a c (loaded count s a) delta
  t.setReg "a_offsets" .nat [2,2] (advances 7 a delta)

theorem outer_decomposition (count : Nat) (cols : Nat) :
    outer count cols = outerPrefix count cols ++
      [.forRange "j" 0 2 1 (middle count cols)] ++
      outerSuffix count cols := by
  rfl

theorem outer_prefix_step (M : Algebra α) (count : Nat) (u : Unique)
    (cols : Nat) (s : State α) (a : OffsetTile)
    (ha : s.regs .nat [2,2] "a_offsets" = some a) :
    run M (outerPrefix count cols) s =
      some (s.setReg "a1" .int [2,2] (loaded count s a)) := by
  all_goals simp [ComputeDType.eraseDType, Option.bind, outerPrefix, outer, source, kernel, ComputeKernel.surfaceBody,
    run, step, evalExpr, evalComputeOp, evalOp_unfold, natLt, natLe,
    ha, loaded, ← min_def_lt]
  all_goals unfold FP.Structural.bop
  all_goals simp [ComputeDType.eraseDType, Option.bind, Broadcast.leftIndex,
    Broadcast.rightIndex, ha, loaded, ← min_def_lt]
  all_goals rfl

theorem outer_suffix_step (M : Algebra α) (count : Nat) (u : Unique)
    (cols : Nat) (s : State α) (a : OffsetTile)
    (ha : s.regs .nat [2,2] "a_offsets" = some a) :
    run M (outerSuffix count cols) s =
      some (s.setReg "a_offsets" .nat [2,2] (bump a 2)) := by
  all_goals simp [ComputeDType.eraseDType, Option.bind, outerSuffix, outer, source, kernel, ComputeKernel.surfaceBody,
    run, step, evalExpr, evalComputeOp, evalOp_unfold, numeric, natLt, natLe, ha, bump]
  all_goals unfold FP.Structural.bop
  all_goals simp [ComputeDType.eraseDType, Option.bind, Broadcast.leftIndex, Broadcast.rightIndex, ha, bump]
  all_goals rfl

theorem outer_step (M : Algebra α) (count : Nat) (u : Unique)
    (cols : Nat) (s : State α) (a c : OffsetTile)
    (ha : s.regs .nat [2,2] "a_offsets" = some a)
    (hc : s.regs .nat [2,2] "c_offsets" = some c)
    (hu : s.regs .bool [2,2] "unique" = some u) :
    run M (outer count cols) s =
      some (outerState count u s a c 2) := by
  have hp := outer_prefix_step M count u cols s a ha
  have hl := middle_loop M count u cols
    (s.setReg "a1" .int [2,2] (loaded count s a))
    a c (loaded count s a) (by simpa using ha) (by simpa using hc) (by simp) (by simpa using hu)
  have regs := middleLoop_registers count u
    (s.setReg "a1" .int [2,2] (loaded count s a))
    a c (loaded count s a) 2
  have hs := outer_suffix_step M count u cols _ _ regs.1
  rw [outer_decomposition]
  simp only [run_append, hp, run, step, hl, Option.bind_eq_bind, Option.bind_some]
  exact hs

end VeriTile.Bench.Repairs.NestedLoopsProcessing.TypedExecution

#guard_msgs (drop info) in
#auditModuleAxioms
