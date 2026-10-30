import bench.repairs.tritonbench_g.nested_loops_processing.Proofs.SignedMiddle

namespace VeriTile.Bench.Repairs.NestedLoopsProcessing.SignedExecution
open VeriTile.Triton FP.Structural Kernels
set_option maxHeartbeats 1000000
set_option maxRecDepth 10000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def advances : Nat → OffsetTile → Nat → OffsetTile
  | 0, p, _ => p
  | n + 1, p, delta => bump (advances n p delta) delta

theorem middleLoop_registers (count : Nat) (u : Unique) (s : State α)
    (a c : OffsetTile) (v1 : Payload) (delta : Nat) :
    (middleLoopState count u s a c v1 delta).regs .int [2,2] "a_offsets" = some (signed (advances 6 a delta)) ∧
    (middleLoopState count u s a c v1 delta).regs .int [2,2] "c_offsets" = some (signed (advances 12 c delta)) := by
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
  t.setReg "a_offsets" .int [2,2] (signed (advances 7 a delta))

theorem outer_decomposition (count : Nat) (cols : Nat) :
    outer count cols = outerPrefix count cols ++
      [.forRangeDyn "j" (.constNat 0) (.constNat 2) (.constNat 1) (middle count cols)] ++
      outerSuffix count cols := by
  rfl

theorem outer_prefix_step (M : Algebra α) (count : Nat) (u : Unique)
    (cols : Nat) (s : State α) (a : OffsetTile)
    (ha : s.regs .int [2,2] "a_offsets" = some (signed a))
    (context : Context count s) :
    run M (outerPrefix count cols) s =
      some (s.setReg "a1" .int [2,2] (loaded count s a)) := by
  rcases context with ⟨hn,hs⟩
  all_goals simp [ComputeDType.eraseDType, Option.bind, outerPrefix, outer, source, kernel, originalKernel, quotedOriginal, ComputeKernel.surfaceBody,
    run, step, evalExpr, evalComputeOp, evalOp_unfold, numeric, numericLt, numericLe, natLt, natLe,
    ha,hn,hs, loaded, ← min_def_lt, signed, Context, Int.natCast_add, Int.natCast_mul, Int.ofNat_lt, Int.ofNat_le, clamp_toNat]
  all_goals unfold FP.Structural.bop
  all_goals simp [ComputeDType.eraseDType, Option.bind, Broadcast.leftIndex,
    Broadcast.rightIndex, ha,hn,hs, loaded, ← min_def_lt, signed, Context, Int.natCast_add, Int.natCast_mul, Int.ofNat_lt, Int.ofNat_le, clamp_toNat]
  all_goals try simp only [signed,bump,cast_bump_two,Int.ofNat_lt,clamp_toNat,index_eta]
  all_goals rfl

theorem outer_suffix_step (M : Algebra α) (count : Nat) (u : Unique)
    (cols : Nat) (s : State α) (a : OffsetTile)
    (ha : s.regs .int [2,2] "a_offsets" = some (signed a))
    (context : Context count s) :
    run M (outerSuffix count cols) s =
      some (s.setReg "a_offsets" .int [2,2] (signed (bump a 2))) := by
  rcases context with ⟨hn,hs⟩
  all_goals simp [ComputeDType.eraseDType, Option.bind, outerSuffix, outer, source, kernel, originalKernel, quotedOriginal, ComputeKernel.surfaceBody,
    run, step, evalExpr, evalComputeOp, evalOp_unfold, numeric, numericLt,numericLe,natLt, natLe, ha,hn,hs, bump, signed, Context, Int.natCast_add]
  all_goals unfold FP.Structural.bop
  all_goals simp [ComputeDType.eraseDType, Option.bind, Broadcast.leftIndex, Broadcast.rightIndex, ha,hn,hs, bump, signed, Context, Int.natCast_add]
  all_goals try simp only [signed,bump,cast_bump_two,Int.ofNat_lt,clamp_toNat,index_eta]
  all_goals rfl

theorem outer_step (M : Algebra α) (count : Nat) (u : Unique)
    (cols : Nat) (s : State α) (a c : OffsetTile)
    (ha : s.regs .int [2,2] "a_offsets" = some (signed a))
    (hc : s.regs .int [2,2] "c_offsets" = some (signed c))
    (hu : s.regs .bool [2,2] "unique" = some u)
    (context : Context count s) :
    run M (outer count cols) s =
      some (outerState count u s a c 2) := by
  have hp := outer_prefix_step M count u cols s a ha context
  have hl := middle_loop M count u cols
    (s.setReg "a1" .int [2,2] (loaded count s a))
    a c (loaded count s a) (by simpa using ha) (by simpa using hc) (by simp) (by simpa using hu) (by simpa [Context] using context)
  have regs := middleLoop_registers count u
    (s.setReg "a1" .int [2,2] (loaded count s a))
    a c (loaded count s a) 2
  have hs := outer_suffix_step M count u cols _ _ regs.1 (by simpa [Context,innerState,innerLoopState,middlePrefixState,middleState,middleLoopState,outerState] using context)
  rw [outer_decomposition]
  simp only [run_append, hp, run, step, evalOp_unfold, hl, Option.bind_eq_bind, Option.bind_some]
  exact hs

end VeriTile.Bench.Repairs.NestedLoopsProcessing.SignedExecution

#guard_msgs (drop info) in
#auditModuleAxioms
