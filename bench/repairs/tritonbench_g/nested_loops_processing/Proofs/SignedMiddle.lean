import bench.repairs.tritonbench_g.nested_loops_processing.Proofs.SignedLoop

namespace VeriTile.Bench.Repairs.NestedLoopsProcessing.SignedExecution
open VeriTile.Triton FP.Structural Kernels
set_option maxHeartbeats 1000000
set_option maxRecDepth 10000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def middlePrefix (count : Nat) (cols : Nat) : List ComputeStmt :=
  (middle count cols).take 2

def middlePrefixState (count : Nat) (u : Unique)
    (s : State α) (a : OffsetTile) (delta : Nat) : State α :=
  let t := s.setReg "a_offsets" .int [2,2] (signed (bump a delta))
  t.setReg "a2" .int [2,2] (loaded count s (bump a delta))

def middleState (count : Nat) (u : Unique) (s : State α)
    (a c : OffsetTile) (v1 : Payload) (delta : Nat) : State α :=
  innerLoopState count u (middlePrefixState count u s a delta) (bump a delta) c v1
    (loaded count s (bump a delta)) delta

theorem middle_decomposition (count : Nat) (cols : Nat) :
    middle count cols = middlePrefix count cols ++
      [.forRangeDyn "k" (.constNat 0) (.constNat 2) (.constNat 1) (inner count cols)] := by
  rfl

theorem middle_prefix_step (M : Algebra α) (count : Nat) (u : Unique)
    (cols : Nat) (s : State α) (a : OffsetTile)
    (ha : s.regs .int [2,2] "a_offsets" = some (signed a))
    (context : Context count s) :
    run M (middlePrefix count cols) s =
      some (middlePrefixState count u s a 2) := by
  rcases context with ⟨hn,hs⟩
  all_goals simp only [middlePrefix, middle, outer, source,
    kernel, originalKernel, quotedOriginal, ComputeKernel.surfaceBody, List.getElem?_cons_zero, List.getElem?_cons_succ,
    List.take_succ_cons, List.take_zero, Bool.false_eq_true, ↓reduceIte, run, step, evalExpr, evalComputeOp]
  all_goals simp [ComputeDType.eraseDType, Option.bind, evalOp_unfold, numeric, numericLt, numericLe, natLt, natLe, ha, hn,hs,
    middlePrefixState, loaded, bump, ← min_def_lt, signed, Context, Int.natCast_add, Int.natCast_mul, Int.ofNat_lt, Int.ofNat_le, clamp_toNat]
  all_goals unfold FP.Structural.bop
  all_goals simp [ComputeDType.eraseDType, Option.bind, Broadcast.leftIndex, Broadcast.rightIndex, ha, hn,hs,
    middlePrefixState, loaded, bump, ← min_def_lt, signed, Context, Int.natCast_add, Int.natCast_mul, Int.ofNat_lt, Int.ofNat_le, clamp_toNat]
  all_goals simp only [signed,bump,cast_bump_two,Int.ofNat_lt,clamp_toNat,index_eta]
  all_goals rfl

theorem middle_step (M : Algebra α) (count : Nat) (u : Unique)
    (cols : Nat) (s : State α) (a c : OffsetTile)
    (v1 : Payload)
    (ha : s.regs .int [2,2] "a_offsets" = some (signed a))
    (hc : s.regs .int [2,2] "c_offsets" = some (signed c))
    (h1 : s.regs .int [2,2] "a1" = some v1)
    (hu : s.regs .bool [2,2] "unique" = some u)
    (context : Context count s) :
    run M (middle count cols) s =
      some (middleState count u s a c v1 2) := by
  rw [middle_decomposition, run_append, middle_prefix_step M count u cols s a ha context]
  have h := inner_loop M count u cols (middlePrefixState count u s a 2)
    (bump a 2) c v1 (loaded count s (bump a 2))
    (by simp [middlePrefixState]) (by simpa [middlePrefixState] using hc)
    (by simpa [middlePrefixState] using h1) (by simp [middlePrefixState]) (by simpa [middlePrefixState] using hu) (by simpa [Context,middlePrefixState] using context)
  simpa [run, step, evalOp_unfold, h, middleState] using h

theorem middle_registers (count : Nat) (u : Unique) (s : State α)
    (a c : OffsetTile) (v1 : Payload) (delta : Nat)
    (h1 : s.regs .int [2,2] "a1" = some v1) :
    (middleState count u s a c v1 delta).regs .int [2,2] "a_offsets" =
      some (signed (bump (bump (bump a delta) delta) delta)) ∧
    (middleState count u s a c v1 delta).regs .int [2,2] "c_offsets" =
      some (signed (bump (bump (bump (bump (bump (bump c delta) delta) delta) delta) delta) delta)) ∧
    (middleState count u s a c v1 delta).regs .int [2,2] "a1" = some v1 := by
  simp [middleState, middlePrefixState, innerLoopState, innerState, scatter_regs, h1]

def middleLoopState (count : Nat) (u : Unique) (s : State α)
    (a c : OffsetTile) (v1 : Payload) (delta : Nat) : State α :=
  let t := middleState count u (s.setReg "j" .nat [] (fun _ ↦ 0)) a c v1 delta
  middleState count u (t.setReg "j" .nat [] (fun _ ↦ 1))
    (bump (bump (bump a delta) delta) delta)
    (bump (bump (bump (bump (bump (bump c delta) delta) delta) delta) delta) delta) v1 delta

theorem middle_loop (M : Algebra α) (count : Nat) (u : Unique)
    (cols : Nat) (s : State α) (a c : OffsetTile)
    (v1 : Payload)
    (ha : s.regs .int [2,2] "a_offsets" = some (signed a))
    (hc : s.regs .int [2,2] "c_offsets" = some (signed c))
    (h1 : s.regs .int [2,2] "a1" = some v1)
    (hu : s.regs .bool [2,2] "unique" = some u)
    (context : Context count s) :
    range M "j" 0 2 1 (middle count cols) s =
      some (middleLoopState count u s a c v1 2) := by
  rw [range_two]
  have e0 := middle_step M count u cols (s.setReg "j" .nat [] (fun _ ↦ 0)) a c v1
    (by simpa using ha) (by simpa using hc) (by simpa using h1) (by simpa using hu) (by simpa [Context] using context)
  rw [e0]
  have regs := middle_registers count u (s.setReg "j" .nat [] (fun _ ↦ 0)) a c v1
    2 (by simpa using h1)
  exact middle_step M count u cols _ _ _ _
    (by simpa using regs.1) (by simpa using regs.2.1) (by simpa using regs.2.2)
    (by simpa [middleState, middlePrefixState, innerLoopState, innerState] using hu) (by simpa [Context,innerState,innerLoopState,middlePrefixState,middleState,middleLoopState] using context)

end VeriTile.Bench.Repairs.NestedLoopsProcessing.SignedExecution

#guard_msgs (drop info) in
#auditModuleAxioms
