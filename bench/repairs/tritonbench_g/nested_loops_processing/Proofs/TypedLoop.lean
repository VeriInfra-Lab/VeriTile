import bench.repairs.tritonbench_g.nested_loops_processing.Proofs.TypedStep

namespace VeriTile.Bench.Repairs.NestedLoopsProcessing.TypedExecution
open VeriTile.Triton FP.Structural Kernels.LegacyPastedSurface
set_option maxHeartbeats 1000000
set_option maxRecDepth 10000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

 theorem range_two (M : Algebra α) (idx : RegName) (body : List ComputeStmt) (s : State α) :
    range M idx 0 2 1 body s =
      (run M body (s.setReg idx .nat [] (fun _ ↦ 0)) >>= fun t ↦
        run M body (t.setReg idx .nat [] (fun _ ↦ 1))) := by
  rw [range]
  simp only [Nat.one_ne_zero, Nat.reduceLT, ↓reduceIte]
  cases h0 : run M body (s.setReg idx .nat [] (fun _ ↦ 0)) with
  | none => rfl
  | some t =>
    change range M idx 1 2 1 body t = run M body (t.setReg idx .nat [] (fun _ ↦ 1))
    rw [range]
    simp only [Nat.one_ne_zero, Nat.reduceLT, ↓reduceIte]
    cases h1 : run M body (t.setReg idx .nat [] (fun _ ↦ 1)) with
    | none => simp [h1]
    | some u => simp [range, h1]

theorem inner_registers (count : Nat) (u : Unique) (s : State α)
    (a c : OffsetTile) (v1 v2 : Payload) (delta : Nat)
    (h1 : s.regs .int [2,2] "a1" = some v1)
    (h2 : s.regs .int [2,2] "a2" = some v2) :
    (innerState count u s a c v1 v2 delta).regs .nat [2,2] "a_offsets" = some (bump a delta) ∧
    (innerState count u s a c v1 v2 delta).regs .nat [2,2] "c_offsets" =
      some (bump (bump (bump c delta) delta) delta) ∧
    (innerState count u s a c v1 v2 delta).regs .int [2,2] "a1" = some v1 ∧
    (innerState count u s a c v1 v2 delta).regs .int [2,2] "a2" = some v2 := by
  simp [innerState, scatter_regs, h1, h2]

def innerLoopState (count : Nat) (u : Unique) (s : State α)
    (a c : OffsetTile) (v1 v2 : Payload) (delta : Nat) : State α :=
  let t := innerState count u (s.setReg "k" .nat [] (fun _ ↦ 0)) a c v1 v2 delta
  innerState count u (t.setReg "k" .nat [] (fun _ ↦ 1))
    (bump a delta) (bump (bump (bump c delta) delta) delta) v1 v2 delta

theorem inner_loop (M : Algebra α) (count : Nat) (u : Unique) (cols : Nat)
    (s : State α) (a c : OffsetTile) (v1 v2 : Payload)
    (ha : s.regs .nat [2,2] "a_offsets" = some a)
    (hc : s.regs .nat [2,2] "c_offsets" = some c)
    (h1 : s.regs .int [2,2] "a1" = some v1)
    (h2 : s.regs .int [2,2] "a2" = some v2)
    (hu : s.regs .bool [2,2] "unique" = some u) :
    range M "k" 0 2 1 (inner count cols) s =
      some (innerLoopState count u s a c v1 v2 2) := by
  rw [range_two]
  have e0 := inner_step M count cols (s.setReg "k" .nat [] (fun _ ↦ 0)) a c v1 v2 u
    (by simpa using ha) (by simpa using hc) (by simpa using h1) (by simpa using h2) (by simpa using hu)
  rw [e0]
  have regs := inner_registers count u (s.setReg "k" .nat [] (fun _ ↦ 0)) a c v1 v2
    2 (by simpa using h1) (by simpa using h2)
  exact inner_step M count cols _ _ _ _ _ u
    (by simpa using regs.1) (by simpa using regs.2.1)
    (by simpa using regs.2.2.1) (by simpa using regs.2.2.2)
    (by simpa [innerState] using hu)

end VeriTile.Bench.Repairs.NestedLoopsProcessing.TypedExecution

#guard_msgs (drop info) in
#auditModuleAxioms
