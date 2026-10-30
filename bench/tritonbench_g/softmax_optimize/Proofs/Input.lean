import bench.tritonbench_g.softmax_optimize.Proofs.Primitive

namespace VeriTile.Bench.Repairs.SoftmaxOptimize.Input
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.SoftmaxOptimize
open SignedTransport Primitive
set_option maxHeartbeats 4000000
noncomputable section

def fullLoad (s : BlockState) (X : RegionName) (N B start row : Nat) : Tile .real [B] :=
  ⟨fun lane => (s.readMem X (row * N + (start + lane.1.val)) : WithBot ℝ)⟩

def fullState (s : BlockState) (X : RegionName) (N B start row : Nat)
    (m z : Tile .real [B]) : BlockState :=
  updateState ((prefixIntState X N B start row s).setReg "inp" .real [B]
    (fullLoad s X N B start row)) m z (fullLoad s X N B start row)

theorem full_run (s : BlockState) (X : RegionName) (N B start row : Nat)
    (m z : Tile .real [B])
    (hpid : s.regs .int [] "pid_m" = some (Tile.scalar (row : Int)))
    (hcols : s.regs .int [] "cols" = some (Tile.scalar (N : Int)))
    (hm : s.regs .real [B] "m" = some m)
    (hz : s.regs .real [B] "z" = some z) :
    stepStmts (Source.inputFull X N B)
      (s.setReg "start_n" .int [] (Tile.scalar (start : Int))) =
      some (fullState s X N B start row m z) := by
  unfold Source.inputFull
  simp only [List.append_assoc]
  rw [ProgramRewrite.append, prefix_run X N B start row s hpid hcols]
  simp only [Option.bind_some, loopAbody, List.drop, List.take, List.singleton_append]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (v := fullLoad s X N B start row) (by
    rw [evalOp_load_ptrRef_none "input_ptrs" _ (pointers X N B start row)
      (by simp only [evalOp_ref, prefixIntState, BlockState.setReg_same])]
    apply congrArg some
    apply Tile.ext
    intro lane
    simp only [Tile.mk]
    rw [BlockState.readMemValue_real]
    rfl))]
  exact update_run _ m z (fullLoad s X N B start row)
    (by simpa only [prefixIntState, BlockState.setReg_ne_name, ne_eq,
      String.reduceEq, not_false_eq_true] using hm)
    (by simpa only [prefixIntState, BlockState.setReg_ne_name, ne_eq,
      String.reduceEq, not_false_eq_true] using hz)
    (by simp only [BlockState.setReg_same])

def tailMask (N B start : Nat) : Tile .bool [B] :=
  ⟨fun lane => decide (start + lane.1.val < N)⟩

def maskedPrefix (s : BlockState) (X : RegionName) (N B start row : Nat) : BlockState :=
  (prefixIntState X N B start row s).setReg "mask" .bool [B] (tailMask N B start)

def tailLoad (s : BlockState) (X : RegionName) (N B start row : Nat) : Tile .real [B] :=
  ⟨fun lane => if start + lane.1.val < N then
    (s.readMem X (row * N + (start + lane.1.val)) : WithBot ℝ) else ⊥⟩

def tailState (s : BlockState) (X : RegionName) (N B start row : Nat)
    (m z : Tile .real [B]) : BlockState :=
  updateState ((maskedPrefix s X N B start row).setReg "inp" .real [B]
    (tailLoad s X N B start row)) m z (tailLoad s X N B start row)

theorem mask_run (s : BlockState) (X : RegionName) (N B start row : Nat)
    (hcols : s.regs .int [] "cols" = some (Tile.scalar (N : Int))) :
    stepStmt (Source.maskStatement B) (prefixIntState X N B start row s) =
      some (maskedPrefix s X N B start row) := by
  apply stepStmt_assign_eq_some
  simp only [Source.maskStatement, evalOp.eq_def, prefixIntState,
    BlockState.setReg_ne_name, BlockState.setReg_same, hcols,
    ne_eq, String.reduceEq, not_false_eq_true, Option.bind_some, Option.bind_eq_bind]
  apply congrArg some
  apply Tile.ext
  intro lane
  change decide (((start + lane.1.val : Nat) : Int) < (N : Int)) = decide (start + lane.1.val < N)
  simp <;> omega

theorem tail_run (s : BlockState) (X : RegionName) (N B start row : Nat)
    (m z : Tile .real [B])
    (hpid : s.regs .int [] "pid_m" = some (Tile.scalar (row : Int)))
    (hcols : s.regs .int [] "cols" = some (Tile.scalar (N : Int)))
    (hm : s.regs .real [B] "m" = some m)
    (hz : s.regs .real [B] "z" = some z) :
    stepStmts (Source.inputTail X N B)
      (s.setReg "start_n" .int [] (Tile.scalar (start : Int))) =
      some (tailState s X N B start row m z) := by
  unfold Source.inputTail
  simp only [List.append_assoc]
  rw [ProgramRewrite.append, prefix_run X N B start row s hpid hcols]
  simp only [Option.bind_some, List.singleton_append, loopBbody, List.drop, List.take]
  rw [stepStmts.cons_some (mask_run s X N B start row hcols)]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (v := tailLoad s X N B start row) (by
    rw [evalOp_load_ptrRef_maskOther "input_ptrs" _ _ _ (pointers X N B start row)
      (tailMask N B start) (⟨fun _ => (⊥ : WithBot ℝ)⟩ : Tile .real [B])
      (by simp only [evalOp_ref, maskedPrefix, prefixIntState,
        BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true])
      (by simp only [evalOp_ref, maskedPrefix, BlockState.setReg_same])
      (by simp only [evalOp.eq_def, Option.bind_some]; rfl)]
    apply congrArg some
    apply Tile.ext
    intro lane
    simp only [tailMask, tailLoad, pointers, BlockState.readMemValue_real, decide_eq_true_eq]
    rfl))]
  exact update_run _ m z (tailLoad s X N B start row)
    (by simpa only [maskedPrefix, prefixIntState, BlockState.setReg_ne_name, ne_eq,
      String.reduceEq, not_false_eq_true] using hm)
    (by simpa only [maskedPrefix, prefixIntState, BlockState.setReg_ne_name, ne_eq,
      String.reduceEq, not_false_eq_true] using hz)
    (by simp only [BlockState.setReg_same])

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SoftmaxOptimize.Input
