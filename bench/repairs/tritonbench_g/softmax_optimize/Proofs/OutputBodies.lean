import bench.repairs.tritonbench_g.softmax_optimize.Proofs.Output

namespace VeriTile.Bench.Repairs.SoftmaxOptimize.OutputBodies
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.SoftmaxOptimize
open SignedTransport Primitive Input Output
set_option maxHeartbeats 5000000
noncomputable section

def fullLoaded (s : BlockState) (X : RegionName) (N B start row : Nat) : BlockState :=
  (prefixIntState X N B start row s).setReg "inp" .real [B] (fullLoad s X N B start row)

def tailLoaded (s : BlockState) (X : RegionName) (N B start row : Nat) : BlockState :=
  (maskedPrefix s X N B start row).setReg "inp" .real [B] (tailLoad s X N B start row)

def fullBeforeStore (s : BlockState) (Y X : RegionName) (N B start row : Nat)
    (shift denominator : WithBot ℝ) : BlockState :=
  (arithmeticState (fullLoaded s X N B start row) (fullLoad s X N B start row)
    shift denominator).setReg "output_ptrs" .ptr [B] (pointers Y N B start row)

def tailBeforeStore (s : BlockState) (Y X : RegionName) (N B start row : Nat)
    (shift denominator : WithBot ℝ) : BlockState :=
  (arithmeticState (tailLoaded s X N B start row) (tailLoad s X N B start row)
    shift denominator).setReg "output_ptrs" .ptr [B] (pointers Y N B start row)

def fullResult (s : BlockState) (Y X : RegionName) (N B start row : Nat)
    (shift denominator : WithBot ℝ) : BlockState :=
  scatter (fullBeforeStore s Y X N B start row shift denominator) Y
    (fun i => row * N + (start + i.1.val))
    (quotient (fullLoad s X N B start row) shift denominator) (fun _ => Bool.true)

def tailResult (s : BlockState) (Y X : RegionName) (N B start row : Nat)
    (shift denominator : WithBot ℝ) : BlockState :=
  scatter (tailBeforeStore s Y X N B start row shift denominator) Y
    (fun i => row * N + (start + i.1.val))
    (quotient (tailLoad s X N B start row) shift denominator) (tailMask N B start).data

theorem full_run (s : BlockState) (Y X : RegionName) (N B start row : Nat)
    (shift denominator : WithBot ℝ)
    (hpid : s.regs .int [] "pid_m" = some (Tile.scalar (row : Int)))
    (hcols : s.regs .int [] "cols" = some (Tile.scalar (N : Int)))
    (hm : s.regs .real [] "m" = some (Tile.scalar shift))
    (hz : s.regs .real [] "z" = some (Tile.scalar denominator)) :
    stepStmts (Source.outputFull Y X N B)
      (s.setReg "start_n" .int [] (Tile.scalar (start : Int))) =
      some (fullResult s Y X N B start row shift denominator) := by
  unfold Source.outputFull
  simp only [List.append_assoc]
  rw [ProgramRewrite.append, prefix_run X N B start row s hpid hcols]
  simp only [Option.bind_some, loopApass2body, List.drop, List.take,
    List.cons_append, List.nil_append]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (v := fullLoad s X N B start row) (by
    rw [evalOp_load_ptrRef_none "input_ptrs" _ (pointers X N B start row)
      (by simp only [evalOp_ref, prefixIntState, BlockState.setReg_same])]
    apply congrArg some
    apply Tile.ext
    intro lane
    simp only [Tile.mk]
    rw [BlockState.readMemValue_real]
    rfl))]
  change stepStmts (arithmetic B ++ [SignedTransport.outputPointer Y B,
    .store .real [B] (.ptr (.ref .ptr [B] "output_ptrs")) (.ref .real [B] "out") .none])
    (fullLoaded s X N B start row) = _
  rw [ProgramRewrite.append, arithmetic_run _ (fullLoad s X N B start row) shift denominator
    (by simp only [fullLoaded, BlockState.setReg_same])
    (by simpa only [fullLoaded, prefixIntState, BlockState.setReg_ne_name, ne_eq,
      String.reduceEq, not_false_eq_true] using hm)
    (by simpa only [fullLoaded, prefixIntState, BlockState.setReg_ne_name, ne_eq,
      String.reduceEq, not_false_eq_true] using hz)]
  simp only [Option.bind_some, SignedTransport.outputPointer]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (v := pointers Y N B start row) (by
    simp only [SignedTransport.outputPointer, evalOp.eq_def, arithmeticState, fullLoaded,
      prefixIntState, BlockState.setReg_ne_name, BlockState.setReg_same,
      ne_eq, String.reduceEq, not_false_eq_true, Option.bind_eq_bind, Option.bind_some]
    apply congrArg some
    apply Tile.ext
    intro lane
    change (Y, 0 + Int.toNat ((row * N + (start + lane.1.val) : Nat) : Int)) =
      (Y, row * N + (start + lane.1.val))
    rw [Int.toNat_natCast, Nat.zero_add]))]
  rw [stepStmts.cons_some (store_full_run _ Y _
    (quotient (fullLoad s X N B start row) shift denominator)
    (by simp only [BlockState.setReg_same]; rfl)
    (by simp only [arithmeticState, BlockState.setReg_ne_name, BlockState.setReg_same,
      ne_eq, String.reduceEq, not_false_eq_true]))]
  rw [stepStmts.nil]
  rfl

theorem tail_run (s : BlockState) (Y X : RegionName) (N B start row : Nat)
    (shift denominator : WithBot ℝ)
    (hpid : s.regs .int [] "pid_m" = some (Tile.scalar (row : Int)))
    (hcols : s.regs .int [] "cols" = some (Tile.scalar (N : Int)))
    (hm : s.regs .real [] "m" = some (Tile.scalar shift))
    (hz : s.regs .real [] "z" = some (Tile.scalar denominator)) :
    stepStmts (Source.outputTail Y X N B)
      (s.setReg "start_n" .int [] (Tile.scalar (start : Int))) =
      some (tailResult s Y X N B start row shift denominator) := by
  unfold Source.outputTail
  simp only [List.append_assoc]
  rw [ProgramRewrite.append, prefix_run X N B start row s hpid hcols]
  simp only [Option.bind_some, List.singleton_append, loopBpass2body, List.drop, List.take,
    List.cons_append, List.nil_append]
  rw [stepStmts.cons_some (mask_run s X N B start row hcols)]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (v := tailLoad s X N B start row) (by
    rw [evalOp_load_ptrRef_maskOther "input_ptrs" _ _ _ (pointers X N B start row)
      (tailMask N B start) (⟨fun _ => (⊥ : WithBot ℝ)⟩ : Tile .real [B])
      (by simp only [evalOp_ref, maskedPrefix, prefixIntState,
        BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true])
      (by simp only [evalOp_ref, maskedPrefix, BlockState.setReg_same])
      (by simp only [evalOp.eq_def]; rfl)]
    apply congrArg some
    apply Tile.ext
    intro lane
    simp only [tailMask, tailLoad, pointers, BlockState.readMemValue_real, decide_eq_true_eq]
    rfl))]
  change stepStmts (arithmetic B ++ [SignedTransport.outputPointer Y B,
    .store .real [B] (.ptr (.ref .ptr [B] "output_ptrs")) (.ref .real [B] "out")
      (.mask (.ref .bool [B] "mask"))]) (tailLoaded s X N B start row) = _
  rw [ProgramRewrite.append, arithmetic_run _ (tailLoad s X N B start row) shift denominator
    (by simp only [tailLoaded, BlockState.setReg_same])
    (by simpa only [tailLoaded, maskedPrefix, prefixIntState, BlockState.setReg_ne_name,
      ne_eq, String.reduceEq, not_false_eq_true] using hm)
    (by simpa only [tailLoaded, maskedPrefix, prefixIntState, BlockState.setReg_ne_name,
      ne_eq, String.reduceEq, not_false_eq_true] using hz)]
  simp only [Option.bind_some, SignedTransport.outputPointer]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (v := pointers Y N B start row) (by
    simp only [SignedTransport.outputPointer, evalOp.eq_def, arithmeticState, tailLoaded,
      maskedPrefix, prefixIntState, BlockState.setReg_ne_name, BlockState.setReg_same,
      ne_eq, String.reduceEq, not_false_eq_true, Option.bind_eq_bind, Option.bind_some]
    apply congrArg some
    apply Tile.ext
    intro lane
    change (Y, 0 + Int.toNat ((row * N + (start + lane.1.val) : Nat) : Int)) =
      (Y, row * N + (start + lane.1.val))
    rw [Int.toNat_natCast, Nat.zero_add]))]
  rw [stepStmts.cons_some (store_tail_run _ Y _
    (quotient (tailLoad s X N B start row) shift denominator) (tailMask N B start)
    (by simp only [BlockState.setReg_same]; rfl)
    (by simp only [arithmeticState, BlockState.setReg_ne_name, BlockState.setReg_same,
      ne_eq, String.reduceEq, not_false_eq_true])
    (by simp only [arithmeticState, tailLoaded, maskedPrefix, BlockState.setReg_ne_name,
      BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true]))]
  rw [stepStmts.nil]
  rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SoftmaxOptimize.OutputBodies
