import bench.repairs.tritonbench_g.attention_fwd_triton1_tile_checked.Proofs.IOComputeSafety

namespace VeriTile.Bench.Repairs.AttentionFwdTriton1TileChecked.KernelIO.SourceSafety
open VeriTile.Triton FP.Structural
open AttentionFwdTriton1.Whole001 TermBody SharedStep
noncomputable section
set_option maxHeartbeats 800000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false

theorem body_safe [Inhabited α] (c : Shape) (scale : ℝ) (head chunk : Nat)
    (storeFlag ifcond : Bool) (carry : Matrix α c.D c.D)
    (bounds : RegionName → Nat) (bounded : Bounds c storeFlag bounds)
    (headBound : head<c.heads) (chunkBound : chunk<c.chunks) (s : State (Term α))
    (hh : s.regs .nat [] "i_bh"=some (fun _=>head))
    (hi : s.regs .nat [] "i"=some (fun _=>chunk))
    (hc : s.regs .real [c.D,c.D] "b_h"=some carry) :
    Safety.traceSafeList algebra bounds (Program.body c scale storeFlag ifcond) s := by
  simp only [Program.body,List.append_assoc,Safety.traceSafeList_append]
  refine ⟨pointers_safe c bounds s,?_⟩
  intro t ran
  have equal := Option.some.inj (ran.symm.trans (pointer_run c head chunk s hh hi))
  subst t
  refine ⟨save_safe c head chunk storeFlag bounds bounded headBound chunkBound s,?_⟩
  intro t ran
  have equal := Option.some.inj (ran.symm.trans (save_run c head chunk storeFlag carry s hc))
  subst t
  refine ⟨loads_safe c scale head chunk storeFlag carry bounds bounded headBound s,?_⟩
  intro t ran
  have equal := Option.some.inj (ran.symm.trans (loads_run c scale head chunk storeFlag carry s))
  subst t
  apply Safety.traceSafeList_cons_of_step (update_safe c ifcond bounds _)
    (update_step c scale head chunk storeFlag ifcond carry s hc hi)
  apply Safety.traceSafeList_cons_intro ?_ (fun t _=>Safety.traceSafeList_nil _ _ _)
  apply store_safe _ _ _ _ (mainPtr c head chunk "o") _ _ _
    (updated_pointer c scale head chunk storeFlag ifcond carry s)
  intro i active
  change (mainPtr c head chunk "o").inBounds [i.1.val,i.2.1.val] [0]=Bool.true at active
  rw [main_bound,decide_eq_true_eq] at active
  change (mainPtr c head chunk "o").address [i.1.val,i.2.1.val]<bounds "o"
  rw [main_address]
  exact bounded.main "o" (by simp) _
    (c.main_bound head (chunk*32+i.1.val) i.2.1.val headBound active i.2.1.isLt)

theorem range_safe [Inhabited α] (c : Shape) (scale : ℝ) (head : Nat)
    (storeFlag ifcond : Bool) (s : State (Term α)) (bounds : RegionName → Nat)
    (bounded : Bounds c storeFlag bounds) (headBound : head<c.heads) (n : Nat) :
    Safety.rangeSafe algebra bounds "i" n c.chunks 1 (Program.body c scale storeFlag ifcond)
      (SharedLoop.state c scale head storeFlag ifcond s n) := by
  rw [Safety.rangeSafe_unfold]
  simp only [Nat.one_ne_zero,if_false]
  by_cases active : n<c.chunks
  · simp only [active,if_true]
    refine ⟨body_safe c scale head n storeFlag ifcond
      (Graph.carry c (read s "k") (read s "v") head ifcond n)
      bounds bounded headBound active _ ?_ ?_ ?_,?_⟩
    · simpa using SharedLoop.head_reg c scale head storeFlag ifcond s n
    · simp
    · simpa using SharedLoop.carry_reg c scale head storeFlag ifcond s n
    · rw [SharedLoop.iteration]
      exact range_safe c scale head storeFlag ifcond s bounds bounded headBound (n+1)
  · simp [active]
termination_by c.chunks-n
decreasing_by omega

theorem kernel_safe [Inhabited α] (c : Shape) (scale : ℝ) (head : Nat)
    (storeFlag ifcond : Bool) (s : State (Term α)) (pid : s.pids 0=head)
    (bounds : RegionName → Nat) (bounded : Bounds c storeFlag bounds) (headBound : head<c.heads) :
    TraceSafe algebra bounds (Program.kernel c scale storeFlag ifcond) s := by
  unfold TraceSafe
  rw [Program.quoted]
  change Safety.traceSafeList algebra bounds (Program.code c scale storeFlag ifcond) s
  unfold Program.code
  apply Safety.traceSafeList_cons_of_step (t:=s.setReg "i_bh" .nat [] (fun _=>head))
  · simp only [Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeOp_unfold]
  · simp [step,evalExpr,evalOp_unfold,pid]
  apply Safety.traceSafeList_cons_of_step (t:=SharedLoop.initial c head s)
  · simp only [Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold]
  · simp [step,evalExpr,evalComputeOp,evalOp_unfold,algebra,SharedLoop.initial,TermBody.num]
    rfl
  apply Safety.traceSafeList_cons_intro ?_ (fun t _=>Safety.traceSafeList_nil _ _ _)
  simp only [Safety.stepSafe_unfold,Safety.safeOp_unfold,evalOp_unfold,true_and]
  exact range_safe c scale head storeFlag ifcond s bounds bounded headBound 0

end
end VeriTile.Bench.Repairs.AttentionFwdTriton1TileChecked.KernelIO.SourceSafety
