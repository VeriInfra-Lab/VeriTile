import bench.repairs.tritonbench_g.kldiv_ops.Proofs.SafetyBackwardSteps

namespace VeriTile.Bench.Repairs.KldivOps.SafetyBackward
open VeriTile Triton FP.Structural FP.ProfiledRewrite Safety
noncomputable section
set_option maxHeartbeats 500000
set_option maxRecDepth 18000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

/-- A list of source statements with no accesses is safe in every predecessor
state; no execution result is needed for this purely syntactic prefix. -/
theorem pure_trace (M : Algebra α) (bounds : RegionName → Nat) (code : List ComputeStmt)
    (pure : ∀ st∈code,∀ s,stepSafe M bounds st s) (s : State α) :
    traceSafeList M bounds code s := by
  induction code generalizing s with
  | nil => exact traceSafeList_nil _ _ _
  | cons st rest ih =>
    apply traceSafeList_cons_intro (pure st (by simp) s)
    intro t _
    exact ih (fun cmd member => pure cmd (by simp [member])) t

theorem body_commands (p : Parameters) (flag : Bool) : Backward.body p flag =
    [command p flag 0,command p flag 1,command p flag 2,command p flag 3,command p flag 4] := by
  cases flag <;> rfl

theorem chunk_safe (M : Algebra α) (p : Parameters) (flag : Bool) (row start : Nat)
    (s : State α) (h : Backward.Registers p row s) (bounds : RegionName → Nat)
    (ht : ∀ j : Fin p.block,start+j.val<p.cols → row*p.cols+(start+j.val)<bounds p.T)
    (ho : ∀ j : Fin p.block,start+j.val<p.cols → row*p.cols+(start+j.val)<bounds p.O) :
    traceSafeList (engine M) bounds (Backward.body p flag) (b0 s start) := by
  rw [body_commands]
  refine traceSafeList_cons_of_step ?_ (step0 M p flag s start)
    (traceSafeList_cons_of_step ?_ (step1 M p flag s start row h)
      (traceSafeList_cons_of_step ?_ (step2 M p flag s row start h)
        (traceSafeList_cons_of_step ?_ (step3 M p flag s row start)
          (traceSafeList_cons_intro ?_ (by intro t _;exact traceSafeList_nil _ _ _)))))
  all_goals cases flag
  all_goals simp only [command,Backward.body,backward,KernelDSL.backward,KernelDSL.backwardDefault,KernelDSL.backwardLog,KernelDSL.backwardBody,
    VeriTile.Bench.TritonBenchG.KldivOps.TypedSource.backwardLog,
    VeriTile.Bench.TritonBenchG.KldivOps.TypedSource.backwardDefault,
    VeriTile.Bench.TritonBenchG.KldivOps.TypedSource.backwardDefaultOriginal,
    VeriTile.Bench.TritonBenchG.KldivOps.TypedSource.backwardLogOriginal,
    VeriTile.Bench.TritonBenchG.KldivOps.TypedSource.commuteOutput,
    ComputeKernel.surfaceBody,List.getD,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD,
    stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,safeOp_unfold,safeAddress_unfold,
    safeMask_unfold,addressSafe_unfold,maskActive_unfold,precisionContext,
    evalOp_unfold,Option.bind_eq_bind,Option.pure_def,Option.bind,
    engine,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
    b0,b1,b2,b3,b4,State.setReg,h.target,h.output,h.cols,
    numeric,numericLt,natLt,FP.Structural.bop,Region.cast,
    ne_eq,not_false_eq_true,not_true_eq_false,String.reduceEq,reduceCtorEq,
    decide_eq_true_eq,ite_true,ite_false,dite_true,dite_false,Bool.false_eq_true,Bool.true_eq,
    Option.some.injEq,forall_eq,forall_eq',exists_eq_left,exists_eq_left',true_and,and_true,
    Broadcast.leftIndex,Broadcast.rightIndex,←Int.natCast_add,Int.toNat_natCast,Int.ofNat_lt]
  all_goals try simp only [Bool.not_false,Bool.not_true,Bool.false_eq_true,Bool.true_eq,
    ite_true,ite_false,traceSafeList_cons,traceSafeList_nil,stepSafe_unfold,safeExpr_unfold,
    safeCompute_unfold,safeOp_unfold,precisionContext,true_and,and_true,implies_true,forall_const]
  all_goals first | exact fun j hj => ht j.1 hj | exact fun j hj => ho j.1 hj

theorem prefix_safe (M : Algebra α) (p : Parameters) (flag : Bool)
    (bounds : RegionName → Nat) (s : State α) :
    traceSafeList (engine M) bounds (Backward.backwardPrefix p flag) s := by
  apply pure_trace
  intro cmd member state
  cases flag
  all_goals simp only [Backward.backwardPrefix,backward,KernelDSL.backward,KernelDSL.backwardDefault,KernelDSL.backwardLog,KernelDSL.backwardBody,
    VeriTile.Bench.TritonBenchG.KldivOps.TypedSource.backwardLog,
    VeriTile.Bench.TritonBenchG.KldivOps.TypedSource.backwardDefault,
    VeriTile.Bench.TritonBenchG.KldivOps.TypedSource.backwardDefaultOriginal,
    VeriTile.Bench.TritonBenchG.KldivOps.TypedSource.backwardLogOriginal,
    VeriTile.Bench.TritonBenchG.KldivOps.TypedSource.commuteOutput,
    ComputeKernel.surfaceBody,List.take,Bool.false_eq_true,Bool.true_eq,ite_true,ite_false,
    List.mem_cons,List.not_mem_nil,or_false] at member
  all_goals rcases member with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
  all_goals simp only [stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,safeOp_unfold,
    precisionContext,and_true]

theorem range_safe (M : Algebra α) (p : Parameters) (flag : Bool) (row start : Nat)
    (s : State α) (h : Backward.Registers p row s) (positive : 0<p.block)
    (bounds : RegionName → Nat)
    (ht : ∀ col,col<p.cols → row*p.cols+col<bounds p.T)
    (ho : ∀ col,col<p.cols → row*p.cols+col<bounds p.O) :
    rangeIntSafe (engine M) bounds "i" (start:Int) (p.cols:Int) p.block (Backward.body p flag) s := by
  by_cases active : start<p.cols
  · rw [rangeIntSafe_unfold,if_neg (Nat.ne_of_gt positive),if_pos (by exact_mod_cast active)]
    refine ⟨chunk_safe M p flag row start s h bounds (fun j hj => ht _ hj) (fun j hj => ho _ hj),?_⟩
    rw [(Backward.chunkRun M p flag row start s h).property]
    simp only [←Int.natCast_add]
    exact range_safe M p flag row (start+p.block) _ (Backward.chunk_registers M p flag row start s h)
      positive bounds ht ho
  · rw [rangeIntSafe_unfold,if_neg (Nat.ne_of_gt positive),if_neg (by exact_mod_cast active)]
    trivial
termination_by p.cols-start
decreasing_by omega

/-- Actual complete signed-loop source, preserving both target branches. -/
theorem source_safe (M : Algebra α) (p : Parameters) (flag : Bool) (s : State α)
    (positive : 0<p.block) (bounds : RegionName → Nat)
    (ht : ∀ col,col<p.cols → s.pids 0*p.cols+col<bounds p.T)
    (ho : ∀ col,col<p.cols → s.pids 0*p.cols+col<bounds p.O) :
    FP.Structural.TraceSafe (engine M) bounds (KernelDSL.backward p flag) s := by
  rw [KernelDSL.backward_eq]
  unfold FP.Structural.TraceSafe
  rw [Backward.source_decomposition,traceSafeList_append]
  refine ⟨prefix_safe M p flag bounds s,?_⟩
  intro prepared executed
  have same : prepared=(Backward.prefixRun M p flag s).val :=
    Option.some.inj (executed.symm.trans (Backward.prefixRun M p flag s).property)
  subst prepared
  apply traceSafeList_cons_intro
  · simp only [stepSafe_unfold,safeOp_unfold,evalOp_unfold,Option.bind_eq_bind,
      Option.pure_def,Option.bind,true_and,(Backward.prefix_registers M p flag s).cols]
    exact range_safe M p flag (s.pids 0) 0 _ (Backward.prefix_registers M p flag s) positive bounds ht ho
  · intro final _
    exact traceSafeList_nil _ _ _

#axiomsClean source_safe
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KldivOps.SafetyBackward
