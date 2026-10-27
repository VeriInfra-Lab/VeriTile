import bench.repairs.tritonbench_g.kldiv_ops.Proofs.SafetyForwardSteps
import bench.repairs.tritonbench_g.kldiv_ops.Proofs.SafetyBackward

namespace VeriTile.Bench.Repairs.KldivOps.SafetyForward
open VeriTile Triton FP.Structural FP.ProfiledRewrite Safety
noncomputable section
set_option maxHeartbeats 160000
set_option maxRecDepth 18000
set_option linter.unusedSimpArgs false
attribute [local irreducible] Tile.chunkRun Tile.prefixRun
variable {α : Type} [Inhabited α]

theorem body_commands (p : Parameters) (flag : Bool) (red : Nat) : forwardBody p flag red =
    [command p flag red 0,command p flag red 1,command p flag red 2,
      command p flag red 3,command p flag red 4,command p flag red 5] := by
  cases flag <;> rfl

theorem chunk_safe (M : Algebra α) (p : Parameters) (flag : Bool) (red row start : Nat)
    (eps carry : α) (s : State α) (h : Tile.Registers p red row eps carry s) (bounds : RegionName → Nat)
    (hy : ∀ j : Fin p.block,start+j.val<p.cols → row*p.cols+(start+j.val)<bounds p.Y)
    (ht : ∀ j : Fin p.block,start+j.val<p.cols → row*p.cols+(start+j.val)<bounds p.T)
    (ho : red=0 → ∀ j : Fin p.block,start+j.val<p.cols → row*p.cols+(start+j.val)<bounds p.O) :
    traceSafeList (engine M) bounds (forwardBody p flag red) (f0 s start) := by
  rw [body_commands]
  refine traceSafeList_cons_of_step ?_ (step0 M p flag red row start eps carry s h)
    (traceSafeList_cons_of_step ?_ (step1 M p flag red row start eps carry s h)
      (traceSafeList_cons_of_step ?_ (step2 M p flag red row start eps carry s h)
        (traceSafeList_cons_of_step ?_ (step3 M p flag red row start eps carry s h)
          (traceSafeList_cons_of_step ?_ (step4 M p flag red row start eps carry s h)
            (traceSafeList_cons_intro ?_ (by intro t _;exact traceSafeList_nil _ _ _))))))
  all_goals cases flag
  all_goals simp only [command,forwardBody,forwardTemplateBody,forwardTemplate,KernelDSL.forward,KernelDSL.forwardDefault,KernelDSL.forwardLog,KernelDSL.forwardBody,epsilonLoss,
    ComputeKernel.surfaceBody,List.getD,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD,
    List.take,List.drop,List.nil_append,List.cons_append,
    stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,safeOp_unfold,safeAddress_unfold,
    safeMask_unfold,addressSafe_unfold,maskActive_unfold,precisionContext,
    evalOp_unfold,Option.bind_eq_bind,Option.pure_def,Option.bind,
    engine,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
    f0,f1,f2,SafetyBackward.b0,SafetyBackward.b1,SafetyBackward.b2,f3,f4,f5,State.setReg,
    h.y,h.t,h.o,h.base,h.epsilon,h.sum,h.cols,
    numeric,numericLt,numericEq,natLt,FP.Structural.bop,Region.cast,
    ne_eq,not_false_eq_true,not_true_eq_false,String.reduceEq,reduceCtorEq,
    decide_eq_true_eq,ite_true,ite_false,dite_true,dite_false,Bool.false_eq_true,Bool.true_eq,
    Option.some.injEq,forall_eq,forall_eq',exists_eq_left,exists_eq_left',true_and,and_true,
    Broadcast.leftIndex,Broadcast.rightIndex,←Int.natCast_add,Int.toNat_natCast,Int.ofNat_lt]
  all_goals try (solve | simp only [Bool.not_false,Bool.not_true,Bool.false_eq_true,Bool.true_eq,
    ite_true,ite_false,traceSafeList_cons,traceSafeList_nil,stepSafe_unfold,safeExpr_unfold,
    safeCompute_unfold,safeOp_unfold,precisionContext,true_and,and_true,implies_true,forall_const])
  all_goals first | exact fun j hj => hy j.1 hj | exact fun j hj => ht j.1 hj | skip
  all_goals by_cases hr : red=0
  all_goals simp only [hr,ite_true,ite_false]
  all_goals solve
    | (apply SafetyBackward.pure_trace
       intro cmd member state
       simp only [List.mem_cons,List.not_mem_nil,or_false] at member
       rcases member with rfl | rfl
       all_goals simp only [stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,safeOp_unfold,
         precisionContext,true_and,and_true])
    | (apply traceSafeList_cons_intro
       · simp only [stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,safeOp_unfold,safeAddress_unfold,
           safeMask_unfold,addressSafe_unfold,maskActive_unfold,precisionContext,
           evalOp_unfold,Option.bind_eq_bind,Option.pure_def,Option.bind,
           State.setReg,h.o,hr,ite_true,ite_false,
           ne_eq,not_false_eq_true,not_true_eq_false,String.reduceEq,reduceCtorEq,
           decide_eq_true_eq,dite_true,dite_false,
           Option.some.injEq,forall_eq,forall_eq',exists_eq_left,exists_eq_left',true_and,and_true,
           FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,
           ←Int.natCast_add,Int.toNat_natCast,Int.ofNat_lt]
         exact fun j hj => ho hr j.1 hj
       · intro t _;exact traceSafeList_nil _ _ _)

 theorem prefix_safe (M : Algebra α) (p : Parameters) (flag : Bool) (red : Nat)
    (bounds : RegionName → Nat) (s : State α) :
    traceSafeList (engine M) bounds (forwardPrefix p flag red) s := by
  apply SafetyBackward.pure_trace
  intro cmd member state
  cases flag
  all_goals simp only [forwardPrefix,forwardTemplate,KernelDSL.forward,KernelDSL.forwardDefault,KernelDSL.forwardLog,KernelDSL.forwardBody,ComputeKernel.surfaceBody,List.take,
    Bool.false_eq_true,Bool.true_eq,ite_true,ite_false,
    List.mem_cons,List.not_mem_nil,or_false] at member
  all_goals rcases member with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
  all_goals simp only [stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,safeOp_unfold,
    precisionContext,and_true]

theorem range_safe (M : Algebra α) (p : Parameters) (flag : Bool) (red row start : Nat)
    (eps carry : α) (s : State α) (h : Tile.Registers p red row eps carry s) (positive : 0<p.block)
    (bounds : RegionName → Nat)
    (hy : ∀ col,col<p.cols → row*p.cols+col<bounds p.Y)
    (ht : ∀ col,col<p.cols → row*p.cols+col<bounds p.T)
    (ho : red=0 → ∀ col,col<p.cols → row*p.cols+col<bounds p.O) :
    rangeIntSafe (engine M) bounds "i" (start:Int) (p.cols:Int) p.block (forwardBody p flag red) s := by
  by_cases active : start<p.cols
  · rw [rangeIntSafe_unfold,if_neg (Nat.ne_of_gt positive),if_pos (by exact_mod_cast active)]
    refine ⟨chunk_safe M p flag red row start eps carry s h bounds
      (fun j hj => hy _ hj) (fun j hj => ht _ hj) (fun hr j hj => ho hr _ hj),?_⟩
    rw [(Tile.chunkRun M p flag red row start eps carry s h).property]
    simp only [←Int.natCast_add]
    exact range_safe M p flag red row (start+p.block) eps _ _
      (Tile.chunk_registers M p flag red row start eps carry s h) positive bounds hy ht ho
  · rw [rangeIntSafe_unfold,if_neg (Nat.ne_of_gt positive),if_neg (by exact_mod_cast active)]
    trivial
termination_by p.cols-start
decreasing_by omega

/-- The reduced path stores once after all chunks; zero columns still reaches
this scalar store. -/
theorem finish_safe (M : Algebra α) (p : Parameters) (flag : Bool) (red row : Nat)
    (eps carry : α) (s : State α) (h : Tile.Registers p red row eps carry s)
    (bounds : RegionName → Nat) (ho : red≠0 → row<bounds p.O) :
    traceSafeList (engine M) bounds (forwardFinish p flag red) s := by
  cases flag <;> by_cases hr : red=0
  all_goals simp only [forwardFinish,forwardTemplate,KernelDSL.forward,KernelDSL.forwardDefault,KernelDSL.forwardLog,KernelDSL.forwardBody,ComputeKernel.surfaceBody,List.drop,
    Bool.false_eq_true,Bool.true_eq,ite_true,ite_false]
  all_goals apply traceSafeList_cons_intro
  all_goals try (intro final _;exact traceSafeList_nil _ _ _)
  all_goals simp only [stepSafe_unfold,safeOp_unfold,evalOp_unfold,Option.bind_eq_bind,
    Option.pure_def,Option.bind,numericEq,natLt,FP.Structural.bop,
    engine,Algebra.withDefaultPrecision,resolvePrecision,Broadcast.leftIndex,Broadcast.rightIndex,
    Bool.not_eq_true',decide_eq_false_iff_not,hr,decide_true,decide_false,Bool.not_true,Bool.not_false,Bool.false_eq_true,Bool.true_eq,
    ite_true,ite_false,true_and]
  all_goals apply traceSafeList_cons_intro
  all_goals try (intro final _;exact traceSafeList_nil _ _ _)
  all_goals simp only [stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,safeOp_unfold,safeAddress_unfold,
    safeMask_unfold,addressSafe_unfold,maskActive_unfold,precisionContext,
    evalOp_unfold,Option.bind_eq_bind,Option.pure_def,Option.bind,h.o,
    hr,ite_false,Nat.mul_one,Option.some.injEq,forall_eq,forall_eq',true_and,and_true]
  all_goals exact fun _ _ => ho hr

/-- Safety follows the same prefix, every signed-loop iteration, and the final
row store as the complete source execution theorem. -/
theorem source_safe (M : Algebra α) (p : Parameters) (flag : Bool) (red : Nat)
    (eps : α) (s : State α) (positive : 0<p.block) (bounds : RegionName → Nat)
    (hy : ∀ col,col<p.cols → s.pids 0*p.cols+col<bounds p.Y)
    (ht : ∀ col,col<p.cols → s.pids 0*p.cols+col<bounds p.T)
    (ho : red=0 → ∀ col,col<p.cols → s.pids 0*p.cols+col<bounds p.O)
    (hs : red≠0 → s.pids 0<bounds p.O) :
    FP.Structural.TraceSafe (engine M) bounds (KernelDSL.forward p flag red) (Tile.bindEpsilon eps s) := by
  rw [KernelDSL.forward_eq]
  unfold FP.Structural.TraceSafe
  rw [forward_decomposition]
  simp only [List.append_assoc,List.singleton_append]
  rw [traceSafeList_append]
  refine ⟨prefix_safe M p flag red bounds _,?_⟩
  intro prepared executed
  have same : prepared=(Tile.prefixRun M p flag red eps s).val :=
    Option.some.inj (executed.symm.trans (Tile.prefixRun M p flag red eps s).property)
  subst prepared
  have ready := Tile.prefix_registers M p flag red eps s
  obtain ⟨looped,ran,_,registers⟩ := Program.loop_complete M p flag red (s.pids 0) eps 0
    (Tile.zero M) _ ready positive
  have stepRun : step (engine M) (.forRangeInt "i" (.constInt 0) (.ref .int [] "n_cols")
      (.constNat p.block) (forwardBody p flag red)) (Tile.prefixRun M p flag red eps s).val=some looped := by
    simpa only [step,evalOp_unfold,Option.bind_eq_bind,Option.pure_def,Option.bind,ready.cols] using ran
  refine traceSafeList_cons_of_step ?_ stepRun
    (finish_safe M p flag red (s.pids 0) eps _ looped registers bounds hs)
  simp only [stepSafe_unfold,safeOp_unfold,evalOp_unfold,Option.bind_eq_bind,
    Option.pure_def,Option.bind,true_and,ready.cols]
  exact range_safe M p flag red (s.pids 0) 0 eps (Tile.zero M) _ ready positive bounds hy ht ho

#axiomsClean source_safe
#guard_msgs (drop info) in
#auditModuleAxioms

end
end VeriTile.Bench.Repairs.KldivOps.SafetyForward
