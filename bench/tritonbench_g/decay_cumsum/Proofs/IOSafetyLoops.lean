import bench.tritonbench_g.decay_cumsum.Proofs.IOSafetyBodies
import bench.tritonbench_g.decay_cumsum.Proofs.IOSafetyBackward

namespace VeriTile.Bench.Repairs.DecayCumsum.IOSafety
open VeriTile.Triton FP.Structural FP.ProfiledRewrite FPPrefix FPOps Geometry
open FPBackwardBase
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem backward_body_safe (M : Algebra α) (h : Call) (p : Phase) (prog : Program h p)
    (initial : State α) (n : Nat) (u : State α) (hn : n<p.span)
    (hi : u.regs .int [] "__rev_j"=some (fun _ => Int.ofNat n-Int.ofNat (p.span-1)))
    (inv : Holds M h p prog initial n u) (bounds : RegionName → Nat)
    (bounded : ∀ r∈inputRegions++outputRegions, ∀ j,
      off h p prog (FPBackwardBase.time p n) j<bounds r) :
    Safety.traceSafeList (engine M) bounds (backwardBody h p) u := by
  let addr := off h p prog (FPBackwardBase.time p n)
  have gp := inv.pointers "p_g" "g" (by simp [pointerNames]) hn
  change Safety.traceSafeList (engine M) bounds
    (FPBackwardControl.firstCode h p++backwardArithmetic h) u
  rw [Safety.traceSafeList_append]
  refine ⟨backward_control_safe M h p n addr u hi gp inv.mask bounds
    (bounded "g" (by simp [inputRegions,outputRegions])),?_⟩
  intro next ran
  have eq := Option.some.inj ((FPBackwardControl.prefix_executes M h p n hn addr u hi gp inv.mask).symm.trans ran)
  subst next
  let gval := FPBackwardControl.gateValue M h addr u
  let last := if n=0 then gval else gate M h p prog initial (p.span-1)
  apply backward_arithmetic_safe M h addr gval last (carry M h p prog initial n) _
  · intro name r member
    have pin := inv.pointers name r member hn
    simp only [pointerNames,List.mem_cons,List.not_mem_nil,or_false,Prod.mk.injEq] at member
    rcases member with ⟨rfl,rfl⟩|⟨rfl,rfl⟩|⟨rfl,rfl⟩|⟨rfl,rfl⟩|⟨rfl,rfl⟩|⟨rfl,rfl⟩|⟨rfl,rfl⟩|⟨rfl,rfl⟩
    all_goals by_cases hz : n=0 <;> simpa [FPBackwardControl.prefixState,hz,addr] using pin
  · by_cases hz : n=0 <;> simp [FPBackwardControl.prefixState,hz,inv.mask]
  · by_cases hz : n=0 <;> simp [FPBackwardControl.prefixState,hz,gval]
  · by_cases hz : n=0
    · simp [FPBackwardControl.prefixState,last,hz,gval]
    · simpa [FPBackwardControl.prefixState,last,hz] using inv.last hz
  · by_cases hz : n=0 <;> simp [FPBackwardControl.prefixState,hz,inv.acc]
  · exact bounded

 theorem backward_loop_safe (M : Algebra α) (h : Call) (p : Phase) (positive : 0<p.span)
    (prog : Program h p) (initial : State α) (n : Nat) (u : State α)
    (inv : Holds M h p prog initial n u) (bounds : RegionName → Nat)
    (bounded : ∀ r∈inputRegions++outputRegions, ∀ (t : Fin p.span) (j : Fin h.BK),
      off h p prog t.val j<bounds r) :
    Safety.rangeIntSafe (engine M) bounds "__rev_j" (Int.ofNat n-Int.ofNat (p.span-1))
      1 1 (backwardBody h p) u := by
  by_cases hn : n<p.span
  · have lt : Int.ofNat n-Int.ofNat (p.span-1)<1 := by simp only [Int.ofNat_eq_natCast];omega
    have inc : Int.ofNat n-Int.ofNat (p.span-1)+1=Int.ofNat (n+1)-Int.ofNat (p.span-1) := by
      simp only [Int.ofNat_eq_natCast,Nat.cast_add,Nat.cast_one];omega
    rw [Safety.rangeIntSafe_unfold]
    simp only [Nat.one_ne_zero,if_false,lt,if_true]
    refine ⟨?_,?_⟩
    · apply backward_body_safe M h p prog initial n _ hn (by simp)
        (bind_holds M h p prog initial n u inv) bounds
      intro r member j
      exact bounded r member ⟨FPBackwardBase.time p n,by dsimp [FPBackwardBase.time];omega⟩ j
    · obtain ⟨v,ran,next⟩ := FPBackwardIteration.iteration M h p prog initial n
        (u.setReg "__rev_j" .int [] (fun _ => Int.ofNat n-Int.ofNat (p.span-1))) hn
        (by simp) (bind_holds M h p prog initial n u inv)
      rw [ran]
      simpa only [Nat.cast_one,inc] using backward_loop_safe M h p positive prog initial (n+1) v next bounds bounded
  · have bound := inv.bound
    have done : ¬Int.ofNat n-Int.ofNat (p.span-1)<1 := by simp only [Int.ofNat_eq_natCast];omega
    rw [Safety.rangeIntSafe_unfold]
    simp only [Nat.one_ne_zero,if_false,done]
termination_by p.span-n

 theorem source_safe (M : Algebra α) (kind : Kind) (h : Call) (p : Phase)
    (positive : 0<p.span) (prog : Program h p) (scale : ℝ) (u : State α)
    (hf : u.pids 0=prog.feature) (hc : u.pids 1=prog.chunk) (hr : u.pids 2=prog.row)
    (bounds : RegionName → Nat)
    (bounded : ∀ r∈FPTile.inputs kind++FPTile.outputs kind,
      ∀ (t : Fin p.span) (j : Fin h.BK), off h p prog t.val j<bounds r) :
    TraceSafe (engine M) bounds (Kernels.source kind h p scale) u := by
  unfold TraceSafe
  rw [FPProgram.source_body,FPPrefix.source_init,Safety.traceSafeList_append]
  refine ⟨prefix_safe M kind h p positive prog bounds u hf hc hr ?_,?_⟩
  · intro j
    exact bounded "g" (by cases kind <;> simp [FPTile.inputs,FPTile.outputs,inputRegions,outputRegions])
      ⟨p.span-1,by omega⟩ j
  · intro start ran
    have eq := Option.some.inj ((FPPrefix.executes M kind h p prog u hf hc hr).symm.trans ran)
    subst start
    apply Safety.traceSafeList_cons_intro ?_ (by intro;intro;exact Safety.traceSafeList_nil _ _ _)
    cases kind
    case forward =>
      simp only [FPProgram.loopCode,forward_body,Safety.stepSafe_unfold,
        Safety.safeOp_unfold,evalOp_unfold,Option.bind_some,true_and]
      apply Safety.rangeSafe_of_invariant (engine M) bounds "i" p.span 1 (forwardBody h)
        (FPForward.Holds M h p prog u)
      · intro n active current inv
        apply forward_safe M h p prog u n _ (FPForward.bind_holds M h p prog u n current inv) bounds
        · exact bounded "g" (by simp [FPTile.inputs,FPTile.outputs]) ⟨n,active⟩
        · exact bounded "go" (by simp [FPTile.inputs,FPTile.outputs]) ⟨n,active⟩
      · intro n active current inv next executed
        obtain ⟨v,he,hv⟩ := FPForward.iteration M h p prog u n
          (current.setReg "i" .nat [] (fun _ => n)) active (FPForward.bind_holds M h p prog u n current inv)
        have eq := Option.some.inj (he.symm.trans executed)
        subst next
        exact hv
      · exact FPForward.initial_holds M h p prog u
    case prepare =>
      simp only [FPProgram.loopCode,prepare_body,Safety.stepSafe_unfold,
        Safety.safeOp_unfold,evalOp_unfold,Option.bind_some,true_and]
      apply Safety.rangeSafe_of_invariant (engine M) bounds "i" p.span 1 (prepareBody h scale)
        (FPPrepare.Holds M h p prog scale u)
      · intro n active current inv
        apply prepare_safe M h p prog scale u n _ (FPPrepare.bind_holds M h p prog scale u n current inv) bounds
        all_goals exact bounded _ (by simp [FPTile.inputs,FPTile.outputs]) ⟨n,active⟩
      · intro n active current inv next executed
        obtain ⟨v,he,hv⟩ := FPPrepare.iteration M h p prog scale u n
          (current.setReg "i" .nat [] (fun _ => n)) active (FPPrepare.bind_holds M h p prog scale u n current inv)
        have eq := Option.some.inj (he.symm.trans executed)
        subst next
        exact hv
      · exact FPPrepare.initial_holds M h p positive prog scale u
    case backward =>
      have hs : (0:Int)-(Int.ofNat p.span-1)=Int.ofNat 0-Int.ofNat (p.span-1) := by
        simp only [Int.ofNat_eq_natCast];omega
      simp only [FPProgram.loopCode,backward_body,Safety.stepSafe_unfold,
        Safety.safeOp_unfold,evalOp_unfold,Option.bind_some,true_and]
      change Safety.rangeIntSafe (engine M) bounds "__rev_j" ((0:Int)-(Int.ofNat p.span-1))
        1 1 (backwardBody h p) (FPPrefix.initial M .backward h p prog u)
      rw [hs]
      exact backward_loop_safe M h p positive prog u 0 _ (initial_holds M h p positive prog u) bounds bounded

#axiomsClean source_safe
end
end VeriTile.Bench.Repairs.DecayCumsum.IOSafety
