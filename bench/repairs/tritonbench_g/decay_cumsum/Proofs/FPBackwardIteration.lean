import bench.repairs.tritonbench_g.decay_cumsum.Proofs.FPBackwardMemory

namespace VeriTile.Bench.Repairs.DecayCumsum.FPBackwardIteration
open VeriTile.Triton FP.Structural FP.ProfiledRewrite FPOps FPPrefix Geometry FPBackwardBase
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 7000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem iteration (M : Algebra α) (h : Call) (p : Phase) (prog : Program h p)
    (initial : State α) (n : Nat) (u : State α) (hn : n<p.span)
    (hi : u.regs .int [] "__rev_j"=some (fun _ => Int.ofNat n-Int.ofNat (p.span-1)))
    (inv : Holds M h p prog initial n u) :
    ∃ v, run (engine M) (backwardBody h p) u=some v ∧ Holds M h p prog initial (n+1) v := by
  let t := FPBackwardBase.time p n
  let addr := off h p prog t
  let gval := gate M h p prog initial t
  let last := gate M h p prog initial (p.span-1)
  let acc := carry M h p prog initial n
  let control := FPBackwardControl.prefixState M h p n addr u
  let v := FPBackwardStep.result M h addr gval last acc control
  have current : t<p.span-n := by dsimp [t,FPBackwardBase.time]; omega
  have inside : t<p.span := by omega
  have cmem : control.mem=u.mem := by
    dsimp [control,FPBackwardControl.prefixState]
    split <;> rfl
  have rawU (r : RegionName) (hr : r∈inputRegions) : raw h.BK r addr u=raw h.BK r addr initial := by
    funext i
    exact congrArg (Cell.read .real) (inv.reads r hr t current i.1)
  have rawC (r : RegionName) (hr : r∈inputRegions) : raw h.BK r addr control=raw h.BK r addr initial := by
    unfold raw
    rw [cmem]
    exact rawU r hr
  have cpointers (name : RegName) (r : RegionName) (member : (name,r)∈pointerNames) :
      control.regs .ptr [h.BK] name=some (ptr h.BK r addr) := by
    have orig := inv.pointers name r member hn
    simp only [pointerNames,List.mem_cons,List.not_mem_nil,or_false,Prod.mk.injEq] at member
    rcases member with ⟨rfl,rfl⟩|⟨rfl,rfl⟩|⟨rfl,rfl⟩|⟨rfl,rfl⟩|⟨rfl,rfl⟩|⟨rfl,rfl⟩|⟨rfl,rfl⟩|⟨rfl,rfl⟩
    all_goals by_cases hz : n=0 <;> simpa [control,FPBackwardControl.prefixState,hz,addr,t] using orig
  have cg : control.regs .real [h.BK] "_g"=some gval := by
    have eq : FPBackwardControl.gateValue M h addr u=gval := by
      unfold FPBackwardControl.gateValue gval gate
      rw [rawU "g" (by simp [inputRegions])]
    by_cases hz : n=0 <;> simp [control,FPBackwardControl.prefixState,eq,hz]
  have cl : control.regs .real [h.BK] "last_g"=some last := by
    by_cases zero : n=0
    · have eq : FPBackwardControl.gateValue M h addr u=last := by
        unfold FPBackwardControl.gateValue last gate
        rw [rawU "g" (by simp [inputRegions])]
        simp [addr,t,FPBackwardBase.time,zero]
      simp [control,FPBackwardControl.prefixState,zero,eq]
    · simpa [control,FPBackwardControl.prefixState,zero] using inv.last zero
  have ca : control.regs .real [h.BK] "cum_grad_dg"=some acc := by
    by_cases hz : n=0 <;> simpa [control,FPBackwardControl.prefixState,acc,hz] using inv.acc
  have cm : control.regs .bool [h.BK] "mask"=some (fun _ => Bool.true) := by
    by_cases hz : n=0 <;> simpa [control,FPBackwardControl.prefixState,hz] using inv.mask
  have qe : FPBackwardStep.qvalue M h addr gval control=qValue M h p prog initial t := by
    unfold FPBackwardStep.qvalue FPBackwardStep.qterm FPBackwardStep.direct
    rw [rawC "dqi" (by simp [inputRegions]),rawC "dqo" (by simp [inputRegions])]
    rfl
  have ke : FPBackwardStep.kvalue M h addr gval last control=kValue M h p prog initial t := by
    unfold FPBackwardStep.kvalue FPBackwardStep.kterm FPBackwardStep.direct
    rw [rawC "dki" (by simp [inputRegions]),rawC "dko" (by simp [inputRegions])]
    rfl
  have ce : FPBackwardStep.nextCarry M h addr gval last acc control=carry M h p prog initial (n+1) := by
    unfold FPBackwardStep.nextCarry FPBackwardStep.delta
    rw [qe,ke]
    unfold FPBackwardStep.direct
    rw [rawC "q" (by simp [inputRegions]),rawC "k" (by simp [inputRegions])]
    rfl
  have memory : v.mem=(FPBackwardMemory.stores M h p prog t (qValue M h p prog initial t)
      (kValue M h p prog initial t) (carry M h p prog initial (n+1)) u).mem := by
    dsimp [v]
    rw [FPStoreFrame.backward_memory,qe,ke,ce]
    apply FPStoreFrame.scatter_mem_congr
    apply FPStoreFrame.directScatter_mem_congr
    apply FPStoreFrame.directScatter_mem_congr
    exact cmem
  have execution : run (engine M) (backwardBody h p) u=some v := by
    change run (engine M) (FPBackwardControl.firstCode h p++backwardArithmetic h) u=some v
    rw [run_append,FPBackwardControl.prefix_executes M h p n hn addr u hi
      (inv.pointers "p_g" "g" (by simp [pointerNames]) hn) inv.mask]
    simp only [Option.bind_some]
    exact FPBackwardStep.executes M h addr gval last acc control
      (cpointers "p_q" "q" (by simp [pointerNames])) (cpointers "p_k" "k" (by simp [pointerNames]))
      (cpointers "p_g" "g" (by simp [pointerNames])) (cpointers "p_dq_inner" "dqi" (by simp [pointerNames]))
      (cpointers "p_dk_inner" "dki" (by simp [pointerNames])) (cpointers "p_dq_inter" "dqo" (by simp [pointerNames]))
      (cpointers "p_dk_inter" "dko" (by simp [pointerNames])) (cpointers "p_dg" "dg" (by simp [pointerNames])) cm cg cl ca
  refine ⟨v,execution,Nat.succ_le_of_lt hn,?_,?_,?_,?_,?_,?_,?_⟩
  · intro name r member next
    have pos : 0<t := by dsimp [t,FPBackwardBase.time]; omega
    have eq : t-1=FPBackwardBase.time p (n+1) := by dsimp [t,FPBackwardBase.time]; omega
    have pointers : (fun j => addr j-h.width)=off h p prog (FPBackwardBase.time p (n+1)) := by
      funext j
      exact (FPMemory.off_pred h p prog t pos j).trans (congrArg (fun x => off h p prog x j) eq)
    simp only [pointerNames,List.mem_cons,List.not_mem_nil,or_false,Prod.mk.injEq] at member
    rcases member with ⟨rfl,rfl⟩|⟨rfl,rfl⟩|⟨rfl,rfl⟩|⟨rfl,rfl⟩|⟨rfl,rfl⟩|⟨rfl,rfl⟩|⟨rfl,rfl⟩|⟨rfl,rfl⟩
    all_goals simp [v,FPBackwardStep.result,pointers]
  · simp [v,FPBackwardStep.result,cm]
  · simp [v,FPBackwardStep.result]
    exact ce
  · intro _
    simpa [v,FPBackwardStep.result] using cl
  · intro r member s hs j
    rw [memory,FPBackwardMemory.stores_other_time M h p prog t s (by dsimp [t,FPBackwardBase.time]; omega)]
    exact inv.reads r member s (by omega) j
  · intro r member s lower upper j
    rw [memory]
    by_cases same : s=t
    · subst s
      rw [FPBackwardMemory.stores_cell M h p prog t _ _ _ u r member j]
      have steps : p.span-t=n+1 := by dsimp [t,FPBackwardBase.time]; omega
      simp [FPBackwardBase.output,steps]
    · rw [FPBackwardMemory.stores_other_time M h p prog t s (Ne.symm same)]
      exact inv.output r member s (by dsimp [t,FPBackwardBase.time] at same; omega) upper j
  · intro r o outside
    rw [memory]
    exact (FPBackwardMemory.stores_frame M h p prog ⟨t,inside⟩ _ _ _ u r o outside).trans (inv.frame r o outside)

#axiomsClean iteration
end
end VeriTile.Bench.Repairs.DecayCumsum.FPBackwardIteration
