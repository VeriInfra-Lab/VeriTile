import bench.repairs.tritonbench_g.decay_cumsum.Proofs.FPMemory

namespace VeriTile.Bench.Repairs.DecayCumsum.FPForward
open VeriTile.Triton FP.Structural FP.ProfiledRewrite FPOps FPPrefix Geometry
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 6000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def carry (M : Algebra α) (h : Call) (p : Phase) (prog : Program h p) (u : State α)
    (n : Nat) : Values α .real [h.BK] := fun i =>
  FPReference.forwardCarry M (fun t => raw h.BK "g" (off h p prog t) u i) n

structure Holds (M : Algebra α) (h : Call) (p : Phase) (prog : Program h p)
    (initial : State α) (n : Nat) (u : State α) : Prop where
  bound : n≤p.span
  sourcePtr : u.regs .ptr [h.BK] "p_g"=some (ptr h.BK "g" (off h p prog n))
  targetPtr : u.regs .ptr [h.BK] "p_go"=some (ptr h.BK "go" (off h p prog n))
  mask : u.regs .bool [h.BK] "mask"=some (fun _ => Bool.true)
  acc : u.regs .real [h.BK] "cum_decay"=some (carry M h p prog initial n)
  reads : ∀ o, u.mem "g" o=initial.mem "g" o
  output : ∀ t, t<n → ∀ j : Fin h.BK,
    u.mem "go" (off h p prog t j)=FPReference.stored M (carry M h p prog initial (t+1) (j,PUnit.unit))
  frame : FPMemory.Frame h p prog ["go"] initial u

theorem iteration (M : Algebra α) (h : Call) (p : Phase) (prog : Program h p)
    (initial : State α) (n : Nat) (u : State α) (hn : n<p.span)
    (inv : Holds M h p prog initial n u) :
    ∃ v, run (engine M) (forwardBody h) u=some v ∧ Holds M h p prog initial (n+1) v := by
  let acc := carry M h p prog initial n
  let value := FPStep.fwdValue M h (off h p prog n) acc u
  let pre := (u.setReg "_g" .real [h.BK] (fun i => FPReference.gate M (raw h.BK "g" (off h p prog n) u i))).setReg
    "cum_decay" .real [h.BK] value
  let v := FPStep.forwardResult M h (off h p prog n) acc u
  have values : value=carry M h p prog initial (n+1) := by
    funext i
    simp [value,FPStep.fwdValue,carry,FPReference.forwardCarry,raw,inv.reads,acc]
  have pointers : (fun j => off h p prog n j+h.width)=off h p prog (n+1) :=
    funext (FPMemory.off_succ h p prog n)
  refine ⟨v,FPStep.forward M h _ acc u inv.sourcePtr inv.targetPtr inv.mask inv.acc,
    Nat.succ_le_of_lt hn,?_,?_,?_,?_,?_,?_,?_⟩
  · simp [v,FPStep.forwardResult,pointers]
  · simp [v,FPStep.forwardResult,pointers]
  · simp [v,FPStep.forwardResult,inv.mask]
  · simp [v,FPStep.forwardResult]
    exact values
  · intro o
    change (scatter M h.BK "go" (off h p prog n) value pre).mem "g" o=initial.mem "g" o
    simpa [pre] using (scatter_other M h.BK "go" (off h p prog n) value pre "g" o (by decide)).trans (inv.reads o)
  · intro t ht j
    change (scatter M h.BK "go" (off h p prog n) value pre).mem "go" _=_
    by_cases same : t=n
    · subst t
      rw [FPMemory.cell,values]
    · rw [FPMemory.other_step M h p prog n t (Ne.symm same)]
      exact inv.output t (by omega) j
  · intro r o outside
    change (scatter M h.BK "go" (off h p prog n) value pre).mem r o=initial.mem r o
    exact (FPMemory.frame M h p prog ⟨n,hn⟩ "go" ["go"] (by simp) value pre r o outside).trans
      (inv.frame r o outside)

theorem initial_holds (M : Algebra α) (h : Call) (p : Phase) (prog : Program h p) (u : State α) :
    Holds M h p prog u 0 (FPPrefix.initial M .forward h p prog u) := by
  have mask : (fun i : TileIndex [h.BK] => decide (prog.feature*h.BK+i.1.val<h.width))=
      (fun _ => Bool.true) := by
    funext i
    exact decide_eq_true (column_inside h p prog i.1)
  refine ⟨Nat.zero_le _,?_,?_,?_,?_,by intros; rfl,?_,fun _ _ _ => rfl⟩
  · simp [FPPrefix.initial]
  · simp [FPPrefix.initial]
  · simp [FPPrefix.initial,mask]
  · simp [FPPrefix.initial,carry,FPReference.forwardCarry]
    rfl
  · intro t ht
    omega

theorem bind_holds (M : Algebra α) (h : Call) (p : Phase) (prog : Program h p)
    (initial : State α) (n : Nat) (u : State α) (inv : Holds M h p prog initial n u) :
    Holds M h p prog initial n (u.setReg "i" .int [] (fun _ => (n : Int))) := by
  refine ⟨inv.bound,?_,?_,?_,?_,inv.reads,inv.output,inv.frame⟩
  all_goals simp [inv.sourcePtr,inv.targetPtr,inv.mask,inv.acc]

theorem loop_executes (M : Algebra α) (h : Call) (p : Phase) (prog : Program h p)
    (initial : State α) (n : Nat) (u : State α) (inv : Holds M h p prog initial n u) :
    ∃ v, FP.Structural.rangeInt (engine M) "i" (n : Int) (p.span : Int) 1 (forwardBody h) u=some v ∧
      Holds M h p prog initial p.span v := by
  by_cases hn : n<p.span
  · obtain ⟨v,he,hv⟩ := iteration M h p prog initial n
      (u.setReg "i" .int [] (fun _ => (n : Int))) hn (bind_holds M h p prog initial n u inv)
    obtain ⟨w,hw,hf⟩ := loop_executes M h p prog initial (n+1) v hv
    refine ⟨w,?_,hf⟩
    have hi : (n : Int)<(p.span : Int) := by exact_mod_cast hn
    rw [FP.Structural.rangeInt]
    simpa [hi,he] using hw
  · have bound := inv.bound
    have eq : n=p.span := by omega
    refine ⟨u,?_,by simpa [eq] using inv⟩
    rw [FP.Structural.rangeInt]
    simp [eq]
termination_by p.span-n

theorem source_correct (M : Algebra α) (h : Call) (p : Phase) (prog : Program h p)
    (scale : ℝ) (u : State α) (hf : u.pids 0=prog.feature) (hc : u.pids 1=prog.chunk)
    (hr : u.pids 2=prog.row) :
    ∃ v, FP.Structural.exec (engine M) (Kernels.source .forward h p scale) u=some v ∧
      (∀ (t : Fin p.span) (j : Fin h.BK), v.mem "go" (off h p prog t.val j)=
        FPReference.forward M (fun n => raw h.BK "g" (off h p prog n) u (j,PUnit.unit)) t.val) ∧
      FPMemory.Frame h p prog ["go"] u v := by
  obtain ⟨v,he,inv⟩ := loop_executes M h p prog u 0 (FPPrefix.initial M .forward h p prog u)
    (initial_holds M h p prog u)
  refine ⟨v,?_,fun t j => inv.output t.val t.isLt j,inv.frame⟩
  change run (engine M) (Kernels.source .forward h p scale).surfaceBody u=some v
  rw [FPProgram.source_body,run_append,FPPrefix.source_init]
  rw [FPPrefix.executes M .forward h p prog u hf hc hr]
  simpa [FPProgram.loopCode,FPOps.forward_body,run,step,FP.Structural.evalOp] using he

#axiomsClean source_correct
end
end VeriTile.Bench.Repairs.DecayCumsum.FPForward
