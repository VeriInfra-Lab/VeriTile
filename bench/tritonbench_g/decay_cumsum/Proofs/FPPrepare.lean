import bench.tritonbench_g.decay_cumsum.Proofs.FPMemory

namespace VeriTile.Bench.Repairs.DecayCumsum.FPPrepare
open VeriTile.Triton FP.Structural FP.ProfiledRewrite FPOps FPPrefix Geometry
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 6000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def lastValue (M : Algebra α) (h : Call) (p : Phase) (prog : Program h p) (u : State α) : Values α .real [h.BK] :=
  fun i => M.fp32Load (raw h.BK "g" (off h p prog (p.span-1)) u i)
def qValue (M : Algebra α) (h : Call) (p : Phase) (prog : Program h p) (scale : ℝ) (u : State α) (n : Nat) : Values α .real [h.BK] :=
  fun i => FPReference.prepareQ M scale (raw h.BK "q" (off h p prog n) u i) (raw h.BK "g" (off h p prog n) u i)
def kValue (M : Algebra α) (h : Call) (p : Phase) (prog : Program h p) (u : State α) (n : Nat) : Values α .real [h.BK] :=
  fun i => FPReference.prepareK M (raw h.BK "k" (off h p prog n) u i)
    (raw h.BK "g" (off h p prog (p.span-1)) u i) (raw h.BK "g" (off h p prog n) u i)

structure Holds (M : Algebra α) (h : Call) (p : Phase) (prog : Program h p) (scale : ℝ)
    (initial : State α) (n : Nat) (u : State α) : Prop where
  bound : n≤p.span
  qPtr : u.regs .ptr [h.BK] "p_q"=some (ptr h.BK "q" (off h p prog n))
  kPtr : u.regs .ptr [h.BK] "p_k"=some (ptr h.BK "k" (off h p prog n))
  gPtr : u.regs .ptr [h.BK] "p_g"=some (ptr h.BK "g" (off h p prog n))
  qoPtr : u.regs .ptr [h.BK] "p_qg"=some (ptr h.BK "qg" (off h p prog n))
  koPtr : u.regs .ptr [h.BK] "p_kg"=some (ptr h.BK "kg" (off h p prog n))
  mask : u.regs .bool [h.BK] "mask"=some (fun _ => Bool.true)
  last : u.regs .real [h.BK] "last_decay"=some (lastValue M h p prog initial)
  reads : ∀ r∈(["q","k","g"] : List RegionName), ∀ o, u.mem r o=initial.mem r o
  qo : ∀ t, t<n → ∀ j : Fin h.BK, u.mem "qg" (off h p prog t j)=FPReference.stored M (qValue M h p prog scale initial t (j,PUnit.unit))
  ko : ∀ t, t<n → ∀ j : Fin h.BK, u.mem "kg" (off h p prog t j)=FPReference.stored M (kValue M h p prog initial t (j,PUnit.unit))
  frame : FPMemory.Frame h p prog ["qg","kg"] initial u

theorem iteration (M : Algebra α) (h : Call) (p : Phase) (prog : Program h p) (scale : ℝ)
    (initial : State α) (n : Nat) (u : State α) (hn : n<p.span) (inv : Holds M h p prog scale initial n u) :
    ∃ v, run (engine M) (prepareBody h scale) u=some v ∧ Holds M h p prog scale initial (n+1) v := by
  let last := lastValue M h p prog initial
  let qv := FPStep.prepareQValue M h (off h p prog n) scale u
  let kv := FPStep.prepareKValue M h (off h p prog n) last u
  let pre := ((((u.setReg "_q" .real [h.BK] (fun i => M.fp32Load (raw h.BK "q" (off h p prog n) u i))).setReg
    "_k" .real [h.BK] (fun i => M.fp32Load (raw h.BK "k" (off h p prog n) u i))).setReg
    "_g" .real [h.BK] (fun i => FPReference.gate M (raw h.BK "g" (off h p prog n) u i))).setReg
    "_q" .real [h.BK] qv).setReg "_k" .real [h.BK] kv
  let stores := scatter M h.BK "qg" (off h p prog n) qv (scatter M h.BK "kg" (off h p prog n) kv pre)
  let v := FPStep.prepareResult M h (off h p prog n) scale last u
  have qeq : qv=qValue M h p prog scale initial n := by
    funext i
    simp [qv,FPStep.prepareQValue,qValue,raw,inv.reads "q" (by simp),inv.reads "g" (by simp)]
  have keq : kv=kValue M h p prog initial n := by
    funext i
    simp [kv,FPStep.prepareKValue,kValue,FPReference.prepareK,last,lastValue,raw,
      inv.reads "k" (by simp),inv.reads "g" (by simp)]
  have pointers : (fun j => off h p prog n j+h.width)=off h p prog (n+1) := funext (FPMemory.off_succ h p prog n)
  refine ⟨v,FPStep.prepare M h _ scale last u inv.qPtr inv.kPtr inv.gPtr inv.qoPtr inv.koPtr inv.mask inv.last,
    Nat.succ_le_of_lt hn,?_,?_,?_,?_,?_,?_,?_,?_,?_,?_,?_⟩
  · simp [v,FPStep.prepareResult,pointers]
  · simp [v,FPStep.prepareResult,pointers]
  · simp [v,FPStep.prepareResult,pointers]
  · simp [v,FPStep.prepareResult,pointers]
  · simp [v,FPStep.prepareResult,pointers]
  · simp [v,FPStep.prepareResult,inv.mask]
  · simp [v,FPStep.prepareResult,inv.last]
  · intro r member o
    have rq : r≠("qg" : RegionName) := by simp only [List.mem_cons,List.not_mem_nil,or_false] at member; rcases member with rfl|rfl|rfl <;> decide
    have rk : r≠("kg" : RegionName) := by simp only [List.mem_cons,List.not_mem_nil,or_false] at member; rcases member with rfl|rfl|rfl <;> decide
    change stores.mem r o=initial.mem r o
    simpa [stores,pre,rq,rk] using inv.reads r member o
  · intro t ht j
    change stores.mem "qg" _=_
    dsimp [stores]
    by_cases same : t=n
    · subst t
      rw [FPMemory.cell,qeq]
    · rw [FPMemory.other_step M h p prog n t (Ne.symm same)]
      simpa [pre] using inv.qo t (by omega) j
  · intro t ht j
    change stores.mem "kg" _=_
    rw [scatter_other M h.BK "qg" _ qv _ "kg" _ (by decide)]
    by_cases same : t=n
    · subst t
      rw [FPMemory.cell,keq]
    · rw [FPMemory.other_step M h p prog n t (Ne.symm same)]
      exact inv.ko t (by omega) j
  · intro r o outside
    change stores.mem r o=initial.mem r o
    exact (FPMemory.frame M h p prog ⟨n,hn⟩ "qg" ["qg","kg"] (by simp) qv _ r o outside).trans
      ((FPMemory.frame M h p prog ⟨n,hn⟩ "kg" ["qg","kg"] (by simp) kv pre r o outside).trans (inv.frame r o outside))

theorem initial_holds (M : Algebra α) (h : Call) (p : Phase) (positive : 0<p.span)
    (prog : Program h p) (scale : ℝ) (u : State α) :
    Holds M h p prog scale u 0 (FPPrefix.initial M .prepare h p prog u) := by
  have mask : (fun i : TileIndex [h.BK] => decide (prog.feature*h.BK+i.1.val<h.width))=(fun _ => Bool.true) := by
    funext i
    exact decide_eq_true (column_inside h p prog i.1)
  have last : (fun j : Fin h.BK => prog.row*h.headStride+(prog.chunk*p.span+p.span-1)*h.width+prog.feature*h.BK+j.val)=off h p prog (p.span-1) := by
    funext j
    have timeEq : prog.chunk*p.span+p.span-1=prog.chunk*p.span+(p.span-1) := by omega
    simp [off,timeEq,Nat.add_mul]
    omega
  refine ⟨Nat.zero_le _,?_,?_,?_,?_,?_,?_,?_,by intros; rfl,?_,?_,fun _ _ _ => rfl⟩
  all_goals first | (intro t ht; omega) | (simp [FPPrefix.initial,mask,last,lastValue] <;> rfl)

theorem bind_holds (M : Algebra α) (h : Call) (p : Phase) (prog : Program h p) (scale : ℝ)
    (initial : State α) (n : Nat) (u : State α) (inv : Holds M h p prog scale initial n u) :
    Holds M h p prog scale initial n (u.setReg "i" .nat [] (fun _ => n)) := by
  refine ⟨inv.bound,?_,?_,?_,?_,?_,?_,?_,inv.reads,inv.qo,inv.ko,inv.frame⟩
  all_goals simp [inv.qPtr,inv.kPtr,inv.gPtr,inv.qoPtr,inv.koPtr,inv.mask,inv.last]

theorem loop_executes (M : Algebra α) (h : Call) (p : Phase) (prog : Program h p) (scale : ℝ)
    (initial : State α) (n : Nat) (u : State α) (inv : Holds M h p prog scale initial n u) :
    ∃ v, FP.Structural.range (engine M) "i" n p.span 1 (prepareBody h scale) u=some v ∧ Holds M h p prog scale initial p.span v := by
  by_cases hn : n<p.span
  · obtain ⟨v,he,hv⟩ := iteration M h p prog scale initial n (u.setReg "i" .nat [] (fun _ => n)) hn (bind_holds M h p prog scale initial n u inv)
    obtain ⟨w,hw,hf⟩ := loop_executes M h p prog scale initial (n+1) v hv
    refine ⟨w,?_,hf⟩
    rw [FP.Structural.range]
    simpa [hn,he] using hw
  · have bound := inv.bound
    have eq : n=p.span := by omega
    refine ⟨u,FP.Structural.range_done _ _ _ _ _ _ _ (by omega),by simpa [eq] using inv⟩
termination_by p.span-n

theorem source_correct (M : Algebra α) (h : Call) (p : Phase) (positive : 0<p.span) (prog : Program h p)
    (scale : ℝ) (u : State α) (hf : u.pids 0=prog.feature) (hc : u.pids 1=prog.chunk) (hr : u.pids 2=prog.row) :
    ∃ v, FP.Structural.exec (engine M) (Kernels.source .prepare h p scale) u=some v ∧
      (∀ (t : Fin p.span) (j : Fin h.BK), v.mem "qg" (off h p prog t.val j)=FPReference.stored M (qValue M h p prog scale u t.val (j,PUnit.unit))) ∧
      (∀ (t : Fin p.span) (j : Fin h.BK), v.mem "kg" (off h p prog t.val j)=FPReference.stored M (kValue M h p prog u t.val (j,PUnit.unit))) ∧
      FPMemory.Frame h p prog ["qg","kg"] u v := by
  obtain ⟨v,he,inv⟩ := loop_executes M h p prog scale u 0 (FPPrefix.initial M .prepare h p prog u) (initial_holds M h p positive prog scale u)
  refine ⟨v,?_,fun t j => inv.qo t.val t.isLt j,fun t j => inv.ko t.val t.isLt j,inv.frame⟩
  change run (engine M) (Kernels.source .prepare h p scale).surfaceBody u=some v
  rw [FPProgram.source_body,run_append,FPPrefix.source_init,FPPrefix.executes M .prepare h p prog u hf hc hr]
  simpa [FPProgram.loopCode,FPOps.prepare_body,run,step,FP.Structural.evalOp] using he

#axiomsClean source_correct
end
end VeriTile.Bench.Repairs.DecayCumsum.FPPrepare
