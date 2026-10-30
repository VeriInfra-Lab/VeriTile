import bench.repairs.tritonbench_g.triton_linear_activation.Proofs.KernelLoop

namespace VeriTile.Bench.Repairs.TritonLinearActivation.KernelTail
open VeriTile.Triton FP.Structural KernelSource KernelValues KernelPrefix KernelLoop
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 35000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def saveCode (mode : Mode) (p : Parameters) := (tailCode mode p).take 1
def activationCode (p : Parameters) := ((tailCode .fp32 p).drop 1).take 4
def storeCode (mode : Mode) (p : Parameters) := (tailCode mode p).drop 5
theorem tail_split (mode : Mode) (p : Parameters) :
    tailCode mode p=saveCode mode p++activationCode p++storeCode mode p := by rfl

structure TailReady (p : Parameters) (s t : State α) (acc : Accumulator α p) : Prop where
  rawm : t.regs .nat [p.blockRows] "m_offs_untagged"=some (fun i => KernelValues.row p s i.1)
  rawn : t.regs .nat [p.blockColumns] "n_offs_untagged"=some (fun i => KernelValues.col p s i.1)
  acc : t.regs .real [p.blockRows,p.blockColumns] "acc"=some acc

theorem fold_memory_congr {ι : Type} (l : List ι) (region : RegionName) (off : ι → Nat)
    (active : ι → Prop) [DecidablePred active] (v : ι → Cell α) (s t : State α) (hm : s.mem=t.mem) :
    (l.foldl (fun u i => if active i then u.write region (off i) (v i) else u) s).mem=
      (l.foldl (fun u i => if active i then u.write region (off i) (v i) else u) t).mem := by
  induction l generalizing s t with
  | nil => exact hm
  | cons i rest ih =>
    simp only [List.foldl_cons]
    by_cases ha : active i
    · simp only [ha,if_true]
      exact ih _ _ (by simp only [State.write,hm])
    · simp only [ha,if_false]
      exact ih s t hm

theorem save_run (M : Algebra α) (D : Dot α) (mode : Mode) (p : Parameters)
    (s t : State α) (acc : Accumulator α p) (h : TailReady p s t acc) (hm : t.mem=s.mem) :
    ∃ u, run (engine M D) (saveCode mode p) t=some u ∧ TailReady p s u acc ∧
      u.mem=(if p.saveInputs then scatter M mode p s s p.saved (savedOffset p s) acc else s).mem := by
  cases mode <;> cases saved : p.saveInputs
  all_goals
    simp [saveCode,tailCode,source,typed,dtype,ComputeKernel.surfaceBody,run,step,
      evalExpr,evalComputeOp,evalOp_unfold,Option.bind,ComputeDType.eraseDType,
      numeric,bop,natLt,numericLt,Region.cast,ofFloat,toFloat,State.setReg,store,
      h.rawm,h.rawn,h.acc,saved,engine,Algebra.withDefaultPrecision,resolvePrecision]
  all_goals refine ⟨?_,?_⟩
  all_goals first
    | (constructor <;> simp [State.setReg,h.rawm,h.rawn,h.acc,State.masked_scatter_eq_filter])
    | exact hm
    | (simp only [scatter,active,savedOffset,outputCell,dtype,ofFloat,decide_eq_true_eq,Bool.and_eq_true];
       apply fold_memory_congr; exact hm)

private theorem activation_cases (name : String) :
    name="tanh" ∨ name="gelu" ∨ name="fast_gelu" ∨ name="relu" ∨
      (name≠"tanh" ∧ name≠"gelu" ∧ name≠"fast_gelu" ∧ name≠"relu") := by
  by_cases ht : name="tanh" <;> by_cases hg : name="gelu" <;>
    by_cases hf : name="fast_gelu" <;> by_cases hr : name="relu" <;> tauto

theorem activation_run (M : Algebra α) (D : Dot α) (p : Parameters) (s t : State α)
    (acc : Accumulator α p) (h : TailReady p s t acc) :
    ∃ u, run (engine M D) (activationCode p) t=some u ∧
      TailReady p s u (fun i => KernelValues.activation M p.activation (acc i)) ∧ u.mem=t.mem := by
  cases name : p.activation with
  | none =>
    simp [activationCode,tailCode,source,typed,dtype,ComputeKernel.surfaceBody,run,step,
      evalExpr,evalComputeOp,evalOp_unfold,Option.bind,ComputeDType.eraseDType,
      numeric,bop,ofFloat,toFloat,State.setReg,name,KernelValues.activation]
    exact h
  | some value =>
    rcases activation_cases value with ht | hg | hf | hr | ⟨ht,hg,hf,hr⟩
    all_goals
      simp_all [activationCode,tailCode,source,typed,dtype,ComputeKernel.surfaceBody,run,step,
        evalExpr,evalComputeOp,evalOp_unfold,Option.bind,ComputeDType.eraseDType,
        numeric,bop,ofFloat,toFloat,State.setReg,h.rawm,h.rawn,h.acc,
        engine,Algebra.withDefaultPrecision,resolvePrecision,KernelValues.activation,zero]
    all_goals constructor
    all_goals simp [State.setReg,h.rawm,h.rawn,h.acc,bop]
    all_goals rfl

theorem store_run (M : Algebra α) (D : Dot α) (mode : Mode) (p : Parameters)
    (s t base : State α) (acc : Accumulator α p) (h : TailReady p s t acc) (hm : t.mem=base.mem) :
    ∃ u, run (engine M D) (storeCode mode p) t=some u ∧
      u.mem=(scatter M mode p s base p.output (KernelValues.outputOffset p s) acc).mem := by
  cases mode
  all_goals
    simp [storeCode,tailCode,source,typed,dtype,ComputeKernel.surfaceBody,run,step,
      evalExpr,evalComputeOp,evalOp_unfold,Option.bind,ComputeDType.eraseDType,
      numeric,bop,natLt,numericLt,Region.cast,ofFloat,toFloat,State.setReg,store,
      h.rawm,h.rawn,h.acc,engine,Algebra.withDefaultPrecision,resolvePrecision]
  all_goals simp only [scatter,active,KernelValues.outputOffset,outputCell,dtype,ofFloat,decide_eq_true_eq,Bool.and_eq_true]
  all_goals apply fold_memory_congr
  all_goals exact hm

/-- Successful execution of the actual readable kernel; no execution or
completed-output premise is used. Both stores are derived from the syntax. -/
theorem source_executes (M : Algebra α) (D : Dot α) (mode : Mode) (p : Parameters)
    (s : State α) (positive : 0<p.blockInner) :
    ∃ final, exec (engine M D) (source mode p) s=some final ∧
      final.mem=(written M mode p s (accumulated M D mode p s)).mem := by
  obtain ⟨a,ha,oa⟩ := scalars_run (engine M D) p s
  obtain ⟨b,hb,rb⟩ := seeds_run M D mode p s a oa
  obtain ⟨c,hc,rc⟩ := loop_run M D mode p s b positive 0 (Nat.zero_le _) rb
  have tailReady : TailReady p s c (accumulated M D mode p s) := ⟨rc.rawm,rc.rawn,rc.acc⟩
  obtain ⟨d,hd,rd,md⟩ := save_run M D mode p s c _ tailReady rc.mem
  obtain ⟨e,he,re,me⟩ := activation_run M D p s d _ rd
  obtain ⟨f,hf,mf⟩ := store_run M D mode p s e
    (if p.saveInputs then scatter M mode p s s p.saved (savedOffset p s) (accumulated M D mode p s) else s)
    _ re (me.trans md)
  refine ⟨f,?_,mf⟩
  change run (engine M D) (source mode p).surfaceBody s=some f
  rw [source_split,front_split,tail_split]
  simp only [run_append,ha,hb,Option.bind_some,run,statement_eval,hc,hd,he,hf,bind,Option.bind]

#axiomsClean source_executes
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.TritonLinearActivation.KernelTail
