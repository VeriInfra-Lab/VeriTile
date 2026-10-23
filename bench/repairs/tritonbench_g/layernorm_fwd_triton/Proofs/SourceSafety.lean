import bench.repairs.tritonbench_g.layernorm_fwd_triton.Proofs.SourceSafetySteps

namespace VeriTile.Bench.Repairs.LayernormFwdTriton.KernelSafety
open VeriTile.Triton FP.Structural Source
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 22000

/-- Compose actual range successor states with the existing accumulator
invariants. This is only a proof combinator for the shared safety predicate. -/
private theorem range_safe {α δ : Type} [Inhabited α] (M : Algebra α)
    (bounds : RegionName → Nat) (code : List ComputeStmt) (stride stop : Nat)
    (Ready : State α → δ → Prop) (update : Nat → δ → δ)
    (bodySafe : ∀ off t a,Ready t a →
      Safety.traceSafeList M bounds code (t.setReg "off" .nat [] (fun _ => off)))
    (bodyRun : ∀ off t a,Ready t a → ∃ u,
      run M code (t.setReg "off" .nat [] (fun _ => off))=some u ∧ Ready u (update off a))
    (start : Nat) (t : State α) (a : δ) (ready : Ready t a) :
    Safety.rangeSafe M bounds "off" start stop stride code t := by
  rw [Safety.rangeSafe_unfold]
  by_cases zero : stride=0
  · simp [zero]
  by_cases active : start<stop
  · simp only [zero,active,if_false,if_true]
    refine ⟨bodySafe start t a ready,?_⟩
    obtain ⟨u,executed,next⟩ := bodyRun start t a ready
    rw [executed]
    exact range_safe M bounds code stride stop Ready update bodySafe bodyRun (start+stride) u _ next
  · simp [zero,active]
termination_by stop-start
 decreasing_by omega

/-- Source safety includes each iteration of all three original loops, using
existing actual-execution invariants. No numerical or successful-run assumption
is introduced; the runtime epsilon is an ordinary register ABI binding. -/
theorem trace_safe {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (bounds : RegionName → Nat) (s : State α) (epsilon : α)
    (heps : s.regs .real [] "eps_arg"=some (fun _ => epsilon))
    (separateX : p.x≠p.y) (separateW : p.weight≠p.y)
    (input : ∀ j,j<p.columns → xBase p s+j*p.xDimStride<bounds p.x)
    (weight : ∀ j,j<p.columns → wBase p s+j*p.weightDimStride<bounds p.weight)
    (output : ∀ j,j<p.columns → yBase p s+j*p.yDimStride<bounds p.y) :
    TraceSafe M bounds (Source.kernel p) s := by
  unfold TraceSafe
  rw [source_split]
  obtain ⟨a,ha,hr⟩ := prelude_run M p s
  obtain ⟨b,hb,hs⟩ := mean_range_run M p s a 0 p.columns (initial M p) hr
  obtain ⟨c,hc,ht⟩ := mean_post_run M p s b hs
  obtain ⟨d,hd,hu⟩ := var_range_run M p s c 0 p.columns (initial M p) ht
  obtain ⟨e,he,hv⟩ := var_post_run M p s d epsilon heps hu
  have meanSafe : Safety.rangeSafe M bounds "off" 0 p.columns p.block (loopBody p 6) a :=
    range_safe M bounds (loopBody p 6) p.block p.columns (MeanReady M p s)
      (meanStep M p s) (fun off t acc h => mean_body_safe M p bounds s t off acc h input)
      (fun off t acc h => mean_body_run M p s t off acc h) 0 a _ hr
  have varSafe : Safety.rangeSafe M bounds "off" 0 p.columns p.block (loopBody p 9) c :=
    range_safe M bounds (loopBody p 9) p.block p.columns (VarReady M p s)
      (varStep M p s) (fun off t acc h => var_body_safe M p bounds s t off acc h input)
      (fun off t acc h => var_body_run M p s t off acc h) 0 c _ ht
  have outputSafe : Safety.rangeSafe M bounds "off" 0 p.columns p.block (loopBody p 12) e :=
    range_safe M bounds (loopBody p 12) p.block p.columns (OutputReady M p s)
      (outputStep M p s) (fun off t mem h => output_body_safe M p bounds s t off mem h input weight output)
      (fun off t mem h => output_body_run M p s t off mem separateX separateW h) 0 e _ hv
  rw [List.append_assoc,List.append_assoc,List.append_assoc,List.append_assoc,
    Safety.traceSafeList_append]
  refine ⟨prelude_safe M p bounds s,?_⟩
  intro a' ha'
  rw [ha] at ha'
  cases ha'
  try simp only [List.singleton_append]
  apply Safety.traceSafeList_cons_of_step (by simpa only [Safety.stepSafe_unfold] using meanSafe) (by simpa only [step] using hb)
  rw [Safety.traceSafeList_append]
  refine ⟨mean_post_safe M p bounds b,?_⟩
  intro c' hc'
  rw [hc] at hc'
  cases hc'
  try simp only [List.singleton_append]
  apply Safety.traceSafeList_cons_of_step (by simpa only [Safety.stepSafe_unfold] using varSafe) (by simpa only [step] using hd)
  rw [Safety.traceSafeList_append]
  refine ⟨var_post_safe M p bounds d,?_⟩
  intro e' he'
  rw [he] at he'
  cases he'
  apply Safety.traceSafeList_cons_intro (by simpa only [Safety.stepSafe_unfold] using outputSafe)
  intro t _
  exact Safety.traceSafeList_nil M bounds t

#axiomsClean trace_safe
end
end VeriTile.Bench.Repairs.LayernormFwdTriton.KernelSafety
