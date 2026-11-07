import bench.tritonbench_g.layernorm_fwd_triton.Proofs.PairSource

namespace VeriTile.Bench.Repairs.LayernormFwdTriton.PairExecution
open VeriTile.Triton FP.Structural Source PairSource KernelSafety
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

theorem candidate_output_range_run {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (s t : State α) (start stop : Nat) (mem : RegionName → Nat → Cell α)
    (hx : p.x ≠ p.y) (hw : p.weight ≠ p.y) (h : OutputReady M p s t mem) :
    ∃ u, FP.Structural.rangeInt M "off" start stop p.block (outputBody p) t = some u ∧
      OutputReady M p s u (foldRange (outputStep M p s) start stop p.block mem) := by
  apply range_int_values M "off" (outputBody p) (outputStep M p s)
    (fun _ mem t => OutputReady M p s t mem) (fun mem t => OutputReady M p s t mem)
    start stop p.block mem t
  · intro n a u hu _
    exact candidate_output_body_run M p s u n a hx hw hu
  · intro _ _ _ hu
    exact hu
  · exact h

theorem candidate_split (p : Parameters) : (candidateKernel p).surfaceBody = prelude p ++
    [ComputeStmt.forRangeInt "off" (.ref .int [] "zero") (.castNatToInt (.constNat p.columns)) (.constNat p.block) (loopBody p 18)] ++ meanPost p ++
    [ComputeStmt.forRangeInt "off" (.ref .int [] "zero") (.castNatToInt (.constNat p.columns)) (.constNat p.block) (loopBody p 21)] ++ varPost p ++
    [ComputeStmt.forRangeInt "off" (.ref .int [] "zero") (.castNatToInt (.constNat p.columns)) (.constNat p.block) (outputBody p)] := by rfl

theorem candidate_complete {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α)
    (epsilon : α) (heps : s.regs .real [] "eps"=some (fun _ => epsilon))
    (hx : p.x ≠ p.y) (hw : p.weight ≠ p.y) :
    ∃ t, FP.Structural.exec M (candidateKernel p) s = some t ∧ t.mem = referenceMem M p s ∧ t.undef = s.undef := by
  obtain ⟨a,ha,hr⟩ := prelude_run M p s
  obtain ⟨b,hb,hs⟩ := mean_range_run M p s a 0 p.columns (initial M p) hr
  obtain ⟨c,hc,ht⟩ := mean_post_run M p s b hs
  obtain ⟨d,hd,hu⟩ := var_range_run M p s c 0 p.columns (initial M p) ht
  obtain ⟨e,he,hv⟩ := var_post_run M p s d epsilon heps hu
  obtain ⟨f,hf,hw'⟩ := candidate_output_range_run M p s e 0 p.columns s.mem hx hw hv
  simp only [Nat.cast_zero] at hb hd hf
  refine ⟨f,?_,hw'.memory,hw'.undef⟩
  change run M (candidateKernel p).surfaceBody s = some f
  rw [candidate_split]
  simp only [run_append, ha, run, List.foldlM_cons, List.foldlM_nil, bind, pure, Option.bind,
    step, FP.Structural.evalOp, hr.integers.1, ht.integers.1, hv.integers.1,
    Option.bind_some, Int.ofNat_eq_natCast, hb, hc, hd, he, hf]

/-- Both complete programs succeed and agree on every typed memory cell.
Generated load registers are private; their values and formats are not equated. -/
theorem exact {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α)
    (epsilon : α) (heps : s.regs .real [] "eps"=some (fun _ => epsilon))
    (hx : p.x≠p.y) (hw : p.weight≠p.y) :
    ∃ left right, exec M (kernel p) s=some left ∧
      exec M (candidateKernel p) s=some right ∧
      left.mem=right.mem ∧ left.undef=right.undef := by
  obtain ⟨left,hl,hm,hu⟩ := complete_run_undef M p s epsilon heps hx hw
  obtain ⟨right,hr,hn,hv⟩ := candidate_complete M p s epsilon heps hx hw
  exact ⟨left,right,hl,hr,hm.trans hn.symm,hu.trans hv.symm⟩

theorem candidate_body_safe {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (bounds : RegionName → Nat) (s t : State α) (off : Nat)
    (mem : RegionName → Nat → Cell α) (ready : OutputReady M p s t mem)
    (input : ∀ j,j<p.columns → xBase p s+j*p.xDimStride<bounds p.x)
    (weight : ∀ j,j<p.columns → wBase p s+j*p.weightDimStride<bounds p.weight)
    (output : ∀ j,j<p.columns → yBase p s+j*p.yDimStride<bounds p.y) :
    Safety.traceSafeList M bounds (outputBody p) (t.setReg "off" .int [] (fun _ => (off : Int))) := by
  simp [outputBody,candidateKernel,runtimeSourceCandidateKernel,ComputeKernel.surfaceBody,
    Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
    Safety.safeCompute_unfold,Safety.safeOp_unfold,Safety.safeAddress_unfold,
    Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
    step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,store,Option.bind,ComputeDType.eraseDType,
    FloatDType.toTileDType,FloatDType.numericDType,ofFloat,toFloat,
    numeric,bop,numericLt,natLt,Region.cast,ready.integers.1,ready.integers.2.1,ready.integers.2.2,ready.x,ready.y,ready.w,ready.mean,ready.rstd]
  exact ⟨fun j active => input (off+j.val) active,
    fun j active => weight (off+j.val) active,fun j active => output (off+j.val) active⟩

private theorem range_safe {α δ : Type} [Inhabited α] (M : Algebra α)
    (bounds : RegionName → Nat) (code : List ComputeStmt) (stride stop : Nat)
    (Ready : State α → δ → Prop) (update : Nat → δ → δ)
    (bodySafe : ∀ (off : Nat) t a,Ready t a →
      Safety.traceSafeList M bounds code (t.setReg "off" .int [] (fun _ => (off : Int))))
    (bodyRun : ∀ (off : Nat) t a,Ready t a → ∃ u,
      run M code (t.setReg "off" .int [] (fun _ => (off : Int)))=some u ∧ Ready u (update off a))
    (start : Nat) (t : State α) (a : δ) (ready : Ready t a) :
    Safety.rangeIntSafe M bounds "off" start stop stride code t := by
  rw [Safety.rangeIntSafe_unfold]
  by_cases zero : stride=0
  · simp [zero]
  by_cases active : start<stop
  · have hi : (start : Int) < stop := by exact_mod_cast active
    simp only [zero,hi,if_false,if_true]
    refine ⟨bodySafe start t a ready,?_⟩
    obtain ⟨u,executed,next⟩ := bodyRun start t a ready
    rw [executed]
    simpa only [Nat.cast_add] using
      range_safe M bounds code stride stop Ready update bodySafe bodyRun (start+stride) u _ next
  · simp [zero,active]
termination_by stop-start
 decreasing_by omega

/-- Source safety includes each iteration of all three original loops, using
existing actual-execution invariants. No numerical or successful-run assumption
is introduced; the runtime epsilon is an ordinary register ABI binding. -/
theorem candidate_safe {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (bounds : RegionName → Nat) (s : State α) (epsilon : α)
    (heps : s.regs .real [] "eps"=some (fun _ => epsilon))
    (separateX : p.x≠p.y) (separateW : p.weight≠p.y)
    (input : ∀ j,j<p.columns → xBase p s+j*p.xDimStride<bounds p.x)
    (weight : ∀ j,j<p.columns → wBase p s+j*p.weightDimStride<bounds p.weight)
    (output : ∀ j,j<p.columns → yBase p s+j*p.yDimStride<bounds p.y) :
    TraceSafe M bounds (candidateKernel p) s := by
  unfold TraceSafe
  rw [candidate_split]
  obtain ⟨a,ha,hr⟩ := prelude_run M p s
  obtain ⟨b,hb,hs⟩ := mean_range_run M p s a 0 p.columns (initial M p) hr
  obtain ⟨c,hc,ht⟩ := mean_post_run M p s b hs
  obtain ⟨d,hd,hu⟩ := var_range_run M p s c 0 p.columns (initial M p) ht
  obtain ⟨e,he,hv⟩ := var_post_run M p s d epsilon heps hu
  have meanSafe : Safety.rangeIntSafe M bounds "off" 0 p.columns p.block (loopBody p 18) a :=
    range_safe M bounds (loopBody p 18) p.block p.columns (MeanReady M p s)
      (meanStep M p s) (fun off t acc h => mean_body_safe M p bounds s t off acc h input)
      (fun off t acc h => mean_body_run M p s t off acc h) 0 a _ hr
  have varSafe : Safety.rangeIntSafe M bounds "off" 0 p.columns p.block (loopBody p 21) c :=
    range_safe M bounds (loopBody p 21) p.block p.columns (VarReady M p s)
      (varStep M p s) (fun off t acc h => var_body_safe M p bounds s t off acc h input)
      (fun off t acc h => var_body_run M p s t off acc h) 0 c _ ht
  have outputSafe : Safety.rangeIntSafe M bounds "off" 0 p.columns p.block (outputBody p) e :=
    range_safe M bounds (outputBody p) p.block p.columns (OutputReady M p s)
      (outputStep M p s) (fun off t mem h => candidate_body_safe M p bounds s t off mem h input weight output)
      (fun off t mem h => candidate_output_body_run M p s t off mem separateX separateW h) 0 e _ hv
  simp only [Nat.cast_zero] at hb hd meanSafe varSafe outputSafe
  rw [List.append_assoc,List.append_assoc,List.append_assoc,List.append_assoc,
    Safety.traceSafeList_append]
  refine ⟨prelude_safe M p bounds s,?_⟩
  intro a' ha'
  rw [ha] at ha'
  cases ha'
  try simp only [List.singleton_append]
  apply Safety.traceSafeList_cons_of_step (by simpa [Safety.stepSafe_unfold,Safety.safeOp_unfold,FP.Structural.evalOp,hr.integers.1,Int.ofNat_eq_natCast] using meanSafe) (by simpa [step,FP.Structural.evalOp,hr.integers.1,Int.ofNat_eq_natCast] using hb)
  rw [Safety.traceSafeList_append]
  refine ⟨mean_post_safe M p bounds b,?_⟩
  intro c' hc'
  rw [hc] at hc'
  cases hc'
  try simp only [List.singleton_append]
  apply Safety.traceSafeList_cons_of_step (by simpa [Safety.stepSafe_unfold,Safety.safeOp_unfold,FP.Structural.evalOp,ht.integers.1,Int.ofNat_eq_natCast] using varSafe) (by simpa [step,FP.Structural.evalOp,ht.integers.1,Int.ofNat_eq_natCast] using hd)
  rw [Safety.traceSafeList_append]
  refine ⟨var_post_safe M p bounds d,?_⟩
  intro e' he'
  rw [he] at he'
  cases he'
  apply Safety.traceSafeList_cons_intro (by simpa [Safety.stepSafe_unfold,Safety.safeOp_unfold,FP.Structural.evalOp,hv.integers.1,Int.ofNat_eq_natCast] using outputSafe)
  intro t _
  exact Safety.traceSafeList_nil M bounds t

#axiomsClean candidate_split
#axiomsClean candidate_output_range_run
#axiomsClean candidate_complete
#axiomsClean exact
#axiomsClean candidate_body_safe
#axiomsClean candidate_safe
end
end VeriTile.Bench.Repairs.LayernormFwdTriton.PairExecution
