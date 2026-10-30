import bench.repairs.tritonbench_g.layernorm_fwd_triton.Proofs.PairSource

namespace VeriTile.Bench.Repairs.LayernormFwdTriton.PairExecution
open VeriTile.Triton FP.Structural Source PairSource KernelSafety
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

theorem candidate_split (p : Parameters) : (candidateKernel p).surfaceBody = prelude p ++
    [ComputeStmt.forRangeInt "off" (.ref .int [] "zero") (.ref .int [] "N") (.constNat p.block) (loopBody p 18)] ++ meanPost p ++
    [ComputeStmt.forRangeInt "off" (.ref .int [] "zero") (.ref .int [] "N") (.constNat p.block) (loopBody p 21)] ++ varPost p ++
    [ComputeStmt.forRangeInt "off" (.ref .int [] "zero") (.ref .int [] "N") (.constNat p.block) (outputBody p)] := by rfl

theorem output_range_eq {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (s t : State α) (start stop : Nat) (mem : RegionName → Nat → Cell α)
    (hx : p.x ≠ p.y) (hw : p.weight ≠ p.y) (ready : OutputReady M p s t mem) :
    FP.Structural.rangeInt M "off" (start : Int) (stop : Int) p.block (loopBody p 24) t =
      FP.Structural.rangeInt M "off" (start : Int) (stop : Int) p.block (outputBody p) t := by
  rw [FP.Structural.rangeInt, FP.Structural.rangeInt]
  by_cases zero : p.block = 0
  · simp [zero]
  by_cases active : start < stop
  · simp only [Int.ofNat_lt, zero, active, if_false, if_true]
    rw [← output_body_eq M p s t start mem ready]
    obtain ⟨u, ran, next⟩ := output_body_run M p s t start mem hx hw ready
    simp only [ran, bind, Option.bind_some]
    simpa only [Int.natCast_add] using output_range_eq M p s u (start+p.block) stop _ hx hw next
  · simp [zero, active]
termination_by stop-start
 decreasing_by omega

/-- Both complete three-pass programs have identical full final state and
failure behavior at every row; only an independent load has been scheduled. -/
theorem exact {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α)
    (epsilon : α) (heps : s.regs .real [] "eps" = some (fun _ => epsilon))
    (hx : p.x ≠ p.y) (hw : p.weight ≠ p.y) :
    FP.Structural.exec M (kernel p) s = FP.Structural.exec M (candidateKernel p) s := by
  obtain ⟨a,ha,hr⟩ := prelude_run M p s
  obtain ⟨b,hb,hs⟩ := mean_range_run M p s a 0 p.columns (initial M p) hr
  obtain ⟨c,hc,ht⟩ := mean_post_run M p s b hs
  obtain ⟨d,hd,hu⟩ := var_range_run M p s c 0 p.columns (initial M p) ht
  obtain ⟨e,he,hv⟩ := var_post_run M p s d epsilon heps hu
  simp only [Int.natCast_zero] at hb hd
  change run M (kernel p).surfaceBody s = run M (candidateKernel p).surfaceBody s
  rw [source_split, candidate_split]
  simp only [run_append, ha, run, bind, pure, Option.bind, step, evalOp_unfold, hr.zeroValue, hr.nValue, ht.zeroValue, ht.nValue, hv.zeroValue, hv.nValue, hb, hc, hd, he]
  exact congrArg (fun v : Option (State α) => v.bind (fun t => some t))
    (output_range_eq M p s e 0 p.columns s.mem hx hw hv)

theorem candidate_body_safe {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (bounds : RegionName → Nat) (s t : State α) (off : Nat)
    (mem : RegionName → Nat → Cell α) (ready : OutputReady M p s t mem)
    (input : ∀ j,j<p.columns → xBase p s+j*p.xDimStride<bounds p.x)
    (weight : ∀ j,j<p.columns → wBase p s+j*p.weightDimStride<bounds p.weight)
    (output : ∀ j,j<p.columns → yBase p s+j*p.yDimStride<bounds p.y) :
    Safety.traceSafeList M bounds (outputBody p) (t.setReg "off" .int [] (fun _ => (off : Int))) := by
  simp [outputBody,candidateKernel,runtimeSourceCandidate,ComputeKernel.surfaceBody, Random.i32ArangeSize,
    Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
    Safety.safeCompute_unfold,Safety.safeOp_unfold,Safety.safeAddress_unfold,
    Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
    step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,store,Option.bind,ComputeDType.eraseDType,
    FloatDType.toTileDType,FloatDType.numericDType,ofFloat,toFloat,
    numeric,bop,numericLt,natLt,Region.cast,-Nat.cast_mul,-Nat.cast_add,← Int.natCast_mul,← Int.natCast_add,Int.toNat_add,Int.toNat_mul,Int.natCast_nonneg,Int.toNat_of_nonpos,ready.x, ready.nValue, ready.zeroValue, ready.stepValue, ready.lanes, ready.xStride, ready.yStride, ready.wStride,ready.y,ready.w,ready.mean,ready.rstd]
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
  · simp only [Int.ofNat_lt,zero,active,if_false,if_true]
    refine ⟨bodySafe start t a ready,?_⟩
    obtain ⟨u,executed,next⟩ := bodyRun start t a ready
    rw [executed]
    simpa only [Int.natCast_add] using
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
      (fun off t mem h => by
        obtain ⟨u, ran, next⟩ := output_body_run M p s t off mem separateX separateW h
        exact ⟨u, (output_body_eq M p s t off mem h).symm.trans ran, next⟩) 0 e _ hv
  rw [List.append_assoc,List.append_assoc,List.append_assoc,List.append_assoc,
    Safety.traceSafeList_append]
  refine ⟨prelude_safe M p bounds s,?_⟩
  intro a' ha'
  rw [ha] at ha'
  cases ha'
  try simp only [List.singleton_append]
  apply Safety.traceSafeList_cons_of_step (by simpa only [Safety.stepSafe_unfold, Safety.safeOp_unfold, evalOp_unfold, hr.zeroValue, hr.nValue, Option.bind, and_self, true_and] using meanSafe) (by simpa only [step, evalOp_unfold, hr.zeroValue, hr.nValue, Option.bind] using hb)
  rw [Safety.traceSafeList_append]
  refine ⟨mean_post_safe M p bounds b,?_⟩
  intro c' hc'
  rw [hc] at hc'
  cases hc'
  try simp only [List.singleton_append]
  apply Safety.traceSafeList_cons_of_step (by simpa only [Safety.stepSafe_unfold, Safety.safeOp_unfold, evalOp_unfold, ht.zeroValue, ht.nValue, Option.bind, and_self, true_and] using varSafe) (by simpa only [step, evalOp_unfold, ht.zeroValue, ht.nValue, Option.bind] using hd)
  rw [Safety.traceSafeList_append]
  refine ⟨var_post_safe M p bounds d,?_⟩
  intro e' he'
  rw [he] at he'
  cases he'
  apply Safety.traceSafeList_cons_intro (by simpa only [Safety.stepSafe_unfold, Safety.safeOp_unfold, evalOp_unfold, hv.zeroValue, hv.nValue, Option.bind, and_self, true_and] using outputSafe)
  intro t _
  exact Safety.traceSafeList_nil M bounds t

#axiomsClean candidate_split
#axiomsClean output_range_eq
#axiomsClean exact
#axiomsClean candidate_body_safe
#axiomsClean candidate_safe
end
end VeriTile.Bench.Repairs.LayernormFwdTriton.PairExecution
