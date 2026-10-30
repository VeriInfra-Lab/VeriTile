import bench.repairs.tritonbench_g.triton_linear_activation.Proofs.KernelTail
import VeriTile.Triton.Float.StructuralSafety

namespace VeriTile.Bench.Repairs.TritonLinearActivation.SourceSafety
open VeriTile.Triton FP.Structural KernelSource KernelValues KernelPrefix KernelLoop KernelTail
noncomputable section
set_option maxHeartbeats 18000000
set_option maxRecDepth 35000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

/-- Bounds for the concrete operand windows read by this program. They include
wrapped row/column reads even when every output lane is masked off. -/
structure AccessBounds (p : Parameters) (s : State α) (bounds : RegionName → Nat) : Prop where
  bias : p.hasBias = Bool.true → ∀ j : Fin p.blockColumns,
    KernelValues.col p s j % p.columns < p.columns → KernelValues.col p s j % p.columns < bounds p.bias
  input : ∀ cur, cur < ceilDiv p.inner p.blockInner →
    ∀ i : TileIndex [p.blockRows,p.blockInner],
      (p.inner % p.blockInner = 0 ∨ (i.2.1.val : Int) < (p.inner : Int) - (cur*p.blockInner : Int)) →
      aOffset p s cur i < bounds p.input
  weight : ∀ cur, cur < ceilDiv p.inner p.blockInner →
    ∀ i : TileIndex [p.blockInner,p.blockColumns],
      (p.inner % p.blockInner = 0 ∨ (i.1.val : Int) < (p.inner : Int) - (cur*p.blockInner : Int)) →
      bOffset p s cur i < bounds p.weight
  output : ∀ i, active p s i = Bool.true → outputOffset p s i < bounds p.output
  saved : p.saveInputs = Bool.true → ∀ i, active p s i = Bool.true → savedOffset p s i < bounds p.saved

theorem scalars_safe (M : Algebra α) (p : Parameters) (s : State α)
    (bounds : RegionName → Nat) : Safety.traceSafeList M bounds (scalars p) s := by
  simp [scalars,source,typed,ComputeKernel.surfaceBody,
    Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,
    Safety.safeOp_unfold,run,step,evalExpr,evalOp_unfold,Option.bind,
    numeric,bop,natLt,numericLt,integralFloorDiv,integralMod,State.setReg]

theorem seeds_safe (M : Algebra α) (D : Dot α) (mode : Mode) (p : Parameters)
    (s t : State α) (bounds : RegionName → Nat) (h : Offsets p s t)
    (hb : p.hasBias = Bool.true → ∀ j : Fin p.blockColumns,
      KernelValues.col p s j % p.columns < p.columns → KernelValues.col p s j % p.columns < bounds p.bias) :
    Safety.traceSafeList (engine M D) bounds (seeds mode p) t := by
  cases mode <;> cases bias : p.hasBias
  all_goals
    simp [seeds,source,typed,dtype,ComputeKernel.surfaceBody,
      Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,
      Safety.safeOp_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,ComputeDType.eraseDType,
      numeric,bop,natLt,numericLt,Region.cast,ofFloat,toFloat,State.setReg,
      h.mem,h.pids,h.undef,h.rawm,h.rawn,h.offm,h.offn,h.offk,bias,
      engine,Algebra.withDefaultPrecision,resolvePrecision]
  all_goals exact hb bias

theorem body_safe (M : Algebra α) (D : Dot α) (mode : Mode) (p : Parameters)
    (s t : State α) (bounds : RegionName → Nat) (cur : Nat) (a : Accumulator α p)
    (h : Ready p s t cur a) (inside : cur < ceilDiv p.inner p.blockInner)
    (backed : AccessBounds p s bounds) :
    Safety.traceSafeList (engine M D) bounds (body mode p)
      (t.setReg "__rev_k" .int [] (fun _ => mirror p cur)) := by
  cases mode <;> by_cases mask : p.inner % p.blockInner = 0
  all_goals
    simp [body,statement,source,typed,dtype,ComputeKernel.surfaceBody,
      Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,
      Safety.safeOp_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,ComputeDType.eraseDType,
      numeric,bop,natLt,numericLt,integralFloorDiv,integralMod,
      Region.cast,ofFloat,toFloat,State.setReg,h.mem,h.pids,h.undef,
      h.rawm,h.rawn,h.offm,h.offn,h.offk,h.aptr,h.bptr,h.acc,
      engine,Algebra.withDefaultPrecision,resolvePrecision,mask,mirror,aPointer,bPointer]
  all_goals constructor
  all_goals first
    | (intro i j u; exact backed.input cur inside (i,j,u) (Or.inl mask))
    | (intro i j u; exact backed.weight cur inside (i,j,u) (Or.inl mask))
    | (intro i j u hij; exact backed.input cur inside (i,j,u) (Or.inr hij))
    | (intro i j u hij; exact backed.weight cur inside (i,j,u) (Or.inr hij))

theorem loop_safe (M : Algebra α) (D : Dot α) (mode : Mode) (p : Parameters)
    (s t : State α) (bounds : RegionName → Nat) (positive : 0 < p.blockInner)
    (cur : Nat) (bound : cur ≤ ceilDiv p.inner p.blockInner)
    (h : Ready p s t cur (foldN (update M D mode p s) cur (initial M mode p s)))
    (backed : AccessBounds p s bounds) :
    Safety.rangeIntSafe (engine M D) bounds "__rev_k" (mirror p cur) 0 p.blockInner
      (body mode p) t := by
  by_cases inside : cur < ceilDiv p.inner p.blockInner
  · have upper := ceil_upper p.inner p.blockInner positive
    have product := Nat.mul_le_mul_right p.blockInner (Nat.succ_le_of_lt inside)
    have active : mirror p cur < 0 := by simp only [mirror]; push_cast; nlinarith
    obtain ⟨v,he,hv⟩ := body_run M D mode p s t cur _ h
    have next : mirror p cur + (p.blockInner : Int) = mirror p (cur+1) := by
      simp [mirror,Nat.add_mul]; ring
    have nextReady : Ready p s v (cur+1)
        (foldN (update M D mode p s) (cur+1) (initial M mode p s)) := by
      simpa only [foldN] using hv
    rw [Safety.rangeIntSafe_unfold]
    simp only [Nat.ne_of_gt positive,if_false,active,if_true,he,next]
    exact ⟨body_safe M D mode p s t bounds cur _ h inside backed,
      loop_safe M D mode p s v bounds positive (cur+1) (by omega) nextReady backed⟩
  · have last : cur = ceilDiv p.inner p.blockInner := by omega
    have cover := ceil_cover p.inner p.blockInner positive
    have inactive : ¬mirror p cur < 0 := by simp only [mirror,last]; push_cast; omega
    rw [Safety.rangeIntSafe_unfold]
    simp [Nat.ne_of_gt positive,inactive]
termination_by ceilDiv p.inner p.blockInner - cur

theorem statement_safe (M : Algebra α) (D : Dot α) (mode : Mode) (p : Parameters)
    (t : State α) (bounds : RegionName → Nat)
    (h : Safety.rangeIntSafe (engine M D) bounds "__rev_k" (mirror p 0) 0 p.blockInner
      (body mode p) t) : Safety.stepSafe (engine M D) bounds (statement mode p) t := by
  simpa [statement,body,source,typed,ComputeKernel.surfaceBody,
    Safety.stepSafe_unfold,Safety.safeOp_unfold,evalOp_unfold,mirror,
    numeric,bop,integralFloorDiv,integralMod,Broadcast.leftIndex,Broadcast.rightIndex] using h

theorem save_safe (M : Algebra α) (D : Dot α) (mode : Mode) (p : Parameters)
    (s t : State α) (bounds : RegionName → Nat) (acc : Accumulator α p)
    (h : TailReady p s t acc)
    (backed : p.saveInputs = Bool.true → ∀ i, active p s i = Bool.true → savedOffset p s i < bounds p.saved) :
    Safety.traceSafeList (engine M D) bounds (saveCode mode p) t := by
  cases mode <;> cases saved : p.saveInputs
  all_goals
    simp [saveCode,tailCode,source,typed,dtype,ComputeKernel.surfaceBody,
      Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,
      Safety.safeOp_unfold,Safety.safeAddress_unfold,Safety.safeMask_unfold,
      Safety.addressSafe_unfold,Safety.maskActive_unfold,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,ComputeDType.eraseDType,
      numeric,bop,natLt,numericLt,Region.cast,ofFloat,toFloat,State.setReg,store,
      h.rawm,h.rawn,h.acc,saved,engine,Algebra.withDefaultPrecision,resolvePrecision]
  all_goals
    intro i j hi hj
    exact backed saved (i,j,PUnit.unit) (by simp [active,hi,hj])

theorem activation_safe (M : Algebra α) (D : Dot α) (p : Parameters)
    (s t : State α) (bounds : RegionName → Nat) (acc : Accumulator α p)
    (h : TailReady p s t acc) :
    Safety.traceSafeList (engine M D) bounds (activationCode p) t := by
  by_cases ht : p.activation.getD "" = "tanh" <;>
    by_cases hg : p.activation.getD "" = "gelu" <;>
    by_cases hf : p.activation.getD "" = "fast_gelu" <;>
    by_cases hr : p.activation.getD "" = "relu"
  all_goals
    simp_all [activationCode,tailCode,source,typed,dtype,ComputeKernel.surfaceBody,
      Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,
      Safety.safeOp_unfold,run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
      ComputeDType.eraseDType,numeric,bop,State.setReg,h.rawm,h.rawn,h.acc,ofFloat,toFloat,engine,Algebra.withDefaultPrecision,resolvePrecision]

theorem store_safe (M : Algebra α) (D : Dot α) (mode : Mode) (p : Parameters)
    (s t : State α) (bounds : RegionName → Nat) (acc : Accumulator α p)
    (h : TailReady p s t acc)
    (backed : ∀ i, active p s i = Bool.true → outputOffset p s i < bounds p.output) :
    Safety.traceSafeList (engine M D) bounds (storeCode mode p) t := by
  cases mode
  all_goals
    simp [storeCode,tailCode,source,typed,dtype,ComputeKernel.surfaceBody,
      Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,
      Safety.safeOp_unfold,Safety.safeAddress_unfold,Safety.safeMask_unfold,
      Safety.addressSafe_unfold,Safety.maskActive_unfold,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,ComputeDType.eraseDType,
      numeric,bop,natLt,numericLt,Region.cast,ofFloat,toFloat,State.setReg,store,
      h.rawm,h.rawn,h.acc,engine,Algebra.withDefaultPrecision,resolvePrecision]
  all_goals
    intro i j hi hj
    exact backed (i,j,PUnit.unit) (by simp [active,hi,hj])

/-- Actual complete source safety: the descending K range and both optional
branches are followed in the same intermediate states as successful execution. -/
theorem source_safe (M : Algebra α) (D : Dot α) (mode : Mode) (p : Parameters)
    (s : State α) (bounds : RegionName → Nat) (positive : 0 < p.blockInner)
    (backed : AccessBounds p s bounds) : TraceSafe (engine M D) bounds (source mode p) s := by
  obtain ⟨a,ha,oa⟩ := scalars_run (engine M D) p s
  obtain ⟨b,hb,rb⟩ := seeds_run M D mode p s a oa
  obtain ⟨c,hc,rc⟩ := loop_run M D mode p s b positive 0 (Nat.zero_le _) rb
  have tr : TailReady p s c (accumulated M D mode p s) := ⟨rc.rawm,rc.rawn,rc.acc⟩
  obtain ⟨d,hd,rd,md⟩ := save_run M D mode p s c _ tr rc.mem
  obtain ⟨e,he,re,me⟩ := activation_run M D p s d _ rd
  unfold TraceSafe
  rw [source_split,front_split,tail_split]
  simp only [List.append_assoc, List.singleton_append]
  apply (Safety.traceSafeList_append _ _ _ _ _).mpr
  constructor
  · exact scalars_safe _ _ _ _
  · intro t ht
    have : t = a := Option.some.inj (ht.symm.trans ha)
    subst t
    apply (Safety.traceSafeList_append _ _ _ _ _).mpr
    constructor
    · exact seeds_safe M D mode p s a bounds oa backed.bias
    · intro t ht
      have : t = b := Option.some.inj (ht.symm.trans hb)
      subst t
      apply Safety.traceSafeList_cons_of_step
        (statement_safe M D mode p b bounds
          (loop_safe M D mode p s b bounds positive 0 (Nat.zero_le _) rb backed))
        ((statement_eval M D mode p b).trans hc)
      apply (Safety.traceSafeList_append _ _ _ _ _).mpr
      constructor
      · exact save_safe M D mode p s c bounds _ tr backed.saved
      · intro t ht
        have : t = d := Option.some.inj (ht.symm.trans hd)
        subst t
        apply (Safety.traceSafeList_append _ _ _ _ _).mpr
        constructor
        · exact activation_safe M D p s d bounds _ rd
        · intro t ht
          have : t = e := Option.some.inj (ht.symm.trans he)
          subst t
          exact store_safe M D mode p s e bounds _ re backed.output

#axiomsClean source_safe
end
end VeriTile.Bench.Repairs.TritonLinearActivation.SourceSafety
