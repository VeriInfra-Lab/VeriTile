import bench.repairs.tritonbench_g.int8_matmul_kernel_modular_checked.Proofs.TileComplete
import VeriTile.Triton.Float.StructuralSafety

namespace VeriTile.Bench.Repairs.Int8MatmulKernelModularChecked.TileModel
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.Int8MatmulKernel.Whole001 (Byte)
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

private theorem safe_of_steps (M : Algebra Word32) (bounds : RegionName → Nat)
    (body : List ComputeStmt) (safe : ∀ st∈body,∀ s,Safety.stepSafe M bounds st s)
    (s : State Word32) : Safety.traceSafeList M bounds body s := by
  induction body generalizing s with
  | nil => exact Safety.traceSafeList_nil M bounds s
  | cons st tail ih =>
    apply Safety.traceSafeList_cons_intro (safe st (by simp) s)
    intro next _
    exact ih (fun st h => safe st (by simp [h])) next

private theorem prelude_safe (cast : Word32 → Byte) (M : Algebra Word32)
    (p : Parameters) (bounds : RegionName → Nat) (s : State Word32) :
    Safety.traceSafeList (backend cast M) bounds (prelude p) s := by
  apply safe_of_steps
  intro st member current
  simp only [prelude,source,compileAssertionProjection,quotedSource,kernel,
    ComputeKernel.surfaceBody,List.take,List.mem_cons,List.not_mem_nil,or_false] at member
  rcases member with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl
  all_goals simp [Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,
    Safety.safeOp_unfold,evalOp_unfold,Safety.traceSafeList_unfold]

private theorem inner_safe (cast : Word32 → Byte) (M : Algebra Word32)
    (p : Parameters) (initial current : State Word32) (bounds : RegionName → Nat)
    (f k : Nat) (hf : f<4) (hk : k<p.steps) (defined : InputDefined p initial)
    (h : InnerInvariant cast p initial f k current)
    (ha : ∀ i : AIndex p,i.2.1.val<p.inner-(f*p.steps+k)*p.blockInner →
      aOffset p initial f k i<bounds p.input)
    (hb : ∀ i : BIndex p,i.1.val<p.inner → bOffset p initial k i<bounds p.weight) :
    Safety.traceSafeList (backend cast M) bounds (innerBody p)
      (current.setReg "j" .nat [] (fun _ => k)) := by
  have collected : collectValues? (fun i => convert8 cast (aWord p initial f k i)) =
      some (aValue cast p initial f k) := by
    have hc (i : AIndex p) := convert8_defined cast _ (defined f hf k hk i)
    simp only [hc,collectValues?_some]
    rfl
  simp only [aWord,Parameters.steps] at collected
  rw [←List.take_append_drop 3 (innerBody p),Safety.traceSafeList_append]
  constructor
  · simp [innerBody,outerBody,source,compileAssertionProjection,quotedSource,kernel,
      ComputeKernel.surfaceBody,Safety.traceSafeList_unfold,Safety.stepSafe_unfold,
      Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,
      Safety.safeAddress_unfold,Safety.safeMask_unfold,Safety.addressSafe_unfold,
      Safety.maskActive_unfold,Safety.precisionContext,run,step,evalExpr,evalComputeOp,
      evalOp_unfold,Option.bind,numeric,bop,natLt,Region.cast,State.setReg,
      h.fieldReg,h.aptrs,h.bptrs,h.offsk,h.memory,backend,integralFloorDiv,Parameters.steps,
      FloatDType.toTileDType,FloatDType.numericDType,toFloat,ofFloat,
      Broadcast.leftIndex,Broadcast.rightIndex]
    simp only [BitVec.ofNat_eq_ofNat] at collected
    rw [collected]
    simp only [Option.bind_some,State.setReg]
    simp [evalOp_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,h.bptrs,h.offsk,
      Option.bind,natLt,bop,Broadcast.leftIndex,Broadcast.rightIndex]
    constructor
    · intro r q u active
      exact ha (r,q,u) active
    · intro q n u active
      exact hb (q,n,u) active
  · intro next _
    apply safe_of_steps
    intro st member current
    simp [innerBody,outerBody,source,compileAssertionProjection,quotedSource,kernel,
      ComputeKernel.surfaceBody] at member
    rcases member with rfl|rfl|rfl|rfl|rfl|rfl|rfl
    all_goals simp [Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,
      Safety.safeOp_unfold]

private theorem inner_range_safe (cast : Word32 → Byte) (M : Algebra Word32)
    (p : Parameters) (initial current : State Word32) (bounds : RegionName → Nat)
    (f k : Nat) (hf : f<4) (hk : k≤p.steps) (defined : InputDefined p initial)
    (h : InnerInvariant cast p initial f k current)
    (ha : ∀ k,k<p.steps → ∀ i : AIndex p,i.2.1.val<p.inner-(f*p.steps+k)*p.blockInner →
      aOffset p initial f k i<bounds p.input)
    (hb : ∀ k,k<p.steps → ∀ i : BIndex p,i.1.val<p.inner →
      bOffset p initial k i<bounds p.weight) :
    Safety.rangeSafe (backend cast M) bounds "j" k p.steps 1 (innerBody p) current := by
  have aux : ∀ n k (current : State Word32),p.steps-k=n → k≤p.steps →
      InnerInvariant cast p initial f k current →
      Safety.rangeSafe (backend cast M) bounds "j" k p.steps 1 (innerBody p) current := by
    intro n
    induction n using Nat.strong_induction_on with
    | h n ih =>
      intro k current hn hk h
      by_cases active : k<p.steps
      · rw [Safety.rangeSafe_unfold,if_neg (by decide),if_pos active]
        refine ⟨inner_safe cast M p initial current bounds f k hf active defined h (ha k active) (hb k active),?_⟩
        rw [(innerExecution cast M p initial current f k hf active defined h).property]
        exact ih (p.steps-(k+1)) (by omega) (k+1) _ rfl (by omega)
          (innerInvariant cast M p initial current f k hf active defined h)
      · simp [Safety.rangeSafe_unfold,active]
  exact aux _ k current rfl hk h

private theorem outer_step_safe (cast : Word32 → Byte) (M : Algebra Word32)
    (p : Parameters) (initial current : State Word32) (bounds : RegionName → Nat)
    (f : Nat) (hf : f<4) (defined : InputDefined p initial)
    (h : OuterInvariant cast p initial f current)
    (ha : ∀ k,k<p.steps → ∀ i : AIndex p,i.2.1.val<p.inner-(f*p.steps+k)*p.blockInner →
      aOffset p initial f k i<bounds p.input)
    (hb : ∀ k,k<p.steps → ∀ i : BIndex p,i.1.val<p.inner →
      bOffset p initial k i<bounds p.weight) :
    Safety.traceSafeList (backend cast M) bounds (outerBody p)
      (current.setReg "i" .nat [] (fun _ => f)) := by
  rw [outer_decomposition,Safety.traceSafeList_append]
  constructor
  · simp [outerPrelude,outerBody,source,compileAssertionProjection,quotedSource,kernel,
      ComputeKernel.surfaceBody,Safety.traceSafeList_unfold,Safety.stepSafe_unfold,
      Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold]
    split <;> trivial
  · intro next ran
    rw [(outerPreludeExecution cast M p initial current f h).property] at ran
    cases ran
    have hs := inner_range_safe cast M p initial
      (outerPreludeExecution cast M p initial current f h).val bounds f 0 hf (by omega)
      defined (outerPreludeInvariant cast M p initial current f h) ha hb
    apply Safety.traceSafeList_cons_intro
    · simpa [Safety.stepSafe_unfold,Safety.safeOp_unfold,
      evalOp_unfold,stepsExpr,Parameters.steps,numeric,bop,Broadcast.leftIndex,
        Broadcast.rightIndex,integralFloorDiv] using hs
    · intro next _; exact Safety.traceSafeList_nil _ _ _

private theorem outer_range_safe (cast : Word32 → Byte) (M : Algebra Word32)
    (p : Parameters) (initial current : State Word32) (bounds : RegionName → Nat)
    (f : Nat) (hf : f≤4) (defined : InputDefined p initial)
    (h : OuterInvariant cast p initial f current)
    (ha : ∀ f,f<4 → ∀ k,k<p.steps → ∀ i : AIndex p,
      i.2.1.val<p.inner-(f*p.steps+k)*p.blockInner → aOffset p initial f k i<bounds p.input)
    (hb : ∀ k,k<p.steps → ∀ i : BIndex p,i.1.val<p.inner →
      bOffset p initial k i<bounds p.weight) :
    Safety.rangeSafe (backend cast M) bounds "i" f 4 1 (outerBody p) current := by
  have aux : ∀ n f (current : State Word32),4-f=n → f≤4 →
      OuterInvariant cast p initial f current →
      Safety.rangeSafe (backend cast M) bounds "i" f 4 1 (outerBody p) current := by
    intro n
    induction n using Nat.strong_induction_on with
    | h n ih =>
      intro f current hn hf h
      by_cases active : f<4
      · rw [Safety.rangeSafe_unfold,if_neg (by decide),if_pos active]
        refine ⟨outer_step_safe cast M p initial current bounds f active defined h (ha f active) hb,?_⟩
        obtain ⟨next,ran,inv⟩ := outerStep cast M p initial current f active defined h
        rw [ran]
        exact ih (4-(f+1)) (by omega) (f+1) next rfl (by omega) inv
      · simp [Safety.rangeSafe_unfold,active]
  exact aux _ f current rfl hf h

private theorem epilogue_safe (cast : Word32 → Byte) (M : Algebra Word32)
    (p : Parameters) (initial current : State Word32) (bounds : RegionName → Nat)
    (h : OuterInvariant cast p initial 4 current)
    (hout : ∀ i,active p initial i → outputOffset p initial i<bounds p.output) :
    Safety.traceSafeList (backend cast M) bounds (epilogue p) current := by
  simp [epilogue,source,compileAssertionProjection,quotedSource,kernel,ComputeKernel.surfaceBody,
    Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
    Safety.safeCompute_unfold,Safety.safeOp_unfold,Safety.safeAddress_unfold,
    Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,numeric,bop,natLt,
    Region.cast,State.setReg,h.pidm,h.pidn,h.acc,Broadcast.leftIndex,Broadcast.rightIndex]
  refine ⟨?_,?_⟩
  · intro r c hm hn
    exact hout (r,c,PUnit.unit) ⟨hm,hn⟩
  · split <;> trivial

/-- Access safety of the actual nested source loops and store, through the
same predecessor states used by the retained execution proof. -/
theorem source_safe (cast : Word32 → Byte) (M : Algebra Word32)
    (p : Parameters) (initial : State Word32) (bounds : RegionName → Nat)
    (defined : InputDefined p initial)
    (ha : ∀ f,f<4 → ∀ k,k<p.steps → ∀ i : AIndex p,
      i.2.1.val<p.inner-(f*p.steps+k)*p.blockInner → aOffset p initial f k i<bounds p.input)
    (hb : ∀ k,k<p.steps → ∀ i : BIndex p,i.1.val<p.inner →
      bOffset p initial k i<bounds p.weight)
    (hout : ∀ i,active p initial i → outputOffset p initial i<bounds p.output) :
    TraceSafe (backend cast M) bounds (source p) initial := by
  unfold TraceSafe
  rw [source_decomposition,Safety.traceSafeList_append,Safety.traceSafeList_append]
  refine ⟨⟨prelude_safe cast M p bounds initial,?_⟩,?_⟩
  · intro current ran
    rw [(preludeExecution cast M p initial).property] at ran
    cases ran
    have hs := outer_range_safe cast M p initial (preludeExecution cast M p initial).val
      bounds 0 (by omega) defined (preludeInvariant cast M p initial) ha hb
    apply Safety.traceSafeList_cons_intro
    · simpa [Safety.stepSafe_unfold,Safety.safeOp_unfold,
        evalOp_unfold, Safety.loopSafe_eq_rangeSafe] using hs
    · intro next _; exact Safety.traceSafeList_nil _ _ _
  · intro current ran
    obtain ⟨mid,hm,hi⟩ := outerLoop cast M p initial (preludeExecution cast M p initial).val
      0 (by omega) defined (preludeInvariant cast M p initial)
    rw [run_append,(preludeExecution cast M p initial).property] at ran
    simp only [Option.bind_some,run,step,evalOp_unfold,loop_eq_range] at ran
    change (FP.Structural.range (backend cast M) "i" 0 4 1 (outerBody p)
      (preludeExecution cast M p initial).val).bind (fun x => some x) = some current at ran
    rw [hm] at ran
    simp only [Option.bind_some,run] at ran
    cases ran
    exact epilogue_safe cast M p initial current bounds hi hout

#axiomsClean source_safe
end
end VeriTile.Bench.Repairs.Int8MatmulKernelModularChecked.TileModel
