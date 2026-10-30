import bench.tritonbench_g.token_attn_llama2.Proofs.GridRun
import VeriTile.Triton.Float.StructuralSafety

namespace VeriTile.Bench.Repairs.TokenAttnLlama2GridAxesChecked.SourceSafety
open VeriTile Triton FP.Structural Safety
open VeriTile.Bench.Repairs.TokenAttnLlama2.FPValue
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

/-- Bounds for the reads and writes actually reached by the 0/1 source loop. -/
structure Bounds (p : Parameters) (s : State α) (bounds : RegionName → Nat) : Prop where
  lengths : s.pids 0 < bounds p.lengths.cast
  starts : s.pids 0 < bounds p.starts.cast
  query : blockActive p s → ∀ d, queryOffset p s d < bounds p.q
  locations : ∀ i, active p s i →
    p.locationBatchStride*s.pids 0+p.locationSequenceStride*windowIndex p s i <
      bounds p.locations.cast
  keys : ∀ i, active p s i → ∀ d, keyOffset p s i d < bounds p.k
  output : ∀ i, active p s i → outputOffset p s i < bounds p.output

theorem prelude_safe (M : Algebra α) (p : Parameters) (s : State α)
    (bounds : RegionName → Nat) (hb : Bounds p s bounds) :
    traceSafeList M bounds (prelude p) s := by
  simp [prelude,source,typed,ComputeKernel.surfaceBody]
  simp [traceSafeList_unfold,stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,
    safeOp_unfold,safeAddress_unfold,safeMask_unfold,addressSafe_unfold,maskActive_unfold,
    step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,Option.bind,numeric,bop,
    natLt,numericEq,ComputeDType.eraseDType,ofFloat,toFloat,
    FloatDType.numericDType,FloatDType.toTileDType,Region.cast,
    integralFloorDiv,integralMod]
  exact ⟨hb.lengths,hb.starts⟩

theorem body_safe (M : Algebra α) (p : Parameters) (s : State α)
    (bounds : RegionName → Nat) (hb : Bounds p s bounds) (activeBlock : blockActive p s) :
    traceSafeList M bounds (body p)
      ((preExecution M p s).val.setReg "start_mark" .nat [] (fun _ => 0)) := by
  simp only [body,source,typed,ComputeKernel.surfaceBody]
  simp [traceSafeList_unfold,stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,
    safeOp_unfold,safeAddress_unfold,safeMask_unfold,addressSafe_unfold,maskActive_unfold,
    step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,Option.bind,numeric,bop,
    natLt,numericEq,ComputeDType.eraseDType,ofFloat,toFloat,
    FloatDType.numericDType,FloatDType.toTileDType,Region.cast,
    integralFloorDiv,integralMod,preExecution,store]
  refine ⟨hb.query activeBlock,?_,?_,?_⟩
  · intro i hi
    exact hb.locations (i,PUnit.unit) hi
  · intro i d hi
    simpa [keyOffset,keyLocation,active,windowIndex,length,token,Region.cast,
      show active p s (i,PUnit.unit) from hi] using hb.keys (i,PUnit.unit) hi d
  · intro i hi
    exact hb.output (i,PUnit.unit) hi

theorem source_safe (M : Algebra α) (p : Parameters) (s : State α)
    (bounds : RegionName → Nat) (hb : Bounds p s bounds) :
    TraceSafe M bounds (source p) s := by
  unfold TraceSafe
  rw [source_body,traceSafeList_append]
  refine ⟨prelude_safe M p s bounds hb,?_⟩
  intro next ran
  have eqNext := Option.some.inj (ran.symm.trans (preExecution M p s).property)
  subst next
  apply traceSafeList_cons_intro
  · simp only [stepSafe_unfold,safeOp_unfold,evalOp_unfold,preExecution_mask]
    by_cases ha : blockActive p s
    · simp only [ha,ite_true,true_and]
      rw [rangeSafe_unfold]
      simp only [Nat.one_ne_zero,ite_false,Nat.zero_lt_one,ite_true,Nat.zero_add]
      rw [(bodyExecution M p s).property]
      dsimp only
      rw [rangeSafe_unfold]
      simpa using body_safe M p s bounds hb ha
    · simp only [ha,ite_false,true_and]
      rw [rangeSafe_unfold]
      simp
  · intro final _
    exact traceSafeList_nil _ _ _

#axiomsClean source_safe

end
end VeriTile.Bench.Repairs.TokenAttnLlama2GridAxesChecked.SourceSafety
