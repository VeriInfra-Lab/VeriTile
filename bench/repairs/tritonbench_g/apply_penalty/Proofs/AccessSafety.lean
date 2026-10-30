import bench.repairs.tritonbench_g.apply_penalty.IO

namespace VeriTile.Bench.Repairs.ApplyPenalty.AccessSafety
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

/-- Actual source access safety for both variants. All memory conditions are
the scalar slots and the masked gather/scatter windows; no arithmetic property
or execution result is hidden in the bounds. -/
theorem source (M : Algebra α) (lt : α→α→Bool)
    (hlt : M.compareLt (some .fp32) .real = some lt)
    (candidate : Bool) (stride B : Nat) (s : State α) (bounds : RegionName→Nat)
    (hp : s.pids 0<bounds "presence") (hf : s.pids 0<bounds "frequency")
    (hr : s.pids 0<bounds "repetition") (hs : s.pids 0<bounds "cumsum")
    (he : s.pids 0+1<bounds "cumsum")
    (hi : ∀ j : TileIndex [B],Rewrite.active s j → Rewrite.start s+j.1.val<bounds "ids")
    (hc : ∀ j : TileIndex [B],Rewrite.active s j → Rewrite.start s+j.1.val<bounds "counts")
    (hx : ∀ j : TileIndex [B],Rewrite.active s j → Rewrite.address stride s j<bounds "logits") :
    TraceSafe (engine M) bounds
      (if candidate then Rewrite.reciprocalSource stride B else Rewrite.originalSource stride B) s := by
  cases candidate <;>
    simp [Rewrite.originalSource,Rewrite.reciprocalSource,kernel,reciprocalKernel,
      TraceSafe,ComputeKernel.surfaceBody,Safety.traceSafeList_cons,Safety.stepSafe_unfold,
      Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,
      Safety.safeAddress_unfold,Safety.safeMask_unfold,Safety.addressSafe_unfold,
      Safety.maskActive_unfold,run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
      ComputeDType.eraseDType,numeric,numericLt,natLt,bop,Region.cast,State.setReg,
      engine,Algebra.withDefaultPrecision,resolvePrecision,hlt,hp,hf,hr,hs,he]
  all_goals
    simp only [←Int.natCast_mul,←Int.natCast_add,Int.toNat_natCast,Int.zero_sub,Int.toNat_neg_natCast,Nat.sub_zero,Int.ofNat_lt]
    have hi' (j : Fin B) := hi (j,PUnit.unit)
    have hc' (j : Fin B) := hc (j,PUnit.unit)
    have hx' (j : Fin B) := hx (j,PUnit.unit)
    simp only [Rewrite.active,Rewrite.start,Rewrite.stop,Rewrite.address] at hi' hc' hx'
    refine ⟨hi',hc',?_,?_⟩
    all_goals
      intro j hj
      have bound := hx' j hj
      simpa only [hj,↓reduceIte,←Int.natCast_mul,←Int.natCast_add,Int.toNat_natCast,Int.zero_sub,Int.toNat_neg_natCast,Nat.sub_zero] using bound

#axiomsClean source
end
end VeriTile.Bench.Repairs.ApplyPenalty.AccessSafety
