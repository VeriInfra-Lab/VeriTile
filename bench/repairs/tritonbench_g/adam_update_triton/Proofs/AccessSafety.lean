import bench.repairs.tritonbench_g.adam_update_triton.Proofs.Tile
import VeriTile.Triton.Float.StructuralSafety

/-! Safety of the actual runtime-scalar source. All numerical backends share
the same natural-number addresses and active masks. Only active accesses need
backing; pointer formation in masked-off lanes remains unrestricted. -/
namespace VeriTile.Bench.Repairs.AdamUpdateTriton.Tile
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

theorem source_traceSafe {α : Type} [Inhabited α]
    (sem : Semantics α) (params : Scalars α) (total n B pid : Nat)
    (bounds : RegionName → Nat) (s : State α)
    (hp : ∀ j : Fin B, pid * B + j.val < n → pid * B + j.val < bounds "P")
    (hg : ∀ j : Fin B, pid * B + j.val < n → pid * B + j.val < bounds "G")
    (hm : ∀ j : Fin B, pid * B + j.val < n → pid * B + j.val < bounds "M") :
    FP.Structural.TraceSafe sem.backend bounds (source total n B)
      (programState params pid s) := by
  simp (maxSteps := 1400000) [FP.Structural.TraceSafe, source,
    sourceWithArgumentDeclarations, sourceBody, ComputeKernel.surfaceBody,
    Safety.traceSafeList_unfold, Safety.stepSafe_unfold, Safety.safeExpr_unfold,
    Safety.safeOp_unfold, Safety.safeCompute_unfold, Safety.safeAddress_unfold, Safety.safeMask_unfold,
    Safety.addressSafe_unfold, Safety.maskActive_unfold,
    step, FP.Structural.evalExpr, evalComputeOp, ComputeDType.eraseDType, evalOp_unfold,
    store, Option.bind, numeric, bop, natLt, numericLt, numericEq,
    programState, bindArguments, ofFloat, toFloat, Region.cast, State.setReg,
    Semantics.backend, engine, Algebra.withDefaultPrecision, resolvePrecision]
  exact ⟨hp, hg, hm, hp, hm⟩

#axiomsClean source_traceSafe
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AdamUpdateTriton.Tile
