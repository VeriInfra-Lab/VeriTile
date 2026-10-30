import bench.repairs.tritonbench_g.index_select_bwd.Proofs.Kernel
import VeriTile.Triton.Float.StructuralSafety

/-! Access safety of the selected FP32 slice source. The signed lane cast,
clamping expression and both load/store masks are checked in the actual
Structural execution states. No floating arithmetic law is needed. -/
namespace VeriTile.Bench.Repairs.IndexSelectBwd.KernelProof
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

theorem source_traceSafe {A : Type} [Inhabited A] (rest : Algebra A)
    (add : A → A → A) (p : Launch) (bounds : RegionName → Nat) (s : State A)
    (destination : ∀ i : Fin p.block, i.val < p.count →
      i.val * p.stride < bounds p.destination)
    (incoming : ∀ i : Fin p.block, i.val < p.count → i.val < bounds p.incoming) :
    FP.Structural.TraceSafe (Source.backend (arithmetic rest add)) bounds (kernel p) s := by
  simp (maxSteps := 1400000) [FP.Structural.TraceSafe, kernel, ComputeKernel.surfaceBody,
    Safety.traceSafeList, Safety.stepSafe, Safety.safeExpr, Safety.safeCompute,
    Safety.safeOp.eq_def, Safety.safeAddress, Safety.safeMask, Safety.addressSafe,
    Safety.maskActive, step, FP.Structural.evalExpr, evalComputeOp, evalOp_unfold,
    store, Option.bind, numeric, bop, natLt, numericLt, numericEq,
    Source.backend, arithmetic, ofFloat, toFloat, Region.cast, State.setReg,
    ComputeDType.eraseDType, Broadcast.leftIndex, Broadcast.rightIndex,
    Int.ofNat_lt, Int.toNat_mul]
  refine ⟨?_, ?_, ?_⟩
  · intro i active
    simpa [active, Int.toNat_mul] using destination i active
  · intro i active
    simpa [active] using incoming i active
  · intro i active
    simpa [active, Int.toNat_mul] using destination i active

#axiomsClean source_traceSafe
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.IndexSelectBwd.KernelProof
