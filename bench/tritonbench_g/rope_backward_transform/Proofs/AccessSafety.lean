import bench.tritonbench_g.rope_backward_transform.Proofs.KernelInterface
import VeriTile.Triton.Float.StructuralSafety

/-! Every reached access in the complete backward Q/K source. Coefficient
loads, both input halves and all four stores use their actual evaluated masks.
The proof is independent of floating values and intermediate memory contents. -/
namespace VeriTile.Bench.Repairs.RopeBackwardTransform.SourceSafety
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.IsfiniteKernel.TypedBits
noncomputable section
set_option maxHeartbeats 20000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

theorem source_safe {A : Type} [Inhabited A] (M : Algebra A)
    (r : Request) (s : State A) (bounds : RegionName → Nat)
    (coefficient : ∀ sine (j : Fin (r.paddedDim / 2)), j.val < r.half →
      s.pids 0 % r.sequence * r.rowStride sine + j.val * r.colStride sine <
        bounds (if sine then "S" else "C"))
    (input : ∀ key side (i : Lane r key), Source.active (r.heads key) r.half i →
      address r (s.pids 0) key side i < bounds (region key)) :
    FP.Structural.TraceSafe M bounds (Source.kernel .fp32 .fp32 (parameters r)) s := by
  simp (maxSteps := 3000000) [FP.Structural.TraceSafe, Source.kernel, Source.source,
    parameters, ComputeKernel.surfaceBody,
    Safety.traceSafeList, Safety.stepSafe, Safety.safeExpr, Safety.safeCompute,
    Safety.safeOp.eq_def, Safety.safeAddress, Safety.safeMask, Safety.addressSafe,
    Safety.maskActive, run, step, FP.Structural.evalExpr, evalComputeOp, evalOp_unfold,
    store, Option.bind, ComputeDType.eraseDType, numeric, bop, natLt, numericLt,
    numericEq, Region.cast, State.setReg, integralFloorDiv, integralMod,
    FloatDType.numericDType, FloatDType.toTileDType, ofFloat, toFloat,
    fold_writes_regs]
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro j hj
    exact coefficient Bool.false j hj
  · intro j hj
    exact coefficient Bool.true j hj
  all_goals intro h j hh hj
  all_goals first
    | simpa [address, Source.address, region, Request.heads, Source.active, Request.half, Nat.add_assoc]
        using input Bool.false Bool.false (h, j, PUnit.unit) (by simpa [Source.active, Request.heads, Request.half] using And.intro hh hj)
    | simpa [address, Source.address, region, Request.heads, Source.active, Request.half, Nat.add_assoc]
        using input Bool.true Bool.false (h, j, PUnit.unit) (by simpa [Source.active, Request.heads, Request.half] using And.intro hh hj)
    | simpa [address, Source.address, region, Request.heads, Source.active, Request.half, Nat.add_assoc]
        using input Bool.false Bool.true (h, j, PUnit.unit) (by simpa [Source.active, Request.heads, Request.half] using And.intro hh hj)
    | simpa [address, Source.address, region, Request.heads, Source.active, Request.half, Nat.add_assoc]
        using input Bool.true Bool.true (h, j, PUnit.unit) (by simpa [Source.active, Request.heads, Request.half] using And.intro hh hj)

#axiomsClean source_safe
end
end VeriTile.Bench.Repairs.RopeBackwardTransform.SourceSafety
