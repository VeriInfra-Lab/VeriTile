import bench.repairs.tritonbench_g.quantize_copy_kv.Proofs.RawKernel
import VeriTile.Triton.Float.StructuralSafety

/-! Actual-source safety for both integer-address spellings. The destination
metadata is read from the current execution state; FP32 arithmetic, FP16 scale
rounding and int8 conversion introduce no access-safety assumption. -/
namespace VeriTile.Bench.Repairs.QuantizeCopyKv.KernelExecution
open VeriTile.Triton FP.Structural KernelSource
noncomputable section
set_option maxHeartbeats 14000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

theorem source_traceSafe {A : Type} [Inhabited A] (M : Algebra A) (convert : A → Int)
    (equal : A → A → Bool) (p : Parameters) (version : Bool) (s : State A)
    (bounds : RegionName → Nat) (dest : Nat) (width : 0 < p.width)
    (metadata : (s.mem "destination" (s.pids 0)).read .int = (dest : Int))
    (metadataBound : s.pids 0 < bounds "destination")
    (input : ∀ h : Fin p.height, h.val < p.heads → ∀ c : Fin p.width,
      KernelResult.inputOffset p (s.pids 0) h.val c.val < bounds "input")
    (codes : ∀ h : Fin p.height, h.val < p.heads → ∀ c : Fin p.width,
      KernelResult.codeOffset p dest h.val c.val < bounds "output")
    (scales : ∀ h : Fin p.height, h.val < p.heads →
      KernelResult.scaleOffset p dest h.val < bounds "scale") :
    FP.Structural.TraceSafe (backend M convert equal) bounds (kernel version p) s := by
  cases version
  all_goals simp (maxSteps := 2200000) [FP.Structural.TraceSafe, kernel, ComputeKernel.surfaceBody,
    Safety.traceSafeList, Safety.stepSafe, Safety.safeExpr, Safety.safeCompute,
    Safety.safeOp.eq_def, Safety.safeAddress, Safety.safeMask, Safety.addressSafe,
    Safety.maskActive, run, step, FP.Structural.evalExpr, evalComputeOp, evalOp_unfold,
    store, Option.bind, ComputeDType.eraseDType, numeric, bop, natLt, numericLt,
    numericEq, Region.cast, State.setReg, backend, TileShape.axisDim, width, metadata,
    ofFloat, toFloat, Broadcast.leftIndex, Broadcast.rightIndex,
    Int.ofNat_lt, Int.toNat_mul, Int.toNat_add]
  all_goals simp only [← Int.natCast_mul, ← Int.natCast_add, Int.toNat_natCast]
  all_goals refine ⟨metadataBound, ?_, ?_, ?_⟩
  all_goals first
    | intro h c active
      simpa [KernelResult.inputOffset, Nat.mul_assoc] using input h active c
    | intro h c active
      simpa [KernelResult.codeOffset, Nat.mul_comm, Nat.mul_left_comm, Nat.mul_assoc] using codes h active c
    | simpa [KernelResult.scaleOffset] using scales

#axiomsClean source_traceSafe
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.QuantizeCopyKv.KernelExecution
