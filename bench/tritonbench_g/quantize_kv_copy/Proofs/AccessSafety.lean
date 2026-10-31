import bench.tritonbench_g.quantize_kv_copy.Proofs.KernelGridResult
import VeriTile.Triton.Float.StructuralSafety

namespace VeriTile.Bench.Repairs.QuantizeKvCopy.AccessSafety
open VeriTile.Triton FP.Structural Kernel
noncomputable section
set_option autoImplicit true
set_option maxHeartbeats 18000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false

/-- Actual source access safety: the unmasked signed destination load has its
own bound; only active groups access input, INT8 code and FP32 scale storage. -/
theorem source_safe [Inhabited α] (op : Ops α) (p : Parameters)
    (bounds : RegionName → Nat) (s : State α) (dest : Nat)
    (hd : (s.mem "D" (s.pids 0)).read .int = (dest : Int))
    (hD : s.pids 0 < bounds "D")
    (hK : ∀ g c, g < p.groups → c < 8 →
      inputOffset p (s.pids 0*p.heads+s.pids 1) g c < bounds "K")
    (hO : ∀ g c, g < p.groups → c < 8 →
      codeOffset p (dest*p.heads+s.pids 1) g c < bounds "O")
    (hS : ∀ g, g < p.groups → scaleOffset p (dest*p.heads+s.pids 1) g < bounds "S") :
    FP.Structural.TraceSafe (backend op) bounds (kernel p) s := by
  have hzero : (0.0 : ℝ) = 0 := by norm_num
  have hone : (1.0 : ℝ) = 1 := by norm_num
  have hlevels : (127.0 : ℝ) = 127 := by norm_num
  simp (maxSteps := 1600000) [FP.Structural.TraceSafe, kernel, KernelSyntax.source,
    ComputeKernel.surfaceBody, Safety.traceSafeList_unfold, Safety.stepSafe_unfold,
    Safety.safeExpr_unfold, Safety.safeCompute_unfold, Safety.safeOp_unfold,
    Safety.safeAddress_unfold, Safety.safeMask_unfold, Safety.addressSafe_unfold,
    Safety.maskActive_unfold, step, evalExpr, evalComputeOp, evalOp_unfold,
    store, Option.bind, numeric, bop, natLt, numericLt, numericEq,
    ofFloat, toFloat, Region.cast, State.setReg, backend, ComputeDType.eraseDType,
    hd, hzero, hone, hlevels, reduction, TileShape.axisDim,
    index_toNat, index8_toNat, -Nat.cast_mul, -Nat.cast_add,
    ← Int.natCast_mul, ← Int.natCast_add, Int.ofNat_lt, Int.toNat_natCast]
  refine ⟨hD, ?_, ?_, ?_⟩
  · intro g c hg
    simp only [Value] at *
    norm_cast
    simpa only [inputOffset, Nat.add_mul, Nat.mul_assoc, Nat.add_assoc] using hK g.val c.val hg c.isLt
  · intro g c hg
    simp only [Value] at *
    norm_cast
    simpa only [codeOffset, Nat.add_mul, Nat.mul_assoc, Nat.add_assoc] using hO g.val c.val hg c.isLt
  · intro g hg
    simpa only [scaleOffset, Nat.add_mul, Nat.mul_assoc, Nat.add_assoc] using hS g.val hg

#axiomsClean source_safe
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.QuantizeKvCopy.AccessSafety
