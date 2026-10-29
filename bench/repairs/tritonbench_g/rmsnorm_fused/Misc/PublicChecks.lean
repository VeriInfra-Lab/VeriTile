import bench.repairs.tritonbench_g.rmsnorm_fused.Correct
import bench.repairs.tritonbench_g.rmsnorm_fused.Misc.TensorBridge

namespace VeriTile.Bench.Repairs.RmsnormFused.Audit
open VeriTile Triton Binding
noncomputable section

/-- The admitted mathematical formula has a strictly positive radicand.
This deliberately imposes no analogous condition on native FP words. -/
theorem positive_radicand (x : Tensor) (memory : Nat → ℝ) (N row : Nat)
    (epsilon : ℝ) (positive : 0<epsilon) :
    0<(∑ j ∈ Finset.range N,(x.read memory (row*N+j))^2)/(N:ℝ)+epsilon := by
  have squares : 0≤∑ j ∈ Finset.range N,(x.read memory (row*N+j))^2 :=
    Finset.sum_nonneg (fun j _ => sq_nonneg _)
  exact add_pos_of_nonneg_of_pos (div_nonneg squares (Nat.cast_nonneg N)) positive

theorem original_signed_overflow :
    ((BitVec.ofNat 32 (2^17))*(BitVec.ofNat 32 (2^14))).toInt=(-2147483648 : Int) := by decide

theorem repaired_widening :
    ((Index.rowWord (2^17))*(BitVec.ofNat 64 (2^14))).toInt=(2147483648 : Int) := by decide

#axiomsClean Correct.kernel_real
#axiomsClean TensorBridge.kernel_correct
#axiomsClean FPEquiv.kernel_native
#axiomsClean TensorBridge.kernel_native_equivalence
#axiomsClean Runtime.complete
#axiomsClean Runtime.source_complete
#axiomsClean Bytes.run_bytes
#axiomsClean Ordered.source_lowering
#axiomsClean Ordered.singleton_fma
#specNonCircular Ideal.result avoiding [source,Runtime.value,Runtime.run,FP.Structural.exec]
#specNonCircular Bytes.nativeResult avoiding [source,Ordered.lower,Ordered.interpret,Runtime.value,Runtime.run,FP.Structural.exec]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RmsnormFused.Audit
