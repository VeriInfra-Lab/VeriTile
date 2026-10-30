import bench.tritonbench_g.matmul_leakyrelu.Proofs.SourceIO

namespace VeriTile.Bench.Repairs.MatmulLeakyrelu.KernelOnly.FPEquiv
open VeriTile.Triton FP.Structural
open SourcePair
open scoped VeriTile.Spec
noncomputable section
set_option maxRecDepth 40000
set_option maxHeartbeats 6000000

/-- Actual original separate dot/add versus existing live-accumulator dot
candidate. The local TF32/FP32 contraction rule is explicitly pending numerical
validation; strict FP32 slope commutativity reuses the shared software theorem.
The conditional conclusion includes both safe successful complete schedules,
all FP16-tagged output cells and both exact exterior frames. -/
specification source_pair_equiv (R : Rules) (p : KernelInterface.Parameters) (order : Schedule p.request)
    (separateA : p.C ≠ p.A) (separateB : p.C ≠ p.B) :
    original p order ≡[R] contracted p order := by
  apply Spec.FloatingPoint.ofNumerical
  · rfl
  · rfl
  exact io_equivalent R p order separateA separateB

#axiomsClean source_pair_equiv
#print_fp_assumptions source_pair_equiv
end
end VeriTile.Bench.Repairs.MatmulLeakyrelu.KernelOnly.FPEquiv
