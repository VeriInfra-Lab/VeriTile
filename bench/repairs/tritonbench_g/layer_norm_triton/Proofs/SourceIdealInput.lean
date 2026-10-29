import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.KernelContract
import VeriTile.Triton.Float.Structural

namespace VeriTile.Bench.Repairs.LayerNormTriton.SourceIdealInput
noncomputable section

/-- Input-only ideal interpretation of host binary64 words. This ordinary
input abstraction contains no kernel, memory, output, or execution equation;
it does not assert equality between native FP32 arithmetic and Real arithmetic. -/
structure ScalarProjection where
  binary64 : UInt64 → ℝ

/-- The ideal scalar is projected from the exact epsilon stored in the actual
forward-call contract. Source execution and the independent target consume
this same value; the public call has no separate free Real epsilon. -/
def epsilon (projection : ScalarProjection) (c : KernelContract.Forward M N) : ℝ :=
  projection.binary64 c.epsilon.toBits

theorem epsilon_input (projection : ScalarProjection) (c : KernelContract.Forward M N) :
    epsilon projection c=projection.binary64 c.epsilon.toBits := rfl

#axiomsClean epsilon_input
#specNonCircular epsilon avoiding [VeriTile.Triton.FP.Structural.exec]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormTriton.SourceIdealInput
