import VeriTile.Meta.StatementAudit
import bench.repairs.tritonbench_g.swiglu_backward.Kernels

namespace VeriTile.Bench.Repairs.SwigluBackward.SourcePreservation

/-- Equality of the entire parameterized original program, including its public
input/output signature and both recomputation branches. -/
theorem kernel_preserved :
    kernel = sourceKernel .fp32 .fp32 .fp32 .fp32 Bool.false := by rfl

/-- Equality of the entire existing candidate, with every other parameter
universally quantified by function equality. -/
theorem commutedKernel_preserved :
    commutedKernel = sourceKernel .fp32 .fp32 .fp32 .fp32 Bool.true := by rfl

#axiomsClean kernel_preserved
#axiomsClean commutedKernel_preserved
end VeriTile.Bench.Repairs.SwigluBackward.SourcePreservation
