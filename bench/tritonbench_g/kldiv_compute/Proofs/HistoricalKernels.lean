import bench.tritonbench_g.kldiv_compute.Proofs.Sources.Mathematical_Kernels
import bench.tritonbench_g.kldiv_compute.Proofs.Support.Parameters
import VeriTile.Meta.StatementAudit

/-! The retained mathematical view selects the canonical raw source's FP32
profile. Host packing and copyback are explicit in Proofs.Host. Grid-product
compatibility remains outside the ordinary launch contract. -/
namespace VeriTile.Bench.Repairs.KldivCompute
open VeriTile.Triton
abbrev source (p : Parameters) :=
  VeriTile.Bench.TritonBenchG.KldivCompute.TypedSource.kernel
    .fp32 .fp32 .fp32 Bool.false p.x p.y p.output p.n p.block
end VeriTile.Bench.Repairs.KldivCompute

namespace VeriTile.Bench.Repairs.KldivCompute.KernelDSL
/-- The retained mathematical view is the canonical raw program. -/
theorem source_eq (p : Parameters) : kernel p = KldivCompute.source p := rfl
#axiomsClean source_eq
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.KldivCompute.KernelDSL
