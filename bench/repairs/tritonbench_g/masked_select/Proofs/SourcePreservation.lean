import bench.repairs.tritonbench_g.masked_select.Kernels
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.MaskedSelect.SourcePreservation
open VeriTile.Triton

/-- Complete kernel identity for every existing invocation. Natural prefix
values keep the same mathematical carrier; signed-int64 storage bounds remain
part of the unchanged prefix/IO refinement. No FP operation or state premise
is introduced by moving metadata out of the source body. -/
theorem complete_kernel (p : VeriTile.Bench.TritonBenchG.MaskedSelect.TypedCopy.Parameters) :
    kernel p = VeriTile.Bench.TritonBenchG.MaskedSelect.TypedCopy.source .fp32 p := rfl

#axiomsClean complete_kernel
end VeriTile.Bench.Repairs.MaskedSelect.SourcePreservation
