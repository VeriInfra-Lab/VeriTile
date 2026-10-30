import bench.repairs.tritonbench_g.masked_select.Kernels
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.MaskedSelect.ReadableSource
/-- Complete selected FP32 source function, including IO and every parameter. -/
theorem kernel_preserved : kernel=VeriTile.Bench.TritonBenchG.MaskedSelect.Whole.kernel := rfl
#axiomsClean kernel_preserved
end VeriTile.Bench.Repairs.MaskedSelect.ReadableSource
