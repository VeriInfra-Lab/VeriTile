import bench.tritonbench_g.rotary_transform_ops.KernelSource
import bench.tritonbench_g.rotary_transform_ops.XorSource
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.RotaryTransformOps
/-- The canonical entry is the entire original constructor exported by the
compatibility module, with every runtime and configuration argument retained. -/
theorem original_source_entry : Kernels.original = KernelSource.source := rfl
/-- Moving the source entry does not change the complete candidate constructor. -/
theorem xor_source_entry : Kernels.xorIndexed = XorSource.source := rfl
#axiomsClean original_source_entry
#axiomsClean xor_source_entry
end VeriTile.Bench.Repairs.RotaryTransformOps
