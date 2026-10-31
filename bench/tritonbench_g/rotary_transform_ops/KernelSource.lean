import bench.tritonbench_g.rotary_transform_ops.Kernels

namespace VeriTile.Bench.Repairs.RotaryTransformOps.KernelSource

/-- The exact signed-value interpreter shares an Int carrier for the two
storage widths. The Tensor read/finite-index ABI must bind the actual signed
int32/int64 bytes; this equality is not a dtype-conversion permission. -/
theorem signed_carrier (cumulative64 offsets64 : Bool) :
    source cumulative64 offsets64 = source64_64 := by
  cases cumulative64 <;> cases offsets64 <;> rfl

end VeriTile.Bench.Repairs.RotaryTransformOps.KernelSource
