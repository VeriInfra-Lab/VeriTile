import bench.tritonbench_g.masked_select.Sources.Mathematical_Kernels

/-! The checked wrapper preserves the complete original kernel and all twenty
source autotune configurations. Payload storage is the selected FP32 profile. -/
namespace VeriTile.Bench.Repairs.MaskedSelect
open VeriTile.Triton

abbrev configs := VeriTile.Bench.TritonBenchG.MaskedSelect.Whole.configs
end VeriTile.Bench.Repairs.MaskedSelect
