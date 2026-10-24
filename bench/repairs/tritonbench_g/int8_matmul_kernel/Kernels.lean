import bench.tritonbench_g.int8_matmul_kernel.Whole001.Kernels

/-! The checked wrapper calls the complete accepted original program. Its
four field passes, packed weights, all fifteen configurations, FP32-to-int8
conversion and signed32 instruction behavior are retained without a rewrite. -/
namespace VeriTile.Bench.Repairs.Int8MatmulKernel
open VeriTile.Bench.TritonBenchG.Int8MatmulKernel.Whole001

abbrev source := @programCode
abbrev sourceGrid := @gridCode
abbrev mathematicalFormula := quantizedDenseTarget
abbrev finiteFormula := finiteTarget

end VeriTile.Bench.Repairs.Int8MatmulKernel
