import bench.repairs.tritonbench_g.int8_matmul_quantization.Proofs.Helper
import bench.repairs.tritonbench_g.int8_matmul_quantization.Proofs.Ingress

namespace VeriTile.Bench.Repairs.Int8MatmulQuantization
noncomputable section
abbrev matmul_int8 := @compiledWrapper
abbrev quantize_int8_perrow := @compiledQuantWrapper
abbrev matmul_quantize_int8 := @compiledComposed
abbrev quantize_int8 := @compiledHelper
end
end VeriTile.Bench.Repairs.Int8MatmulQuantization
