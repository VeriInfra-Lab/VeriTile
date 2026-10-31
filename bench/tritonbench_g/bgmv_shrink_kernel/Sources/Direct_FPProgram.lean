import bench.tritonbench_g.bgmv_shrink_kernel.Sources.Kernels
import VeriTile.Triton.Float.Structural

namespace VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.FPProgram
open VeriTile.Triton
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000

def guard : Op .bool [] := .ne .int .nil (.ref .int [] "lora_index") (.constInt (-1))

end VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.FPProgram
