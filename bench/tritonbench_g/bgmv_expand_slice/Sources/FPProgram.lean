import bench.tritonbench_g.bgmv_expand_slice.Sources.Kernels
import VeriTile.Triton.Float.Structural

namespace VeriTile.Bench.Repairs.BgmvExpandSlice.FPProgram
open VeriTile.Triton
set_option maxHeartbeats 2000000
set_option maxRecDepth 12000

def guard : Op .bool [] := .ne .int .nil (.ref .int [] "lora_index") (.constInt (-1))

end VeriTile.Bench.Repairs.BgmvExpandSlice.FPProgram
