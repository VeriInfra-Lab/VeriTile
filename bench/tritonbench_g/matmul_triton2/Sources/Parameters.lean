import bench.tritonbench_g.matmul_triton1.Sources.Parameters

namespace VeriTile.Bench.Repairs.MatmulTriton2.Revision001.Core
open VeriTile.Triton
abbrev Parameters := VeriTile.Bench.Repairs.MatmulTriton1.Revision001.Core.Parameters
def inner (p : Parameters) : Nat := p.innerSize
def steps (p : Parameters) : Nat := (inner p+p.blockInner-1)/p.blockInner

end VeriTile.Bench.Repairs.MatmulTriton2.Revision001.Core
