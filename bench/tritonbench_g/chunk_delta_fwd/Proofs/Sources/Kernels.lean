import bench.tritonbench_g.chunk_delta_fwd.Kernels
import VeriTile.Triton.Source.CountedLoop
import VeriTile.Triton.Float.CountedLoop
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.Repairs.ChunkDeltaFwd
open VeriTile.Triton
def outer (p : Parameters) : Source.DynamicRange :=
  (Source.countedLoop? (sourceKernel p).surfaceBody "i_t").getD ⟨[],.constNat 0,.constNat 0,.constNat 1,[],[]⟩
def inner (p : Parameters) : Source.DynamicRange :=
  (Source.countedLoop? (outer p).body "i_c").getD ⟨[],.constNat 0,.constNat 0,.constNat 1,[],[]⟩

theorem outer_parts (p : Parameters) :
    (sourceKernel p).surfaceBody=(outer p).before++
      [.forLoop "i_t" p.NT (outer p).body]++(outer p).after := rfl
theorem inner_parts (p : Parameters) :
    (outer p).body=(inner p).before++
      [.forLoop "i_c" ((p.BT+p.BC-1)/p.BC) (inner p).body]++(inner p).after := rfl
#axiomsClean outer_parts
#axiomsClean inner_parts
end VeriTile.Bench.Repairs.ChunkDeltaFwd
