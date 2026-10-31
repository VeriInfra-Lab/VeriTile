import bench.tritonbench_g.chunk_gated_attention.Proofs.HErasureDot
import bench.tritonbench_g.chunk_gated_attention.Proofs.HErasureGate
import bench.tritonbench_g.chunk_gated_attention.Proofs.HErasureHistory
import bench.tritonbench_g.chunk_gated_attention.Proofs.HErasureSeed
import bench.tritonbench_g.chunk_gated_attention.Proofs.HErasureFinal
namespace VeriTile.Bench.Repairs.ChunkGatedAttention.HSource
open VeriTile.Triton
open VeriTile.Bench.Repairs.ChunkGatedAttention
open VeriTile.Bench.TritonBenchG.ChunkGatedAttention
noncomputable section

def algorithm (p : Parameters) (r : RealHProgram.Regions) : Kernel :=
  (projection p r .real .real .real .real .real).val.eraseDType

theorem erased_algorithm (p : Parameters) (r : RealHProgram.Regions)
    (dot gate history seed final : FloatDType) :
    (projection p r dot gate history seed final).val.eraseDType=algorithm p r := by
  calc
    _ = (projection p r .real gate history seed final).val.eraseDType :=
      erase_dot p r dot gate history seed final
    _ = (projection p r .real .real history seed final).val.eraseDType :=
      erase_gate p r .real gate history seed final
    _ = (projection p r .real .real .real seed final).val.eraseDType :=
      erase_history p r .real .real history seed final
    _ = (projection p r .real .real .real .real final).val.eraseDType :=
      erase_seed p r .real .real .real seed final
    _ = algorithm p r := erase_final p r .real .real .real .real final

#axiomsClean erased_algorithm
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.HSource
