import bench.repairs.tritonbench_g.rms_rbe_matmul.Proofs.KernelGraph

namespace VeriTile.Bench.Repairs.RmsRbeMatmul.KernelSource
open VeriTile.Triton
set_option maxHeartbeats 12000000
set_option maxRecDepth 20000

theorem reshape_to_pairs : ∀ i : TileIndex [16,64],
    TileShape.reshapeIndex (input:=[16,32,2]) (output:=[16,64]) (by decide) i=
      (i.1,⟨i.2.1.val/2,by omega⟩,⟨i.2.1.val%2,by omega⟩,PUnit.unit) := by decide

theorem reshape_from_pairs : ∀ i : TileIndex [16,32], ∀ side : Fin 2,
    pairIndex i side=(i.1,⟨2*i.2.1.val+side.val,by omega⟩,PUnit.unit) := by decide

#axiomsClean reshape_to_pairs
#axiomsClean reshape_from_pairs
end VeriTile.Bench.Repairs.RmsRbeMatmul.KernelSource
