import bench.repairs.tritonbench_g.chunk_bwd_dqkg.Proofs.KernelContract

namespace VeriTile.Bench.Repairs.ChunkBwdDqkg
abbrev kernel_check := KernelContract.check
abbrev kernel_checked := @KernelContract.check_iff
abbrev kernel_scale := KernelContract.Scale
#axiomsClean kernel_checked
end VeriTile.Bench.Repairs.ChunkBwdDqkg
