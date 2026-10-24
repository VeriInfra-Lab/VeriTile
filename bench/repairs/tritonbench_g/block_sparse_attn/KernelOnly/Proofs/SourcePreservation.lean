import bench.repairs.tritonbench_g.block_sparse_attn.KernelOnly.Proofs.LegacySurface
namespace VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly.SourcePreservation
/-- Whole interface and executed AST are unchanged by explicit runtime scale. -/
theorem original (p : Parameters) : source p=LegacySurface.source p := rfl
#axiomsClean original
end VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly.SourcePreservation
