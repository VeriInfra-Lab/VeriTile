import bench.repairs.tritonbench_g.mixed_sparse_attention.Proofs.KernelGrid
import bench.repairs.tritonbench_g.mixed_sparse_attention.Proofs.KernelSafety
namespace VeriTile.Bench.Repairs.MixedSparseAttention
abbrev kernel_source := Whole002.Source.kernel
abbrev kernel_source_exact := Whole002.Source.kernel_exact
abbrev kernel_source_exec := @KernelGrid.source
abbrev kernel_source_safety := @KernelSafety.source_safe
#axiomsClean kernel_source_exact
#axiomsClean kernel_source_exec
#axiomsClean kernel_source_safety
end VeriTile.Bench.Repairs.MixedSparseAttention
