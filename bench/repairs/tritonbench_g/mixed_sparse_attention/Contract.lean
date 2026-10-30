import bench.repairs.tritonbench_g.mixed_sparse_attention.Proofs.KernelContract
namespace VeriTile.Bench.Repairs.MixedSparseAttention
abbrev kernel_admission := KernelContract.Admission
abbrev kernel_empty_admission := KernelContract.EmptyAdmission
abbrev kernel_sparse_check := KernelContract.checkSparse
abbrev kernel_sparse_checked := KernelContract.sparse_iff
#axiomsClean kernel_sparse_checked
end VeriTile.Bench.Repairs.MixedSparseAttention
