import bench.tritonbench_g.kldiv_compute.Proofs.HostContract
import bench.tritonbench_g.kldiv_compute.Kernels
import bench.tritonbench_g.vector_addition.Proofs.Mathematical_FiniteIndices

namespace VeriTile.Bench.Repairs.KldivCompute.Contiguous
abbrev checkTensorInputs_iff := AddExample.Contiguous.checkTensorInputs_iff
abbrev checkInputs_iff := AddExample.Contiguous.checkInputs_iff
abbrev check_tensor_arguments_iff := AddExample.Contiguous.check_tensor_arguments_iff
abbrev same_numel := AddExample.Contiguous.same_numel
end VeriTile.Bench.Repairs.KldivCompute.Contiguous
