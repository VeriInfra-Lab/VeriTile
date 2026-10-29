import bench.repairs.tritonbench_g.block_sparse_attn.KernelOnly.Tensor
import bench.repairs.tritonbench_g.block_sparse_attn.KernelOnly.Raw
import bench.repairs.tritonbench_g.block_sparse_attn.KernelOnly.InputSafety
import bench.repairs.tritonbench_g.block_sparse_attn.KernelOnly.Proofs.PhysicalApplications

/-! Optional Tensor-coordinate and byte readback applications of the complete
Real/native source results. These exports are ordinary supporting theorems. -/
namespace VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly.TensorBridge

export PhysicalKernel (whole_real_tensor whole_tensor whole_bytes)

export PhysicalKernel (whole_real whole_native whole_input_frame)
export PhysicalKernel (raw_whole_real raw_whole_native)
export PublicIO (physical_program_trace native_physical_inputs real_physical_inputs physical_in_bounds)

open PhysicalKernel
/-- Unchanged Tensor guards discharge only these raw address/CSR premises. -/
theorem raw_abi {α : Type} [Inhabited α] (p : Parameters) (batches : Nat)
    (access : Access α Store Byte) (buffers : Physical.Buffers) (initial : Store)
    (q k v out : Host.Tensor) (rows cols : Nat)
    (abi : ABI p batches access buffers initial q k v out rows cols) : RawABI p batches access buffers initial :=
  PublicIO.rawABI_of_abi p batches access buffers initial q k v out rows cols abi

#axiomsClean whole_real_tensor
#axiomsClean whole_tensor
#axiomsClean whole_bytes
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly.TensorBridge
