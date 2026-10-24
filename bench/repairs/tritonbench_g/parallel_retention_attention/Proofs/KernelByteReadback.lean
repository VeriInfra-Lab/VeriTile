import bench.repairs.tritonbench_g.parallel_retention_attention.Proofs.KernelAddressBounds
import bench.repairs.tritonbench_g.parallel_retention_attention.Proofs.NativeBytes

namespace VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel
open NativeBytes
noncomputable section

theorem scratch_byte (s : Shape) (kind : OutputKind) (base device : Nat)
    (tensor : ScratchTensor s kind base device) (after : Bytes) (offset : Nat) (word : FP32Word)
    (read : NativeBytes.load after (base+offset)=word) (j : Fin 4) :
    after (tensor.metadata.dataPtr.toNat+4*offset+j.val)=encode word j := by
  rw [tensor.pointer]
  have bytes := original_bytes after (base+offset) j
  rw [read] at bytes
  simpa only [Nat.mul_add,Nat.add_assoc] using bytes.symm

#axiomsClean scratch_byte
end
end VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel
