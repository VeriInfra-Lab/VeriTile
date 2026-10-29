import bench.repairs.tritonbench_g.quantize_kv_transform.Proofs.KernelInputs
import bench.repairs.tritonbench_g.quantize_kv_transform.Proofs.KernelResult
import bench.repairs.tritonbench_g.quantize_kv_transform.Proofs.Source

namespace VeriTile.Bench.Repairs.QuantizeKvTransform.KernelAddresses
open KernelInputs
noncomputable section
set_option linter.unusedSimpArgs false

/-- The exact native INT64 index/pointer arithmetic selects the same byte
addresses as the actual shared kernel's logical offsets. The existing checked
pointer/extent proof is reused; no new host or allocator execution is modeled. -/
theorem pointers (m : Inputs) (words : Nat → MetadataWord) (h : InputContract m words)
    (objects : Role → Nat) (heap : Heap) (binding : TensorBinding m words heap)
    (token head column : Nat) (ht : token < (request m words).tokens)
    (hh : head < (request m words).heads) (hc : column < (request m words).columns) :
    (Native.pointer (base m .input) (Native.inputIndex (request m words) token head column) 4).toNat =
        base m .input + 4 * KernelResult.inputOffset (parameters m words) token head column ∧
    (Native.pointer (base m .output)
      (Native.outputIndex (request m words) ((request m words).destination token) head column) 1).toNat =
        base m .output + KernelResult.codeOffset (parameters m words) ((request m words).destination token) head column ∧
    (∀ byte : Fin 4,
      (Native.pointer (base m .scale)
        (Native.scaleIndex (request m words) ((request m words).destination token) head) 4).toNat + byte.val =
        base m .scale + 4 * KernelResult.scaleOffset (parameters m words) ((request m words).destination token) head + byte.val) := by
  refine ⟨?_, ?_, ?_⟩
  · simpa [directLayout, Tensor.data, inputAddress, KernelResult.inputOffset,
      parameters, Nat.add_mul, Nat.mul_add, Nat.mul_assoc, Nat.add_assoc] using
      Native.input_pointer m words h objects heap binding token head column ht hh hc
  · simpa [directLayout, Tensor.data, outputAddress, KernelResult.codeOffset,
      parameters, Nat.add_mul, Nat.mul_add, Nat.mul_assoc, Nat.mul_comm, Nat.mul_left_comm,
      Nat.add_assoc] using Native.output_pointer m words h objects heap binding token head column ht hh hc
  · intro byte
    simpa [directLayout, Tensor.data, scaleAddress, KernelResult.scaleOffset,
      parameters, Nat.add_mul, Nat.mul_add, Nat.mul_assoc, Nat.add_assoc] using
      Native.scale_pointer m words h objects heap binding token head byte ht hh

theorem destination (m : Inputs) (words : Nat → MetadataWord) (h : InputContract m words)
    (objects : Role → Nat) (heap : Heap) (binding : TensorBinding m words heap)
    (token : Nat) (ht : token < (request m words).tokens) :
    Native.destination (request m words) (directLayout m objects) heap token =
      (request m words).destination token :=
  Native.destination_exact m words h objects heap binding token ht

#axiomsClean pointers
#axiomsClean destination
end
end VeriTile.Bench.Repairs.QuantizeKvTransform.KernelAddresses
