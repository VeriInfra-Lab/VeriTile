import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.KernelBoundary
import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.NativeIndex

namespace VeriTile.Bench.Repairs.LayerNormLiger.KernelContract
open VeriTile.Triton FP.Structural
open VeriTile.Torch
noncomputable section

 theorem valid_capacity (h : Valid b shape) : b.size ≤ 2^62 := by
  by_cases hn : 0 < b.size
  · have cap := Torch.FP32.contiguous_capacity b.tensor h.tensor h.contiguous hn
    have span := b.tensor.storage.spanFits
    change b.tensor.storage.base + 4*b.tensor.storage.elements ≤ 2^64 at span
    change b.tensor.storageOffset + b.size ≤ b.tensor.storage.elements at cap
    omega
  · omega

 theorem valid_access (h : Valid b shape) (i : Nat) (hi : i < b.size) :
    b.pointer+4*i < 2^64 ∧ (b.pointer+4*i)%4 = 0 := by
  have range := h.tensor.inBounds i hi
  have span := b.tensor.storage.spanFits
  have equal := valid_address h i hi
  have aligned := h.aligned (Nat.zero_lt_of_lt hi)
  simp only [Torch.FP32.Storage.address] at equal
  constructor
  · omega
  · omega

 theorem forward_native (c : ForwardCall) (h : ForwardDomain c pointers) :
    NativeIndex.Safe c.rows c.columns 1 c.block := by
  apply NativeIndex.safe _ _ _ _ (Nat.zero_lt_of_lt h.pid_bound) h.columns_pos (by decide) h.block_bound
  have cap := valid_capacity h.x
  rw [valid_size h.x] at cap
  simp only [List.prod_cons,List.prod_nil,Nat.mul_one] at cap
  exact ⟨cap,by nlinarith [h.pid_bound]⟩

 theorem backward_native (c : BackwardCall) (h : BackwardDomain c pointers) :
    NativeIndex.Safe c.rows c.columns c.programs c.block := by
  apply NativeIndex.safe _ _ _ _ h.rows_pos h.columns_pos h.programs_pos h.block_bound
  have input := valid_capacity h.x
  have partialCapacity := valid_capacity h.dw
  rw [valid_size h.x] at input
  rw [valid_size h.dw] at partialCapacity
  constructor
  · simpa only [List.prod_cons,List.prod_nil,Nat.mul_one] using input
  · simpa only [List.prod_cons,List.prod_nil,Nat.mul_one] using partialCapacity

 theorem matrix_bound (row col rows columns : Nat) (hr : row < rows) (hc : col < columns) :
    row*columns+col < rows*columns := by
  have step := Nat.mul_le_mul_right columns hr
  nlinarith

 theorem input_bound (h : Valid b [rows,columns]) (row col : Nat)
    (hr : row < rows) (hc : col < columns) : row*columns+col < b.size := by
  rw [valid_size h]
  simpa using matrix_bound row col rows columns hr hc

 theorem vector_bound (h : Valid b [columns]) (col : Nat) (hc : col < columns) : col < b.size := by
  simpa [valid_size h] using hc

end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.LayerNormLiger.KernelContract
