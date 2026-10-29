import bench.repairs.tritonbench_g.triton_mul2.Contract
import VeriTile.Torch.FP32

/-! The input contract has storage meaning through the shared Tensor API
binding. These lemmas describe kernel inputs and do not execute a wrapper. -/
namespace VeriTile.Bench.Repairs.TritonMul2.MemoryContract
open VeriTile.Triton VeriTile.Torch.FP32

theorem direct_read (read : Nat → Word) (x : Tensor) (laws : TensorLaws x)
    (B : Nat) (h : InputContract x.metadata B) (i : Nat) (hi : i < x.metadata.numel) :
    read ((dataView x laws h.contiguous).address i) = logicalValue read x laws i :=
  direct_read_eq_logical read x laws h.contiguous h.noNeg i hi

theorem actual_input_pointer (x : Tensor) (laws : TensorLaws x)
    (B : Nat) (h : InputContract x.metadata B) (nonempty : 0 < x.metadata.numel) :
    BitVec.ofNat 64 ((dataView x laws h.contiguous).address 0) = x.dataPtr :=
  actual_pointer x laws h.contiguous nonempty

/-- Every active FP32 access is naturally aligned, remains within the backing
allocation, and uses the logical element's physical address, including offset
views. The shared API binding documents the pointer/contiguity assumptions. -/
theorem active_bytes (x : Tensor) (laws : TensorLaws x)
    (B : Nat) (h : InputContract x.metadata B) (i byte : Nat)
    (hi : i < x.metadata.numel) (hb : byte < 4) :
    let address := (dataView x laws h.contiguous).address i
    address % 4 = 0 ∧ address = x.storage.address (x.physicalIndex i) ∧
      x.storage.base ≤ address + byte ∧
      address + byte < x.storage.base + 4*x.storage.elements ∧ address + byte < 2^64 := by
  have aligned := dataView_aligned x laws h.contiguous h.alignment
  have inside := laws.inBounds i hi
  have span := x.storage.spanFits
  have nonempty : x.metadata.numel ≠ 0 := by omega
  have eq : (dataView x laws h.contiguous).address i = x.storage.address (x.physicalIndex i) := by
    rw [laws.contiguous h.contiguous i hi]
    simp only [dataView, DataView.address, DataView.base, if_neg nonempty, Storage.address]
    omega
  dsimp only
  have alignment : (dataView x laws h.contiguous).address i % 4 = 0 := by
    simp only [DataView.address]
    omega
  refine ⟨alignment, eq, ?_, ?_, ?_⟩ <;> rw [eq] <;>
    simp only [Storage.address] at * <;> omega

#axiomsClean direct_read
#axiomsClean actual_input_pointer
#axiomsClean active_bytes
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.TritonMul2.MemoryContract
