import bench.repairs.tritonbench_g.triton_mul2.Contract
import bench.repairs.tritonbench_g.triton_mul2.Kernels
import bench.tritonbench_g.triton_mul2.Proofs.FiniteSource

namespace VeriTile.Bench.Repairs.TritonMul2

theorem tensor_contract_iff (x : TensorMetadata) (B : Nat) : InputContract x B ↔
    x.layout=.strided ∧ x.dtype=.floating .fp32 ∧ x.device.kind="cuda" ∧
    x.isContiguous=Bool.true ∧ x.isNeg=Bool.false ∧ x.numel<2^31 ∧
    (0<x.numel → x.dataPtr.toNat%4=0) ∧ B∈supportedBlocks := by
  constructor
  · intro h; exact ⟨h.layout,h.dtype,h.cuda,h.contiguous,h.noNeg,h.extent,h.alignment,h.block⟩
  · rintro ⟨a,b,c,d,e,f,g,h⟩; exact ⟨a,b,c,d,e,f,g,h⟩

theorem checkTensorInputs_iff (x : TensorMetadata) (B : Nat) :
    checkTensorInputs x B=.ok () ↔ InputContract x B := by
  rw [tensor_contract_iff]
  simp [checkTensorInputs,VeriTile.Torch.TensorMetadata.isCuda,ite_eq_iff]

theorem checkInputs_iff (x : ArgumentMetadata) (b : BlockArgument) :
    checkInputs x b=.ok () ↔ AdmittedArguments x b := by
  cases x <;> cases b <;> simp [checkInputs,AdmittedArguments,ite_eq_iff,checkTensorInputs_iff]

theorem block_positive (B : Nat) (h : B∈supportedBlocks) : 0<B := by
  simp [supportedBlocks] at h
  rcases h with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> decide

/-- The admitted block divisors round at most to 2^31, so masked tail lanes
also remain signed-int32 representable. -/
theorem index_bound (n B pid lane : Nat) (hn : n<2^31) (hb : B∈supportedBlocks)
    (hp : pid<grid n B) (hl : lane<B) : pid*B+lane<2^31 := by
  simp [supportedBlocks] at hb
  rcases hb with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> unfold grid at hp <;> omega

theorem nat32_signed (n : Nat) (h : n<2^31) : (BitVec.ofNat 32 n).toInt=(n : Int) :=
  BitVec.toInt_ofInt_eq_self (w:=32) (n:=(n : Int)) (by decide) (by omega) (by norm_num; omega)

theorem native_indices (n B pid lane : Nat) (hn : n<2^31) (hb : B∈supportedBlocks)
    (hp : pid<grid n B) (hl : lane<B) :
    (TritonMul2Finite.offset32 B pid lane).toInt=(pid*B+lane : Nat) ∧
    TritonMul2Finite.active32 n B pid lane=decide (pid*B+lane<n) := by
  have eq : TritonMul2Finite.offset32 B pid lane=BitVec.ofNat 32 (pid*B+lane) := by
    simp [TritonMul2Finite.offset32,BitVec.ofNat_mul_ofNat,BitVec.ofNat_add_ofNat]
  have exactIndex := nat32_signed _ (index_bound n B pid lane hn hb hp hl)
  rw [←eq] at exactIndex
  refine ⟨exactIndex,?_⟩
  simp only [TritonMul2Finite.active32,exactIndex,nat32_signed n hn,Int.ofNat_lt]

#axiomsClean checkInputs_iff
#axiomsClean native_indices
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.TritonMul2
