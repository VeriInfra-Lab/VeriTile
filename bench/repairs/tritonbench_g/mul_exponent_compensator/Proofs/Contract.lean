import bench.repairs.tritonbench_g.mul_exponent_compensator.Contract
import bench.repairs.tritonbench_g.mul_exponent_compensator.Kernels
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.MulExponentCompensator.Contiguous
theorem block_positive (B : Nat) (h : B∈supportedBlocks) : 0<B := by
  obtain ⟨k,hk,rfl⟩ := List.mem_map.mp h
  exact Nat.pow_pos (by decide)
theorem block_bound (B : Nat) (h : B∈supportedBlocks) : B≤2^20 := by
  obtain ⟨k,hk,rfl⟩ := List.mem_map.mp h
  have hk := List.mem_range.mp hk
  exact Nat.pow_le_pow_right (by decide) (by omega)

theorem contract_iff (x : TensorMetadata) (current B : Nat) : InputContract x current B ↔
    x.shape≠[] ∧ B∈supportedBlocks ∧ x.layout=.strided ∧ x.dtype=.floating .fp32 ∧
    x.device.kind="cuda" ∧ x.isContiguous=Bool.true ∧ x.isNeg=Bool.false ∧
    (x.numel+B-1)/B≤2^31-1 ∧ (0<x.numel → x.device=⟨"cuda",current⟩) ∧
    (0<x.numel → x.dataPtr.toNat%4=0) := by
  constructor
  · intro h;exact ⟨h.rank,h.block,h.layout,h.dtype,h.cuda,h.contiguous,h.noNeg,h.grid,h.device,h.alignment⟩
  · rintro ⟨a,b,c,d,e,f,g,h,i,j⟩;exact ⟨a,b,c,d,e,f,g,h,i,j⟩

theorem checkTensorInputs_iff (x : TensorMetadata) (current B : Nat) :
    checkTensorInputs x current B=.ok () ↔ InputContract x current B := by
  rw [contract_iff]
  simp [checkTensorInputs,VeriTile.Torch.TensorMetadata.isCuda,ite_eq_iff]

theorem checkInputs_iff (x : ArgumentMetadata) (current : Nat) (b : BlockArgument) :
    checkInputs x current b=.ok () ↔ AdmittedArguments x current b := by
  cases x with
  | nonTensor => cases b <;> simp [checkInputs,AdmittedArguments]
  | tensor x =>
    cases b with
    | nonInteger => simp [checkInputs,AdmittedArguments];split <;> simp
    | integer b =>
      by_cases rank : x.shape=[]
      · simp [checkInputs,rank,AdmittedArguments,contract_iff]
      · simp [checkInputs,rank,AdmittedArguments,checkTensorInputs_iff,ite_eq_iff,not_lt]

#axiomsClean checkInputs_iff
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.MulExponentCompensator.Contiguous
