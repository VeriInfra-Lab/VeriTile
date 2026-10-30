import bench.repairs.tritonbench_g.add_value.Contract
import bench.repairs.tritonbench_g.add_value.Kernels
import bench.tritonbench_g.vector_addition.Proofs.FiniteIndices

namespace VeriTile.Bench.Repairs.AddValue.Contiguous

theorem tensor_contract_iff (x : TensorMetadata) : InputContract x ↔
    x.layout = .strided ∧ x.dtype = .floating .fp32 ∧
    x.device.kind = "cuda" ∧ x.isContiguous = Bool.true ∧
    x.isNeg = Bool.false ∧ x.numel < 2^31 ∧
    (0 < x.numel → x.dataPtr.toNat % 4 = 0) := by
  constructor
  · intro h; exact ⟨h.layout,h.dtype,h.cuda,h.contiguous,h.noNeg,h.extent,h.alignment⟩
  · rintro ⟨a,b,c,d,e,f,g⟩; exact ⟨a,b,c,d,e,f,g⟩

theorem checkTensorInput_iff (x : TensorMetadata) :
    checkTensorInput x = .ok () ↔ InputContract x := by
  rw [tensor_contract_iff]
  simp [checkTensorInput, VeriTile.Torch.TensorMetadata.isCuda, ite_eq_iff]

theorem checkInput_iff (x : ArgumentMetadata) :
    checkInput x = .ok () ↔ AdmittedArgument x := by
  cases x <;> simp [checkInput, AdmittedArgument, checkTensorInput_iff]

theorem native_indices (x : TensorMetadata) (h : InputContract x)
    (pid lane : Nat) (hp : pid < gridCount x.numel) (hl : lane < 1024) :
    (VeriTile.Bench.TritonBenchG.VectorAddition.Finite.offset32 pid lane).toInt =
      (pid * 1024 + lane : Nat) ∧
    VeriTile.Bench.TritonBenchG.VectorAddition.Finite.mask32 x.numel pid lane =
      decide (pid * 1024 + lane < x.numel) :=
  (VeriTile.Bench.TritonBenchG.VectorAddition.Finite.source_index_arithmetic
    x.numel pid lane h.extent hp hl).2

#axiomsClean checkTensorInput_iff
#axiomsClean checkInput_iff
#axiomsClean native_indices
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.AddValue.Contiguous
