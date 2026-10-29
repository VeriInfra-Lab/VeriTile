import bench.repairs.tritonbench_g.relu_triton_kernel.Contract
import bench.repairs.tritonbench_g.relu_triton_kernel.Kernels
import bench.tritonbench_g.relu_triton_kernel.Proofs.FiniteAddress

namespace VeriTile.Bench.Repairs.ReluTritonKernel

theorem tensor_contract_iff (x : TensorMetadata) : InputContract x ↔
    x.layout = .strided ∧ x.dtype = .floating .fp32 ∧ x.device.kind = "cuda" ∧
    x.isContiguous = Bool.true ∧ x.isNeg = Bool.false ∧ x.numel < 2^31 ∧
    (0 < x.numel → x.dataPtr.toNat % 4 = 0) := by
  constructor
  · intro h
    exact ⟨h.layout,h.dtype,h.cuda,h.contiguous,h.noNeg,h.extent,h.alignment⟩
  · rintro ⟨hl,hd,hc,ht,hn,he,ha⟩
    exact ⟨hl,hd,hc,ht,hn,he,ha⟩

theorem checkTensorInputs_iff (x : TensorMetadata) :
    checkTensorInputs x = .ok () ↔ InputContract x := by
  rw [tensor_contract_iff]
  simp [checkTensorInputs, VeriTile.Torch.TensorMetadata.isCuda, ite_eq_iff]

theorem checkInputs_iff (x : ArgumentMetadata) :
    checkInputs x = .ok () ↔ AdmittedArguments x := by
  cases x <;> simp [checkInputs,AdmittedArguments,checkTensorInputs_iff]

/-- All native multiply/add intermediates, including masked tail lanes, are
signed-int32 exact. This reuses the original source's index proof. -/
theorem native_indices (x : TensorMetadata) (h : InputContract x)
    (pid lane : Nat) (hp : pid < gridCount x.numel) (hl : lane < 1024) :
    let F := VeriTile.Bench.TritonBenchG.ReluTritonKernel.FiniteAddress.offset32 pid lane
    F.toInt = (pid*1024+lane : Nat) ∧
    VeriTile.Bench.TritonBenchG.ReluTritonKernel.FiniteAddress.mask32 x.numel pid lane =
      decide (pid*1024+lane < x.numel) := by
  exact ⟨(VeriTile.Bench.TritonBenchG.ReluTritonKernel.FiniteAddress.lane_arithmetic
    x.numel pid lane h.extent hp hl).2.1,
    VeriTile.Bench.TritonBenchG.ReluTritonKernel.FiniteAddress.mask_exact
      x.numel pid lane h.extent hp hl⟩

#axiomsClean checkInputs_iff
#axiomsClean native_indices
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.ReluTritonKernel
