import bench.tritonbench_g.vector_addition_custom.Contract
import bench.tritonbench_g.vector_addition_custom.Kernels
import bench.tritonbench_g.vector_addition_custom.Proofs.Mathematical_NativeArithmetic

namespace VeriTile.Bench.Repairs.VectorAdditionCustom.Contiguous

/-- Named fields describe exactly the concrete supported metadata conditions. -/
theorem tensor_contract_iff (x y : TensorMetadata) :
    InputContract x y ↔
      (x.layout = .strided ∧ y.layout = .strided) ∧
      (x.dtype = .floating .fp32 ∧ y.dtype = .floating .fp32) ∧
      (x.device.kind = "cuda" ∧ y.device.kind = "cuda") ∧
      x.device = y.device ∧ x.shape = y.shape ∧ x.shape.length = 1 ∧
      (x.isContiguous = Bool.true ∧ y.isContiguous = Bool.true) ∧
      (x.isNeg = Bool.false ∧ y.isNeg = Bool.false) ∧ x.numel < 2^31 ∧
      (0 < x.numel → x.dataPtr.toNat % 4 = 0 ∧ y.dataPtr.toNat % 4 = 0) := by
  constructor
  · intro h
    exact ⟨h.layout, h.dtype, h.cuda, h.sameDevice, h.sameShape, h.vector, h.contiguous, h.noNeg,
      h.extent, h.alignment⟩
  · rintro ⟨hl, hd, hc, hv, hs, hr, ht, hn, he, ha⟩
    exact ⟨hl, hd, hc, hv, hs, hr, ht, hn, he, ha⟩

/-- The attribute checks accept exactly the named input contract. -/
theorem checkTensorInputs_iff (x y : TensorMetadata) :
    checkTensorInputs x y = .ok () ↔ InputContract x y := by
  rw [tensor_contract_iff]
  simp [checkTensorInputs, VeriTile.Torch.TensorMetadata.isCuda, ite_eq_iff]

/-- Complete dynamic admission, including rejection before attribute access. -/
theorem checkInputs_iff (x y : ArgumentMetadata) :
    checkInputs x y = .ok () ↔ AdmittedArguments x y := by
  cases x <;> cases y <;> simp [checkInputs, AdmittedArguments, checkTensorInputs_iff]

/-- Typed callers use the same dynamic checks through the Tensor constructor. -/
theorem check_tensor_arguments_iff (x y : TensorMetadata) :
    checkInputs x y = .ok () ↔ InputContract x y := checkTensorInputs_iff x y

theorem same_numel (x y : TensorMetadata) (h : InputContract x y) :
    x.numel = y.numel := congrArg List.prod h.sameShape

theorem bounded_inputs (x y : TensorMetadata) (h : InputContract x y) :
    x.numel < 2^31 ∧ y.numel < 2^31 :=
  ⟨h.extent, (same_numel x y h) ▸ h.extent⟩

/-- Signed-int32 offsets and masks agree with natural-number indexing. -/
theorem native_indices (x y : TensorMetadata) (h : InputContract x y)
    (pid lane : Nat) (hp : pid < gridCount x.numel) (hl : lane < 16) :
    (VeriTile.Bench.TritonBenchG.VectorAdditionCustom.Native.offset pid lane).toInt =
      (pid * 16 + lane : Nat) ∧
    VeriTile.Bench.TritonBenchG.VectorAdditionCustom.Native.mask x.numel pid lane =
      decide (pid * 16 + lane < x.numel) := by
  exact VeriTile.Bench.TritonBenchG.VectorAdditionCustom.Native.bounded_arithmetic
    x.numel pid lane h.extent hp hl

#axiomsClean tensor_contract_iff
#axiomsClean checkTensorInputs_iff
#axiomsClean checkInputs_iff
#axiomsClean bounded_inputs
#axiomsClean native_indices
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.VectorAdditionCustom.Contiguous
