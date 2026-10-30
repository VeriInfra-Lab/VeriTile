import bench.repairs.tritonbench_g.dequantize_rowwise.Correct
import bench.repairs.tritonbench_g.dequantize_rowwise.Proofs.InputBinding
import bench.repairs.tritonbench_g.dequantize_rowwise.Proofs.NativeIndex

/-! Optional checked Tensor applications and input-binding correspondence.
The primary mathematical/native kernel entries do not import this module. -/
namespace VeriTile.Bench.Repairs.DequantizeRowwise.TensorBridge
open VeriTile VeriTile.Triton
open VeriTile.Bench.TritonBenchG.DequantizeRowwise
noncomputable section

/-- The kernel formula on the admitted caller tensors, including nonzero
storage offsets and read-only aliases. This proves no wrapper execution.
`interpret` is the Real specification's interpretation of scale words; it
does not claim exact native floating arithmetic or restrict admitted words. -/
theorem tensor_kernel_correct (c : InputBinding.Request)
    (heap : InputBinding.Heap) (interpret : VeriTile.Torch.FP32.Word → ℝ)
    (scratch : BlockState) (row : scratch.pid < rows c.x.metadata)
    (positive : 0 < columns c.x.metadata) :
    Spec.Real (∃ v,
      exec (kernel (parameters c.x.metadata)).eraseDType.toAlgKernel
        (InputBinding.initial heap interpret c scratch) = some v ∧
      (∀ column, column < columns c.x.metadata →
        v.readMem outRegion (scratch.pid * columns c.x.metadata + column) =
          KernelCorrectness.dequantized
            (interpret (VeriTile.Torch.FP32.logicalValue (InputBinding.readScale heap)
              c.scale c.scaleLaws scratch.pid))
            (InputBinding.Int8.logicalValue heap c.x (scratch.pid * columns c.x.metadata + column))) ∧
      (∀ r o, (r ≠ outRegion ∨ ∀ column, column < columns c.x.metadata →
        o ≠ scratch.pid * columns c.x.metadata + column) →
        v.mem r o = (InputBinding.initial heap interpret c scratch).mem r o)) := by
  let u := InputBinding.initial heap interpret c scratch
  obtain ⟨v, he, ho, hf⟩ := KernelCorrectness.correct (parameters c.x.metadata) u
  have nonempty : 0 < c.x.metadata.numel := by
    rw [numel_eq _ _ c.contract]
    exact Nat.mul_pos (Nat.zero_lt_of_lt row) positive
  have scaleBound := lt_of_lt_of_le row (c.contract.scales nonempty)
  refine ⟨v, he, ?_, hf⟩
  intro column hc
  have hi : scratch.pid * columns c.x.metadata + column < c.x.metadata.numel := by
    rw [numel_eq _ _ c.contract]
    have step := Nat.mul_le_mul_right (columns c.x.metadata) (Nat.succ_le_of_lt row)
    simp only [Nat.succ_mul] at step
    omega
  have read := ho column hc
  change v.readMem outRegion (scratch.pid * columns c.x.metadata + column) =
    KernelCorrectness.dequantized
      ((InputBinding.initial heap interpret c scratch).readMem scaleRegion scratch.pid)
      ((InputBinding.initial heap interpret c scratch).readMemValue .int xRegion
        (scratch.pid * columns c.x.metadata + column)) at read
  rw [InputBinding.input_read heap interpret c scratch _ hi,
    InputBinding.scale_read heap interpret c scratch _ scaleBound] at read
  exact read

theorem python_checks_iff_preconditions (x s : ArgumentMetadata) :
    checkInputs x s = .ok () ↔ AdmittedArguments x s := checkInputs_iff x s

abbrev original_input_binding := InputBinding.Request
abbrev int8_logical_read := InputBinding.input_read
abbrev fp32_logical_read := InputBinding.scale_read
abbrev native_lane_addresses := NativeIndex.lane_exact

#axiomsClean tensor_kernel_correct
#axiomsClean python_checks_iff_preconditions
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.DequantizeRowwise.TensorBridge
