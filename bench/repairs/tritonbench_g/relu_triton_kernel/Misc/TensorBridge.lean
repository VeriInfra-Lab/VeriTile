import bench.repairs.tritonbench_g.relu_triton_kernel.Correct
import bench.repairs.tritonbench_g.relu_triton_kernel.Proofs.Binding

/-! Tensor input checks and application of the mathematical kernel proof.
This optional interface is separate from the primary specifications in
`Correct.lean` and `FPEquiv.lean`; neither primary entry imports this file. -/
namespace VeriTile.Bench.Repairs.ReluTritonKernel.TensorBridge
open VeriTile.Triton
noncomputable section

/-- Public kernel result for the actual checked Tensor. Its observed pointer
supplies the input state, its numel supplies N, and shared Tensor laws derive
the logical input values. No input-read or output equation is assumed. -/
theorem relu_checked_kernel_correctness
    (x : VeriTile.Torch.FP32.Tensor) (laws : VeriTile.Torch.FP32.TensorLaws x)
    (h : InputContract x.metadata) (read : Nat → VeriTile.Torch.FP32.Word)
    (decode : VeriTile.Torch.FP32.Word → ℝ) (surrounding : BlockState) :
    let initial := Binding.realState x read decode surrounding
    Spec.Real (∃ final,
      exec (originalKernel x.metadata.numel).toAlgKernel initial = some final ∧
      (∀ i : Fin blockSize, surrounding.pid*blockSize+i.val < x.metadata.numel →
        final.readMem "out" (surrounding.pid*blockSize+i.val) =
          max 0 (decode (VeriTile.Torch.FP32.logicalValue read x laws
            (surrounding.pid*blockSize+i.val)))) ∧
      (∀ (r : RegionName) o, Binding.Outside x.metadata.numel surrounding.pid r o →
        final.mem r o = initial.mem r o)) := by
  exact KernelReal.region_run x.metadata.numel (Binding.realState x read decode surrounding)
    (fun i => decode (VeriTile.Torch.FP32.logicalValue read x laws
      (surrounding.pid*blockSize+i.val)))
    (fun i hi => Binding.realState_input x laws h read decode surrounding _ hi)

/-- The checked Python arguments admit exactly the public input contract. -/
theorem python_checks_iff_preconditions (x : ArgumentMetadata) :
    checkInputs x = .ok () ↔ AdmittedArguments x := checkInputs_iff x

#axiomsClean relu_checked_kernel_correctness
#axiomsClean python_checks_iff_preconditions
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ReluTritonKernel.TensorBridge
