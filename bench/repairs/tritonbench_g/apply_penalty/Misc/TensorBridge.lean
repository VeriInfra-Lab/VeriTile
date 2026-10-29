import bench.repairs.tritonbench_g.apply_penalty.Correct
import bench.repairs.tritonbench_g.apply_penalty.Proofs.Contract

/-! Optional checked Tensor/metadata applications of the shared penalty kernel. -/
namespace VeriTile.Bench.Repairs.ApplyPenalty.TensorBridge
open VeriTile.Triton
noncomputable section

/-- The checked tensor/int32 contract supplies the bounds of the canonical
kernel. ReadOnlyBinding identifies metadata observations with kernel memory;
no host execution or final result is assumed. -/
theorem checked_kernel_trace_safe (a : Arguments) (h : InputContract a)
    (input : Inputs ℝ) (s : BlockState) (binding : RealProof.ReadOnlyBinding a input s)
    (b : Nat) (hb : b < a.batches) :
    Kernel.TraceSafe (RealProof.capacities a)
      (kernel "logits" "presence" "frequency" "repetition" "ids" "counts" "cumsum"
        a.vocab 1 a.block).toAlgKernel (RealProof.programState a s b) := by
  rw [Projection.algorithm]
  exact RealProof.source_traceSafe a h input s binding b hb

/-- The Python admission checks expose exactly the declared tensor contract. -/
theorem python_checks_iff (a : Arguments) : checkInputs a = .ok () ↔ InputContract a :=
  checkInputs_iff a

#axiomsClean checked_kernel_trace_safe
#axiomsClean python_checks_iff
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ApplyPenalty.TensorBridge
