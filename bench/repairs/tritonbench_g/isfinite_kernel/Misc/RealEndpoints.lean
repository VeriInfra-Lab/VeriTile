import bench.repairs.tritonbench_g.isfinite_kernel.Kernels
import bench.repairs.tritonbench_g.isfinite_kernel.Proofs.Kernel
import bench.repairs.tritonbench_g.isfinite_kernel.Proofs.Storage
import VeriTile.Meta.StatementAudit

/-! Kernel-only classification, including every FP32/FP64 bit pattern.
`Spec.Real` denotes the independent mathematical bit formula here: it does not
erase NaNs or infinities into reals. The scalar primitive is interpreted by
its exponent-field meaning; native libdevice implementation is a model boundary.
The wrapper provides the checked input contract, not a host-proof obligation. -/
namespace VeriTile.Bench.Repairs.IsfiniteKernel.RealEndpoints
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000

/-- Complete strided classification on raw input/output views and initial
word memory. The only address premise is active output-address injectivity. -/
theorem isfinite_kernel_real (precision : Precision) (input output : View) (n B : Nat)
    (before : State (Bytes.Word precision.width))
    (injective : ∀ i, i<n → ∀ j, j<n → output.address i=output.address j → i=j) :
    Spec.Real (∃ after, FP.Structural.exec (Source.algebra precision (Bytes.reference precision))
        (source precision input output n B) before=some after ∧
      Kernel.Post (Bytes.reference precision) input output n B before after) := by
  obtain ⟨after,executed,memory⟩ := Source.executes precision (Bytes.reference precision) input output n B before
  refine ⟨after,executed,?_⟩
  have post := Kernel.stores_post (Bytes.reference precision) input output n B before injective
  simpa only [Kernel.Post,memory] using post

#specNonCircular Kernel.Post avoiding [source,indexCommuted]
#axiomsClean isfinite_kernel_real
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.IsfiniteKernel.RealEndpoints
