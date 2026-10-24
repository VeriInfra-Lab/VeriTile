import bench.repairs.tritonbench_g.isfinite_kernel.Proofs.IO
import bench.repairs.tritonbench_g.isfinite_kernel.Proofs.Storage
import VeriTile.Meta.StatementAudit

/-! Retained independent all-word IEEE exponent-field classification. This
bit-domain result is separate from the primary mathematical Real specification. -/
namespace VeriTile.Bench.Repairs.IsfiniteKernel.Correct
open VeriTile.Triton FP.Structural
noncomputable section

/-- The exact source classifies every active logical input word and preserves
all cells outside its output window. The output-address condition is the same
explicit condition required by the established strided store proof. -/
theorem isfinite_kernel_real (precision : Precision) (input output : View)
    (n B pid : Nat)
    (injective : ∀ i,i<n → ∀ j,j<n → output.address i=output.address j → i=j) :
    Spec.Real ((classificationIO precision (Bytes.reference precision) input output n B pid).Implements
      (classification precision (Bytes.reference precision) input output n B pid)) :=
  IOProof.implements precision (Bytes.reference precision) input output n B pid injective

#specNonCircular Bytes.reference avoiding [source,indexCommuted,classificationIO]
#axiomsClean isfinite_kernel_real
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.IsfiniteKernel.Correct
