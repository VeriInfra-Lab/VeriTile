import bench.tritonbench_g.matmul_dequant_int4.Proofs.UnitStrideEquivalence

namespace VeriTile.Bench.Repairs.MatmulDequantInt4GridYChecked.FPEquiv
open VeriTile Triton FP.Structural
open scoped VeriTile.Spec
noncomputable section
variable {α : Type} [Inhabited α]

/-- Complete original selected-width dequantization and public grid. -/
def original (f : Source.Profile) (M : Algebra α) (p : Source.Parameters) :=
  ProgramIO.Implementation.original (PublicIO.dequantIO f M p) (fun _ => True)

/-- Actual contiguous-output specialization removes the column multiply;
typed FP operations and their source order stay identical for any backend. -/
def unitStride (f : Source.Profile) (M : Algebra α) (p : Source.Parameters) :=
  ProgramIO.Implementation.ofVariant (PublicIO.unit f M p) (fun _ => True)

specification unit_stride_equiv (f : Source.Profile) (M : Algebra α) (p : Source.Parameters)
    (valid : KernelGrid.Valid p) :
    original f M p ≡[([] : Spec.Assumptions _)] unitStride f M p :=
  ProgramIO.Implementation.of_equivalent _ (UnitStrideEquivalence.equivalent f M p valid)

#axiomsClean unit_stride_equiv
end
end VeriTile.Bench.Repairs.MatmulDequantInt4GridYChecked.FPEquiv
