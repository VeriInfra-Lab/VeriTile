import bench.tritonbench_g.rowwise_quantization_triton.Proofs.StoreIO

namespace VeriTile.Bench.Repairs.RowwiseQuantizationTriton.FPEquiv
open VeriTile.Triton FP.Structural
open IntegerProgramIO
open scoped VeriTile.Spec
noncomputable section
variable {α : Type} [Inhabited α]

/-- Original full grid with the existing input/INT8/FP16 typed IO. -/
def original (ops : Operations α) (half : α → α) (round : α → BitVec 64)
    (rest : Algebra α) (p : Launch) (schedule : Schedule p) :=
  Implementation.original
    (IO.quantizerIO (KernelProfile.algebra ops half p.tree rest) round (KernelProfile.equal ops) p schedule)
    (fun _ => True)

/-- Same actual row permutation, with the FP16 maximum store moved before
quantization to shorten its live range. All numerical operations are retained. -/
def earlyMaximum (ops : Operations α) (half : α → α) (round : α → BitVec 64)
    (rest : Algebra α) (p : Launch) (schedule : Schedule p) :=
  Implementation.fromVariant (fun _ => True)
    (StoreSchedule.variant (KernelProfile.algebra ops half p.tree rest) round (KernelProfile.equal ops) p schedule)

/-- Exact store scheduling equivalence over the full original grid and every
input word. Both source schedules are safe and succeed, both typed output
channels agree and both preserve all exterior cells. No numerical rule is used;
llrint, INT8 low-byte representation and the FP16 maximum cast are unchanged. -/
specification rowwise_grid_equiv (ops : Operations α) (half : α → α)
    (round : α → BitVec 64) (rest : Algebra α) (p : Launch) (schedule : Schedule p) :
    original ops half round rest p schedule ≡[[]] earlyMaximum ops half round rest p schedule := by
  exact Implementation.equiv_of_equivalent
    (StoreSchedule.variant (KernelProfile.algebra ops half p.tree rest) round (KernelProfile.equal ops) p schedule)
    (StoreSchedule.equivalent ops half round rest p schedule)

#axiomsClean rowwise_grid_equiv
end
end VeriTile.Bench.Repairs.RowwiseQuantizationTriton.FPEquiv
