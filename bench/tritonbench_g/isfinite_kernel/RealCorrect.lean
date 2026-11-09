import bench.tritonbench_g.isfinite_kernel.Proofs.LogicalReader

namespace VeriTile.Bench.Repairs.IsfiniteKernel.Correct
open VeriTile.Triton FP.Structural
open scoped LogicalProgramIO
noncomputable section


/-- One masked tile invocation; every mathematical Real is finite. -/
specification isfinite_real_correctness (precision : Precision) (input output : View) (n B pid : Nat)
    (injective : ∀ i,i<n → ∀ j,j<n → output.address i=output.address j → i=j) :
    Spec.Real (logicalIO precision RealStorage.reference input output n B pid ⊨ fun _ _ => Bool.true) :=
  (LogicalReader.full_judgment_iff precision input output n B pid).mpr
    (LogicalReader.isfinite_real_correctness precision input output n B pid injective)
#axiomsClean isfinite_real_correctness

end
end VeriTile.Bench.Repairs.IsfiniteKernel.Correct
