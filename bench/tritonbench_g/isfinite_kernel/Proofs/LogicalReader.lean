import VeriTile.Triton.Float.LogicalProgramIO
import bench.tritonbench_g.isfinite_kernel.IO
import bench.tritonbench_g.isfinite_kernel.Proofs.IO
import bench.tritonbench_g.isfinite_kernel.Proofs.Storage
import VeriTile.Meta.StatementAudit

/-! Independent mathematical Real classification for both source precisions.
`classificationIO` in IO.lean displays the exact source, typed channels,
signed byte-address windows and masks. Every mathematical Real is finite.
The separate all-word IEEE exponent-field result is in `Misc/WordCorrect.lean`.
-/
namespace VeriTile.Bench.Repairs.IsfiniteKernel.LogicalReader
open VeriTile.Triton FP.Structural
open scoped ProgramIO
noncomputable section

/-- In the ideal Real interpretation, the exact source accepts every active
mathematical Real in this single masked tile invocation and preserves
all cells outside its output window. The output-address condition is the same
explicit condition required by the established strided store proof. -/
theorem isfinite_real_correctness (precision : Precision) (input output : View)
    (n B pid : Nat)
    (injective : ∀ i,i<n → ∀ j,j<n → output.address i=output.address j → i=j) :
    Spec.Real (classificationIO precision RealStorage.reference input output n B pid ⊨
      fun _ o => Fin.cases (fun _ => Bool.true) (fun z => Fin.elim0 z) o) := by
  have target : classification precision RealStorage.reference input output n B pid =
      (fun _ o => Fin.cases (fun _ => Bool.true) (fun z => Fin.elim0 z) o) := by
    funext xs o
    have ho : o=(0 : Fin 1) := Fin.eq_zero o
    subst o
    rfl
  rw [← target]
  exact IOProof.implements precision RealStorage.reference input output n B pid injective

end
end VeriTile.Bench.Repairs.IsfiniteKernel.LogicalReader

namespace VeriTile.Bench.Repairs.IsfiniteKernel.LogicalReader
open VeriTile.Triton FP.Structural
open scoped LogicalProgramIO
noncomputable section

theorem full_judgment_iff (precision : Precision) (input output : View) (n B pid : Nat) :
    (logicalIO precision RealStorage.reference input output n B pid ⊨ fun _ _ => Bool.true) ↔
      (classificationIO precision RealStorage.reference input output n B pid).Implements
        (fun _ o => Fin.cases (fun _ => Bool.true) (fun z => Fin.elim0 z) o) := by
  have target : (logicalIO precision RealStorage.reference input output n B pid).lift (fun _ _ => Bool.true) =
      (fun _ o => Fin.cases (fun _ => Bool.true) (fun z => Fin.elim0 z) o) := by
    funext xs o; fin_cases o; funext j; rfl
  change (classificationIO precision RealStorage.reference input output n B pid).Implements _ ↔ _
  rw [target]
  rfl

end
end VeriTile.Bench.Repairs.IsfiniteKernel.LogicalReader
