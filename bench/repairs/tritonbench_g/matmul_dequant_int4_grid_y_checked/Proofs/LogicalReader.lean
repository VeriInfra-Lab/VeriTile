import bench.repairs.tritonbench_g.matmul_dequant_int4_grid_y_checked.Proofs.ScalarIO
import bench.repairs.tritonbench_g.matmul_dequant_int4_grid_y_checked.IO
namespace VeriTile.Bench.Repairs.MatmulDequantInt4GridYChecked.ReaderProof
open VeriTile.Triton FP.Structural ReaderIO
open scoped LogicalProgramIO
noncomputable section

theorem target_eq (f : Source.Profile) (p : Source.Parameters) :
    (view f p).lift (fun xs row column =>
      ((Source.nibble (xs.weight row column) row.val -
        Source.nibble (xs.zero row column) column.val : Int) : ℝ) * xs.scale row column) = (fun xs => PublicIO.result f p (fun x => ((Source.nibble (PublicIO.packed xs Bool.false x) x.1.val - Source.nibble (PublicIO.packed xs Bool.true x) x.2.val : Int) : ℝ) * PublicIO.scale f xs x)) := by
  cases f <;> funext xs o j <;> fin_cases o <;> rfl

/-- Full original judgment equivalence at every logical coordinate, with the
selected profile, complete two-axis grid, bounds and exact frame unchanged. -/
theorem judgment_iff (f : Source.Profile) (p : Source.Parameters) :
    (view f p ⊨ fun xs row column =>
      ((Source.nibble (xs.weight row column) row.val -
        Source.nibble (xs.zero row column) column.val : Int) : ℝ) * xs.scale row column) ↔
      (PublicIO.dequantIO f Spec.real p).Implements (fun xs => PublicIO.result f p (fun x => ((Source.nibble (PublicIO.packed xs Bool.false x) x.1.val - Source.nibble (PublicIO.packed xs Bool.true x) x.2.val : Int) : ℝ) * PublicIO.scale f xs x)) := by
  rw [LogicalProgramIO.implements_iff,target_eq] <;> rfl

theorem correct (f : Source.Profile) (p : Source.Parameters) (valid : KernelGrid.Valid p) :
    (view f p ⊨ fun xs row column =>
      ((Source.nibble (xs.weight row column) row.val -
        Source.nibble (xs.zero row column) column.val : Int) : ℝ) * xs.scale row column) := by
  apply (judgment_iff f p).mpr
  have proved := PublicIO.implements f Spec.real p valid
  have same : (fun xs => PublicIO.result f p (PublicIO.scalarFormula f Spec.real p xs)) =
      (fun xs => PublicIO.result f p (fun x => ((Source.nibble (PublicIO.packed xs Bool.false x) x.1.val - Source.nibble (PublicIO.packed xs Bool.true x) x.2.val : Int) : ℝ) * PublicIO.scale f xs x)) := by
    funext xs
    apply congrArg (PublicIO.result f p)
    funext x
    exact PublicIO.scalar_real f p xs x
  rw [same] at proved
  exact proved
end
end VeriTile.Bench.Repairs.MatmulDequantInt4GridYChecked.ReaderProof
