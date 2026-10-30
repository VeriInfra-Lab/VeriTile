import bench.repairs.tritonbench_g.matmul_dequant_int4_grid_y_checked.Proofs.LogicalReader
namespace VeriTile.Bench.Repairs.MatmulDequantInt4GridYChecked.Correct
open VeriTile.Triton FP.Structural
open scoped LogicalProgramIO
noncomputable section

/-- Independent nibble-difference dequantization over complete logical output
coordinates; original geometry, source precision and full frame are retained. -/
specification kernel_real (f : Source.Profile) (p : Source.Parameters)
    (valid : KernelGrid.Valid p) :
    VeriTile.Spec.Real (ReaderIO.view f p ⊨ fun xs row column =>
      ((Source.nibble (xs.weight row column) row.val -
        Source.nibble (xs.zero row column) column.val : Int) : ℝ) * xs.scale row column) :=
  ReaderProof.correct f p valid

#axiomsClean kernel_real
end
end VeriTile.Bench.Repairs.MatmulDequantInt4GridYChecked.Correct
