import bench.tritonbench_g.int8_dequant_matmul.Proofs.LogicalReader

namespace VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.Correct
open VeriTile Triton FP.Structural
open scoped VeriTile.Triton.LogicalDomainIO
open scoped LogicalProgramIO
noncomputable section


/-- Full grouped tile/split grid, optional bias and the original typed split-K zero domain. -/
specification kernel_real (s : Shape) (c : Config) (admitted : Source.Admitted c) (selected : Selection s c)
    (hasBias : Bool) (schedule : Schedule s c) :
    Spec.Real (IO.logicalIO s c selected hasBias schedule ⊨{IO.logicalDomain s c} fun xs row col =>
      let input := IO.mathematicalInput s xs
      input.columnScale col * (input.rowScale row *
        ((List.ofFn (fun k : Fin s.K => ((input.a row k).toInt : ℝ) * ((input.b k col).toInt : ℝ))).sum / 16129)) +
        if hasBias then input.bias col else 0) :=
  (LogicalReader.full_judgment_iff s c selected hasBias schedule).mpr
    (LogicalReader.kernel_real s c admitted selected hasBias schedule)
#axiomsClean kernel_real

end
end VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.Correct
