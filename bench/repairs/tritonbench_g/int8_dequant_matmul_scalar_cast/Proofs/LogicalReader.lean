import VeriTile.Triton.Float.LogicalProgramIO
import bench.repairs.tritonbench_g.int8_dequant_matmul_scalar_cast.IO
import bench.repairs.tritonbench_g.int8_dequant_matmul_scalar_cast.Proofs.PublicIO

namespace VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.LogicalReader
open VeriTile Triton FP.Structural
noncomputable section

/-- Six actual source regions, complete grouped tile/split grid and independent
per-address split orders. C is an in-place input only for split K. -/
abbrev io (s : Shape) (c : Config) (selected : Selection s c)
    (hasBias : Bool) (schedule : Schedule s c) :=
  IO.programIO (Quote.idealInteger s c selected) s c hasBias (1/16129) schedule

/-- The original split-K FP16 zero initialization, interpreted over Real. -/
abbrev inputDomain (s : Shape) (c : Config) := IOLayout.domain Quote.integerRealOps s c

/-- Independent dense dequantized matmul, optional bias, all output cells and
exterior memory. Source access safety and complete execution are conclusions
of the shared IO judgment. Real is the mathematical interpretation. -/
theorem kernel_real (s : Shape) (c : Config) (admitted : Source.Admitted c)
    (selected : Selection s c) (hasBias : Bool) (schedule : Schedule s c) :
    Spec.Real ((io s c selected hasBias schedule).ImplementsOnCells
      (inputDomain s c) (fun vals => Fin.cases (fun j =>
        let input := IOLayout.data realOps s vals
        input.columnScale j.2.1 * (input.rowScale j.1 *
          ((List.ofFn (fun k : Fin s.K => ((input.a j.1 k).toInt : ℝ) *
            ((input.b k j.2.1).toInt : ℝ))).sum / 16129)) +
          if hasBias then input.bias j.2.1 else 0)
        (fun i => Fin.elim0 i))) :=
  PublicIO.real s c selected hasBias schedule

end
end VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.LogicalReader

namespace VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.LogicalReader
open VeriTile Triton FP.Structural
open scoped VeriTile.Triton.LogicalDomainIO
open scoped LogicalProgramIO
noncomputable section

open scoped VeriTile.Triton.LogicalDomainIO
def logicalFormula (s : Shape) (hasBias : Bool) (xs : IO.LogicalInputs s) (row : Fin s.M) (col : Fin s.N) : ℝ :=
  let input := IO.mathematicalInput s xs
  input.columnScale col * (input.rowScale row *
    ((List.ofFn (fun k : Fin s.K => ((input.a row k).toInt : ℝ) * ((input.b k col).toInt : ℝ))).sum / 16129)) +
    if hasBias then input.bias col else 0

theorem full_judgment_iff (s : Shape) (c : Config) (selected : Selection s c) (hasBias : Bool) (schedule : Schedule s c) :
    (IO.logicalIO s c selected hasBias schedule ⊨{IO.logicalDomain s c} logicalFormula s hasBias) ↔
      (io s c selected hasBias schedule).ImplementsOnCells (inputDomain s c) (IOLayout.real s hasBias) := by
  have domain : (fun raw => IO.logicalDomain s c ((IO.logicalInputs s) raw)) = inputDomain s c := by
    funext raw
    simp only [IO.logicalDomain,Equiv.symm_apply_apply,inputDomain]
  have target : (IO.logicalIO s c selected hasBias schedule).lift (logicalFormula s hasBias) = IOLayout.real s hasBias := by
    funext xs o; fin_cases o; funext j; rfl
  change (io s c selected hasBias schedule).ImplementsOnCells
    (fun raw => IO.logicalDomain s c ((IO.logicalInputs s) raw))
    ((IO.logicalIO s c selected hasBias schedule).lift (logicalFormula s hasBias)) ↔ _
  simpa only [domain,target]

end
end VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.LogicalReader
