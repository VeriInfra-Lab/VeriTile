import bench.tritonbench_g.int8_matmul_quantization.Proofs.LogicalReader
namespace VeriTile.Bench.Repairs.Int8MatmulQuantization.I32Safe.KernelOnly.Correct
open VeriTile.Triton FP.Structural KernelExecution
open KernelProducerGrid (Geometry)
open scoped LogicalProgramIO VeriTile.Triton.LogicalDomainIO BigOperators
noncomputable section
def rowMaximum (values : Nat → ℝ) : Nat → ℝ
  | 0 => 0
  | n+1 => max (rowMaximum values n) |values n|

def producerFormula (p : Geometry) (xs : Fin p.rows → Fin p.columns → ℝ) : PublicIO.Producer.LogicalOutputs p :=
  { codes := fun row col =>
      let x := xs row col / (rowMaximum (PublicIO.Producer.extendRow xs row) p.columns / 127)
      (BitVec.ofInt 8 (if x<0 then -Int.floor (-x) else Int.floor x)).toInt
    scales := fun row => rowMaximum (PublicIO.Producer.extendRow xs row) p.columns / 127 }

def gemmFormula (s : Shape) (xs : PublicIO.Gemm.LogicalInputs s) (row : Fin s.M) (col : Fin s.N) : ℝ :=
  (∑ k : Fin s.K, (((BitVec.ofInt 8 (xs.a row k)).toInt : ℝ) *
    ((BitVec.ofInt 8 (xs.b k col)).toInt : ℝ))) * xs.rowScale row * xs.columnScale col

/-- Complete producer stage with signed8 codes and FP16 row scales; all row orders and source frames. -/
specification producer_correct (p : Geometry) (selection : Producer.Selection p) (covers : p.columns≤p.width)
    (rest : Algebra ℝ) (order : Producer.Schedule p) :
    Spec.Real (PublicIO.Producer.logicalIO p selection rest order ⊨ producerFormula p) :=
  LogicalReader.producer_with_max p selection covers rest order rowMaximum (fun _ => rfl) (fun _ _ => rfl)

/-- Complete dense scaled GEMM, preserving the signed-byte domain, typed zero initialization,
all original precisions/layouts and independent per-cell split orders. -/
specification gemm_correct (tree : KernelScalarProfile.Tree) (rest : Algebra ℝ) (plans : Plans)
    (rowScale output : Precision) (s : Shape) (c : Config) (columnMajor : Bool) (q : Schedule s c) :
    Spec.Real (PublicIO.Gemm.logicalIO tree rest plans rowScale output s c columnMajor q
      (PublicIO.Gemm.initialValues (0:ℝ) s output) ⊨{PublicIO.Gemm.logicalSignedInputs s rowScale} gemmFormula s) :=
  (LogicalReader.gemm_full_iff tree rest plans rowScale output s c columnMajor q).mpr
    (LogicalReader.gemm_correct tree rest plans rowScale output s c columnMajor q)
#axiomsClean producer_correct
#axiomsClean gemm_correct

end
end VeriTile.Bench.Repairs.Int8MatmulQuantization.I32Safe.KernelOnly.Correct
