import bench.repairs.tritonbench_g.matrix_vector_multip.IO
import bench.repairs.tritonbench_g.matrix_vector_multip.Proofs.RealResult

namespace VeriTile.Bench.Repairs.MatrixVectorMultip.LogicalIOProof
open VeriTile VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.MatrixVectorMultip
open scoped VeriTile.Triton.FP.Structural.LogicalProgramIO
noncomputable section

def target (N M BN pid : Nat) (av : (Fin N → Fin M → ℝ) × (Fin M → ℝ)) (row : Fin BN) : ℝ :=
  ∑ col : Fin M, IO.logicalRow av.1 (pid*BN+row.val) col * av.2 col

theorem headline_iff (N M BN BM pid : Nat)
    (rowShape : Random.ArangeSizeDomain BN) (columnShape : Random.ArangeSizeDomain BM) :
    (IO.logicalIO WholeMath.algebra N M BN BM pid rowShape columnShape ⊨ target N M BN pid) ↔
    (IO.io WholeMath.algebra N M BN BM pid rowShape columnShape id id).Implements
      (fun vals => IO.outputValues (fun i => KernelRealResult.matrixVector M
        (IO.matrixValues vals) (IO.vectorValues vals) (pid*BN+i.val))) := by
  have formula : (IO.logicalIO WholeMath.algebra N M BN BM pid rowShape columnShape).lift (target N M BN pid) =
      (fun vals => IO.outputValues (fun i => KernelRealResult.matrixVector M
        (IO.matrixValues vals) (IO.vectorValues vals) (pid*BN+i.val))) := by
    funext xs port i
    fin_cases port
    change (∑ col : Fin M, IO.logicalRow (IO.logicalInputs N M xs).1 (pid*BN+i.1.val) col *
      (IO.logicalInputs N M xs).2 col) = _
    apply Finset.sum_congr rfl
    intro col _
    have zero : (default : ℝ) = 0 := rfl
    by_cases active : pid*BN+i.1.val<N
    · simp [IO.logicalRow,IO.logicalInputs,IO.matrixValues,IO.vectorValues,active,col.isLt,zero]
    · simp [IO.logicalRow,IO.logicalInputs,IO.matrixValues,IO.vectorValues,active,col.isLt,zero]
  change (IO.io _ _ _ _ _ _ _ _ _ _).Implements _ ↔ _
  rw [formula]

theorem implements (N M BN BM pid : Nat) (width : 0<BM)
    (rowShape : Random.ArangeSizeDomain BN) (columnShape : Random.ArangeSizeDomain BM) (domain : SourceDomain N M BN BM pid) :
    (IO.logicalIO WholeMath.algebra N M BN BM pid rowShape columnShape ⊨ target N M BN pid) :=
  (headline_iff N M BN BM pid rowShape columnShape).2 (KernelRealResult.io_correct N M BN BM pid width rowShape columnShape domain)

#axiomsClean headline_iff
#axiomsClean implements
end
end VeriTile.Bench.Repairs.MatrixVectorMultip.LogicalIOProof
