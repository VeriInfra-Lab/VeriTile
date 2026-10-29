import bench.repairs.tritonbench_g.cross_entropy_ops.IO
import bench.repairs.tritonbench_g.cross_entropy_ops.Proofs.ReaderTargets
import bench.repairs.tritonbench_g.cross_entropy1.Proofs.LogicalTransport

namespace VeriTile.Bench.Repairs.CrossEntropyOps.LogicalProof
open VeriTile.Triton FP.Structural Interface
noncomputable section

theorem forward_unsplit (p : Parameters) (row : Nat)
    (rowBound : row < p.rows) (positive : 0 < p.cols) (covered : p.cols ≤ p.block)
    (split : p.split = Bool.false) (noSmooth : p.hasSmoothing = Bool.false → p.smoothing = 0)
    (start : p.classStart = 0) (total : p.totalClasses = p.cols)
    (separate : p.lse ≠ p.logits) (outputs : p.lse ≠ p.loss)
    (zLse : p.zloss ≠ p.lse) (zLoss : p.zloss ≠ p.loss)  :
    (LogicalIO.forward p row 0).ImplementsOn (fun xs => ((fun values => Forward.RealInputs p values ∧ (Forward.label values = p.ignored ∨ (0 ≤ Forward.label values ∧ Forward.label values < (p.cols : Int))))) ((LogicalIO.forward p row 0).inputs.symm xs))
      (fun xs => (LogicalIO.forward p row 0).outputs ((forwardFormula p) ((LogicalIO.forward p row 0).inputs.symm xs))) :=
  (CrossEntropy1.LogicalProof.reindex_iff (LogicalIO.forward p row 0) ((fun values => Forward.RealInputs p values ∧ (Forward.label values = p.ignored ∨ (0 ≤ Forward.label values ∧ Forward.label values < (p.cols : Int))))) (forwardFormula p)).mpr
    (Forward.real_unsplit p row rowBound positive covered split noSmooth start total separate outputs zLse zLoss)

theorem forward_split (p : Parameters) (row block : Nat)
    (rowBound : row < p.rows) (positive : 0 < p.block) (active : block * p.block < p.cols)
    (split : p.split = Bool.true) (noSmooth : p.hasSmoothing = Bool.false → p.smoothing = 0)
    (separate : p.lse ≠ p.logits) (outputs : p.lse ≠ p.loss)
    (zLse : p.zloss ≠ p.lse) (zLoss : p.zloss ≠ p.loss)  :
    (LogicalIO.forward p row block).ImplementsOn (fun xs => (Forward.RealInputs p) ((LogicalIO.forward p row block).inputs.symm xs))
      (fun xs => (LogicalIO.forward p row block).outputs ((splitFormula p block) ((LogicalIO.forward p row block).inputs.symm xs))) :=
  (CrossEntropy1.LogicalProof.reindex_iff (LogicalIO.forward p row block) (Forward.RealInputs p) (splitFormula p block)).mpr
    (Forward.real_split p row block rowBound positive active split noSmooth separate outputs zLse zLoss)

theorem backward (p : Parameters) (row block : Nat) (rowBound : row < p.rows)
    (lseLogits : p.lse ≠ p.logits) (lseUpstream : p.lse ≠ p.upstream)
    (noSmooth : p.hasSmoothing = Bool.false → p.smoothing = 0)  :
    (LogicalIO.backward p row block).ImplementsOn (fun xs => (Backward.RealInputs p) ((LogicalIO.backward p row block).inputs.symm xs))
      (fun xs => (LogicalIO.backward p row block).outputs ((backwardFormula p block) ((LogicalIO.backward p row block).inputs.symm xs))) :=
  (CrossEntropy1.LogicalProof.reindex_iff (LogicalIO.backward p row block) (Backward.RealInputs p) (backwardFormula p block)).mpr
    (Backward.real_implements p row block rowBound lseLogits lseUpstream noSmooth)

end
end VeriTile.Bench.Repairs.CrossEntropyOps.LogicalProof
