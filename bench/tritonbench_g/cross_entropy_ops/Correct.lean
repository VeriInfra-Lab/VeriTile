import bench.tritonbench_g.cross_entropy_ops.Proofs.LogicalReader

namespace VeriTile.Bench.Repairs.CrossEntropyOps.Correct
open VeriTile.Triton FP.Structural Interface
open scoped BigOperators
open scoped VeriTile.Triton.FP.Structural.LogicalProgramIO
open CrossEntropy1.LogicalData
noncomputable section

/-- IO.lean displays the actual FP32 source, runtime scalar ABI, signed label,
logit row, three forward outputs and masked backward output window. -/
def forwardIO (p : Parameters) (row block : Nat) : ProgramIO (WithBot ℝ) (WithBot ℝ) :=
  Forward.programIO KernelReal.algebra p (KernelReal.arguments p) row block
def backwardIO (p : Parameters) (row block : Nat) : ProgramIO (WithBot ℝ) (WithBot ℝ) :=
  Backward.programIO KernelReal.algebra p (KernelReal.arguments p) row block

def forwardFormula (p : Parameters) (values : ForwardInputs (WithBot ℝ) p.cols) : ForwardOutputsZ (WithBot ℝ) :=
  let x := fun j => (sample values.logits j).getD 0
  let label := values.label
  let L := Real.log (∑ j ∈ Finset.range p.cols, Real.exp (x j*p.logitScale))
  { lse := (L : WithBot ℝ), loss := ((if label = p.ignored then 0 else
      L - (1-p.smoothing) * (x label.toNat*p.logitScale) -
        p.smoothing * (∑ j ∈ Finset.range p.cols, x j*p.logitScale) / p.cols +
        p.squareScale*L*L : ℝ) : WithBot ℝ), zLoss := ((if label = p.ignored then 0 else p.squareScale*L*L : ℝ) : WithBot ℝ) }

/-- The split result retains its masked block range, local label ownership,
and zero z-loss observation; the source does not write the z-loss channel. -/
def splitFormula (p : Parameters) (block : Nat) (values : ForwardInputs (WithBot ℝ) p.cols) : ForwardOutputsZ (WithBot ℝ) :=
  let x := fun j => (sample values.logits j).getD 0 * p.logitScale
  let label := values.label
  { lse := ((Real.log (∑ i ∈ Finset.range p.block,
      if block*p.block+i < p.cols then Real.exp (x (block*p.block+i)) else 0) : ℝ) : WithBot ℝ), loss := ((if label = p.ignored then 0 else
      -p.smoothing * (∑ i ∈ Finset.range p.block,
        if block*p.block+i < p.cols then x (block*p.block+i) else 0) / p.totalClasses -
      if (block*p.block : Int) ≤ label-p.classStart ∧
          label-p.classStart < (min p.cols ((block+1)*p.block) : Nat) then
        (1-p.smoothing) * x (label-p.classStart).toNat else 0 : ℝ) : WithBot ℝ), zLoss := (0 : ℝ) }

/-- Backward uses the supplied saved LSE and upstream, without a premise that
they came from a correct forward run. The formula remains independent of source. -/
def backwardFormula (p : Parameters) (block : Nat) (values : BackwardInputs (WithBot ℝ) p.cols) : Fin p.block → WithBot ℝ :=
  fun i =>
    let col := block * p.block + i.val
    let target := values.label
    let L := values.savedLSE.getD 0
    let g := values.upstream.getD 0
    ((if target = p.ignored then 0 else (g * p.logitScale) *
      ((1 + 2 * p.squareScale * L) * Real.exp ((sample values.logits col).getD 0 * p.logitScale - L) -
        (if (col : Int) = target - p.classStart then 1 - p.smoothing else 0) -
        p.smoothing / p.totalClasses) : ℝ) : WithBot ℝ)

specification forward_unsplit (p : Parameters) (row : Nat)
    (rowBound : row < p.rows) (positive : 0 < p.cols) (covered : p.cols ≤ p.block)
    (split : p.split = Bool.false) (noSmooth : p.hasSmoothing = Bool.false → p.smoothing = 0)
    (start : p.classStart = 0) (total : p.totalClasses = p.cols)
    (separate : p.lse ≠ p.logits) (outputs : p.lse ≠ p.loss)
    (zLse : p.zloss ≠ p.lse) (zLoss : p.zloss ≠ p.loss) :
    Spec.Real ((LogicalIO.forward p row 0) ⊨{fun values => LogicalIO.ForwardReal p values ∧
        (values.label = p.ignored ∨
          (0 ≤ values.label ∧ values.label < (p.cols : Int)))} (forwardFormula p)) :=
  LogicalProof.forward_unsplit p row rowBound positive covered split noSmooth start total separate outputs zLse zLoss

/-- Split mode returns its block LSE and partial loss, and frames all z-loss
cells. Host aggregation is outside this source's result. -/
specification forward_split (p : Parameters) (row block : Nat)
    (rowBound : row < p.rows) (positive : 0 < p.block) (active : block * p.block < p.cols)
    (split : p.split = Bool.true) (noSmooth : p.hasSmoothing = Bool.false → p.smoothing = 0)
    (separate : p.lse ≠ p.logits) (outputs : p.lse ≠ p.loss)
    (zLse : p.zloss ≠ p.lse) (zLoss : p.zloss ≠ p.loss) :
    Spec.Real ((LogicalIO.forward p row block) ⊨{LogicalIO.ForwardReal p} (splitFormula p block)) :=
  LogicalProof.forward_split p row block rowBound positive active split noSmooth separate outputs zLse zLoss

specification backward (p : Parameters) (row block : Nat) (rowBound : row < p.rows)
    (lseLogits : p.lse ≠ p.logits) (lseUpstream : p.lse ≠ p.upstream)
    (noSmooth : p.hasSmoothing = Bool.false → p.smoothing = 0) :
    Spec.Real ((LogicalIO.backward p row block) ⊨{LogicalIO.BackwardReal p} (backwardFormula p block)) :=
  LogicalProof.backward p row block rowBound lseLogits lseUpstream noSmooth

#axiomsClean forward_unsplit
#axiomsClean forward_split
#axiomsClean backward
#specNonCircular forwardFormula avoiding [forwardSyntax, backwardSyntax, exec, run]
#specNonCircular splitFormula avoiding [forwardSyntax, backwardSyntax, exec, run]
#specNonCircular backwardFormula avoiding [forwardSyntax, backwardSyntax, exec, run]

#stmtSurfaceSubset forward_unsplit ⊆
  [VeriTile.Bench.Repairs.CrossEntropy1.LogicalData.ForwardInputs,
   VeriTile.Bench.Repairs.CrossEntropy1.LogicalData.ForwardInputs.label,
   VeriTile.Bench.Repairs.CrossEntropy1.LogicalData.ForwardOutputsZ,
   VeriTile.Bench.Repairs.CrossEntropy1.Parameters.block,
   VeriTile.Bench.Repairs.CrossEntropy1.Parameters.classStart,
   VeriTile.Bench.Repairs.CrossEntropy1.Parameters.cols,
   VeriTile.Bench.Repairs.CrossEntropy1.Parameters.hasSmoothing,
   VeriTile.Bench.Repairs.CrossEntropy1.Parameters.ignored,
   VeriTile.Bench.Repairs.CrossEntropy1.Parameters.logits,
   VeriTile.Bench.Repairs.CrossEntropy1.Parameters.loss,
   VeriTile.Bench.Repairs.CrossEntropy1.Parameters.lse,
   VeriTile.Bench.Repairs.CrossEntropy1.Parameters.rows,
   VeriTile.Bench.Repairs.CrossEntropy1.Parameters.smoothing,
   VeriTile.Bench.Repairs.CrossEntropy1.Parameters.split,
   VeriTile.Bench.Repairs.CrossEntropy1.Parameters.totalClasses,
   VeriTile.Bench.Repairs.CrossEntropyOps.Correct.forwardFormula,
   VeriTile.Bench.Repairs.CrossEntropyOps.LogicalIO.ForwardReal,
   VeriTile.Bench.Repairs.CrossEntropyOps.LogicalIO.forward,
   VeriTile.Bench.Repairs.CrossEntropyOps.Parameters,
   VeriTile.Bench.Repairs.CrossEntropyOps.Parameters.toParameters,
   VeriTile.Bench.Repairs.CrossEntropyOps.Parameters.zloss,
   VeriTile.Spec.Real,
   VeriTile.Triton.FP.Structural.LogicalProgramIO.ImplementsOn,
   VeriTile.Triton.RegionName]
#stmtSurfaceSubset forward_split ⊆
  [VeriTile.Bench.Repairs.CrossEntropy1.LogicalData.ForwardInputs,
   VeriTile.Bench.Repairs.CrossEntropy1.LogicalData.ForwardOutputsZ,
   VeriTile.Bench.Repairs.CrossEntropy1.Parameters.block,
   VeriTile.Bench.Repairs.CrossEntropy1.Parameters.cols,
   VeriTile.Bench.Repairs.CrossEntropy1.Parameters.hasSmoothing,
   VeriTile.Bench.Repairs.CrossEntropy1.Parameters.logits,
   VeriTile.Bench.Repairs.CrossEntropy1.Parameters.loss,
   VeriTile.Bench.Repairs.CrossEntropy1.Parameters.lse,
   VeriTile.Bench.Repairs.CrossEntropy1.Parameters.rows,
   VeriTile.Bench.Repairs.CrossEntropy1.Parameters.smoothing,
   VeriTile.Bench.Repairs.CrossEntropy1.Parameters.split,
   VeriTile.Bench.Repairs.CrossEntropyOps.Correct.splitFormula,
   VeriTile.Bench.Repairs.CrossEntropyOps.LogicalIO.ForwardReal,
   VeriTile.Bench.Repairs.CrossEntropyOps.LogicalIO.forward,
   VeriTile.Bench.Repairs.CrossEntropyOps.Parameters,
   VeriTile.Bench.Repairs.CrossEntropyOps.Parameters.toParameters,
   VeriTile.Bench.Repairs.CrossEntropyOps.Parameters.zloss,
   VeriTile.Spec.Real,
   VeriTile.Triton.FP.Structural.LogicalProgramIO.ImplementsOn,
   VeriTile.Triton.RegionName]
#stmtSurfaceSubset backward ⊆
  [VeriTile.Bench.Repairs.CrossEntropy1.LogicalData.BackwardInputs,
   VeriTile.Bench.Repairs.CrossEntropy1.Parameters.block,
   VeriTile.Bench.Repairs.CrossEntropy1.Parameters.cols,
   VeriTile.Bench.Repairs.CrossEntropy1.Parameters.hasSmoothing,
   VeriTile.Bench.Repairs.CrossEntropy1.Parameters.logits,
   VeriTile.Bench.Repairs.CrossEntropy1.Parameters.lse,
   VeriTile.Bench.Repairs.CrossEntropy1.Parameters.rows,
   VeriTile.Bench.Repairs.CrossEntropy1.Parameters.smoothing,
   VeriTile.Bench.Repairs.CrossEntropy1.Parameters.upstream,
   VeriTile.Bench.Repairs.CrossEntropyOps.Correct.backwardFormula,
   VeriTile.Bench.Repairs.CrossEntropyOps.LogicalIO.BackwardReal,
   VeriTile.Bench.Repairs.CrossEntropyOps.LogicalIO.backward,
   VeriTile.Bench.Repairs.CrossEntropyOps.Parameters,
   VeriTile.Bench.Repairs.CrossEntropyOps.Parameters.toParameters,
   VeriTile.Spec.Real,
   VeriTile.Triton.FP.Structural.LogicalProgramIO.ImplementsOn,
   VeriTile.Triton.RegionName]

end
end VeriTile.Bench.Repairs.CrossEntropyOps.Correct
