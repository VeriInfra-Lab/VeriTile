import bench.tritonbench_g.cross_entropy1.Proofs.LogicalReader

namespace VeriTile.Bench.Repairs.CrossEntropy1.Correct
open VeriTile.Triton FP.Structural
open scoped BigOperators
open scoped VeriTile.Triton.FP.Structural.LogicalProgramIO
open CrossEntropy1.LogicalData
noncomputable section

/-- Canonical FP32 source with the actual scalar ABI, integer label channel,
logit row, two forward outputs and the original row/block addresses. -/
def forwardIO (p : Parameters) (row block : Nat) : ProgramIO ℝ (WithBot ℝ) :=
  IO.forward KernelReal.algebra p row block (KernelReal.arguments p) some RealIO.decode

def backwardIO (p : Parameters) (row block : Nat) : ProgramIO ℝ (WithBot ℝ) :=
  IO.backward KernelReal.algebra p row block (KernelReal.arguments p) some RealIO.decode

/-- Independent unsplit loss and LSE; the existing label-validity domain is
visible in the shared judgment, rather than a hidden memory condition. -/
def unsplitTarget (p : Parameters) (xs : Nat → ℝ) (label : Int) : ℝ × ℝ :=
  let L := Real.log (∑ j ∈ Finset.range p.cols, Real.exp (xs j))
  (L, if label = p.ignored then 0 else
    L - (1-p.smoothing) * xs label.toNat -
      p.smoothing * (∑ j ∈ Finset.range p.cols, xs j) / p.cols + p.squareScale * L * L)

/-- One original split block: masked block LSE and its local target/smoothing
contribution. Host aggregation is outside this kernel specification. -/
def splitTarget (p : Parameters) (block : Nat) (xs : Nat → ℝ) (label : Int) : ℝ × ℝ :=
  (Real.log (∑ i ∈ Finset.range p.block,
    if block*p.block+i < p.cols then Real.exp (xs (block*p.block+i)) else 0),
   if label = p.ignored then 0 else
    -p.smoothing * (∑ i ∈ Finset.range p.block,
      if block*p.block+i < p.cols then xs (block*p.block+i) else 0) / p.totalClasses -
    if (block*p.block : Int) ≤ label-p.classStart ∧
        label-p.classStart < (min p.cols ((block+1)*p.block) : Nat) then
      (1-p.smoothing) * xs (label-p.classStart).toNat else 0)

/-- Saved LSE and upstream are independent inputs to the backward kernel. -/
def backwardTarget (p : Parameters) (block : Nat) (xs : Nat → ℝ) (label : Int)
    (L upstream : ℝ) (lane : Fin p.block) : ℝ :=
  let col := block*p.block+lane.val
  if label = p.ignored then 0 else upstream *
    ((1+2*p.squareScale*L) * Real.exp (xs col-L) -
      (if (col : Int) = label-p.classStart then 1-p.smoothing else 0) -
      p.smoothing/p.totalClasses)

specification forward_unsplit_kernel (p : Parameters) (row : Nat) (hr : row<p.rows)
    (separate : p.lse≠p.logits) (outputs : p.lse≠p.loss)
    (positive : 0<p.cols) (covered : p.cols≤p.block) (splitFlag : p.split=Bool.false)
    (noSmooth : p.hasSmoothing=Bool.false → p.smoothing=0)
    (startZero : p.classStart=0) (total : p.totalClasses=p.cols) :
    Spec.Real ((LogicalIO.forward p row 0) ⊨{LogicalIO.unsplitDomain p}
      (fun xs => let result := unsplitTarget p (sample xs.logits) xs.label
        ; ({lse := result.1, loss := result.2} : ForwardOutputs ℝ))) :=
  LogicalProof.forward_unsplit p row hr separate outputs positive covered splitFlag noSmooth startZero total

/-- The complete original block contribution; no wrapper reduction is added. -/
specification forward_split_kernel (p : Parameters) (row block : Nat) (hr : row<p.rows)
    (separate : p.lse≠p.logits) (outputs : p.lse≠p.loss)
    (positive : 0<p.block) (start : block*p.block<p.cols) (splitFlag : p.split=Bool.true)
    (noSmooth : p.hasSmoothing=Bool.false → p.smoothing=0) :
    Spec.Real ((LogicalIO.forward p row block) ⊨
      (fun xs => let result := splitTarget p block (sample xs.logits) xs.label
        ; ({lse := result.1, loss := result.2} : ForwardOutputs ℝ))) :=
  LogicalProof.forward_split p row block hr separate outputs positive start splitFlag noSmooth

/-- Ignored labels suppress the upstream load. Every active gradient output
and the exterior frame are retained, including in-place regional storage. -/
specification backward_kernel (p : Parameters) (row block : Nat)
    (noSmooth : p.hasSmoothing=Bool.false → p.smoothing=0) :
    Spec.Real ((LogicalIO.backward p row block) ⊨
      (fun xs lane => backwardTarget p block (sample xs.logits) xs.label xs.savedLSE xs.upstream lane)) :=
  LogicalProof.backward p row block noSmooth

#specNonCircular unsplitTarget avoiding [forwardSyntax,backwardSyntax,FP.Structural.exec]
#specNonCircular splitTarget avoiding [forwardSyntax,backwardSyntax,FP.Structural.exec]
#specNonCircular backwardTarget avoiding [forwardSyntax,backwardSyntax,FP.Structural.exec]
#axiomsClean forward_unsplit_kernel
#axiomsClean forward_split_kernel
#axiomsClean backward_kernel
#guard_msgs (drop info) in
#auditModuleAxioms
#stmtSurfaceSubset forward_unsplit_kernel ⊆
  [VeriTile.Bench.Repairs.CrossEntropy1.Correct.unsplitTarget,
   VeriTile.Bench.Repairs.CrossEntropy1.LogicalData.ForwardInputs,
   VeriTile.Bench.Repairs.CrossEntropy1.LogicalData.ForwardInputs.label,
   VeriTile.Bench.Repairs.CrossEntropy1.LogicalData.ForwardInputs.logits,
   VeriTile.Bench.Repairs.CrossEntropy1.LogicalData.ForwardOutputs,
   VeriTile.Bench.Repairs.CrossEntropy1.LogicalData.ForwardOutputs.mk,
   VeriTile.Bench.Repairs.CrossEntropy1.LogicalData.sample,
   VeriTile.Bench.Repairs.CrossEntropy1.LogicalIO.forward,
   VeriTile.Bench.Repairs.CrossEntropy1.LogicalIO.unsplitDomain,
   VeriTile.Bench.Repairs.CrossEntropy1.Parameters,
   VeriTile.Bench.Repairs.CrossEntropy1.Parameters.block,
   VeriTile.Bench.Repairs.CrossEntropy1.Parameters.classStart,
   VeriTile.Bench.Repairs.CrossEntropy1.Parameters.cols,
   VeriTile.Bench.Repairs.CrossEntropy1.Parameters.hasSmoothing,
   VeriTile.Bench.Repairs.CrossEntropy1.Parameters.logits,
   VeriTile.Bench.Repairs.CrossEntropy1.Parameters.loss,
   VeriTile.Bench.Repairs.CrossEntropy1.Parameters.lse,
   VeriTile.Bench.Repairs.CrossEntropy1.Parameters.rows,
   VeriTile.Bench.Repairs.CrossEntropy1.Parameters.smoothing,
   VeriTile.Bench.Repairs.CrossEntropy1.Parameters.split,
   VeriTile.Bench.Repairs.CrossEntropy1.Parameters.totalClasses,
   VeriTile.Spec.Real,
   VeriTile.Triton.FP.Structural.LogicalProgramIO.ImplementsOn,
   VeriTile.Triton.RegionName]
#stmtSurfaceSubset forward_split_kernel ⊆
  [VeriTile.Bench.Repairs.CrossEntropy1.Correct.splitTarget,
   VeriTile.Bench.Repairs.CrossEntropy1.LogicalData.ForwardInputs,
   VeriTile.Bench.Repairs.CrossEntropy1.LogicalData.ForwardInputs.label,
   VeriTile.Bench.Repairs.CrossEntropy1.LogicalData.ForwardInputs.logits,
   VeriTile.Bench.Repairs.CrossEntropy1.LogicalData.ForwardOutputs,
   VeriTile.Bench.Repairs.CrossEntropy1.LogicalData.ForwardOutputs.mk,
   VeriTile.Bench.Repairs.CrossEntropy1.LogicalData.sample,
   VeriTile.Bench.Repairs.CrossEntropy1.LogicalIO.forward,
   VeriTile.Bench.Repairs.CrossEntropy1.Parameters,
   VeriTile.Bench.Repairs.CrossEntropy1.Parameters.block,
   VeriTile.Bench.Repairs.CrossEntropy1.Parameters.cols,
   VeriTile.Bench.Repairs.CrossEntropy1.Parameters.hasSmoothing,
   VeriTile.Bench.Repairs.CrossEntropy1.Parameters.logits,
   VeriTile.Bench.Repairs.CrossEntropy1.Parameters.loss,
   VeriTile.Bench.Repairs.CrossEntropy1.Parameters.lse,
   VeriTile.Bench.Repairs.CrossEntropy1.Parameters.rows,
   VeriTile.Bench.Repairs.CrossEntropy1.Parameters.smoothing,
   VeriTile.Bench.Repairs.CrossEntropy1.Parameters.split,
   VeriTile.Spec.Real,
   VeriTile.Triton.FP.Structural.LogicalProgramIO.Implements,
   VeriTile.Triton.RegionName]
#stmtSurfaceSubset backward_kernel ⊆
  [VeriTile.Bench.Repairs.CrossEntropy1.Correct.backwardTarget,
   VeriTile.Bench.Repairs.CrossEntropy1.LogicalData.BackwardInputs,
   VeriTile.Bench.Repairs.CrossEntropy1.LogicalData.BackwardInputs.savedLSE,
   VeriTile.Bench.Repairs.CrossEntropy1.LogicalData.BackwardInputs.toForwardInputs,
   VeriTile.Bench.Repairs.CrossEntropy1.LogicalData.BackwardInputs.upstream,
   VeriTile.Bench.Repairs.CrossEntropy1.LogicalData.ForwardInputs.label,
   VeriTile.Bench.Repairs.CrossEntropy1.LogicalData.ForwardInputs.logits,
   VeriTile.Bench.Repairs.CrossEntropy1.LogicalData.sample,
   VeriTile.Bench.Repairs.CrossEntropy1.LogicalIO.backward,
   VeriTile.Bench.Repairs.CrossEntropy1.Parameters,
   VeriTile.Bench.Repairs.CrossEntropy1.Parameters.block,
   VeriTile.Bench.Repairs.CrossEntropy1.Parameters.cols,
   VeriTile.Bench.Repairs.CrossEntropy1.Parameters.hasSmoothing,
   VeriTile.Bench.Repairs.CrossEntropy1.Parameters.smoothing,
   VeriTile.Spec.Real,
   VeriTile.Triton.FP.Structural.LogicalProgramIO.Implements]

end
end VeriTile.Bench.Repairs.CrossEntropy1.Correct
