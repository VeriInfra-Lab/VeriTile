import bench.repairs.tritonbench_g.cross_entropy2.Proofs.LogicalReader

namespace VeriTile.Bench.Repairs.CrossEntropy2.Correct
open VeriTile.Triton FP.Structural
open scoped BigOperators
open scoped VeriTile.Triton.FP.Structural.LogicalProgramIO
open CrossEntropy1.LogicalData
noncomputable section

/-- Runtime FP32 scalar ABI and exact forward row/block source, with one
logical logit row, an integer label and three separately located outputs. -/
def forwardIO (p : Parameters) (row block : Nat) : ProgramIO ℝ (WithBot ℝ) :=
  IO.forward KernelReal.algebra p row block (KernelReal.arguments p) some RealIO.decode

/-- Backward channels include saved LSE and upstream gradient. The actual
ignored-label upstream mask and masked column stores are visible in IO. -/
def backwardIO (p : Parameters) (row block : Nat) : ProgramIO ℝ (WithBot ℝ) :=
  IO.backward KernelReal.algebra p row block (KernelReal.arguments p) some RealIO.decode

/-- Independent local cross-entropy contribution, LSE and z-loss. In split
mode this is the original block contribution; no wrapper reduction is added. -/
def forwardTarget (p : Parameters) (block : Nat) (xs : Nat → ℝ) (label : Int) : ℝ × ℝ × ℝ :=
  let x := fun j => xs j * p.logitScale
  let s := if p.hasSmoothing then p.smoothing else 0
  let L := Real.log (∑ i ∈ Finset.range p.block,
    if block*p.block+i < p.cols then Real.exp (x (block*p.block+i)) else 0)
  let sum := ∑ i ∈ Finset.range p.block,
    if block*p.block+i < p.cols then x (block*p.block+i) else 0
  (L,
   if label = p.ignored then 0 else
    (if p.split then 0 else L) - s*sum/p.totalClasses -
    (if (block*p.block : Int) ≤ label-p.classStart ∧
        label-p.classStart < (min p.cols ((block+1)*p.block) : Nat) then
      (1-s) * x (label-p.classStart).toNat else 0) +
    (if p.split then 0 else p.squareScale*L*L),
   if label = p.ignored then 0 else if p.split then 0 else p.squareScale*L*L)

def backwardTarget (p : Parameters) (block : Nat) (xs : Nat → ℝ) (label : Int)
    (L upstream : ℝ) (lane : Fin p.block) : ℝ :=
  let col := block*p.block+lane.val
  let s := if p.hasSmoothing then p.smoothing else 0
  if label = p.ignored then 0 else (upstream*p.logitScale) *
    ((1+2*p.squareScale*L) * Real.exp (xs col*p.logitScale-L) -
      (if p.classStart+(col : Int) = label then 1-s else 0) - s/p.totalClasses)

specification forward (p : Parameters) (row block : Nat) (hr : row<p.rows)
    (positive : 0<p.block) (start : block*p.block<p.cols) (separate : p.lse≠p.logits)
    (outputs : p.lse≠p.loss ∧ p.lse≠p.zLoss ∧ p.loss≠p.zLoss) :
    Spec.Real ((LogicalIO.forward p row block) ⊨
      (fun xs => let result := forwardTarget p block (sample xs.logits) xs.label
        ; ({lse := result.1, loss := result.2.1, zLoss := result.2.2} : ForwardOutputsZ ℝ))) :=
  LogicalProof.forward p row block hr positive start separate outputs

specification backward (p : Parameters) (row block : Nat) (hr : row<p.rows) :
    Spec.Real ((LogicalIO.backward p row block) ⊨
      (fun xs lane => backwardTarget p block (sample xs.logits) xs.label xs.savedLSE xs.upstream lane)) :=
  LogicalProof.backward p row block hr

#axiomsClean forward
#axiomsClean backward
#specNonCircular forwardTarget avoiding [forwardSyntax,backwardSyntax,FP.Structural.exec]
#specNonCircular backwardTarget avoiding [forwardSyntax,backwardSyntax,FP.Structural.exec]

end
end VeriTile.Bench.Repairs.CrossEntropy2.Correct
