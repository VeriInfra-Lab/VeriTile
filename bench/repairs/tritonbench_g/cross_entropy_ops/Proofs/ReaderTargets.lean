import bench.repairs.tritonbench_g.cross_entropy_ops.Proofs.RealIO

namespace VeriTile.Bench.Repairs.CrossEntropyOps.LogicalProof
open VeriTile.Triton FP.Structural Interface
open scoped BigOperators
noncomputable section

def forwardFormula (p : Parameters) (values : Forward.Inputs (WithBot ℝ) p) : Forward.Outputs (WithBot ℝ) p :=
  let x := Forward.realInput values
  let label := Forward.label values
  let L := Real.log (∑ j ∈ Finset.range p.cols, Real.exp (x j*p.logitScale))
  Forward.result p (L : WithBot ℝ)
    ((if label = p.ignored then 0 else
      L - (1-p.smoothing) * (x label.toNat*p.logitScale) -
        p.smoothing * (∑ j ∈ Finset.range p.cols, x j*p.logitScale) / p.cols +
        p.squareScale*L*L : ℝ) : WithBot ℝ)
    ((if label = p.ignored then 0 else p.squareScale*L*L : ℝ) : WithBot ℝ)

/-- The split result retains its masked block range, local label ownership,
and zero z-loss observation; the source does not write the z-loss channel. -/
def splitFormula (p : Parameters) (block : Nat) (values : Forward.Inputs (WithBot ℝ) p) : Forward.Outputs (WithBot ℝ) p :=
  let x := fun j => Forward.realInput values j * p.logitScale
  let label := Forward.label values
  Forward.result p
    ((Real.log (∑ i ∈ Finset.range p.block,
      if block*p.block+i < p.cols then Real.exp (x (block*p.block+i)) else 0) : ℝ) : WithBot ℝ)
    ((if label = p.ignored then 0 else
      -p.smoothing * (∑ i ∈ Finset.range p.block,
        if block*p.block+i < p.cols then x (block*p.block+i) else 0) / p.totalClasses -
      if (block*p.block : Int) ≤ label-p.classStart ∧
          label-p.classStart < (min p.cols ((block+1)*p.block) : Nat) then
        (1-p.smoothing) * x (label-p.classStart).toNat else 0 : ℝ) : WithBot ℝ) (0 : ℝ)

/-- Backward uses the supplied saved LSE and upstream, without a premise that
they came from a correct forward run. The formula remains independent of source. -/
def backwardFormula (p : Parameters) (block : Nat) (values : Backward.Inputs (WithBot ℝ) p) : Backward.Outputs (WithBot ℝ) p :=
  Backward.result p (fun i =>
    let col := block * p.block + i.val
    let target := Backward.label values
    let L := (Backward.lse values).getD 0
    let g := (Backward.upstream values).getD 0
    ((if target = p.ignored then 0 else (g * p.logitScale) *
      ((1 + 2 * p.squareScale * L) * Real.exp (Backward.realInput values col * p.logitScale - L) -
        (if (col : Int) = target - p.classStart then 1 - p.smoothing else 0) -
        p.smoothing / p.totalClasses) : ℝ) : WithBot ℝ))

end
end VeriTile.Bench.Repairs.CrossEntropyOps.LogicalProof
