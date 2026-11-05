import bench.tritonbench_g.adam_update_triton.Proofs.LogicalReader
import bench.tritonbench_g.adam_update_triton.Proofs.IOReal

/-! Public Lion IO and its independent mathematical result. The actual source
uses runtime scalars and updates parameter and momentum in place. Source
execution, active-access safety and memory-frame assembly live in Proofs/. -/
namespace VeriTile.Bench.Repairs.AdamUpdateTriton.Correct
open VeriTile.Triton FP.Structural
open scoped VeriTile.Triton.FP.Structural.LogicalProgramIO
noncomputable section

/-- Parameter decay followed by the Lion sign step. -/
def parameterFormula (params : Scalars ℝ) (p g m : ℝ) : ℝ :=
  (1 - params.lr * params.wd) * p -
    params.lr * Real.sign (params.beta1 * m + (1 - params.beta1) * g)

/-- The next momentum, computed from the entry momentum and gradient. -/
def momentumFormula (params : Scalars ℝ) (g m : ℝ) : ℝ :=
  params.beta2 * m + (1 - params.beta2) * g

/-- Output-indexed form retained for the checked-Tensor application. -/
def formula (params : Scalars ℝ) (output : Output) (p g m : ℝ) : ℝ :=
  match output with
  | .parameter => parameterFormula params p g m
  | .momentum => momentumFormula params g m

/-- Complete kernel IO: active accesses are backed, both output arrays satisfy
independent formulas, and every cell outside the two masked stores is kept. -/
specification lion_kernel_correctness (params : Scalars ℝ) (total n B pid : Nat) :
    Spec.Real (LogicalIO.io params total n B pid ⊨ fun xs => {
      parameter := fun j => parameterFormula params (xs.parameter j) (xs.gradient j) (xs.momentum j)
      momentum := fun j => momentumFormula params (xs.gradient j) (xs.momentum j) }) :=
  LogicalReader.correct params total n B pid _ rfl

#specNonCircular parameterFormula avoiding [source, Tile.scalar, Tile.sourceRun]
#specNonCircular momentumFormula avoiding [source, Tile.scalar, Tile.sourceRun]
#axiomsClean lion_kernel_correctness
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AdamUpdateTriton.Correct
