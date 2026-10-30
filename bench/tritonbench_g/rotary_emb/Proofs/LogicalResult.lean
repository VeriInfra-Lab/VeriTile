import bench.tritonbench_g.rotary_emb.IO
import bench.tritonbench_g.rotary_emb.Proofs.PublicIO

namespace VeriTile.Bench.Repairs.RotaryEmb.RotaryIO
open VeriTile.Triton FP.Structural
open scoped VeriTile.Triton.FP.Structural.LogicalProgramIO
noncomputable section

/-- Full judgment transport for the four logical Q/K output halves. -/
theorem logical_rotation_iff (rest : Algebra ℝ) (shape : Shape) (pids : Nat → Nat)
    (f : RotationInputs ℝ shape → Bool → Bool → Lane shape → ℝ)
    (formula : f = fun xs key odd i =>
      let a := xs.payload key Bool.false i
      let b := xs.payload key Bool.true i
      let c := xs.coefficient Bool.false odd i
      let z := xs.coefficient Bool.true odd i
      if odd then a*z+b*c else a*c-b*z) :
    (logicalIO rest realOps shape pids ⊨ f) ↔
      (kernelIO rest realOps shape pids).Implements (rotation realOps shape) := by
  have connection : (logicalIO rest realOps shape pids).lift f = rotation realOps shape := by
    rw [formula]
    funext raw channel
    fin_cases channel <;> rfl
  change (kernelIO rest realOps shape pids).Implements _ ↔ _
  rw [connection]

#axiomsClean logical_rotation_iff
end
end VeriTile.Bench.Repairs.RotaryEmb.RotaryIO
