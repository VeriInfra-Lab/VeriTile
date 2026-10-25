import bench.repairs.tritonbench_g.rms_norm_triton.Proofs.ActualGrid
import bench.tritonbench_g.rms_norm_triton.Proofs.WholeMath
import VeriTile.Meta.Specification

namespace VeriTile.Bench.Repairs.RmsNormTriton.KernelReal
open VeriTile.Bench.TritonBenchG.RmsNormTriton
open VeriTile.Triton FP.Structural Whole
noncomputable section

/-- Complete RMSNorm grid on raw regional memory, in any complete launch
order. The independent mathematical formula and every exterior typed cell
are preserved, including an empty row grid. -/
theorem kernel_correct (c : Layout) (eps : ℝ)
    (xSeparate : c.X ≠ c.Y) (wSeparate : c.W ≠ c.Y)
    (rows : List Nat) (schedule : rows.Perm (List.range c.M)) (s : State ℝ) :
    Spec.Real (∃ final, ActualGrid.runRows realAlgebra c (ActualRow.argument realAlgebra eps) rows s = some final ∧
      (∀ row : Fin c.M, ∀ i : Fin c.N, final.mem c.Y (row.val*c.N+i.val) =
        Cell.mk .real (realFormula c.N eps
          (fun j => (s.mem c.X (row.val*c.N+j)).read .real)
          (fun j => (s.mem c.W j).read .real) i.val)) ∧
      (∀ region a, (region ≠ c.Y ∨ c.M*c.N ≤ a) → final.mem region a = s.mem region a)) := by
  simpa only [ActualRow.rowValue_argument,rowValue_real] using
    ActualGrid.grid_result realAlgebra c (ActualRow.argument realAlgebra eps) xSeparate wSeparate rows schedule s

#guard_msgs (drop info) in
#specNonCircular realFormula avoiding [kernel,rowValue,sourceRun,runRows]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RmsNormTriton.KernelReal
