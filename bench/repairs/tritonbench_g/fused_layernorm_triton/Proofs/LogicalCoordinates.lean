import bench.repairs.tritonbench_g.fused_layernorm_triton.IO
import bench.repairs.tritonbench_g.fused_layernorm_triton.Proofs.IOResults

namespace VeriTile.Bench.Repairs.FusedLayernormTriton.PublicIO
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.FusedLayernormTriton FPStages Whole
noncomputable section

theorem logical_iff (rest : Algebra ℝ) (X W Bias Y Mean Rstd : RegionName)
    (rows D : Nat) (c : Contract.Configuration)
    (f : (Nat → Nat → ℝ) → (Nat → ℝ) → (Nat → ℝ) → Nat → ℝ×ℝ×(Nat → ℝ)) :
    (logicalIO rest X W Bias Y Mean Rstd rows D c).Implements (logicalExpected rows D f) ↔
    (io (ideal rest) realEqual id X W Bias Y Mean Rstd rows D c).Implements
      (expected X W Bias Y Mean Rstd rows D f) := by
  have eq : (logicalIO rest X W Bias Y Mean Rstd rows D c).lift (logicalExpected rows D f) =
      expected X W Bias Y Mean Rstd rows D f := by
    funext xs o j
    fin_cases o <;> rfl
  change (io (ideal rest) realEqual id X W Bias Y Mean Rstd rows D c).Implements _ ↔ _
  rw [eq]

theorem logical_real (rest : Algebra ℝ) (X W Bias Y Mean Rstd : RegionName)
    (rows D : Nat) (positive : 0<D) (sep : Separation X W Bias Y Mean Rstd)
    (c : Contract.Configuration) (complete : c.tree.Complete) :
    (logicalIO rest X W Bias Y Mean Rstd rows D c).Implements
      (logicalExpected rows D (fun xs ws bs row =>
        let mean := ((List.range D).map (xs row)).sum / D
        let variance := ((List.range D).map (fun col => (xs row col-mean)^2)).sum / D
        let inverse := 1 / Real.sqrt (variance + (1e-5 : ℝ))
        (mean,inverse,fun col => (xs row col-mean)*inverse*ws col+bs col))) :=
  (logical_iff rest X W Bias Y Mean Rstd rows D c _).mpr
    (IOResults.real rest X W Bias Y Mean Rstd rows D positive sep c complete)
end
end VeriTile.Bench.Repairs.FusedLayernormTriton.PublicIO
