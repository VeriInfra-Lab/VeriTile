import bench.tritonbench_g.rms_norm_triton.IO
import bench.tritonbench_g.rms_norm_triton.Proofs.PublicIO

namespace VeriTile.Bench.Repairs.RmsNormTriton.LogicalIO
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.RmsNormTriton Whole
open scoped VeriTile.Triton.FP.Structural.LogicalProgramIO
noncomputable section

/-- Full previous grid judgment iff the formula on genuine matrix inputs. -/
theorem logical_iff (c : Layout) (eps : ℝ) (rows : List Nat) :
    (io c eps rows ⊨ fun values row col =>
      let xs := values.1
      let weights := values.2
      xs row col * (1 / Real.sqrt
        ((∑ k : Fin c.N, xs row k*xs row k)/(c.N:ℝ)+eps)) * weights col) ↔
    (GridPublic.io realAlgebra id (fun _ => id) c eps rows).Implements
      (GridPublic.target realAlgebra id (fun _ => id) c eps rows (realFormula c.N eps)) := by
  unfold LogicalProgramIO.Implements
  have target_eq : (io c eps rows).lift (fun values row col =>
      values.1 row col * (1 / Real.sqrt
        ((∑ k : Fin c.N, values.1 row k*values.1 row k)/(c.N:ℝ)+eps)) * values.2 col) =
      GridPublic.target realAlgebra id (fun _ => id) c eps rows (realFormula c.N eps) := by
    funext values channel
    fin_cases channel
    funext j
    rcases j with ⟨row,col,⟨⟩⟩
    let xs : Fin c.M → Fin c.N → ℝ := ((inputs c eps rows) values).1
    let ws : Fin c.N → ℝ := ((inputs c eps rows) values).2
    change xs row col * (1 / Real.sqrt
      ((∑ k : Fin c.N, xs row k*xs row k)/(c.N:ℝ)+eps)) *
        ws col = realFormula c.N eps
          (GridPublic.inputRow realAlgebra id (fun _ => id) c eps rows values row)
          (GridPublic.weights realAlgebra id (fun _ => id) c eps rows values) col.val
    simp [xs,ws,inputs,realFormula,GridPublic.inputRow,GridPublic.weights,Finset.sum_range]

  rw [target_eq]
  rfl

end
end VeriTile.Bench.Repairs.RmsNormTriton.LogicalIO
