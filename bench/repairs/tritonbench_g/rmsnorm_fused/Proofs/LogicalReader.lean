import bench.repairs.tritonbench_g.rmsnorm_fused.Proofs.IOReal

namespace VeriTile.Bench.Repairs.RmsnormFused.Interface
open VeriTile Triton FP.Structural
open VeriTile.Bench.TritonBenchG.RmsNormTriton.Whole (realAlgebra)
open scoped VeriTile.Triton.FP.Structural.LogicalProgramIO
noncomputable section

/-- Logical input reads use exactly the original finite row-major cells. -/
theorem logicalInput_apply (shape : Shape) (values : Inputs ℝ shape)
    (row : Fin shape.rows) (col : Fin shape.width) :
    (logicalInputs shape values).1 row col = inputAt values (row.val*shape.width+col.val) := by
  have bound : row.val*shape.width+col.val<shape.rows*shape.width := by
    calc row.val*shape.width+col.val < (row.val+1)*shape.width := by nlinarith [col.isLt]
         _ ≤ shape.rows*shape.width := Nat.mul_le_mul_right _ row.isLt
  change input values (finProdFinEquiv (row,col)) = inputAt values (row.val*shape.width+col.val)
  rw [inputAt,dif_pos bound]
  congr 1
  apply Fin.ext
  simp [finProdFinEquiv,Nat.mul_comm,Nat.add_comm]


/-- Full previous judgment iff the independent formula on logical matrices. -/
theorem logical_iff (shape : Shape) (epsilon : ℝ) (rows : List Nat) :
    (logicalIO shape epsilon rows ⊨ fun values row col =>
      let xs := values.1
      let weights := values.2
      xs row col * (Real.sqrt
        ((∑ j : Fin shape.width, (xs row j)^2)/(shape.width:ℝ)+epsilon))⁻¹ * weights col) ↔
    (programIO realAlgebra shape epsilon rows id (fun _ => id)).Implements
      (fun values => Interface.result (fun i => KernelLogical.formula shape.width epsilon
        (fun row col => inputAt values (row*shape.width+col)) (weightAt values)
        (i.val/shape.width) (i.val%shape.width))) := by
  unfold LogicalProgramIO.Implements
  have target_eq : (logicalIO shape epsilon rows).lift (fun values row col =>
      values.1 row col * (Real.sqrt
        ((∑ j : Fin shape.width, (values.1 row j)^2)/(shape.width:ℝ)+epsilon))⁻¹ * values.2 col) =
      (fun values => Interface.result (fun i => KernelLogical.formula shape.width epsilon
        (fun row col => inputAt values (row*shape.width+col)) (weightAt values)
        (i.val/shape.width) (i.val%shape.width))) := by
    funext values channel
    fin_cases channel
    funext j
    rcases j with ⟨i,⟨⟩⟩
    change (logicalInputs shape values).1 i.divNat i.modNat * (Real.sqrt
      ((∑ j : Fin shape.width, ((logicalInputs shape values).1 i.divNat j)^2)/(shape.width:ℝ)+epsilon))⁻¹ *
        (logicalInputs shape values).2 i.modNat = _
    simp only [logicalInput_apply,KernelLogical.formula,result]
    rw [Finset.sum_range]
    congr 1
    change weight values i.modNat = weightAt values (i.val%shape.width)
    have positive : 0<shape.width := by nlinarith [i.isLt]
    rw [weightAt,dif_pos (Nat.mod_lt i.val positive)]
    apply congrArg (weight values)
    apply Fin.ext
    rfl

  rw [target_eq]
  rfl

end
end VeriTile.Bench.Repairs.RmsnormFused.Interface
