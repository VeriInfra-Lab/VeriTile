import bench.tritonbench_g.rmsnorm_fused.Proofs.IOInterface
import bench.tritonbench_g.rmsnorm_fused.Proofs.LogicalInputs

namespace VeriTile.Bench.Repairs.RmsnormFused.Ideal
open VeriTile Triton FP.Structural Interface
open VeriTile.Bench.TritonBenchG.RmsNormTriton.Whole (realAlgebra)
noncomputable section

theorem io_correct (shape : Shape) (valid : shape.Valid) (epsilon : ℝ)
    (rows : List Nat) (all : rows.Perm shape.grid) :
    (programIO realAlgebra shape epsilon rows id (fun _ => id)).Implements
      (fun values => Interface.result (fun i => KernelLogical.formula shape.width epsilon
        (fun row col => inputAt values (row*shape.width+col)) (weightAt values)
        (i.val/shape.width) (i.val%shape.width))) := by
  apply Interface.implements realAlgebra shape valid epsilon rows all id (fun _ => id)
  intro vals initial hx hw
  obtain ⟨final,ran,stored,frame⟩ := KernelInterface.complete_typed realAlgebra shape valid epsilon rows all initial
  refine ⟨final,ran,?_,frame⟩
  intro i
  let row := i.val/shape.width
  let col := i.val%shape.width
  have hr : row<shape.rows := (Nat.div_lt_iff_lt_mul valid.1).mpr i.isLt
  have hc : col<shape.width := Nat.mod_lt _ valid.1
  have idx : row*shape.width+col=i.val := by
    simpa only [row,col,Nat.mul_comm] using Nat.div_add_mod i.val shape.width
  have stored' := stored row col hr hc
  rw [idx] at stored'
  rw [stored']
  change Cell.mk .real (Runtime.value realAlgebra initial "X" "W" row shape.width shape.block epsilon col)=_
  rw [value_real initial shape.width shape.block row col epsilon hc valid.2.1]
  have rowSum : (∑ j ∈ Finset.range shape.width, Runtime.read initial "X" (row*shape.width+j)^2)=
      ∑ j ∈ Finset.range shape.width, (inputAt vals (row*shape.width+j))^2 := by
    apply Finset.sum_congr rfl
    intro j hj
    rw [hx (row*shape.width+j) (offset_bounded shape row j hr (Finset.mem_range.mp hj))]
    rfl
  rw [rowSum,hx (row*shape.width+col) (offset_bounded shape row col hr hc),hw col hc]
  simp only [id_eq,KernelLogical.formula]
  rfl

#axiomsClean io_correct
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RmsnormFused.Ideal
