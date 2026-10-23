import bench.repairs.tritonbench_g.rmsnorm_fused.Proofs.IOInterface
import bench.repairs.tritonbench_g.rmsnorm_fused.Proofs.KernelNative

namespace VeriTile.Bench.Repairs.RmsnormFused.Ordered
open VeriTile Triton FP.Structural Interface
open VeriTile.Bench.TritonBenchG.RmsNormTriton.Observed (Term graphAlgebra)
noncomputable section

theorem io_native (shape : Shape) (valid : shape.Valid) (I : Instructions)
    (P : Policy) (R : Reductions) (epsilon : Word) (rows : List Nat)
    (all : rows.Perm shape.grid) :
    (programIO graphAlgebra shape (.input epsilon) rows Term.input
      (fun i => interpret I P R (i%shape.width))).Implements
      (fun vals => Interface.result (fun i => reference I P R shape.width shape.block epsilon
        (fun j => inputAt vals ((i.val/shape.width)*shape.width+j)) (weightAt vals) (i.val%shape.width))) := by
  apply Interface.implements graphAlgebra shape valid (.input epsilon) rows all Term.input
    (fun i => interpret I P R (i%shape.width))
  intro vals initial hx hw
  obtain ⟨final,ran,stored,frame⟩ := KernelInterface.complete_typed graphAlgebra shape valid (.input epsilon) rows all initial
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
  change Cell.mk .real (interpret I P R col
    (Runtime.value graphAlgebra initial "X" "W" row shape.width shape.block (.input epsilon) col))=_
  let bits : RegionName → Nat → Word := fun r j => if r="X" then inputAt vals j else weightAt vals j
  let canonical : State (Term Word) := {initial with mem := fun r j => .mk .real (.input (bits r j))}
  have same : Runtime.value graphAlgebra initial "X" "W" row shape.width shape.block (.input epsilon) col=
      Runtime.value graphAlgebra canonical "X" "W" row shape.width shape.block (.input epsilon) col := by
    apply Runtime.value_congr_active graphAlgebra initial canonical "X" "W" row shape.width shape.block col (.input epsilon) hc
    · intro j hj
      simpa [Runtime.read,canonical,bits] using hx (row*shape.width+j) (offset_bounded shape row j hr hj)
    · simpa [Runtime.read,canonical,bits] using hw col hc
  rw [same]
  have lowered := source_lowering I P R bits shape.width shape.block row col epsilon canonical
    (fun j => by simp [Runtime.read,canonical]) (fun j => by simp [Runtime.read,canonical]) hc
  simpa [bits] using congrArg (Cell.mk .real) lowered

#axiomsClean io_native
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RmsnormFused.Ordered
