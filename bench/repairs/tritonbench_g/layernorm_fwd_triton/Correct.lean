import bench.repairs.tritonbench_g.layernorm_fwd_triton.Proofs.LogicalIO

namespace VeriTile.Bench.Repairs.LayernormFwdTriton.Correctness
open VeriTile.Triton FP.Structural Address
open scoped VeriTile.Triton.FP.Structural.LogicalProgramIO
noncomputable section

/-- Every row of the complete sequence/head grid implements independent
population normalization. All three source loops, arbitrary strides, runtime
epsilon, empty axes, typed FP32 output and exact exterior frame remain. -/
specification layernorm_fwd_triton_kernel_real (shape : Shape) (st : Strides)
    (injective : Function.Injective (fun q : Coordinate shape => yOffset st q.1 q.2.val))
    (epsilon : ℝ) :
    Spec.Real (LogicalIO.io Mathematics.ideal shape st epsilon ⊨ fun a seq head col =>
      let mean := (∑ j : Fin shape.columns, a.x seq head j)/(shape.columns:ℝ)
      let variance := (∑ j : Fin shape.columns, (a.x seq head j-mean)^2)/(shape.columns:ℝ)
      ((a.x seq head col-mean)*(Real.sqrt (variance+epsilon))⁻¹)*a.weight seq head col) :=
  LogicalIO.implements shape st injective epsilon

#axiomsClean layernorm_fwd_triton_kernel_real
end
end VeriTile.Bench.Repairs.LayernormFwdTriton.Correctness
