import bench.tritonbench_g.layernorm_fwd_triton.IO
import bench.tritonbench_g.layernorm_fwd_triton.Proofs.PublicIO
import bench.tritonbench_g.layernorm_fwd_triton.Proofs.KernelMath

namespace VeriTile.Bench.Repairs.LayernormFwdTriton.LogicalIO
open VeriTile.Triton FP.Structural Address
open scoped VeriTile.Triton.FP.Structural.LogicalProgramIO
noncomputable section

theorem judgment_iff (shape : Shape) (st : Strides) (eps : ℝ) :
    (io Mathematics.ideal shape st eps ⊨ fun a seq head col =>
      let mean := (∑ j : Fin shape.columns, a.x seq head j)/(shape.columns:ℝ)
      let variance := (∑ j : Fin shape.columns, (a.x seq head j-mean)^2)/(shape.columns:ℝ)
      ((a.x seq head col-mean)*(Real.sqrt (variance+eps))⁻¹)*a.weight seq head col) ↔
    (KernelPublicIO.io Mathematics.ideal shape st eps).Implements
      (KernelPublicIO.target Mathematics.ideal shape st eps
        (fun x w j => Contract.Math.output shape.columns x w eps j)) := by
  change (KernelPublicIO.io Mathematics.ideal shape st eps).Implements _ ↔ _
  apply Iff.of_eq
  congr 1
  funext xs o j
  fin_cases o
  dsimp [LogicalProgramIO.lift,io,inputs,outputs]
  simp [io,inputs,outputs,LogicalProgramIO.lift,KernelPublicIO.target,KernelPublicIO.rowOf,
    KernelPublicIO.xRow,KernelPublicIO.wRow,Contract.Math.output,Contract.Math.mean,
    Contract.Math.variance,Contract.Math.inverse]
  rfl

theorem implements (shape : Shape) (st : Strides)
    (injective : Function.Injective (fun q : Coordinate shape => yOffset st q.1 q.2.val)) (eps : ℝ) :
    io Mathematics.ideal shape st eps ⊨ fun a seq head col =>
      let mean := (∑ j : Fin shape.columns, a.x seq head j)/(shape.columns:ℝ)
      let variance := (∑ j : Fin shape.columns, (a.x seq head j-mean)^2)/(shape.columns:ℝ)
      ((a.x seq head col-mean)*(Real.sqrt (variance+eps))⁻¹)*a.weight seq head col := by
  apply (judgment_iff shape st eps).mpr
  exact KernelPublicIO.implements Mathematics.ideal shape st eps injective
    (fun x w j => Contract.Math.output shape.columns x w eps j)
    (fun x w j h => Mathematics.output_real shape.columns x w eps j h)
end
end VeriTile.Bench.Repairs.LayernormFwdTriton.LogicalIO
