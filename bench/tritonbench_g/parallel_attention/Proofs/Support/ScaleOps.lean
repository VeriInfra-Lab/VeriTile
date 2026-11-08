import bench.tritonbench_g.parallel_attention.Proofs.Graph
import bench.tritonbench_g.parallel_attention.Proofs.Specification
namespace VeriTile.Bench.Repairs.ParallelAttentionKernel.Math
open VeriTile.Bench.Repairs.ParallelAttention
noncomputable section
/-- Exact Real arithmetic at an arbitrary incoming kernel scale. No equality
between the rounded ABI scalar and ideal inverse square root is assumed. -/
def scaleOps (σ : ℝ) : Ops ℝ := { realOps with scale := fun _ _ => σ }
variable {σ : ℝ}
theorem dot_scale_independent (p : DotPlan) (x y : Nat → ℝ) (carry : ℝ) :
    p.eval (scaleOps σ) x y carry = p.eval realOps x y carry := by
  induction p with
  | zero | acc | product => rfl
  | add l r hl hr => simpa only [DotPlan.eval, scaleOps, realOps] using congrArg₂ (· + ·) hl hr
  | fma i p ih => simpa only [DotPlan.eval, scaleOps, realOps] using congrArg (fun z => x i*y i+z) ih
theorem sum_scale_independent (p : SumPlan) (x : Nat → ℝ) :
    p.eval (scaleOps σ) x = p.eval realOps x := by
  induction p with
  | zero | input => rfl
  | add l r hl hr => simpa only [SumPlan.eval, scaleOps, realOps] using congrArg₂ (· + ·) hl hr
#axiomsClean dot_scale_independent
#axiomsClean sum_scale_independent
end
end VeriTile.Bench.Repairs.ParallelAttentionKernel.Math
